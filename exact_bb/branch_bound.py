"""
Algoritmo exacto Branch & Bound para el problema de rutas de drones.
Implementa la técnica de ramificación y poda del Tema 4.

El algoritmo explora el árbol de soluciones de forma sistemática,
podando ramas que no pueden mejorar la mejor solución encontrada.

Autor: David Valbuena Segura
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
import time

from common.graph import DroneGraph, Node, Edge
from common.pareto import Solution, get_pareto_front, dominates


@dataclass
class BBNode:
    """
    Nodo del árbol de Branch & Bound.
    
    Representa un estado parcial de la búsqueda:
    - path: camino parcial desde el hub
    - visited: conjunto de nodos visitados
    - lower_bound: cota inferior del coste
    - cost: coste acumulado hasta este punto
    """
    path: List[int]
    visited: Set[int]
    cost: Tuple[float, float, float]  # (distancia, riesgo, batería)
    lower_bound: Tuple[float, float, float]
    depth: int
    
    def __lt__(self, other):
        # Para la cola de prioridad: menor cota primero
        return sum(self.lower_bound) < sum(other.lower_bound)


class BranchAndBound:
    """
    Implementación de Branch & Bound para el problema TSP multiobjetivo.
    
    Basado en el Tema 4: Ramificación y Poda.
    
    Características:
    - Genera nodos siguiendo estrategia de mínimo coste (LC)
    - Utiliza cotas heurísticas para podar ramas no prometedoras
    - Verifica restricciones de zonas no-fly
    - Maneja múltiples objetivos mediante agregación ponderada
    """
    
    def __init__(self, graph: DroneGraph, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Args:
            graph: Grafo del problema
            weights: Pesos para la función escalarizada (distancia, riesgo, batería)
        """
        self.graph = graph
        self.weights = weights
        self.best_solutions: List[Solution] = []
        self.nodes_explored = 0
        self.nodes_pruned = 0
    
    def _weighted_cost(self, cost: Tuple[float, float, float]) -> float:
        """Calcula el coste escalarizado usando los pesos."""
        return sum(w * c for w, c in zip(self.weights, cost))
    
    def _calculate_lower_bound(self, node: BBNode) -> Tuple[float, float, float]:
        """
        Calcula una cota inferior para el coste restante.
        
        Utiliza la heurística de aristas mínimas: para cada nodo no visitado,
        suma el coste de la arista más barata que sale de él.
        
        Esta es una relajación del problema que garantiza ser <= al óptimo.
        """
        remaining = set(self.graph.destination_ids) - node.visited
        
        if not remaining:
            # Solo falta volver al hub
            last = node.path[-1]
            edge = self.graph.get_edge(last, self.graph.hub_id)
            if edge and edge.is_valid:
                return (node.cost[0] + edge.distance,
                       node.cost[1] + edge.risk,
                       node.cost[2] + edge.battery_consumption)
            return (float('inf'), float('inf'), float('inf'))
        
        # Cota: coste actual + suma de aristas mínimas desde cada nodo no visitado
        min_distances = 0.0
        min_risks = 0.0
        min_battery = 0.0
        
        # Arista mínima desde el último nodo del path
        last = node.path[-1]
        min_from_last = float('inf')
        for next_node in remaining:
            edge = self.graph.get_edge(last, next_node)
            if edge and edge.is_valid:
                min_from_last = min(min_from_last, edge.distance)
        
        if min_from_last == float('inf'):
            return (float('inf'), float('inf'), float('inf'))
        
        min_distances += min_from_last
        
        # Aristas mínimas desde cada nodo restante
        for current in remaining:
            min_out = float('inf')
            candidates = (remaining - {current}) | {self.graph.hub_id}
            
            for next_node in candidates:
                edge = self.graph.get_edge(current, next_node)
                if edge and edge.is_valid:
                    min_out = min(min_out, edge.distance)
            
            if min_out == float('inf'):
                return (float('inf'), float('inf'), float('inf'))
            
            min_distances += min_out
        
        # Para riesgo y batería usamos proporciones similares
        # (simplificación: asumimos proporcionalidad con distancia)
        ratio = min_distances / max(1, node.cost[0]) if node.cost[0] > 0 else 1
        min_risks = node.cost[1] * ratio * 0.5
        min_battery = node.cost[2] * ratio * 0.5
        
        return (node.cost[0] + min_distances,
               node.cost[1] + min_risks,
               node.cost[2] + min_battery)
    
    def _should_prune(self, node: BBNode, best_cost: float) -> bool:
        """
        Decide si podar una rama del árbol.
        
        Poda si la cota inferior del nodo es peor que la mejor solución encontrada.
        """
        lb_cost = self._weighted_cost(node.lower_bound)
        return lb_cost >= best_cost
    
    def solve(self, max_solutions: int = 10, time_limit: float = 60.0) -> List[Solution]:
        """
        Resuelve el problema usando Branch & Bound.
        
        Args:
            max_solutions: Número máximo de soluciones a encontrar
            time_limit: Tiempo límite en segundos
        
        Returns:
            Lista de soluciones encontradas (frente de Pareto aproximado)
        """
        start_time = time.time()
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_solutions = []
        best_cost = float('inf')
        
        # Nodo raíz: empezamos en el hub
        root = BBNode(
            path=[self.graph.hub_id],
            visited=set(),
            cost=(0.0, 0.0, 0.0),
            lower_bound=(0.0, 0.0, 0.0),
            depth=0
        )
        root.lower_bound = self._calculate_lower_bound(root)
        
        # Cola de prioridad (estrategia LC - menor cota primero)
        priority_queue = [root]
        heapq.heapify(priority_queue)
        
        while priority_queue and len(self.best_solutions) < max_solutions:
            # Verificar límite de tiempo
            if time.time() - start_time > time_limit:
                break
            
            # Extraer nodo con menor cota
            current = heapq.heappop(priority_queue)
            self.nodes_explored += 1
            
            # Poda: si la cota es peor que la mejor solución, descartar
            if self._should_prune(current, best_cost):
                self.nodes_pruned += 1
                continue
            
            # Verificar si es solución completa
            remaining = set(self.graph.destination_ids) - current.visited
            
            if not remaining:
                # Completar circuito: volver al hub
                last = current.path[-1]
                edge = self.graph.get_edge(last, self.graph.hub_id)
                
                if edge and edge.is_valid:
                    final_cost = (
                        current.cost[0] + edge.distance,
                        current.cost[1] + edge.risk,
                        current.cost[2] + edge.battery_consumption
                    )
                    
                    final_path = current.path + [self.graph.hub_id]
                    solution = Solution(final_path, final_cost)
                    
                    # Añadir a soluciones si no está dominada
                    if not any(dominates(s, solution) for s in self.best_solutions):
                        # Eliminar soluciones dominadas por la nueva
                        self.best_solutions = [
                            s for s in self.best_solutions 
                            if not dominates(solution, s)
                        ]
                        self.best_solutions.append(solution)
                        
                        # Actualizar mejor coste
                        sol_cost = self._weighted_cost(final_cost)
                        if sol_cost < best_cost:
                            best_cost = sol_cost
                
                continue
            
            # Expandir: generar nodos hijos
            last = current.path[-1]
            
            for next_node in remaining:
                edge = self.graph.get_edge(last, next_node)
                
                # Verificar si la arista es válida (no cruza zonas no-fly)
                if edge is None or not edge.is_valid:
                    continue
                
                # Crear nodo hijo
                new_cost = (
                    current.cost[0] + edge.distance,
                    current.cost[1] + edge.risk,
                    current.cost[2] + edge.battery_consumption
                )
                
                child = BBNode(
                    path=current.path + [next_node],
                    visited=current.visited | {next_node},
                    cost=new_cost,
                    lower_bound=(0.0, 0.0, 0.0),
                    depth=current.depth + 1
                )
                child.lower_bound = self._calculate_lower_bound(child)
                
                # Solo añadir si no se poda
                if not self._should_prune(child, best_cost):
                    heapq.heappush(priority_queue, child)
                else:
                    self.nodes_pruned += 1
        
        return self.best_solutions
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas de la ejecución."""
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'solutions_found': len(self.best_solutions),
            'pruning_ratio': self.nodes_pruned / max(1, self.nodes_explored + self.nodes_pruned)
        }


def solve(graph: DroneGraph, 
          weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
          max_solutions: int = 10,
          time_limit: float = 60.0) -> Tuple[List[Solution], dict]:
    """
    Función principal para resolver el problema con Branch & Bound.
    
    Args:
        graph: Grafo del problema
        weights: Pesos para escalarización de objetivos
        max_solutions: Número máximo de soluciones
        time_limit: Límite de tiempo en segundos
    
    Returns:
        Tupla (lista de soluciones, estadísticas)
    """
    solver = BranchAndBound(graph, weights)
    solutions = solver.solve(max_solutions, time_limit)
    stats = solver.get_statistics()
    
    return solutions, stats


# Pruebas del módulo
if __name__ == "__main__":
    from common.graph import DroneGraph, Node, Edge
    from common.geometry import distance
    
    print("=" * 60)
    print("PRUEBAS DEL ALGORITMO BRANCH & BOUND")
    print("=" * 60)
    
    # Crear grafo de prueba pequeño
    graph = DroneGraph()
    
    # Hub en el centro
    graph.nodes[0] = Node(0, 50, 50, is_hub=True)
    graph.hub_id = 0
    
    # 4 destinos
    graph.nodes[1] = Node(1, 20, 30)
    graph.nodes[2] = Node(2, 80, 30)
    graph.nodes[3] = Node(3, 80, 70)
    graph.nodes[4] = Node(4, 20, 70)
    
    # Zona no-fly en el centro
    graph.no_fly_zones.append([(45, 45), (55, 45), (55, 55), (45, 55)])
    
    # Crear aristas
    from common.geometry import segment_passes_through_polygon
    
    for i in graph.nodes:
        for j in graph.nodes:
            if i != j:
                node_i = graph.nodes[i]
                node_j = graph.nodes[j]
                dist = distance(node_i.position, node_j.position)
                
                is_valid = True
                for zone in graph.no_fly_zones:
                    if segment_passes_through_polygon(node_i.position, 
                                                      node_j.position, zone):
                        is_valid = False
                        break
                
                risk = 0.1 if is_valid else 1.0
                graph.edges[(i, j)] = Edge(i, j, dist, risk, dist/100, is_valid)
    
    print(f"\nGrafo creado:")
    print(f"  - Nodos: {graph.num_nodes}")
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")
    
    # Resolver con Branch & Bound
    print("\nEjecutando Branch & Bound...")
    start = time.time()
    solutions, stats = solve(graph, max_solutions=5, time_limit=30.0)
    elapsed = time.time() - start
    
    print(f"\nResultados:")
    print(f"  - Tiempo: {elapsed:.3f} s")
    print(f"  - Nodos explorados: {stats['nodes_explored']}")
    print(f"  - Nodos podados: {stats['nodes_pruned']}")
    print(f"  - Ratio de poda: {stats['pruning_ratio']:.2%}")
    print(f"  - Soluciones encontradas: {len(solutions)}")
    
    if solutions:
        print("\nMejores soluciones (frente de Pareto):")
        for i, sol in enumerate(solutions):
            print(f"  {i+1}. Ruta: {sol.route}")
            print(f"     Objetivos: dist={sol.distance:.2f}, "
                  f"risk={sol.risk:.2f}, bat={sol.recharges:.2f}")
    
    print("\n" + "=" * 60)
