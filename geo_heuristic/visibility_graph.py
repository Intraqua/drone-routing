"""
Algoritmo heurístico geométrico basado en grafo de visibilidad.
Implementa técnicas del Tema 5 y 7 para generar rutas factibles rápidas.

El algoritmo construye un grafo de visibilidad que conecta puntos
que tienen línea de visión directa (sin atravesar zonas no-fly),
y luego usa heurísticas voraces para construir rutas.

Autor: David Valbuena Segura
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import heapq
import time
import random

from common.graph import DroneGraph, Node, Edge
from common.geometry import (
    Point, Polygon, distance, 
    segment_passes_through_polygon,
    segments_intersect
)
from common.pareto import Solution, get_pareto_front, dominates


@dataclass
class VisibilityEdge:
    """Arista en el grafo de visibilidad."""
    from_node: int
    to_node: int
    distance: float
    risk: float
    battery: float


class VisibilityGraph:
    """
    Grafo de visibilidad para navegación evitando obstáculos.
    
    Del Tema 7: El grafo de visibilidad conecta puntos que tienen
    línea de visión directa, es decir, el segmento que los une
    no intersecta ningún obstáculo.
    """
    
    def __init__(self, drone_graph: DroneGraph):
        self.drone_graph = drone_graph
        self.edges: Dict[Tuple[int, int], VisibilityEdge] = {}
        self._build_visibility_graph()
    
    def _build_visibility_graph(self):
        """
        Construye el grafo de visibilidad.
        
        Para cada par de nodos, verifica si hay línea de visión directa
        (no intersecta zonas no-fly) y añade la arista correspondiente.
        """
        nodes = list(self.drone_graph.nodes.values())
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i >= j:
                    continue
                
                # Verificar si hay línea de visión
                has_visibility = True
                
                for zone in self.drone_graph.no_fly_zones:
                    if segment_passes_through_polygon(
                        node_i.position, node_j.position, zone
                    ):
                        has_visibility = False
                        break
                
                if has_visibility:
                    dist = distance(node_i.position, node_j.position)
                    risk = self._calculate_visibility_risk(
                        node_i.position, node_j.position
                    )
                    battery = dist / 100.0
                    
                    # Añadir aristas en ambas direcciones
                    self.edges[(node_i.id, node_j.id)] = VisibilityEdge(
                        node_i.id, node_j.id, dist, risk, battery
                    )
                    self.edges[(node_j.id, node_i.id)] = VisibilityEdge(
                        node_j.id, node_i.id, dist, risk, battery
                    )
    
    def _calculate_visibility_risk(self, p1: Point, p2: Point) -> float:
        """
        Calcula el riesgo de una arista de visibilidad.
        Basado en proximidad a zonas no-fly.
        """
        risk = 0.0
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        for zone in self.drone_graph.no_fly_zones:
            centroid = (
                sum(v[0] for v in zone) / len(zone),
                sum(v[1] for v in zone) / len(zone)
            )
            dist_to_zone = distance(mid, centroid)
            zone_risk = max(0, 30.0 - dist_to_zone) / 30.0
            risk += zone_risk
        
        return risk
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, VisibilityEdge]]:
        """Obtiene los vecinos visibles de un nodo."""
        neighbors = []
        for (from_id, to_id), edge in self.edges.items():
            if from_id == node_id:
                neighbors.append((to_id, edge))
        return neighbors
    
    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Verifica si existe arista de visibilidad entre dos nodos."""
        return (from_id, to_id) in self.edges


class GeometricHeuristic:
    """
    Heurística geométrica para construcción de rutas.
    
    Implementa varias estrategias:
    1. Nearest Neighbor: siempre ir al nodo más cercano no visitado
    2. Insertion: insertar nodos en la posición que minimice el coste
    3. Sweep: barrer el plano por ángulo desde el hub
    """
    
    def __init__(self, graph: DroneGraph):
        self.graph = graph
        self.visibility = VisibilityGraph(graph)
    
    def nearest_neighbor(self, 
                         objective: str = 'distance') -> Optional[Solution]:
        """
        Heurística del vecino más cercano.
        
        Construye la ruta seleccionando siempre el nodo no visitado
        más cercano según el objetivo especificado.
        
        Args:
            objective: 'distance', 'risk', o 'battery'
        """
        route = [self.graph.hub_id]
        visited = set()
        
        current = self.graph.hub_id
        destinations = set(self.graph.destination_ids)
        
        while visited != destinations:
            best_next = None
            best_cost = float('inf')
            
            for next_node in destinations - visited:
                # Verificar si hay arista válida
                if not self.visibility.has_edge(current, next_node):
                    # Intentar encontrar camino indirecto
                    continue
                
                edge = self.visibility.edges.get((current, next_node))
                if edge is None:
                    continue
                
                if objective == 'distance':
                    cost = edge.distance
                elif objective == 'risk':
                    cost = edge.risk
                else:
                    cost = edge.battery
                
                if cost < best_cost:
                    best_cost = cost
                    best_next = next_node
            
            if best_next is None:
                # No se puede completar la ruta
                return None
            
            route.append(best_next)
            visited.add(best_next)
            current = best_next
        
        # Volver al hub
        if not self.visibility.has_edge(current, self.graph.hub_id):
            return None
        
        route.append(self.graph.hub_id)
        
        # Calcular objetivos
        objectives = self._calculate_route_objectives(route)
        
        return Solution(route, objectives)
    
    def sweep_heuristic(self) -> Optional[Solution]:
        """
        Heurística de barrido angular.
        
        Ordena los destinos por ángulo respecto al hub y los visita
        en ese orden. Útil para problemas geográficos.
        """
        import math
        
        hub = self.graph.nodes[self.graph.hub_id]
        destinations = []
        
        for node_id in self.graph.destination_ids:
            node = self.graph.nodes[node_id]
            dx = node.x - hub.x
            dy = node.y - hub.y
            angle = math.atan2(dy, dx)
            destinations.append((angle, node_id))
        
        # Ordenar por ángulo
        destinations.sort()
        
        # Construir ruta
        route = [self.graph.hub_id]
        
        for _, node_id in destinations:
            # Verificar visibilidad con el nodo anterior
            prev = route[-1]
            if self.visibility.has_edge(prev, node_id):
                route.append(node_id)
            else:
                # Intentar inserción en otra posición
                inserted = False
                for i in range(len(route)):
                    if (self.visibility.has_edge(route[i], node_id) and
                        (i == len(route) - 1 or 
                         self.visibility.has_edge(node_id, route[i + 1] if i + 1 < len(route) else self.graph.hub_id))):
                        route.insert(i + 1, node_id)
                        inserted = True
                        break
                
                if not inserted:
                    route.append(node_id)
        
        # Volver al hub
        route.append(self.graph.hub_id)
        
        # Verificar validez
        if not self._is_valid_route(route):
            return None
        
        objectives = self._calculate_route_objectives(route)
        return Solution(route, objectives)
    
    def insertion_heuristic(self) -> Optional[Solution]:
        """
        Heurística de inserción.
        
        Comienza con un ciclo parcial (hub -> nodo más cercano -> hub)
        y va insertando nodos en la posición que minimice el incremento de coste.
        """
        # Encontrar nodo más cercano al hub
        hub = self.graph.nodes[self.graph.hub_id]
        best_first = None
        best_dist = float('inf')
        
        for node_id in self.graph.destination_ids:
            if self.visibility.has_edge(self.graph.hub_id, node_id):
                node = self.graph.nodes[node_id]
                dist = distance(hub.position, node.position)
                if dist < best_dist:
                    best_dist = dist
                    best_first = node_id
        
        if best_first is None:
            return None
        
        # Ciclo inicial
        route = [self.graph.hub_id, best_first, self.graph.hub_id]
        remaining = set(self.graph.destination_ids) - {best_first}
        
        # Insertar nodos restantes
        while remaining:
            best_node = None
            best_pos = None
            best_increase = float('inf')
            
            for node_id in remaining:
                node = self.graph.nodes[node_id]
                
                # Probar inserción en cada posición
                for i in range(1, len(route)):
                    prev_id = route[i - 1]
                    next_id = route[i]
                    
                    # Verificar visibilidad
                    if (self.visibility.has_edge(prev_id, node_id) and
                        self.visibility.has_edge(node_id, next_id)):
                        
                        # Calcular incremento de distancia
                        prev_node = self.graph.nodes[prev_id]
                        next_node = self.graph.nodes[next_id]
                        
                        old_dist = distance(prev_node.position, next_node.position)
                        new_dist = (distance(prev_node.position, node.position) +
                                   distance(node.position, next_node.position))
                        increase = new_dist - old_dist
                        
                        if increase < best_increase:
                            best_increase = increase
                            best_node = node_id
                            best_pos = i
            
            if best_node is None:
                # No se puede insertar ningún nodo
                break
            
            route.insert(best_pos, best_node)
            remaining.remove(best_node)
        
        if remaining:
            return None
        
        objectives = self._calculate_route_objectives(route)
        return Solution(route, objectives)
    
    def _calculate_route_objectives(self, route: List[int]) -> Tuple[float, float, float]:
        """Calcula los objetivos de una ruta."""
        total_dist = 0.0
        total_risk = 0.0
        total_battery = 0.0
        
        for i in range(len(route) - 1):
            edge = self.visibility.edges.get((route[i], route[i + 1]))
            if edge:
                total_dist += edge.distance
                total_risk += edge.risk
                total_battery += edge.battery
            else:
                # Usar distancia directa si no hay arista de visibilidad
                node_i = self.graph.nodes[route[i]]
                node_j = self.graph.nodes[route[i + 1]]
                dist = distance(node_i.position, node_j.position)
                total_dist += dist
                total_risk += 1.0  # Penalización por no tener visibilidad
                total_battery += dist / 100.0
        
        return (total_dist, total_risk, total_battery)
    
    def _is_valid_route(self, route: List[int]) -> bool:
        """Verifica si una ruta es válida."""
        if len(route) < 2:
            return False
        
        for i in range(len(route) - 1):
            if not self.visibility.has_edge(route[i], route[i + 1]):
                return False
        
        return True


def solve(graph: DroneGraph, 
          num_solutions: int = 5) -> Tuple[List[Solution], dict]:
    """
    Resuelve el problema usando heurísticas geométricas.
    
    Ejecuta múltiples heurísticas y retorna el frente de Pareto.
    
    Args:
        graph: Grafo del problema
        num_solutions: Número objetivo de soluciones
    
    Returns:
        Tupla (lista de soluciones, estadísticas)
    """
    start_time = time.time()
    
    heuristic = GeometricHeuristic(graph)
    all_solutions = []
    
    # Ejecutar diferentes heurísticas
    
    # 1. Nearest Neighbor con diferentes objetivos
    for objective in ['distance', 'risk', 'battery']:
        sol = heuristic.nearest_neighbor(objective)
        if sol:
            all_solutions.append(sol)
    
    # 2. Sweep heuristic
    sol = heuristic.sweep_heuristic()
    if sol:
        all_solutions.append(sol)
    
    # 3. Insertion heuristic
    sol = heuristic.insertion_heuristic()
    if sol:
        all_solutions.append(sol)
    
    # Obtener frente de Pareto
    pareto_front = get_pareto_front(all_solutions)
    
    elapsed = time.time() - start_time
    
    stats = {
        'total_solutions_generated': len(all_solutions),
        'pareto_front_size': len(pareto_front),
        'execution_time': elapsed,
        'visibility_edges': len(heuristic.visibility.edges)
    }
    
    return pareto_front, stats


# Pruebas del módulo
if __name__ == "__main__":
    from common.graph import DroneGraph, Node, Edge
    
    print("=" * 60)
    print("PRUEBAS DEL ALGORITMO HEURÍSTICO GEOMÉTRICO")
    print("=" * 60)
    
    # Crear grafo de prueba
    graph = DroneGraph()
    
    # Hub en el centro
    graph.nodes[0] = Node(0, 50, 50, is_hub=True)
    graph.hub_id = 0
    
    # 6 destinos
    graph.nodes[1] = Node(1, 20, 30)
    graph.nodes[2] = Node(2, 80, 30)
    graph.nodes[3] = Node(3, 90, 60)
    graph.nodes[4] = Node(4, 70, 80)
    graph.nodes[5] = Node(5, 30, 80)
    graph.nodes[6] = Node(6, 10, 50)
    
    # Zonas no-fly
    graph.no_fly_zones.append([(40, 35), (60, 35), (60, 45), (40, 45)])
    graph.no_fly_zones.append([(55, 60), (65, 60), (65, 70), (55, 70)])
    
    print(f"\nGrafo creado:")
    print(f"  - Nodos: {graph.num_nodes}")
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")
    
    # Resolver
    print("\nEjecutando heurísticas geométricas...")
    solutions, stats = solve(graph)
    
    print(f"\nResultados:")
    print(f"  - Tiempo: {stats['execution_time']:.3f} s")
    print(f"  - Aristas de visibilidad: {stats['visibility_edges']}")
    print(f"  - Soluciones generadas: {stats['total_solutions_generated']}")
    print(f"  - Frente de Pareto: {stats['pareto_front_size']} soluciones")
    
    if solutions:
        print("\nSoluciones del frente de Pareto:")
        for i, sol in enumerate(solutions):
            print(f"  {i+1}. Ruta: {sol.route}")
            print(f"     Objetivos: dist={sol.distance:.2f}, "
                  f"risk={sol.risk:.2f}, bat={sol.recharges:.2f}")
    
    print("\n" + "=" * 60)
