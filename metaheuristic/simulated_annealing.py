"""
Algoritmo metaheurístico Simulated Annealing para optimización multiobjetivo.
Implementa las técnicas del Tema 8: Algoritmos de aleatorización.

Simulated Annealing permite escapar de óptimos locales aceptando
movimientos que empeoran la solución con una probabilidad que
decrece con la temperatura.

Autor: David Valbuena Segura
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
import math
import time
import copy

from common.graph import DroneGraph, Node, Edge
from common.geometry import distance, segment_passes_through_polygon
from common.pareto import (
    Solution, get_pareto_front, dominates, 
    crowding_distance, hypervolume
)


class SimulatedAnnealing:
    """
    Implementación de Simulated Annealing multiobjetivo.
    
    Del Tema 8: Simulated Annealing es un algoritmo de búsqueda local
    que permite escapar de máximos locales al aceptar movimientos
    hacia posiciones de menor beneficio con probabilidad p = e^((f(S')-f(S))/T).
    
    Para el caso multiobjetivo, se mantiene un archivo de soluciones
    no dominadas (aproximación al frente de Pareto).
    """
    
    def __init__(self, 
                 graph: DroneGraph,
                 initial_temp: float = 10000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.01,
                 iterations_per_temp: int = 50,
                 weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Args:
            graph: Grafo del problema
            initial_temp: Temperatura inicial (alta para exploración)
            cooling_rate: Factor de enfriamiento (0 < rate < 1)
            min_temp: Temperatura mínima para detener
            iterations_per_temp: Iteraciones por nivel de temperatura
            weights: Pesos para escalarización de objetivos
        """
        self.graph = graph
        self.T0 = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self.weights = weights
        
        # Archivo de soluciones no dominadas
        self.archive: List[Solution] = []
        
        # Estadísticas
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.improvements = 0
        self.temperature_history = []
    
    def _weighted_cost(self, objectives: Tuple[float, float, float]) -> float:
        """Calcula el coste escalarizado."""
        return sum(w * o for w, o in zip(self.weights, objectives))
    
    def _generate_initial_solution(self) -> List[int]:
        """
        Genera una solución inicial usando nearest neighbor.
        """
        route = [self.graph.hub_id]
        visited = set()
        current = self.graph.hub_id
        destinations = set(self.graph.destination_ids)
        
        while visited != destinations:
            # Encontrar vecino más cercano no visitado
            best_next = None
            best_dist = float('inf')
            
            for next_node in destinations - visited:
                edge = self.graph.get_edge(current, next_node)
                if edge and edge.is_valid and edge.distance < best_dist:
                    best_dist = edge.distance
                    best_next = next_node
            
            if best_next is None:
                # Si no hay arista válida, elegir aleatorio
                remaining = list(destinations - visited)
                if remaining:
                    best_next = random.choice(remaining)
                else:
                    break
            
            route.append(best_next)
            visited.add(best_next)
            current = best_next
        
        route.append(self.graph.hub_id)
        return route
    
    def _generate_neighbor(self, route: List[int]) -> List[int]:
        """
        Genera una solución vecina aplicando un operador de vecindad.
        
        Operadores disponibles:
        - 2-opt: invierte un segmento de la ruta
        - swap: intercambia dos nodos
        - insert: mueve un nodo a otra posición
        """
        new_route = route.copy()
        n = len(route) - 2  # Excluir hub inicial y final
        
        if n < 2:
            return new_route
        
        operator = random.choice(['2opt', 'swap', 'insert'])
        
        if operator == '2opt':
            # Seleccionar dos puntos de corte
            i = random.randint(1, n - 1)
            j = random.randint(i + 1, n)
            # Invertir segmento
            new_route[i:j+1] = reversed(new_route[i:j+1])
        
        elif operator == 'swap':
            # Intercambiar dos nodos
            i = random.randint(1, n)
            j = random.randint(1, n)
            if i != j:
                new_route[i], new_route[j] = new_route[j], new_route[i]
        
        else:  # insert
            # Mover un nodo a otra posición
            i = random.randint(1, n)
            j = random.randint(1, n)
            if i != j:
                node = new_route.pop(i)
                new_route.insert(j, node)
        
        return new_route
    
    def _evaluate(self, route: List[int]) -> Tuple[float, float, float]:
        """
        Evalúa una ruta y retorna sus objetivos.
        
        Si la ruta es inválida (atraviesa zonas no-fly), 
        retorna costes muy altos como penalización.
        """
        total_dist = 0.0
        total_risk = 0.0
        total_battery = 0.0
        
        for i in range(len(route) - 1):
            edge = self.graph.get_edge(route[i], route[i + 1])
            
            if edge is None:
                # Arista no existe: penalización alta
                node_i = self.graph.nodes[route[i]]
                node_j = self.graph.nodes[route[i + 1]]
                dist = distance(node_i.position, node_j.position)
                total_dist += dist * 2
                total_risk += 10.0
                total_battery += dist / 50.0
            elif not edge.is_valid:
                # Arista atraviesa zona no-fly: penalización
                total_dist += edge.distance * 1.5
                total_risk += edge.risk + 5.0
                total_battery += edge.battery_consumption * 1.5
            else:
                total_dist += edge.distance
                total_risk += edge.risk
                total_battery += edge.battery_consumption
        
        return (total_dist, total_risk, total_battery)
    
    def _acceptance_probability(self, 
                                 current_cost: float, 
                                 new_cost: float, 
                                 temperature: float) -> float:
        """
        Calcula la probabilidad de aceptación según la fórmula del Tema 8:
        p = e^((f(S')-f(S))/T)
        
        Para minimización, si new_cost < current_cost, p > 1 (siempre acepta).
        Si new_cost > current_cost, 0 < p < 1.
        """
        if new_cost < current_cost:
            return 1.0
        
        if temperature <= 0:
            return 0.0
        
        delta = current_cost - new_cost  # Negativo si empeora
        return math.exp(delta / temperature)
    
    def _update_archive(self, solution: Solution):
        """
        Actualiza el archivo de soluciones no dominadas.
        
        Añade la solución si no está dominada y elimina
        las soluciones que pasan a estar dominadas.
        """
        # Verificar si la nueva solución está dominada
        for archived in self.archive:
            if dominates(archived, solution):
                return  # No añadir si está dominada
        
        # Eliminar soluciones dominadas por la nueva
        self.archive = [
            s for s in self.archive 
            if not dominates(solution, s)
        ]
        
        # Añadir la nueva solución
        self.archive.append(solution)
    
    def solve(self, 
              max_iterations: int = 10000,
              time_limit: float = 60.0,
              verbose: bool = False) -> List[Solution]:
        """
        Ejecuta el algoritmo Simulated Annealing.
        
        Args:
            max_iterations: Número máximo de iteraciones
            time_limit: Tiempo límite en segundos
            verbose: Si mostrar progreso
        
        Returns:
            Lista de soluciones no dominadas (frente de Pareto)
        """
        start_time = time.time()
        
        # Inicialización
        self.archive = []
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.improvements = 0
        self.temperature_history = []
        
        # Generar solución inicial
        current_route = self._generate_initial_solution()
        current_objectives = self._evaluate(current_route)
        current_cost = self._weighted_cost(current_objectives)
        
        current_solution = Solution(current_route, current_objectives)
        self._update_archive(current_solution)
        
        best_cost = current_cost
        best_route = current_route.copy()
        
        # Bucle principal
        temperature = self.T0
        iteration = 0
        
        while temperature > self.min_temp and iteration < max_iterations:
            # Verificar límite de tiempo
            if time.time() - start_time > time_limit:
                break
            
            for _ in range(self.iterations_per_temp):
                iteration += 1
                
                # Generar vecino
                new_route = self._generate_neighbor(current_route)
                new_objectives = self._evaluate(new_route)
                new_cost = self._weighted_cost(new_objectives)
                
                # Decidir si aceptar
                acceptance_prob = self._acceptance_probability(
                    current_cost, new_cost, temperature
                )
                
                if random.random() < acceptance_prob:
                    current_route = new_route
                    current_objectives = new_objectives
                    current_cost = new_cost
                    self.accepted_moves += 1
                    
                    # Actualizar archivo
                    new_solution = Solution(new_route, new_objectives)
                    self._update_archive(new_solution)
                    
                    # Actualizar mejor global
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_route = new_route.copy()
                        self.improvements += 1
                else:
                    self.rejected_moves += 1
            
            # Enfriar
            self.temperature_history.append(temperature)
            temperature *= self.cooling_rate
            
            if verbose and iteration % 1000 == 0:
                print(f"  Iteración {iteration}: T={temperature:.2f}, "
                      f"archivo={len(self.archive)}, mejor={best_cost:.2f}")
        
        return self.archive
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas de la ejecución."""
        total_moves = self.accepted_moves + self.rejected_moves
        return {
            'total_moves': total_moves,
            'accepted_moves': self.accepted_moves,
            'rejected_moves': self.rejected_moves,
            'acceptance_rate': self.accepted_moves / max(1, total_moves),
            'improvements': self.improvements,
            'archive_size': len(self.archive),
            'final_temperature': self.temperature_history[-1] if self.temperature_history else self.T0
        }


def solve(graph: DroneGraph,
          initial_temp: float = 10000.0,
          cooling_rate: float = 0.995,
          max_iterations: int = 10000,
          time_limit: float = 60.0,
          num_runs: int = 3) -> Tuple[List[Solution], dict]:
    """
    Resuelve el problema ejecutando múltiples corridas de SA.
    
    Args:
        graph: Grafo del problema
        initial_temp: Temperatura inicial
        cooling_rate: Factor de enfriamiento
        max_iterations: Iteraciones máximas por corrida
        time_limit: Tiempo límite total
        num_runs: Número de corridas independientes
    
    Returns:
        Tupla (frente de Pareto combinado, estadísticas agregadas)
    """
    start_time = time.time()
    all_solutions = []
    all_stats = []
    
    time_per_run = time_limit / num_runs
    
    for run in range(num_runs):
        # Variar pesos para explorar diferentes regiones del frente
        weights = (
            random.uniform(0.5, 1.5),
            random.uniform(0.5, 1.5),
            random.uniform(0.5, 1.5)
        )
        
        sa = SimulatedAnnealing(
            graph,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            weights=weights
        )
        
        solutions = sa.solve(
            max_iterations=max_iterations,
            time_limit=time_per_run
        )
        
        all_solutions.extend(solutions)
        all_stats.append(sa.get_statistics())
    
    # Combinar y obtener frente de Pareto global
    pareto_front = get_pareto_front(all_solutions)
    
    elapsed = time.time() - start_time
    
    # Agregar estadísticas
    stats = {
        'total_time': elapsed,
        'num_runs': num_runs,
        'total_solutions_generated': len(all_solutions),
        'pareto_front_size': len(pareto_front),
        'avg_acceptance_rate': sum(s['acceptance_rate'] for s in all_stats) / len(all_stats),
        'total_improvements': sum(s['improvements'] for s in all_stats)
    }
    
    return pareto_front, stats


# Pruebas del módulo
if __name__ == "__main__":
    from common.graph import DroneGraph, Node, Edge
    
    print("=" * 60)
    print("PRUEBAS DEL ALGORITMO SIMULATED ANNEALING")
    print("=" * 60)
    
    # Crear grafo de prueba
    graph = DroneGraph()
    
    # Hub
    graph.nodes[0] = Node(0, 50, 50, is_hub=True)
    graph.hub_id = 0
    
    # 8 destinos
    positions = [
        (20, 20), (80, 20), (80, 80), (20, 80),
        (50, 10), (90, 50), (50, 90), (10, 50)
    ]
    
    for i, (x, y) in enumerate(positions, 1):
        graph.nodes[i] = Node(i, x, y)
    
    # Zonas no-fly
    graph.no_fly_zones.append([(35, 35), (45, 35), (45, 45), (35, 45)])
    graph.no_fly_zones.append([(55, 55), (65, 55), (65, 65), (55, 65)])
    graph.no_fly_zones.append([(70, 25), (85, 25), (85, 35), (70, 35)])
    
    # Crear aristas
    for i in graph.nodes:
        for j in graph.nodes:
            if i != j:
                node_i = graph.nodes[i]
                node_j = graph.nodes[j]
                dist = distance(node_i.position, node_j.position)
                
                is_valid = True
                for zone in graph.no_fly_zones:
                    if segment_passes_through_polygon(
                        node_i.position, node_j.position, zone
                    ):
                        is_valid = False
                        break
                
                risk = 0.1 if is_valid else 2.0
                graph.edges[(i, j)] = Edge(i, j, dist, risk, dist/100, is_valid)
    
    print(f"\nGrafo creado:")
    print(f"  - Nodos: {graph.num_nodes}")
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")
    
    # Resolver
    print("\nEjecutando Simulated Annealing...")
    solutions, stats = solve(
        graph,
        initial_temp=5000.0,
        cooling_rate=0.99,
        max_iterations=5000,
        time_limit=30.0,
        num_runs=3
    )
    
    print(f"\nResultados:")
    print(f"  - Tiempo total: {stats['total_time']:.3f} s")
    print(f"  - Corridas: {stats['num_runs']}")
    print(f"  - Soluciones generadas: {stats['total_solutions_generated']}")
    print(f"  - Frente de Pareto: {stats['pareto_front_size']} soluciones")
    print(f"  - Tasa de aceptación media: {stats['avg_acceptance_rate']:.2%}")
    print(f"  - Mejoras totales: {stats['total_improvements']}")
    
    if solutions:
        print("\nSoluciones del frente de Pareto:")
        for i, sol in enumerate(solutions[:5]):  # Mostrar máximo 5
            print(f"  {i+1}. Ruta: {sol.route}")
            print(f"     Objetivos: dist={sol.distance:.2f}, "
                  f"risk={sol.risk:.2f}, bat={sol.recharges:.2f}")
    
    print("\n" + "=" * 60)
