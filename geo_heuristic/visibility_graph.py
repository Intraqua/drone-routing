"""
Algoritmo heurístico geométrico basado en grafo de visibilidad.
Implementa técnicas del Tema 7: Algoritmos geométricos para generar
rutas factibles rápidas evitando zonas no-fly.

El algoritmo construye un grafo de visibilidad completo que incluye:
- Nodos del problema (hub, destinos, estaciones de recarga)
- Vértices de los polígonos no-fly como nodos auxiliares de navegación

Esto permite encontrar rutas que "rodeen" los obstáculos cuando no
hay visibilidad directa entre dos puntos.

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
import math

from common.graph import DroneGraph, Node, Edge
from common.geometry import (
    Point, Polygon, distance,
    segment_passes_through_polygon,
    segments_intersect,
    point_in_polygon
)
from common.pareto import Solution, get_pareto_front, dominates


# IDs para nodos auxiliares (empiezan en 10000 para evitar conflictos)
AUXILIARY_NODE_START_ID = 10000


@dataclass
class VisibilityEdge:
    """Arista en el grafo de visibilidad."""
    from_node: int
    to_node: int
    distance: float
    risk: float
    battery: float


@dataclass
class VisibilityNode:
    """Nodo en el grafo de visibilidad (puede ser original o auxiliar)."""
    id: int
    position: Point
    is_auxiliary: bool = False  # True si es vértice de polígono no-fly


class VisibilityGraph:
    """
    Grafo de visibilidad completo para navegación evitando obstáculos.

    Tema 7: Algoritmos geométricos. El grafo de visibilidad conecta
    todos los puntos que tienen línea de visión directa, incluyendo:
    - Nodos originales del problema (hub, destinos)
    - Vértices de los polígonos no-fly (nodos auxiliares)

    Los nodos auxiliares permiten encontrar rutas que rodean los
    obstáculos cuando no hay visibilidad directa.

    Utiliza los tests de intersección de segmentos del Tema 7,
    Entrenamiento 2 para determinar visibilidad.
    """

    def __init__(self, drone_graph: DroneGraph):
        self.drone_graph = drone_graph
        self.edges: Dict[Tuple[int, int], VisibilityEdge] = {}
        self.adjacency: Dict[int, List[int]] = {}
        self.all_nodes: Dict[int, VisibilityNode] = {}
        self._build_visibility_graph()

    def _build_visibility_graph(self):
        """
        Construye el grafo de visibilidad completo.

        1. Añade todos los nodos originales del problema
        2. Añade los vértices de los polígonos no-fly como nodos auxiliares
        3. Calcula visibilidad entre todos los pares de nodos
        """
        # 1. Añadir nodos originales
        for node in self.drone_graph.nodes.values():
            self.all_nodes[node.id] = VisibilityNode(
                id=node.id,
                position=node.position,
                is_auxiliary=False
            )
            self.adjacency[node.id] = []

        # 2. Añadir vértices de polígonos no-fly como nodos auxiliares
        aux_id = AUXILIARY_NODE_START_ID
        for zone in self.drone_graph.no_fly_zones:
            for vertex in zone:
                # Verificar que el vértice no está dentro de otro polígono
                inside_other = False
                for other_zone in self.drone_graph.no_fly_zones:
                    if other_zone is not zone:
                        if point_in_polygon(vertex, other_zone):
                            inside_other = True
                            break

                if not inside_other:
                    # Desplazar ligeramente el vértice hacia afuera del polígono
                    # para evitar problemas de colinealidad
                    offset_vertex = self._offset_vertex(vertex, zone)

                    self.all_nodes[aux_id] = VisibilityNode(
                        id=aux_id,
                        position=offset_vertex,
                        is_auxiliary=True
                    )
                    self.adjacency[aux_id] = []
                    aux_id += 1

        # 3. Construir aristas de visibilidad
        node_list = list(self.all_nodes.values())

        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                if i >= j:
                    continue

                # Verificar si hay línea de visión (Tema 7)
                if self._has_visibility(node_i.position, node_j.position):
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

                    # Actualizar listas de adyacencia
                    self.adjacency[node_i.id].append(node_j.id)
                    self.adjacency[node_j.id].append(node_i.id)

    def _offset_vertex(self, vertex: Point, polygon: Polygon) -> Point:
        """
        Desplaza un vértice ligeramente hacia afuera del polígono.
        Esto evita problemas numéricos con segmentos que tocan el borde.
        """
        # Calcular centroide del polígono
        cx = sum(v[0] for v in polygon) / len(polygon)
        cy = sum(v[1] for v in polygon) / len(polygon)

        # Vector desde centroide hacia el vértice
        dx = vertex[0] - cx
        dy = vertex[1] - cy

        # Normalizar y aplicar pequeño offset (1 unidad)
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            offset = 1.5  # Pequeño desplazamiento hacia afuera
            new_x = vertex[0] + (dx / length) * offset
            new_y = vertex[1] + (dy / length) * offset
            return (new_x, new_y)

        return vertex

    def _has_visibility(self, p1: Point, p2: Point) -> bool:
        """
        Verifica si hay línea de visión entre dos puntos.
        Usa los tests de intersección del Tema 7.
        """
        for zone in self.drone_graph.no_fly_zones:
            if segment_passes_through_polygon(p1, p2, zone):
                return False
        return True

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
        for to_id in self.adjacency.get(node_id, []):
            edge = self.edges.get((node_id, to_id))
            if edge:
                neighbors.append((to_id, edge))
        return neighbors

    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Verifica si existe arista de visibilidad entre dos nodos."""
        return (from_id, to_id) in self.edges

    def shortest_path(self, start: int, end: int,
                      weight: str = 'distance') -> Optional[List[int]]:
        """
        Encuentra el camino más corto entre dos nodos usando Dijkstra.

        El camino puede pasar por nodos auxiliares (vértices de polígonos)
        para rodear obstáculos.

        Args:
            start: Nodo inicial
            end: Nodo destino
            weight: 'distance', 'risk', o 'battery'

        Returns:
            Lista de nodos del camino, o None si no existe
        """
        if start == end:
            return [start]

        # Dijkstra
        dist = {start: 0}
        prev = {start: None}
        pq = [(0, start)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == end:
                # Reconstruir camino
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = prev[current]
                return path[::-1]

            for v, edge in self.get_neighbors(u):
                if v in visited:
                    continue

                if weight == 'distance':
                    w = edge.distance
                elif weight == 'risk':
                    w = edge.risk
                else:
                    w = edge.battery

                new_dist = dist[u] + w
                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        return None

    def get_auxiliary_node_count(self) -> int:
        """Retorna el número de nodos auxiliares añadidos."""
        return sum(1 for n in self.all_nodes.values() if n.is_auxiliary)


class GeometricHeuristic:
    """
    Heurística geométrica para construcción de rutas.

    Implementa varias estrategias del Tema 7:
    1. Nearest Neighbor: siempre ir al nodo más cercano no visitado
    2. Insertion: insertar nodos en la posición que minimice el coste
    3. Sweep: barrer el plano por ángulo desde el hub

    Usa el grafo de visibilidad completo con nodos auxiliares para
    encontrar rutas que rodean los obstáculos.
    """

    def __init__(self, graph: DroneGraph):
        self.graph = graph
        self.visibility = VisibilityGraph(graph)

    def _get_path_cost(self, from_id: int, to_id: int,
                       objective: str = 'distance') -> Tuple[float, List[int]]:
        """
        Obtiene el coste y camino entre dos nodos.
        Si hay visibilidad directa, usa esa arista.
        Si no, busca camino indirecto con Dijkstra (puede usar nodos auxiliares).

        Returns:
            Tupla (coste, camino) o (inf, []) si no hay camino
        """
        if self.visibility.has_edge(from_id, to_id):
            edge = self.visibility.edges[(from_id, to_id)]
            if objective == 'distance':
                return edge.distance, [from_id, to_id]
            elif objective == 'risk':
                return edge.risk, [from_id, to_id]
            else:
                return edge.battery, [from_id, to_id]

        # Buscar camino indirecto (puede pasar por nodos auxiliares)
        path = self.visibility.shortest_path(from_id, to_id, objective)
        if path is None:
            return float('inf'), []

        # Calcular coste del camino
        cost = 0.0
        for i in range(len(path) - 1):
            edge = self.visibility.edges.get((path[i], path[i+1]))
            if edge:
                if objective == 'distance':
                    cost += edge.distance
                elif objective == 'risk':
                    cost += edge.risk
                else:
                    cost += edge.battery

        return cost, path

    def nearest_neighbor(self,
                         objective: str = 'distance') -> Optional[Solution]:
        """
        Heurística del vecino más cercano mejorada.

        Construye la ruta seleccionando siempre el nodo no visitado
        más cercano según el objetivo especificado.
        """
        route = [self.graph.hub_id]
        visited = set()

        current = self.graph.hub_id
        destinations = set(self.graph.destination_ids)

        while visited != destinations:
            best_next = None
            best_cost = float('inf')
            best_path = []

            for next_node in destinations - visited:
                cost, path = self._get_path_cost(current, next_node, objective)

                if cost < best_cost:
                    best_cost = cost
                    best_next = next_node
                    best_path = path

            if best_next is None or best_cost == float('inf'):
                return None

            # Añadir camino (sin duplicar el nodo actual)
            route.extend(best_path[1:])
            visited.add(best_next)
            current = best_next

        # Volver al hub
        cost, path = self._get_path_cost(current, self.graph.hub_id, objective)
        if cost == float('inf'):
            return None

        route.extend(path[1:])

        # Calcular objetivos (filtrando nodos auxiliares de la ruta mostrada)
        objectives = self._calculate_route_objectives(route)

        return Solution(route, objectives)

    def sweep_heuristic(self) -> Optional[Solution]:
        """
        Heurística de barrido angular.

        Ordena los destinos por ángulo respecto al hub y los visita
        en ese orden.
        """
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
        current = self.graph.hub_id

        for _, node_id in destinations:
            cost, path = self._get_path_cost(current, node_id, 'distance')
            if cost == float('inf'):
                continue

            route.extend(path[1:])
            current = node_id

        # Volver al hub
        cost, path = self._get_path_cost(current, self.graph.hub_id, 'distance')
        if cost == float('inf'):
            return None

        route.extend(path[1:])

        # Verificar que visitamos todos los destinos
        visited_destinations = set(n for n in route if n < AUXILIARY_NODE_START_ID)
        if not all(d in visited_destinations for d in self.graph.destination_ids):
            return None

        objectives = self._calculate_route_objectives(route)
        return Solution(route, objectives)

    def insertion_heuristic(self) -> Optional[Solution]:
        """
        Heurística de inserción mejorada.

        Comienza con un ciclo parcial y va insertando nodos en la
        posición que minimice el incremento de coste.
        """
        # Encontrar nodo más cercano al hub con camino válido
        best_first = None
        best_dist = float('inf')

        for node_id in self.graph.destination_ids:
            cost, _ = self._get_path_cost(self.graph.hub_id, node_id, 'distance')
            if cost < best_dist:
                best_dist = cost
                best_first = node_id

        if best_first is None:
            return None

        # Lista de destinos a visitar en orden
        visit_order = [best_first]
        remaining = set(self.graph.destination_ids) - {best_first}

        # Insertar nodos restantes
        while remaining:
            best_node = None
            best_pos = None
            best_increase = float('inf')

            for node_id in remaining:
                for i in range(len(visit_order) + 1):
                    if i == 0:
                        prev_id = self.graph.hub_id
                    else:
                        prev_id = visit_order[i - 1]

                    if i == len(visit_order):
                        next_id = self.graph.hub_id
                    else:
                        next_id = visit_order[i]

                    old_cost, _ = self._get_path_cost(prev_id, next_id, 'distance')
                    cost1, _ = self._get_path_cost(prev_id, node_id, 'distance')
                    cost2, _ = self._get_path_cost(node_id, next_id, 'distance')

                    if cost1 == float('inf') or cost2 == float('inf'):
                        continue

                    increase = cost1 + cost2 - old_cost

                    if increase < best_increase:
                        best_increase = increase
                        best_node = node_id
                        best_pos = i

            if best_node is None:
                # Último recurso: añadir al final
                for node_id in remaining:
                    cost, _ = self._get_path_cost(
                        visit_order[-1] if visit_order else self.graph.hub_id,
                        node_id, 'distance'
                    )
                    if cost < float('inf'):
                        visit_order.append(node_id)
                        remaining.remove(node_id)
                        break
                else:
                    break
            else:
                visit_order.insert(best_pos, best_node)
                remaining.remove(best_node)

        if remaining:
            return None

        # Construir ruta completa
        route = [self.graph.hub_id]
        current = self.graph.hub_id

        for node_id in visit_order:
            _, path = self._get_path_cost(current, node_id, 'distance')
            route.extend(path[1:])
            current = node_id

        _, path = self._get_path_cost(current, self.graph.hub_id, 'distance')
        route.extend(path[1:])

        objectives = self._calculate_route_objectives(route)
        return Solution(route, objectives)

    def randomized_nearest_neighbor(self,
                                    k: int = 3,
                                    seed: Optional[int] = None) -> Optional[Solution]:
        """
        Variante aleatorizada del vecino más cercano.
        """
        if seed is not None:
            random.seed(seed)

        route = [self.graph.hub_id]
        visited = set()

        current = self.graph.hub_id
        destinations = set(self.graph.destination_ids)

        while visited != destinations:
            candidates = []

            for next_node in destinations - visited:
                cost, path = self._get_path_cost(current, next_node, 'distance')
                if cost < float('inf'):
                    candidates.append((cost, next_node, path))

            if not candidates:
                return None

            candidates.sort()
            top_k = candidates[:min(k, len(candidates))]
            _, best_next, best_path = random.choice(top_k)

            route.extend(best_path[1:])
            visited.add(best_next)
            current = best_next

        cost, path = self._get_path_cost(current, self.graph.hub_id, 'distance')
        if cost == float('inf'):
            return None

        route.extend(path[1:])

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
                # Fallback: distancia directa
                node_i = self.visibility.all_nodes.get(route[i])
                node_j = self.visibility.all_nodes.get(route[i + 1])
                if node_i and node_j:
                    dist = distance(node_i.position, node_j.position)
                    total_dist += dist
                    total_risk += 1.0
                    total_battery += dist / 100.0

        return (total_dist, total_risk, total_battery)


def solve(graph: DroneGraph,
          num_solutions: int = 5) -> Tuple[List[Solution], dict]:
    """
    Resuelve el problema usando heurísticas geométricas.

    Ejecuta múltiples heurísticas y retorna el frente de Pareto.
    """
    start_time = time.time()

    heuristic = GeometricHeuristic(graph)
    all_solutions = []

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

    # 4. Variantes aleatorizadas
    for seed in range(3):
        sol = heuristic.randomized_nearest_neighbor(k=3, seed=seed)
        if sol:
            all_solutions.append(sol)

    # Eliminar duplicados
    unique_solutions = []
    seen_objectives = set()
    for sol in all_solutions:
        obj_key = (round(sol.distance, 2), round(sol.risk, 2), round(sol.recharges, 2))
        if obj_key not in seen_objectives:
            seen_objectives.add(obj_key)
            unique_solutions.append(sol)

    # Obtener frente de Pareto
    pareto_front = get_pareto_front(unique_solutions)

    elapsed = time.time() - start_time

    stats = {
        'total_solutions_generated': len(all_solutions),
        'unique_solutions': len(unique_solutions),
        'pareto_front_size': len(pareto_front),
        'execution_time': elapsed,
        'visibility_edges': len(heuristic.visibility.edges),
        'auxiliary_nodes': heuristic.visibility.get_auxiliary_node_count()
    }

    return pareto_front, stats


# Pruebas del módulo
if __name__ == "__main__":
    from common.graph import DroneGraph, Node, Edge

    print("=" * 60)
    print("PRUEBAS DEL GRAFO DE VISIBILIDAD COMPLETO")
    print("Tema 7: Algoritmos geométricos")
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
    print(f"  - Nodos originales: {graph.num_nodes}")
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")

    # Resolver
    print("\nEjecutando heurísticas geométricas...")
    solutions, stats = solve(graph)

    print(f"\nResultados:")
    print(f"  - Tiempo: {stats['execution_time']:.3f} s")
    print(f"  - Nodos auxiliares añadidos: {stats['auxiliary_nodes']}")
    print(f"  - Aristas de visibilidad: {stats['visibility_edges']}")
    print(f"  - Soluciones generadas: {stats['total_solutions_generated']}")
    print(f"  - Soluciones únicas: {stats['unique_solutions']}")
    print(f"  - Frente de Pareto: {stats['pareto_front_size']} soluciones")

    if solutions:
        print("\nSoluciones del frente de Pareto:")
        for i, sol in enumerate(solutions):
            # Filtrar nodos auxiliares para mostrar
            main_route = [n for n in sol.route if n < AUXILIARY_NODE_START_ID]
            print(f"  {i+1}. Ruta (destinos): {main_route}")
            print(f"     Ruta completa: {sol.route}")
            print(f"     Objetivos: dist={sol.distance:.2f}, "
                  f"risk={sol.risk:.2f}, bat={sol.recharges:.2f}")

    print("\n" + "=" * 60)