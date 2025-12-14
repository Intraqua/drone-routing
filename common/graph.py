"""
Módulo de representación de grafos para el problema de rutas de drones.
Implementa el grafo dirigido ponderado con múltiples objetivos.

Autor: David Valbuena Segura
"""

import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from .geometry import Point, Polygon, distance, segment_passes_through_polygon


@dataclass
class Node:
    """Representa un nodo (destino o hub) en el grafo."""
    id: int
    x: float
    y: float
    is_hub: bool = False
    is_charging_station: bool = False
    
    @property
    def position(self) -> Point:
        return (self.x, self.y)


@dataclass
class Edge:
    """
    Representa una arista con pesos vectoriales w = <distancia, riesgo, consumo_batería>.
    Basado en el modelado del problema del enunciado.
    """
    from_node: int
    to_node: int
    distance: float
    risk: float
    battery_consumption: float
    is_valid: bool = True  # False si atraviesa zona no-fly


@dataclass
class DroneGraph:
    """
    Grafo dirigido ponderado para el problema de planificación de drones.
    
    Atributos:
        nodes: Diccionario de nodos indexados por ID
        edges: Matriz de adyacencia con objetos Edge
        no_fly_zones: Lista de polígonos que representan zonas restringidas
        hub_id: ID del nodo hub (origen/destino del circuito)
        charging_stations: IDs de los nodos con estaciones de recarga
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: Dict[Tuple[int, int], Edge] = field(default_factory=dict)
    no_fly_zones: List[Polygon] = field(default_factory=list)
    hub_id: int = 0
    charging_stations: List[int] = field(default_factory=list)
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def destination_ids(self) -> List[int]:
        """Retorna los IDs de los destinos (excluyendo el hub)."""
        return [n.id for n in self.nodes.values() if not n.is_hub]
    
    def get_edge(self, from_id: int, to_id: int) -> Optional[Edge]:
        """Obtiene la arista entre dos nodos."""
        return self.edges.get((from_id, to_id))
    
    def is_edge_valid(self, from_id: int, to_id: int) -> bool:
        """Verifica si una arista es válida (no atraviesa zonas no-fly)."""
        edge = self.get_edge(from_id, to_id)
        return edge is not None and edge.is_valid
    
    def get_route_cost(self, route: List[int]) -> Tuple[float, float, float]:
        """
        Calcula el coste vectorial de una ruta completa.
        
        Args:
            route: Lista de IDs de nodos en orden de visita
        
        Returns:
            Tupla (distancia_total, riesgo_total, consumo_batería_total)
        """
        total_distance = 0.0
        total_risk = 0.0
        total_battery = 0.0
        
        for i in range(len(route) - 1):
            edge = self.get_edge(route[i], route[i + 1])
            if edge is None or not edge.is_valid:
                # Ruta inválida: retornar infinito
                return (float('inf'), float('inf'), float('inf'))
            
            total_distance += edge.distance
            total_risk += edge.risk
            total_battery += edge.battery_consumption
        
        return (total_distance, total_risk, total_battery)
    
    def count_recharges_needed(self, route: List[int], battery_capacity: float = 1.0) -> int:
        """
        Cuenta el número de recargas necesarias en una ruta.
        
        Asume que el dron parte con batería completa y debe recargar
        cuando el consumo acumulado supera la capacidad.
        """
        recharges = 0
        current_battery = battery_capacity
        
        for i in range(len(route) - 1):
            edge = self.get_edge(route[i], route[i + 1])
            if edge is None:
                continue
            
            current_battery -= edge.battery_consumption
            
            # Verificar si necesita recarga
            if current_battery < 0:
                # Buscar estación de recarga más cercana
                if route[i + 1] in self.charging_stations:
                    current_battery = battery_capacity
                    recharges += 1
                else:
                    # Recarga de emergencia (penalización)
                    recharges += 2
                    current_battery = battery_capacity
        
        return recharges
    
    def is_valid_route(self, route: List[int]) -> bool:
        """
        Verifica si una ruta es válida:
        - Comienza y termina en el hub
        - Visita todos los destinos exactamente una vez
        - No atraviesa zonas no-fly
        """
        if len(route) < 2:
            return False
        
        # Debe empezar y terminar en el hub
        if route[0] != self.hub_id or route[-1] != self.hub_id:
            return False
        
        # Verificar que visita todos los destinos exactamente una vez
        destinations = set(self.destination_ids)
        visited = set(route[1:-1])  # Excluir hub inicial y final
        
        if visited != destinations:
            return False
        
        # Verificar que no hay aristas inválidas
        for i in range(len(route) - 1):
            if not self.is_edge_valid(route[i], route[i + 1]):
                return False
        
        return True


def create_graph_from_json(filepath: str) -> DroneGraph:
    """
    Crea un grafo a partir de un archivo JSON de instancia.
    
    Formato esperado del JSON según el enunciado.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    graph = DroneGraph()
    
    # Cargar hub
    hub_data = data['hub']
    hub = Node(
        id=hub_data['id'],
        x=hub_data['x'],
        y=hub_data['y'],
        is_hub=True
    )
    graph.nodes[hub.id] = hub
    graph.hub_id = hub.id
    
    # Cargar destinos
    for dest in data['destinations']:
        node = Node(
            id=dest['id'],
            x=dest['x'],
            y=dest['y'],
            is_charging_station=dest.get('is_charging', False)
        )
        graph.nodes[node.id] = node
    
    # Cargar estaciones de recarga
    if 'charging_stations' in data:
        graph.charging_stations = data['charging_stations']
        for station_id in graph.charging_stations:
            if station_id in graph.nodes:
                graph.nodes[station_id].is_charging_station = True
    
    # Cargar zonas no-fly
    if 'no_fly_zones' in data:
        for zone in data['no_fly_zones']:
            polygon = [tuple(v) for v in zone['vertices']]
            graph.no_fly_zones.append(polygon)
    
    # Crear aristas (grafo completo)
    # Calculamos distancia, riesgo y consumo de batería para cada par
    for i in graph.nodes:
        for j in graph.nodes:
            if i != j:
                node_i = graph.nodes[i]
                node_j = graph.nodes[j]
                
                # Calcular distancia euclidiana
                dist = distance(node_i.position, node_j.position)
                
                # Calcular riesgo (basado en proximidad a zonas no-fly)
                risk = calculate_edge_risk(node_i.position, node_j.position, 
                                          graph.no_fly_zones)
                
                # Consumo de batería proporcional a la distancia
                battery = dist / 100.0  # Normalizado
                
                # Verificar si atraviesa zona no-fly
                is_valid = True
                for zone in graph.no_fly_zones:
                    if segment_passes_through_polygon(node_i.position, 
                                                      node_j.position, zone):
                        is_valid = False
                        break
                
                edge = Edge(
                    from_node=i,
                    to_node=j,
                    distance=dist,
                    risk=risk,
                    battery_consumption=battery,
                    is_valid=is_valid
                )
                graph.edges[(i, j)] = edge
    
    return graph


def calculate_edge_risk(p1: Point, p2: Point, no_fly_zones: List[Polygon]) -> float:
    """
    Calcula el riesgo de una arista basado en su proximidad a zonas no-fly.
    
    El riesgo aumenta cuando la arista pasa cerca de zonas restringidas.
    """
    if not no_fly_zones:
        return 0.0
    
    risk = 0.0
    mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    for zone in no_fly_zones:
        # Calcular distancia al centroide de la zona
        centroid = (sum(v[0] for v in zone) / len(zone),
                   sum(v[1] for v in zone) / len(zone))
        
        dist_to_zone = distance(mid_point, centroid)
        
        # Riesgo inversamente proporcional a la distancia
        # Más cerca = más riesgo
        zone_risk = max(0, 50.0 - dist_to_zone) / 50.0
        risk += zone_risk
    
    return risk


def save_graph_to_json(graph: DroneGraph, filepath: str):
    """Guarda un grafo a un archivo JSON."""
    data = {
        'hub': {
            'id': graph.hub_id,
            'x': graph.nodes[graph.hub_id].x,
            'y': graph.nodes[graph.hub_id].y
        },
        'destinations': [],
        'no_fly_zones': [],
        'charging_stations': graph.charging_stations
    }
    
    for node in graph.nodes.values():
        if not node.is_hub:
            data['destinations'].append({
                'id': node.id,
                'x': node.x,
                'y': node.y,
                'is_charging': node.is_charging_station
            })
    
    for zone in graph.no_fly_zones:
        data['no_fly_zones'].append({
            'vertices': [list(v) for v in zone]
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# Pruebas del módulo
if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBAS DEL MÓDULO DE GRAFOS")
    print("=" * 60)
    
    # Crear grafo de prueba
    graph = DroneGraph()
    
    # Añadir nodos
    graph.nodes[0] = Node(0, 50, 50, is_hub=True)
    graph.nodes[1] = Node(1, 20, 30)
    graph.nodes[2] = Node(2, 80, 40)
    graph.nodes[3] = Node(3, 60, 80)
    
    graph.hub_id = 0
    
    # Añadir zona no-fly
    graph.no_fly_zones.append([(40, 40), (60, 40), (60, 60), (40, 60)])
    
    # Crear aristas
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
                
                graph.edges[(i, j)] = Edge(i, j, dist, 0.1, dist/100, is_valid)
    
    print(f"\nNúmero de nodos: {graph.num_nodes}")
    print(f"Destinos: {graph.destination_ids}")
    print(f"Hub ID: {graph.hub_id}")
    print(f"Zonas no-fly: {len(graph.no_fly_zones)}")
    
    # Verificar aristas válidas
    print("\nAristas válidas:")
    for (i, j), edge in graph.edges.items():
        status = "✓ válida" if edge.is_valid else "✗ inválida (no-fly)"
        print(f"  {i} -> {j}: {status}")
    
    print("\n" + "=" * 60)
