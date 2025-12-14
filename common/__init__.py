"""
Módulo común con utilidades compartidas para el problema de rutas de drones.

Contiene:
- geometry: Funciones de geometría computacional
- graph: Representación del grafo del problema
- pareto: Optimización multiobjetivo y frontera de Pareto
"""

from .geometry import (
    Point, Segment, Polygon,
    cross_product,
    segments_intersect,
    segment_intersects_polygon,
    point_in_polygon,
    segment_passes_through_polygon,
    distance,
    polygon_area,
    polygon_centroid
)

from .graph import (
    Node, Edge, DroneGraph,
    create_graph_from_json,
    save_graph_to_json,
    calculate_edge_risk
)

from .pareto import (
    Solution,
    dominates,
    is_dominated,
    get_pareto_front,
    fast_non_dominated_sort,
    crowding_distance,
    hypervolume,
    spacing,
    diversity
)

__all__ = [
    # Geometry
    'Point', 'Segment', 'Polygon',
    'cross_product', 'segments_intersect', 'segment_intersects_polygon',
    'point_in_polygon', 'segment_passes_through_polygon',
    'distance', 'polygon_area', 'polygon_centroid',
    
    # Graph
    'Node', 'Edge', 'DroneGraph',
    'create_graph_from_json', 'save_graph_to_json', 'calculate_edge_risk',
    
    # Pareto
    'Solution', 'dominates', 'is_dominated', 'get_pareto_front',
    'fast_non_dominated_sort', 'crowding_distance',
    'hypervolume', 'spacing', 'diversity'
]
