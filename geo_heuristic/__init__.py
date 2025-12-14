"""
Módulo de heurísticas geométricas basadas en grafo de visibilidad.
Implementa técnicas del Tema 5 y 7.
"""

from .visibility_graph import (
    VisibilityGraph,
    VisibilityEdge,
    GeometricHeuristic,
    solve
)

__all__ = ['VisibilityGraph', 'VisibilityEdge', 'GeometricHeuristic', 'solve']
