"""
Módulo de algoritmo exacto Branch & Bound.
Implementa la estrategia de ramificación y poda del Tema 4.
"""

from .branch_bound import BranchAndBound, BBNode, solve

__all__ = ['BranchAndBound', 'BBNode', 'solve']
