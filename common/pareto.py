"""
Módulo de optimización multiobjetivo y frontera de Pareto.
Implementa los conceptos del Tema 5 y 9 del curso.

Autor: David Valbuena Segura
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Solution:
    """
    Representa una solución al problema de rutas de drones.
    
    Atributos:
        route: Lista de IDs de nodos en orden de visita
        objectives: Tupla con los valores de los 3 objetivos
                   (distancia, riesgo, recargas)
    """
    route: List[int]
    objectives: Tuple[float, float, float]
    
    @property
    def distance(self) -> float:
        return self.objectives[0]
    
    @property
    def risk(self) -> float:
        return self.objectives[1]
    
    @property
    def recharges(self) -> float:
        return self.objectives[2]
    
    def __eq__(self, other):
        if not isinstance(other, Solution):
            return False
        return self.route == other.route
    
    def __hash__(self):
        return hash(tuple(self.route))


def dominates(sol_a: Solution, sol_b: Solution) -> bool:
    """
    Verifica si la solución A domina a la solución B.
    
    Definición del Tema 9: A domina a B si:
    - Para todos los objetivos: f_i(A) <= f_i(B)
    - Para al menos un objetivo: f_j(A) < f_j(B)
    
    En problemas de minimización, menor es mejor.
    """
    obj_a = sol_a.objectives
    obj_b = sol_b.objectives
    
    # Verificar que A es al menos tan buena como B en todos los objetivos
    all_leq = all(a <= b for a, b in zip(obj_a, obj_b))
    
    # Verificar que A es estrictamente mejor en al menos un objetivo
    any_less = any(a < b for a, b in zip(obj_a, obj_b))
    
    return all_leq and any_less


def is_dominated(solution: Solution, population: List[Solution]) -> bool:
    """
    Verifica si una solución está dominada por alguna otra en la población.
    """
    for other in population:
        if other != solution and dominates(other, solution):
            return True
    return False


def get_pareto_front(population: List[Solution]) -> List[Solution]:
    """
    Obtiene el frente de Pareto (soluciones no dominadas) de una población.
    
    Implementa el concepto de frontera de Pareto del Tema 9:
    El conjunto de soluciones Pareto óptimas P se define como el conjunto
    de todas las soluciones no dominadas.
    """
    if not population:
        return []
    
    pareto_front = []
    
    for solution in population:
        # Verificar si esta solución está dominada
        if not is_dominated(solution, population):
            pareto_front.append(solution)
    
    return pareto_front


def fast_non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
    """
    Implementa el algoritmo Fast Non-Dominated Sorting de NSGA-II.
    
    Del Tema 9: Este algoritmo crea dos conjuntos para cada solución p:
    - n_p: número de soluciones que dominan a p
    - S_p: conjunto de soluciones que p domina
    
    Returns:
        Lista de frentes, donde fronts[0] es el frente de Pareto
    """
    fronts = [[]]
    
    # Para cada solución, calcular n_p y S_p
    n = {id(sol): 0 for sol in population}
    S = {id(sol): [] for sol in population}
    
    for p in population:
        for q in population:
            if p != q:
                if dominates(p, q):
                    S[id(p)].append(q)
                elif dominates(q, p):
                    n[id(p)] += 1
        
        # Si n_p == 0, p pertenece al primer frente
        if n[id(p)] == 0:
            fronts[0].append(p)
    
    # Construir frentes sucesivos
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[id(p)]:
                n[id(q)] -= 1
                if n[id(q)] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    
    return [f for f in fronts if f]  # Eliminar frentes vacíos


def crowding_distance(front: List[Solution]) -> dict:
    """
    Calcula la distancia de amontonamiento (crowding distance) para NSGA-II.
    
    Del Tema 9: Prima a las soluciones no amontonadas para mejorar
    la cobertura de la frontera de Pareto.
    """
    if len(front) <= 2:
        return {id(sol): float('inf') for sol in front}
    
    distances = {id(sol): 0.0 for sol in front}
    num_objectives = len(front[0].objectives)
    
    for m in range(num_objectives):
        # Ordenar por objetivo m
        sorted_front = sorted(front, key=lambda s: s.objectives[m])
        
        # Extremos tienen distancia infinita
        distances[id(sorted_front[0])] = float('inf')
        distances[id(sorted_front[-1])] = float('inf')
        
        # Calcular rango del objetivo
        obj_range = (sorted_front[-1].objectives[m] - 
                    sorted_front[0].objectives[m])
        
        if obj_range == 0:
            continue
        
        # Calcular distancia para soluciones intermedias
        for i in range(1, len(sorted_front) - 1):
            distances[id(sorted_front[i])] += (
                sorted_front[i + 1].objectives[m] - 
                sorted_front[i - 1].objectives[m]
            ) / obj_range
    
    return distances


def hypervolume(front: List[Solution], reference_point: Tuple[float, ...]) -> float:
    """
    Calcula el hipervolumen dominado por un frente de Pareto.
    
    El hipervolumen es una métrica de calidad que mide el espacio
    objetivo dominado por las soluciones del frente.
    
    Para 2D, es el área dominada. Para 3D, el volumen.
    
    Args:
        front: Lista de soluciones del frente
        reference_point: Punto de referencia (peor caso posible)
    
    Returns:
        Valor del hipervolumen
    """
    if not front:
        return 0.0
    
    # Para simplificar, usamos una aproximación 2D (primeros 2 objetivos)
    # considerando distancia y riesgo como los objetivos principales
    points = [(sol.objectives[0], sol.objectives[1]) for sol in front]
    
    # Ordenar por primer objetivo
    points = sorted(points)
    
    # Calcular área dominada
    hv = 0.0
    prev_y = reference_point[1]
    
    for x, y in points:
        if y < prev_y:
            # Añadir rectángulo
            width = reference_point[0] - x
            height = prev_y - y
            hv += width * height
            prev_y = y
    
    return hv


def spacing(front: List[Solution]) -> float:
    """
    Calcula la métrica de espaciado (diversidad) del frente de Pareto.
    
    Mide la uniformidad de la distribución de soluciones en el frente.
    Un valor menor indica mejor distribución.
    """
    if len(front) < 2:
        return 0.0
    
    # Calcular distancias mínimas entre soluciones consecutivas
    distances = []
    
    for i, sol_i in enumerate(front):
        min_dist = float('inf')
        for j, sol_j in enumerate(front):
            if i != j:
                # Distancia euclidiana en el espacio de objetivos
                dist = sum((a - b)**2 for a, b in 
                          zip(sol_i.objectives, sol_j.objectives))**0.5
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calcular desviación estándar
    mean_dist = sum(distances) / len(distances)
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    
    return math.sqrt(variance)


def diversity(front: List[Solution]) -> float:
    """
    Calcula la diversidad del frente como la distancia media entre soluciones.
    Mayor valor indica mejor diversidad (cobertura más amplia).
    """
    if len(front) < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    
    for i, sol_i in enumerate(front):
        for j, sol_j in enumerate(front):
            if i < j:
                dist = sum((a - b)**2 for a, b in 
                          zip(sol_i.objectives, sol_j.objectives))**0.5
                total_dist += dist
                count += 1
    
    return total_dist / count if count > 0 else 0.0


# Pruebas del módulo
if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBAS DEL MÓDULO DE PARETO")
    print("=" * 60)
    
    # Crear soluciones de prueba
    solutions = [
        Solution([0, 1, 2, 0], (10.0, 5.0, 1.0)),   # A
        Solution([0, 2, 1, 0], (8.0, 7.0, 2.0)),    # B
        Solution([0, 1, 2, 0], (12.0, 3.0, 1.0)),   # C
        Solution([0, 2, 1, 0], (15.0, 8.0, 3.0)),   # D (dominada)
        Solution([0, 1, 2, 0], (9.0, 4.0, 2.0)),    # E
    ]
    
    print("\nSoluciones:")
    for i, sol in enumerate(solutions):
        print(f"  {chr(65+i)}: objetivos = {sol.objectives}")
    
    # Test de dominancia
    print("\nTest de dominancia:")
    print(f"  A domina a D: {dominates(solutions[0], solutions[3])}")
    print(f"  B domina a A: {dominates(solutions[1], solutions[0])}")
    
    # Obtener frente de Pareto
    pareto = get_pareto_front(solutions)
    print(f"\nFrente de Pareto ({len(pareto)} soluciones):")
    for sol in pareto:
        print(f"  Objetivos: {sol.objectives}")
    
    # Calcular hipervolumen
    ref_point = (20.0, 10.0, 5.0)
    hv = hypervolume(pareto, ref_point)
    print(f"\nHipervolumen (ref={ref_point[:2]}): {hv:.2f}")
    
    # Calcular diversidad
    div = diversity(pareto)
    print(f"Diversidad: {div:.2f}")
    
    # Fast non-dominated sort
    fronts = fast_non_dominated_sort(solutions)
    print(f"\nFast Non-Dominated Sort:")
    for i, front in enumerate(fronts):
        print(f"  Frente F{i}: {len(front)} soluciones")
    
    print("\n" + "=" * 60)
