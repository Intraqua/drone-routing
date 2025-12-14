"""
Módulo de geometría computacional para detección de intersecciones.
Implementa los algoritmos del Tema 5 y 7 del curso.

Autor: David Valbuena Segura
"""

import math
from typing import Tuple, List, Optional

# Tipo para representar un punto 2D
Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Polygon = List[Point]


def cross_product(o: Point, a: Point, b: Point) -> float:
    """
    Calcula el producto cruzado de los vectores OA y OB.
    Determina el sentido de giro:
    - Positivo: giro antihorario
    - Negativo: giro horario
    - Cero: colineales
    
    Basado en el material del Tema 5: Intersección de segmentos.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """
    Verifica si el punto q está sobre el segmento pr,
    asumiendo que p, q, r son colineales.
    """
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    """
    Determina si dos segmentos (p1,q1) y (p2,q2) se intersectan.
    
    Implementa el algoritmo de intersección de segmentos del Tema 5:
    - Cada segmento cubre la línea del otro si los extremos están en lados opuestos
    - Maneja casos especiales de colinealidad
    
    Returns:
        True si los segmentos se intersectan, False en caso contrario
    """
    # Calcular las orientaciones necesarias
    d1 = cross_product(p2, q2, p1)
    d2 = cross_product(p2, q2, q1)
    d3 = cross_product(p1, q1, p2)
    d4 = cross_product(p1, q1, q2)
    
    # Caso general: los segmentos se cruzan
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Casos especiales de colinealidad
    epsilon = 1e-10
    
    if abs(d1) < epsilon and on_segment(p2, p1, q2):
        return True
    if abs(d2) < epsilon and on_segment(p2, q1, q2):
        return True
    if abs(d3) < epsilon and on_segment(p1, p2, q1):
        return True
    if abs(d4) < epsilon and on_segment(p1, q2, q1):
        return True
    
    return False


def segment_intersects_polygon(p1: Point, p2: Point, polygon: Polygon) -> bool:
    """
    Verifica si un segmento intersecta con algún lado de un polígono.
    
    Args:
        p1, p2: Extremos del segmento a verificar
        polygon: Lista de vértices del polígono (cerrado automáticamente)
    
    Returns:
        True si hay intersección con algún lado del polígono
    """
    n = len(polygon)
    for i in range(n):
        # Lado del polígono: desde vértice i hasta vértice (i+1) % n
        poly_p1 = polygon[i]
        poly_p2 = polygon[(i + 1) % n]
        
        if segments_intersect(p1, p2, poly_p1, poly_p2):
            return True
    
    return False


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Determina si un punto está dentro de un polígono usando el método
    de ray casting (Tema 5: Inclusión de un punto en un polígono).
    
    Traza un rayo horizontal hacia la derecha y cuenta intersecciones.
    Si el número es impar, el punto está dentro.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        # Verificar si el rayo horizontal cruza el lado
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def segment_passes_through_polygon(p1: Point, p2: Point, polygon: Polygon) -> bool:
    """
    Verifica si un segmento atraviesa o está dentro de una zona no-fly.
    
    Un segmento viola una zona no-fly si:
    1. Intersecta algún lado del polígono, O
    2. Algún extremo está dentro del polígono, O
    3. El punto medio está dentro del polígono (para casos donde
       el segmento está completamente contenido)
    """
    # Verificar intersección con los lados
    if segment_intersects_polygon(p1, p2, polygon):
        return True
    
    # Verificar si algún extremo está dentro
    if point_in_polygon(p1, polygon) or point_in_polygon(p2, polygon):
        return True
    
    # Verificar punto medio (para segmentos completamente contenidos)
    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    if point_in_polygon(mid, polygon):
        return True
    
    return False


def distance(p1: Point, p2: Point) -> float:
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def polygon_area(polygon: Polygon) -> float:
    """
    Calcula el área de un polígono usando la fórmula del cordón de zapatos
    (Shoelace formula) del Tema 5.
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def polygon_centroid(polygon: Polygon) -> Point:
    """Calcula el centroide de un polígono."""
    n = len(polygon)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


# Funciones de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBAS DEL MÓDULO DE GEOMETRÍA")
    print("=" * 60)
    
    # Test de intersección de segmentos
    print("\n1. Test de intersección de segmentos:")
    
    # Segmentos que se cruzan
    p1, q1 = (0, 0), (4, 4)
    p2, q2 = (0, 4), (4, 0)
    result = segments_intersect(p1, q1, p2, q2)
    print(f"   Segmentos (0,0)-(4,4) y (0,4)-(4,0): {'INTERSECTAN' if result else 'NO intersectan'}")
    
    # Segmentos paralelos
    p1, q1 = (0, 0), (4, 0)
    p2, q2 = (0, 2), (4, 2)
    result = segments_intersect(p1, q1, p2, q2)
    print(f"   Segmentos paralelos: {'INTERSECTAN' if result else 'NO intersectan'}")
    
    # Test de punto en polígono
    print("\n2. Test de punto en polígono:")
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    
    point_inside = (2, 2)
    point_outside = (5, 5)
    
    print(f"   Punto (2,2) en cuadrado [0,4]x[0,4]: {'DENTRO' if point_in_polygon(point_inside, square) else 'FUERA'}")
    print(f"   Punto (5,5) en cuadrado [0,4]x[0,4]: {'DENTRO' if point_in_polygon(point_outside, square) else 'FUERA'}")
    
    # Test de segmento atravesando polígono
    print("\n3. Test de segmento atravesando zona no-fly:")
    no_fly = [(2, 2), (3, 2), (3, 3), (2, 3)]
    
    seg_through = ((0, 0), (5, 5))
    seg_around = ((0, 0), (1, 1))
    
    print(f"   Segmento (0,0)-(5,5) atraviesa zona: {'SÍ' if segment_passes_through_polygon(*seg_through, no_fly) else 'NO'}")
    print(f"   Segmento (0,0)-(1,1) atraviesa zona: {'SÍ' if segment_passes_through_polygon(*seg_around, no_fly) else 'NO'}")
    
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS COMPLETADAS")
    print("=" * 60)
