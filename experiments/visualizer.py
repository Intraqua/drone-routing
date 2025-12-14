"""
Visualizador de rutas y soluciones para el problema de drones.
Genera gráficas de instancias, rutas y frentes de Pareto.

Autor: David Valbuena Segura
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict, Any, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Advertencia: matplotlib no disponible")

from common.graph import DroneGraph
from common.pareto import Solution


def plot_instance(instance: Dict[str, Any], 
                  output_path: Optional[str] = None,
                  title: Optional[str] = None,
                  show: bool = False) -> None:
    """
    Visualiza una instancia del problema.
    
    Args:
        instance: Diccionario con datos de la instancia
        output_path: Ruta para guardar la imagen
        title: Título personalizado
        show: Si mostrar la figura interactivamente
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib requerido para visualización")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    map_size = instance.get("map_size", 100)
    
    # Dibujar zonas no-fly (primero, para que queden detrás)
    for i, zone in enumerate(instance.get("no_fly_zones", [])):
        vertices = zone["vertices"]
        polygon = patches.Polygon(
            vertices, 
            closed=True,
            facecolor='#ffcccc',
            edgecolor='#cc0000',
            alpha=0.5,
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(polygon)
        
        # Etiqueta de zona
        centroid = (sum(v[0] for v in vertices) / len(vertices),
                   sum(v[1] for v in vertices) / len(vertices))
        ax.annotate(f'NFZ-{i+1}', centroid, ha='center', va='center',
                   fontsize=8, color='#cc0000', fontweight='bold')
    
    # Dibujar hub
    hub = instance["hub"]
    ax.plot(hub["x"], hub["y"], 's', markersize=18, 
            color='#27ae60', markeredgecolor='#1e8449',
            markeredgewidth=2, label='Hub', zorder=5)
    ax.annotate('HUB', (hub["x"], hub["y"]), 
                textcoords="offset points", xytext=(0, -25),
                ha='center', fontsize=10, fontweight='bold',
                color='#1e8449')
    
    # Dibujar destinos
    charging_stations = instance.get("charging_stations", [])
    
    for dest in instance["destinations"]:
        is_charging = dest["id"] in charging_stations or dest.get("is_charging", False)
        
        if is_charging:
            color = '#f39c12'
            marker = '^'
            size = 12
        else:
            color = '#3498db'
            marker = 'o'
            size = 10
        
        ax.plot(dest["x"], dest["y"], marker, markersize=size, 
                color=color, markeredgecolor='white',
                markeredgewidth=1.5, zorder=4)
        ax.annotate(str(dest["id"]), (dest["x"], dest["y"]),
                   textcoords="offset points", xytext=(8, 8),
                   fontsize=9, fontweight='bold')
    
    # Configuración del gráfico
    margin = map_size * 0.05
    ax.set_xlim(-margin, map_size + margin)
    ax.set_ylim(-margin, map_size + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coordenada X', fontsize=11)
    ax.set_ylabel('Coordenada Y', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        n_dest = len(instance["destinations"])
        n_zones = len(instance.get("no_fly_zones", []))
        ax.set_title(f"Instancia: {instance.get('name', 'N/A')}\n"
                    f"{n_dest} destinos, {n_zones} zonas no-fly",
                    fontsize=14, fontweight='bold')
    
    # Leyenda
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
               markersize=12, markeredgecolor='#1e8449', 
               markeredgewidth=2, label='Hub'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, label='Destino'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#f39c12',
               markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, label='Estación recarga'),
        patches.Patch(facecolor='#ffcccc', edgecolor='#cc0000',
                     alpha=0.5, linestyle='--', label='Zona no-fly')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Instancia guardada: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_route(instance: Dict[str, Any],
               route: List[int],
               objectives: tuple = None,
               output_path: Optional[str] = None,
               title: Optional[str] = None,
               show: bool = False) -> None:
    """
    Visualiza una ruta sobre la instancia.
    
    Args:
        instance: Diccionario con datos de la instancia
        route: Lista de IDs de nodos en orden de visita
        objectives: Tupla (distancia, riesgo, batería) opcional
        output_path: Ruta para guardar la imagen
        title: Título personalizado
        show: Si mostrar la figura interactivamente
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib requerido para visualización")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    map_size = instance.get("map_size", 100)
    
    # Crear diccionario de posiciones
    positions = {}
    hub = instance["hub"]
    positions[hub["id"]] = (hub["x"], hub["y"])
    
    for dest in instance["destinations"]:
        positions[dest["id"]] = (dest["x"], dest["y"])
    
    # Dibujar zonas no-fly
    for zone in instance.get("no_fly_zones", []):
        vertices = zone["vertices"]
        polygon = patches.Polygon(
            vertices, 
            closed=True,
            facecolor='#ffcccc',
            edgecolor='#cc0000',
            alpha=0.4,
            linewidth=2
        )
        ax.add_patch(polygon)
    
    # Dibujar ruta
    for i in range(len(route) - 1):
        from_id = route[i]
        to_id = route[i + 1]
        
        if from_id in positions and to_id in positions:
            x1, y1 = positions[from_id]
            x2, y2 = positions[to_id]
            
            # Flecha de la ruta
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                      lw=2, mutation_scale=15),
                       zorder=3)
            
            # Número de orden
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.annotate(str(i + 1), (mid_x, mid_y),
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='circle,pad=0.2', 
                                facecolor='white', edgecolor='gray'),
                       zorder=6)
    
    # Dibujar hub
    ax.plot(hub["x"], hub["y"], 's', markersize=18, 
            color='#27ae60', markeredgecolor='#1e8449',
            markeredgewidth=2, zorder=5)
    ax.annotate('HUB', (hub["x"], hub["y"]), 
                textcoords="offset points", xytext=(0, -25),
                ha='center', fontsize=10, fontweight='bold')
    
    # Dibujar destinos
    charging_stations = instance.get("charging_stations", [])
    
    for dest in instance["destinations"]:
        is_charging = dest["id"] in charging_stations
        color = '#f39c12' if is_charging else '#3498db'
        marker = '^' if is_charging else 'o'
        
        ax.plot(dest["x"], dest["y"], marker, markersize=12, 
                color=color, markeredgecolor='white',
                markeredgewidth=1.5, zorder=4)
        ax.annotate(str(dest["id"]), (dest["x"], dest["y"]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=9, fontweight='bold')
    
    # Configuración
    margin = map_size * 0.05
    ax.set_xlim(-margin, map_size + margin)
    ax.set_ylim(-margin, map_size + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Coordenada X', fontsize=11)
    ax.set_ylabel('Coordenada Y', fontsize=11)
    
    # Título
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        route_str = ' -> '.join(map(str, route))
        title_text = f"Ruta: {route_str}"
        if objectives:
            title_text += f"\nDist={objectives[0]:.1f}, Risk={objectives[1]:.2f}, Bat={objectives[2]:.2f}"
        ax.set_title(title_text, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Ruta guardada: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_pareto_front(solutions: List[Solution],
                      output_path: Optional[str] = None,
                      title: str = "Frente de Pareto",
                      show: bool = False) -> None:
    """
    Visualiza el frente de Pareto en 2D (distancia vs riesgo).
    
    Args:
        solutions: Lista de soluciones del frente
        output_path: Ruta para guardar la imagen
        title: Título del gráfico
        show: Si mostrar interactivamente
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib requerido para visualización")
        return
    
    if not solutions:
        print("No hay soluciones para visualizar")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distances = [sol.distance for sol in solutions]
    risks = [sol.risk for sol in solutions]
    batteries = [sol.recharges for sol in solutions]
    
    # Gráfica 1: Distancia vs Riesgo
    ax1 = axes[0]
    ax1.scatter(distances, risks, c='#3498db', s=100, alpha=0.7,
               edgecolors='white', linewidths=1.5)
    ax1.set_xlabel('Distancia total', fontsize=11)
    ax1.set_ylabel('Riesgo total', fontsize=11)
    ax1.set_title('Distancia vs Riesgo', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Conectar puntos del frente ordenados
    sorted_points = sorted(zip(distances, risks))
    ax1.plot([p[0] for p in sorted_points], [p[1] for p in sorted_points],
            'b--', alpha=0.5, linewidth=1)
    
    # Gráfica 2: Distancia vs Batería
    ax2 = axes[1]
    ax2.scatter(distances, batteries, c='#e74c3c', s=100, alpha=0.7,
               edgecolors='white', linewidths=1.5)
    ax2.set_xlabel('Distancia total', fontsize=11)
    ax2.set_ylabel('Consumo batería', fontsize=11)
    ax2.set_title('Distancia vs Batería', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Riesgo vs Batería
    ax3 = axes[2]
    ax3.scatter(risks, batteries, c='#27ae60', s=100, alpha=0.7,
               edgecolors='white', linewidths=1.5)
    ax3.set_xlabel('Riesgo total', fontsize=11)
    ax3.set_ylabel('Consumo batería', fontsize=11)
    ax3.set_title('Riesgo vs Batería', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f"{title} ({len(solutions)} soluciones)", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Frente de Pareto guardado: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_comparison(results: Dict[str, Any],
                    metric: str = 'time',
                    output_path: Optional[str] = None,
                    show: bool = False) -> None:
    """
    Genera gráfica comparativa de algoritmos.
    
    Args:
        results: Diccionario con resultados de experimentos
        metric: 'time', 'memory', 'hypervolume', o 'diversity'
        output_path: Ruta para guardar
        show: Si mostrar interactivamente
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    instances = []
    n_values = []
    values_bb = []
    values_geo = []
    values_sa = []
    
    metric_map = {
        'time': 'time_mean',
        'memory': 'memory_peak',
        'hypervolume': 'hypervolume',
        'diversity': 'diversity'
    }
    
    metric_key = metric_map.get(metric, 'time_mean')
    
    for instance_name, data in sorted(results.get('instances', {}).items()):
        if 'error' in data:
            continue
        
        instances.append(instance_name)
        n_values.append(data.get('n_destinations', 0))
        
        bb = data.get('algorithms', {}).get('branch_bound', {})
        geo = data.get('algorithms', {}).get('geometric', {})
        sa = data.get('algorithms', {}).get('simulated_annealing', {})
        
        values_bb.append(bb.get(metric_key) if not bb.get('skipped') else None)
        values_geo.append(geo.get(metric_key))
        values_sa.append(sa.get(metric_key))
    
    width = 0.25
    x_pos = np.arange(len(instances))
    
    # Preparar datos
    bb_plot = [v if v is not None else 0 for v in values_bb]
    geo_plot = [v if v is not None else 0 for v in values_geo]
    sa_plot = [v if v is not None else 0 for v in values_sa]
    
    ax.bar(x_pos - width, bb_plot, width, label='Branch & Bound', color='#2ecc71')
    ax.bar(x_pos, geo_plot, width, label='Geométrico', color='#3498db')
    ax.bar(x_pos + width, sa_plot, width, label='Simulated Annealing', color='#e74c3c')
    
    ax.set_xlabel('Instancia', fontsize=11)
    
    ylabel_map = {
        'time': 'Tiempo (s)',
        'memory': 'Memoria (MB)',
        'hypervolume': 'Hipervolumen',
        'diversity': 'Diversidad'
    }
    ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=11)
    
    title_map = {
        'time': 'Tiempo de Ejecución',
        'memory': 'Uso de Memoria',
        'hypervolume': 'Hipervolumen del Frente',
        'diversity': 'Diversidad del Frente'
    }
    ax.set_title(title_map.get(metric, metric), fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={n}' for n in n_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparación guardada: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


# Pruebas del módulo
if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBAS DEL MÓDULO DE VISUALIZACIÓN")
    print("=" * 60)
    
    # Crear instancia de prueba
    test_instance = {
        "name": "test_instance",
        "map_size": 100,
        "hub": {"id": 0, "x": 50, "y": 50},
        "destinations": [
            {"id": 1, "x": 20, "y": 30},
            {"id": 2, "x": 80, "y": 30},
            {"id": 3, "x": 80, "y": 70},
            {"id": 4, "x": 20, "y": 70}
        ],
        "no_fly_zones": [
            {"vertices": [[40, 40], [60, 40], [60, 60], [40, 60]]}
        ],
        "charging_stations": [2]
    }
    
    print("\nGenerando visualización de instancia...")
    plot_instance(test_instance, output_path="test_instance.png")
    
    print("\nGenerando visualización de ruta...")
    test_route = [0, 1, 4, 3, 2, 0]
    plot_route(test_instance, test_route, 
               objectives=(150.0, 0.5, 1.5),
               output_path="test_route.png")
    
    print("\nGenerando visualización de frente de Pareto...")
    test_solutions = [
        Solution([0, 1, 2, 0], (100, 0.3, 1.0)),
        Solution([0, 2, 1, 0], (120, 0.2, 0.8)),
        Solution([0, 1, 2, 0], (90, 0.5, 1.2)),
    ]
    plot_pareto_front(test_solutions, output_path="test_pareto.png")
    
    print("\n" + "=" * 60)
    print("Visualizaciones generadas")
    print("=" * 60)
