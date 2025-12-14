"""
Módulo de visualización de rutas y soluciones.
Genera gráficas de las instancias y las rutas encontradas.

Autor: David Valbuena Segura
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Dict, Any
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Advertencia: matplotlib no disponible. Visualización deshabilitada.")

from common.graph import DroneGraph
from common.pareto import Solution


def plot_instance(graph: DroneGraph, 
                 title: str = "Instancia",
                 output_path: Optional[str] = None,
                 show: bool = True):
    """
    Visualiza una instancia del problema mostrando nodos y zonas no-fly.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determinar límites del mapa
    all_x = [n.x for n in graph.nodes.values()]
    all_y = [n.y for n in graph.nodes.values()]
    
    for zone in graph.no_fly_zones:
        all_x.extend([v[0] for v in zone])
        all_y.extend([v[1] for v in zone])
    
    margin = 10
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    
    # Dibujar zonas no-fly
    for zone in graph.no_fly_zones:
        polygon = patches.Polygon(
            zone,
            closed=True,
            facecolor='red',
            edgecolor='darkred',
            alpha=0.3,
            linewidth=2,
            label='Zona no-fly'
        )
        ax.add_patch(polygon)
    
    # Dibujar nodos
    for node in graph.nodes.values():
        if node.is_hub:
            ax.plot(node.x, node.y, 's', markersize=15, 
                   color='green', zorder=5)
            ax.annotate('HUB', (node.x, node.y),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontweight='bold', fontsize=10)
        elif node.is_charging_station:
            ax.plot(node.x, node.y, '^', markersize=12,
                   color='orange', zorder=4)
            ax.annotate(str(node.id), (node.x, node.y),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=9)
        else:
            ax.plot(node.x, node.y, 'o', markersize=10,
                   color='blue', zorder=4)
            ax.annotate(str(node.id), (node.x, node.y),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=9)
    
    # Configuración
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Leyenda
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markersize=12, label='Hub'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Destino'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
               markersize=10, label='Estación recarga'),
        patches.Patch(facecolor='red', alpha=0.3, edgecolor='darkred',
                     label='Zona no-fly')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_route(graph: DroneGraph,
              solution: Solution,
              title: str = "Ruta",
              output_path: Optional[str] = None,
              show: bool = True):
    """
    Visualiza una ruta sobre la instancia.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Límites
    all_x = [n.x for n in graph.nodes.values()]
    all_y = [n.y for n in graph.nodes.values()]
    
    for zone in graph.no_fly_zones:
        all_x.extend([v[0] for v in zone])
        all_y.extend([v[1] for v in zone])
    
    margin = 10
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    
    # Zonas no-fly
    for zone in graph.no_fly_zones:
        polygon = patches.Polygon(
            zone,
            closed=True,
            facecolor='red',
            edgecolor='darkred',
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(polygon)
    
    # Dibujar ruta
    route = solution.route
    for i in range(len(route) - 1):
        node_i = graph.nodes[route[i]]
        node_j = graph.nodes[route[i + 1]]
        
        ax.annotate('',
                   xy=(node_j.x, node_j.y),
                   xytext=(node_i.x, node_i.y),
                   arrowprops=dict(arrowstyle='->', color='blue',
                                  lw=2, alpha=0.7))
    
    # Nodos
    for node in graph.nodes.values():
        if node.is_hub:
            ax.plot(node.x, node.y, 's', markersize=15,
                   color='green', zorder=5)
            ax.annotate('HUB', (node.x, node.y),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontweight='bold', fontsize=10)
        elif node.is_charging_station:
            ax.plot(node.x, node.y, '^', markersize=12,
                   color='orange', zorder=4)
            ax.annotate(str(node.id), (node.x, node.y),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=9)
        else:
            ax.plot(node.x, node.y, 'o', markersize=10,
                   color='blue', zorder=4)
            ax.annotate(str(node.id), (node.x, node.y),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=9)
    
    # Configuración
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Información de la solución
    info = (f"Distancia: {solution.distance:.2f} | "
            f"Riesgo: {solution.risk:.2f} | "
            f"Batería: {solution.recharges:.2f}")
    ax.set_title(f"{title}\n{info}", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_pareto_front(solutions: List[Solution],
                     title: str = "Frontera de Pareto",
                     output_path: Optional[str] = None,
                     show: bool = True):
    """
    Visualiza el frente de Pareto en el espacio de objetivos.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible")
        return
    
    if not solutions:
        print("No hay soluciones para visualizar")
        return
    
    fig = plt.figure(figsize=(14, 5))
    
    # 2D: Distancia vs Riesgo
    ax1 = fig.add_subplot(131)
    distances = [s.distance for s in solutions]
    risks = [s.risk for s in solutions]
    ax1.scatter(distances, risks, c='blue', s=100, alpha=0.7)
    ax1.set_xlabel('Distancia', fontsize=11)
    ax1.set_ylabel('Riesgo', fontsize=11)
    ax1.set_title('Distancia vs Riesgo', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2D: Distancia vs Batería
    ax2 = fig.add_subplot(132)
    batteries = [s.recharges for s in solutions]
    ax2.scatter(distances, batteries, c='green', s=100, alpha=0.7)
    ax2.set_xlabel('Distancia', fontsize=11)
    ax2.set_ylabel('Consumo Batería', fontsize=11)
    ax2.set_title('Distancia vs Batería', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 2D: Riesgo vs Batería
    ax3 = fig.add_subplot(133)
    ax3.scatter(risks, batteries, c='red', s=100, alpha=0.7)
    ax3.set_xlabel('Riesgo', fontsize=11)
    ax3.set_ylabel('Consumo Batería', fontsize=11)
    ax3.set_title('Riesgo vs Batería', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f"{title} ({len(solutions)} soluciones)", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_algorithm_comparison(results: Dict[str, Any],
                            output_path: Optional[str] = None,
                            show: bool = True):
    """
    Genera gráficas comparativas de los algoritmos.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible")
        return
    
    instances = list(results['instances'].keys())
    algorithms = ['branch_bound', 'geometric', 'simulated_annealing']
    algo_names = ['Branch & Bound', 'Geométrico', 'Simulated Annealing']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Preparar datos
    times = {algo: [] for algo in algorithms}
    hvs = {algo: [] for algo in algorithms}
    divs = {algo: [] for algo in algorithms}
    sols = {algo: [] for algo in algorithms}
    n_values = []
    
    for instance_name in sorted(instances):
        data = results['instances'][instance_name]
        if 'error' in data:
            continue
        
        n_values.append(data.get('n_destinations', 0))
        
        for algo in algorithms:
            algo_data = data['algorithms'].get(algo, {})
            if algo_data.get('skipped') or 'error' in algo_data:
                times[algo].append(None)
                hvs[algo].append(None)
                divs[algo].append(None)
                sols[algo].append(None)
            else:
                times[algo].append(algo_data.get('time_mean'))
                hvs[algo].append(algo_data.get('hypervolume'))
                divs[algo].append(algo_data.get('diversity'))
                sols[algo].append(algo_data.get('num_solutions'))
    
    x = np.arange(len(n_values))
    width = 0.25
    
    # Gráfica 1: Tiempo (escala log)
    ax1 = axes[0, 0]
    for i, (algo, name, color) in enumerate(zip(algorithms, algo_names, colors)):
        valid = [(n, t) for n, t in zip(n_values, times[algo]) if t is not None]
        if valid:
            ax1.semilogy([p[0] for p in valid], [p[1] for p in valid],
                        'o-', label=name, color=color, linewidth=2, markersize=8)
    ax1.set_xlabel('N (destinos)')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempo de Ejecución')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Hipervolumen
    ax2 = axes[0, 1]
    for i, (algo, name, color) in enumerate(zip(algorithms, algo_names, colors)):
        values = [h if h is not None else 0 for h in hvs[algo]]
        ax2.bar(x + i*width - width, values, width, label=name, color=color)
    ax2.set_xlabel('Instancia')
    ax2.set_ylabel('Hipervolumen')
    ax2.set_title('Hipervolumen Dominado')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'N={n}' for n in n_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Gráfica 3: Diversidad
    ax3 = axes[1, 0]
    for i, (algo, name, color) in enumerate(zip(algorithms, algo_names, colors)):
        values = [d if d is not None else 0 for d in divs[algo]]
        ax3.bar(x + i*width - width, values, width, label=name, color=color)
    ax3.set_xlabel('Instancia')
    ax3.set_ylabel('Diversidad')
    ax3.set_title('Diversidad del Frente')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'N={n}' for n in n_values])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Gráfica 4: Número de soluciones
    ax4 = axes[1, 1]
    for i, (algo, name, color) in enumerate(zip(algorithms, algo_names, colors)):
        values = [s if s is not None else 0 for s in sols[algo]]
        ax4.bar(x + i*width - width, values, width, label=name, color=color)
    ax4.set_xlabel('Instancia')
    ax4.set_ylabel('Soluciones')
    ax4.set_title('Soluciones No Dominadas')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'N={n}' for n in n_values])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Módulo de visualización cargado.")
    print(f"matplotlib disponible: {HAS_MATPLOTLIB}")
