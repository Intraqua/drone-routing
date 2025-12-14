"""
Script principal de experimentación.
Ejecuta todos los algoritmos sobre todas las instancias y genera resultados.

Autor: David Valbuena Segura
"""

import sys
import os

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import List, Dict, Any
from datetime import datetime
import traceback

# Importar módulos del proyecto
from common.graph import DroneGraph, create_graph_from_json, Node, Edge
from common.geometry import distance, segment_passes_through_polygon
from common.pareto import Solution, hypervolume, diversity, spacing, get_pareto_front

from exact_bb import solve as solve_bb
from geo_heuristic import solve as solve_geo
from metaheuristic import solve as solve_sa

from experiments.benchmark import (
    benchmark_algorithm, BenchmarkResult, 
    format_time, format_memory, print_benchmark_result
)
from experiments.generate_instances import generate_all_instances, visualize_instance


def load_instance(filepath: str) -> DroneGraph:
    """Carga una instancia desde un archivo JSON y crea el grafo."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    graph = DroneGraph()
    
    # Hub
    hub = data['hub']
    graph.nodes[hub['id']] = Node(hub['id'], hub['x'], hub['y'], is_hub=True)
    graph.hub_id = hub['id']
    
    # Destinos
    for dest in data['destinations']:
        graph.nodes[dest['id']] = Node(
            dest['id'], dest['x'], dest['y'],
            is_charging_station=dest.get('is_charging', False)
        )
    
    # Zonas no-fly
    for zone in data.get('no_fly_zones', []):
        polygon = [tuple(v) for v in zone['vertices']]
        graph.no_fly_zones.append(polygon)
    
    # Estaciones de recarga
    graph.charging_stations = data.get('charging_stations', [])
    
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
    
    return graph


def run_experiments(instances_dir: str = "instances",
                   results_dir: str = "results",
                   num_replicas: int = 5,
                   time_limit: float = 60.0) -> Dict[str, Any]:
    """
    Ejecuta todos los experimentos.
    
    Args:
        instances_dir: Directorio con las instancias
        results_dir: Directorio para guardar resultados
        num_replicas: Número de réplicas por experimento
        time_limit: Tiempo límite por algoritmo (segundos)
    
    Returns:
        Diccionario con todos los resultados
    """
    # Crear directorios de resultados
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    os.makedirs(f"{results_dir}/graphs", exist_ok=True)
    
    # Cargar instancias
    instance_files = sorted([
        f for f in os.listdir(instances_dir) 
        if f.endswith('.json')
    ])
    
    if not instance_files:
        print("No se encontraron instancias. Generando...")
        generate_all_instances(instances_dir)
        instance_files = sorted([
            f for f in os.listdir(instances_dir) 
            if f.endswith('.json')
        ])
    
    print(f"\nInstancias encontradas: {len(instance_files)}")
    for f in instance_files:
        print(f"  - {f}")
    
    # Resultados
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'num_replicas': num_replicas,
        'time_limit': time_limit,
        'instances': {},
        'summary': {}
    }
    
    # Procesar cada instancia
    for instance_file in instance_files:
        instance_path = os.path.join(instances_dir, instance_file)
        instance_name = instance_file.replace('.json', '')
        
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {instance_name}")
        print(f"{'='*60}")
        
        try:
            graph = load_instance(instance_path)
            n_nodes = graph.num_nodes
            n_destinations = len(graph.destination_ids)
            
            print(f"  Nodos: {n_nodes}, Destinos: {n_destinations}")
            print(f"  Zonas no-fly: {len(graph.no_fly_zones)}")
            
            instance_results = {
                'n_destinations': n_destinations,
                'n_no_fly_zones': len(graph.no_fly_zones),
                'algorithms': {}
            }
            
            # Punto de referencia para hipervolumen
            ref_point = (1000.0, 50.0, 10.0)
            
            # 1. Branch & Bound (solo para instancias pequeñas)
            if n_destinations <= 12:
                print(f"\n  [1/3] Ejecutando Branch & Bound...")
                try:
                    times_bb = []
                    memories_bb = []
                    solutions_bb = []
                    stats_bb = []
                    
                    for rep in range(num_replicas):
                        import gc
                        import tracemalloc
                        
                        gc.collect()
                        tracemalloc.start()
                        start = time.perf_counter()
                        
                        sols, stats = solve_bb(
                            graph, 
                            max_solutions=10,
                            time_limit=min(time_limit, 30.0)
                        )
                        
                        elapsed = time.perf_counter() - start
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        times_bb.append(elapsed)
                        memories_bb.append(peak / (1024*1024))
                        solutions_bb = sols
                        stats_bb.append(stats)
                    
                    import statistics
                    
                    # Calcular métricas
                    hv = hypervolume(solutions_bb, ref_point) if solutions_bb else 0
                    div = diversity(solutions_bb) if solutions_bb else 0
                    
                    instance_results['algorithms']['branch_bound'] = {
                        'time_mean': statistics.mean(times_bb),
                        'time_std': statistics.stdev(times_bb) if len(times_bb) > 1 else 0,
                        'memory_peak': max(memories_bb),
                        'num_solutions': len(solutions_bb),
                        'hypervolume': hv,
                        'diversity': div,
                        'nodes_explored': stats_bb[-1].get('nodes_explored', 0),
                        'nodes_pruned': stats_bb[-1].get('nodes_pruned', 0)
                    }
                    
                    print(f"    Tiempo: {format_time(statistics.mean(times_bb))}")
                    print(f"    Soluciones: {len(solutions_bb)}")
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    instance_results['algorithms']['branch_bound'] = {'error': str(e)}
            else:
                print(f"\n  [1/3] Branch & Bound omitido (N > 12)")
                instance_results['algorithms']['branch_bound'] = {
                    'skipped': True,
                    'reason': 'Instance too large (N > 12)'
                }
            
            # 2. Heurística Geométrica
            print(f"\n  [2/3] Ejecutando Heurística Geométrica...")
            try:
                times_geo = []
                memories_geo = []
                solutions_geo = []
                stats_geo = []
                
                for rep in range(num_replicas):
                    import gc
                    import tracemalloc
                    
                    gc.collect()
                    tracemalloc.start()
                    start = time.perf_counter()
                    
                    sols, stats = solve_geo(graph, num_solutions=10)
                    
                    elapsed = time.perf_counter() - start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    times_geo.append(elapsed)
                    memories_geo.append(peak / (1024*1024))
                    solutions_geo = sols
                    stats_geo.append(stats)
                
                hv = hypervolume(solutions_geo, ref_point) if solutions_geo else 0
                div = diversity(solutions_geo) if solutions_geo else 0
                
                instance_results['algorithms']['geometric'] = {
                    'time_mean': statistics.mean(times_geo),
                    'time_std': statistics.stdev(times_geo) if len(times_geo) > 1 else 0,
                    'memory_peak': max(memories_geo),
                    'num_solutions': len(solutions_geo),
                    'hypervolume': hv,
                    'diversity': div,
                    'visibility_edges': stats_geo[-1].get('visibility_edges', 0)
                }
                
                print(f"    Tiempo: {format_time(statistics.mean(times_geo))}")
                print(f"    Soluciones: {len(solutions_geo)}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                instance_results['algorithms']['geometric'] = {'error': str(e)}
            
            # 3. Simulated Annealing
            print(f"\n  [3/3] Ejecutando Simulated Annealing...")
            try:
                times_sa = []
                memories_sa = []
                solutions_sa = []
                stats_sa = []
                
                for rep in range(num_replicas):
                    import gc
                    import tracemalloc
                    
                    gc.collect()
                    tracemalloc.start()
                    start = time.perf_counter()
                    
                    sols, stats = solve_sa(
                        graph,
                        initial_temp=5000.0,
                        cooling_rate=0.995,
                        max_iterations=5000,
                        time_limit=time_limit / num_replicas,
                        num_runs=2
                    )
                    
                    elapsed = time.perf_counter() - start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    times_sa.append(elapsed)
                    memories_sa.append(peak / (1024*1024))
                    solutions_sa = sols
                    stats_sa.append(stats)
                
                hv = hypervolume(solutions_sa, ref_point) if solutions_sa else 0
                div = diversity(solutions_sa) if solutions_sa else 0
                
                instance_results['algorithms']['simulated_annealing'] = {
                    'time_mean': statistics.mean(times_sa),
                    'time_std': statistics.stdev(times_sa) if len(times_sa) > 1 else 0,
                    'memory_peak': max(memories_sa),
                    'num_solutions': len(solutions_sa),
                    'hypervolume': hv,
                    'diversity': div,
                    'acceptance_rate': stats_sa[-1].get('avg_acceptance_rate', 0),
                    'improvements': stats_sa[-1].get('total_improvements', 0)
                }
                
                print(f"    Tiempo: {format_time(statistics.mean(times_sa))}")
                print(f"    Soluciones: {len(solutions_sa)}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                instance_results['algorithms']['simulated_annealing'] = {'error': str(e)}
            
            all_results['instances'][instance_name] = instance_results
            
        except Exception as e:
            print(f"  ERROR cargando instancia: {e}")
            traceback.print_exc()
            all_results['instances'][instance_name] = {'error': str(e)}
    
    return all_results


def generate_tables(results: Dict[str, Any], output_dir: str = "results/tables"):
    """Genera tablas CSV con los resultados."""
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tabla de tiempo
    with open(f"{output_dir}/tiempo_ejecucion.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instancia', 'N', 'Branch&Bound', 'Geométrico', 'Simulated Annealing'])
        
        for instance_name, data in results['instances'].items():
            if 'error' in data:
                continue
            
            n = data.get('n_destinations', 0)
            
            bb_time = data['algorithms'].get('branch_bound', {}).get('time_mean', '-')
            geo_time = data['algorithms'].get('geometric', {}).get('time_mean', '-')
            sa_time = data['algorithms'].get('simulated_annealing', {}).get('time_mean', '-')
            
            if isinstance(bb_time, float):
                bb_time = f"{bb_time:.4f}"
            if isinstance(geo_time, float):
                geo_time = f"{geo_time:.4f}"
            if isinstance(sa_time, float):
                sa_time = f"{sa_time:.4f}"
            
            writer.writerow([instance_name, n, bb_time, geo_time, sa_time])
    
    # Tabla de memoria
    with open(f"{output_dir}/memoria.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instancia', 'N', 'Branch&Bound (MB)', 'Geométrico (MB)', 'SA (MB)'])
        
        for instance_name, data in results['instances'].items():
            if 'error' in data:
                continue
            
            n = data.get('n_destinations', 0)
            
            bb_mem = data['algorithms'].get('branch_bound', {}).get('memory_peak', '-')
            geo_mem = data['algorithms'].get('geometric', {}).get('memory_peak', '-')
            sa_mem = data['algorithms'].get('simulated_annealing', {}).get('memory_peak', '-')
            
            if isinstance(bb_mem, float):
                bb_mem = f"{bb_mem:.2f}"
            if isinstance(geo_mem, float):
                geo_mem = f"{geo_mem:.2f}"
            if isinstance(sa_mem, float):
                sa_mem = f"{sa_mem:.2f}"
            
            writer.writerow([instance_name, n, bb_mem, geo_mem, sa_mem])
    
    # Tabla de calidad (hipervolumen y diversidad)
    with open(f"{output_dir}/calidad.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instancia', 'N', 
                        'HV_BB', 'HV_Geo', 'HV_SA',
                        'Div_BB', 'Div_Geo', 'Div_SA'])
        
        for instance_name, data in results['instances'].items():
            if 'error' in data:
                continue
            
            n = data.get('n_destinations', 0)
            
            def get_metric(algo, metric):
                val = data['algorithms'].get(algo, {}).get(metric, '-')
                return f"{val:.2f}" if isinstance(val, float) else val
            
            writer.writerow([
                instance_name, n,
                get_metric('branch_bound', 'hypervolume'),
                get_metric('geometric', 'hypervolume'),
                get_metric('simulated_annealing', 'hypervolume'),
                get_metric('branch_bound', 'diversity'),
                get_metric('geometric', 'diversity'),
                get_metric('simulated_annealing', 'diversity')
            ])
    
    print(f"\nTablas generadas en {output_dir}/")


def generate_graphs(results: Dict[str, Any], output_dir: str = "results/graphs"):
    """Genera gráficas de los resultados."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib no disponible para generar gráficas")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Preparar datos
    instances = []
    n_values = []
    
    times_bb = []
    times_geo = []
    times_sa = []
    
    hv_bb = []
    hv_geo = []
    hv_sa = []
    
    div_bb = []
    div_geo = []
    div_sa = []
    
    for instance_name, data in sorted(results['instances'].items()):
        if 'error' in data:
            continue
        
        instances.append(instance_name)
        n_values.append(data.get('n_destinations', 0))
        
        # Tiempos
        bb = data['algorithms'].get('branch_bound', {})
        geo = data['algorithms'].get('geometric', {})
        sa = data['algorithms'].get('simulated_annealing', {})
        
        times_bb.append(bb.get('time_mean') if not bb.get('skipped') else None)
        times_geo.append(geo.get('time_mean'))
        times_sa.append(sa.get('time_mean'))
        
        hv_bb.append(bb.get('hypervolume') if not bb.get('skipped') else None)
        hv_geo.append(geo.get('hypervolume'))
        hv_sa.append(sa.get('hypervolume'))
        
        div_bb.append(bb.get('diversity') if not bb.get('skipped') else None)
        div_geo.append(geo.get('diversity'))
        div_sa.append(sa.get('diversity'))
    
    # Gráfica 1: Tiempo vs N (escala log)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array(n_values)
    
    # Filtrar valores None
    bb_valid = [(n, t) for n, t in zip(n_values, times_bb) if t is not None]
    geo_valid = [(n, t) for n, t in zip(n_values, times_geo) if t is not None]
    sa_valid = [(n, t) for n, t in zip(n_values, times_sa) if t is not None]
    
    if bb_valid:
        ax.semilogy([p[0] for p in bb_valid], [p[1] for p in bb_valid], 
                   'o-', label='Branch & Bound', linewidth=2, markersize=8)
    if geo_valid:
        ax.semilogy([p[0] for p in geo_valid], [p[1] for p in geo_valid], 
                   's-', label='Geométrico', linewidth=2, markersize=8)
    if sa_valid:
        ax.semilogy([p[0] for p in sa_valid], [p[1] for p in sa_valid], 
                   '^-', label='Simulated Annealing', linewidth=2, markersize=8)
    
    ax.set_xlabel('Número de destinos (N)', fontsize=12)
    ax.set_ylabel('Tiempo de ejecución (s)', fontsize=12)
    ax.set_title('Tiempo de Ejecución vs Tamaño de Instancia', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tiempo_vs_n.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfica 2: Hipervolumen por técnica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x_pos = np.arange(len(instances))
    
    # Preparar datos para barras
    hv_bb_plot = [h if h is not None else 0 for h in hv_bb]
    hv_geo_plot = [h if h is not None else 0 for h in hv_geo]
    hv_sa_plot = [h if h is not None else 0 for h in hv_sa]
    
    bars1 = ax.bar(x_pos - width, hv_bb_plot, width, label='Branch & Bound', color='#2ecc71')
    bars2 = ax.bar(x_pos, hv_geo_plot, width, label='Geométrico', color='#3498db')
    bars3 = ax.bar(x_pos + width, hv_sa_plot, width, label='Simulated Annealing', color='#e74c3c')
    
    ax.set_xlabel('Instancia', fontsize=12)
    ax.set_ylabel('Hipervolumen', fontsize=12)
    ax.set_title('Hipervolumen Dominado por Técnica', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={n}' for n in n_values], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hipervolumen.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfica 3: Diversidad de la frontera
    fig, ax = plt.subplots(figsize=(10, 6))
    
    div_bb_plot = [d if d is not None else 0 for d in div_bb]
    div_geo_plot = [d if d is not None else 0 for d in div_geo]
    div_sa_plot = [d if d is not None else 0 for d in div_sa]
    
    bars1 = ax.bar(x_pos - width, div_bb_plot, width, label='Branch & Bound', color='#2ecc71')
    bars2 = ax.bar(x_pos, div_geo_plot, width, label='Geométrico', color='#3498db')
    bars3 = ax.bar(x_pos + width, div_sa_plot, width, label='Simulated Annealing', color='#e74c3c')
    
    ax.set_xlabel('Instancia', fontsize=12)
    ax.set_ylabel('Diversidad (distancia media)', fontsize=12)
    ax.set_title('Diversidad de la Frontera de Pareto', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={n}' for n in n_values], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/diversidad.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfica 4: Comparación de número de soluciones
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sols_bb = []
    sols_geo = []
    sols_sa = []
    
    for instance_name, data in sorted(results['instances'].items()):
        if 'error' in data:
            continue
        bb = data['algorithms'].get('branch_bound', {})
        geo = data['algorithms'].get('geometric', {})
        sa = data['algorithms'].get('simulated_annealing', {})
        
        sols_bb.append(bb.get('num_solutions', 0) if not bb.get('skipped') else 0)
        sols_geo.append(geo.get('num_solutions', 0))
        sols_sa.append(sa.get('num_solutions', 0))
    
    bars1 = ax.bar(x_pos - width, sols_bb, width, label='Branch & Bound', color='#2ecc71')
    bars2 = ax.bar(x_pos, sols_geo, width, label='Geométrico', color='#3498db')
    bars3 = ax.bar(x_pos + width, sols_sa, width, label='Simulated Annealing', color='#e74c3c')
    
    ax.set_xlabel('Instancia', fontsize=12)
    ax.set_ylabel('Número de soluciones (Pareto)', fontsize=12)
    ax.set_title('Soluciones No Dominadas Encontradas', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={n}' for n in n_values], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_soluciones.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficas generadas en {output_dir}/")


def generate_report(results: Dict[str, Any], output_path: str = "results/REPORTE.txt"):
    """Genera un reporte textual de los resultados."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DE EXPERIMENTACIÓN\n")
        f.write("Planificación Multiobjetivo de Rutas de Drones\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Fecha: {results['timestamp']}\n")
        f.write(f"Réplicas por experimento: {results['num_replicas']}\n")
        f.write(f"Límite de tiempo: {results['time_limit']} s\n\n")
        
        for instance_name, data in sorted(results['instances'].items()):
            f.write("-" * 70 + "\n")
            f.write(f"INSTANCIA: {instance_name}\n")
            f.write("-" * 70 + "\n")
            
            if 'error' in data:
                f.write(f"ERROR: {data['error']}\n\n")
                continue
            
            f.write(f"Destinos: {data.get('n_destinations', 'N/A')}\n")
            f.write(f"Zonas no-fly: {data.get('n_no_fly_zones', 'N/A')}\n\n")
            
            for algo_name, algo_data in data['algorithms'].items():
                f.write(f"\n  {algo_name.upper()}:\n")
                
                if algo_data.get('skipped'):
                    f.write(f"    Omitido: {algo_data.get('reason', 'N/A')}\n")
                    continue
                
                if 'error' in algo_data:
                    f.write(f"    Error: {algo_data['error']}\n")
                    continue
                
                time_mean = algo_data.get('time_mean', 0)
                time_std = algo_data.get('time_std', 0)
                memory = algo_data.get('memory_peak', 0)
                num_sols = algo_data.get('num_solutions', 0)
                hv = algo_data.get('hypervolume', 0)
                div = algo_data.get('diversity', 0)
                
                f.write(f"    Tiempo: {time_mean:.4f} s (± {time_std:.4f})\n")
                f.write(f"    Memoria pico: {memory:.2f} MB\n")
                f.write(f"    Soluciones Pareto: {num_sols}\n")
                f.write(f"    Hipervolumen: {hv:.2f}\n")
                f.write(f"    Diversidad: {div:.2f}\n")
            
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nReporte generado: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENTACIÓN: Planificación Multiobjetivo de Rutas de Drones")
    print("=" * 70)
    
    # Verificar/generar instancias
    if not os.path.exists("instances") or not os.listdir("instances"):
        print("\nGenerando instancias...")
        generate_all_instances("instances")
    
    # Ejecutar experimentos
    print("\nIniciando experimentos...")
    results = run_experiments(
        instances_dir="instances",
        results_dir="results",
        num_replicas=5,
        time_limit=60.0
    )
    
    # Guardar resultados JSON
    with open("results/resultados_completos.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generar tablas
    print("\nGenerando tablas CSV...")
    generate_tables(results)
    
    # Generar gráficas
    print("\nGenerando gráficas...")
    generate_graphs(results)
    
    # Generar reporte
    print("\nGenerando reporte...")
    generate_report(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTACIÓN COMPLETADA")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  - results/resultados_completos.json")
    print("  - results/tables/*.csv")
    print("  - results/graphs/*.png")
    print("  - results/REPORTE.txt")
