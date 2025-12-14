#!/usr/bin/env python3
"""
Script principal para la planificación multiobjetivo de rutas de drones.
Permite ejecutar los algoritmos de forma individual o comparativa.

Uso:
    python main.py --help
    python main.py generate                    # Generar instancias
    python main.py solve <instancia> --algo bb # Resolver con Branch & Bound
    python main.py solve <instancia> --algo geo # Resolver con Geométrico
    python main.py solve <instancia> --algo sa  # Resolver con Simulated Annealing
    python main.py benchmark                    # Ejecutar todos los experimentos
    python main.py visualize <instancia>        # Visualizar instancia

Autor: David Valbuena Segura
"""

import argparse
import sys
import os
import json
import time

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.graph import DroneGraph, Node, Edge
from common.geometry import distance, segment_passes_through_polygon
from common.pareto import Solution, hypervolume, diversity, get_pareto_front

from exact_bb import solve as solve_bb
from geo_heuristic import solve as solve_geo
from metaheuristic import solve as solve_sa

from experiments.generate_instances import generate_all_instances, visualize_instance
from experiments.run_experiments import (
    run_experiments, generate_tables, generate_graphs, 
    generate_report, load_instance
)


def cmd_generate(args):
    """Genera las instancias de prueba."""
    print("Generando instancias...")
    instances = generate_all_instances(args.output_dir)
    print(f"\nGeneradas {len(instances)} instancias en '{args.output_dir}/'")
    
    if args.visualize:
        print("\nGenerando visualizaciones...")
        for instance in instances:
            output_path = f"{args.output_dir}/{instance['name']}_viz.png"
            visualize_instance(instance, output_path)


def cmd_solve(args):
    """Resuelve una instancia con el algoritmo especificado."""
    # Cargar instancia
    instance_path = args.instance
    if not instance_path.endswith('.json'):
        instance_path = f"instances/{instance_path}.json"
    
    if not os.path.exists(instance_path):
        print(f"Error: No se encuentra la instancia '{instance_path}'")
        sys.exit(1)
    
    print(f"Cargando instancia: {instance_path}")
    graph = load_instance(instance_path)
    
    print(f"  - Nodos: {graph.num_nodes}")
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")
    
    # Seleccionar algoritmo
    algo = args.algo.lower()
    
    print(f"\nEjecutando algoritmo: {algo}")
    start_time = time.time()
    
    if algo == 'bb' or algo == 'branch_bound':
        solutions, stats = solve_bb(
            graph,
            max_solutions=args.max_solutions,
            time_limit=args.time_limit
        )
        algo_name = "Branch & Bound"
        
    elif algo == 'geo' or algo == 'geometric':
        solutions, stats = solve_geo(
            graph,
            num_solutions=args.max_solutions
        )
        algo_name = "Heurística Geométrica"
        
    elif algo == 'sa' or algo == 'simulated_annealing':
        solutions, stats = solve_sa(
            graph,
            initial_temp=5000.0,
            cooling_rate=0.995,
            max_iterations=args.iterations,
            time_limit=args.time_limit,
            num_runs=3
        )
        algo_name = "Simulated Annealing"
        
    else:
        print(f"Error: Algoritmo desconocido '{algo}'")
        print("Algoritmos disponibles: bb, geo, sa")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    # Mostrar resultados
    print(f"\n{'='*60}")
    print(f"RESULTADOS - {algo_name}")
    print(f"{'='*60}")
    print(f"Tiempo de ejecución: {elapsed:.4f} s")
    print(f"Soluciones encontradas: {len(solutions)}")
    
    if solutions:
        ref_point = (1000.0, 50.0, 10.0)
        hv = hypervolume(solutions, ref_point)
        div = diversity(solutions)
        
        print(f"Hipervolumen: {hv:.2f}")
        print(f"Diversidad: {div:.2f}")
        
        print(f"\nFrontera de Pareto ({len(solutions)} soluciones):")
        for i, sol in enumerate(solutions[:10]):
            print(f"\n  Solución {i+1}:")
            print(f"    Ruta: {' -> '.join(map(str, sol.route))}")
            print(f"    Distancia: {sol.distance:.2f}")
            print(f"    Riesgo: {sol.risk:.2f}")
            print(f"    Batería: {sol.recharges:.2f}")
        
        if len(solutions) > 10:
            print(f"\n  ... y {len(solutions) - 10} soluciones más")
    
    if stats:
        print(f"\nEstadísticas del algoritmo:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if args.output:
        results = {
            'instance': instance_path,
            'algorithm': algo_name,
            'execution_time': elapsed,
            'num_solutions': len(solutions),
            'solutions': [
                {
                    'route': sol.route,
                    'objectives': {
                        'distance': sol.distance,
                        'risk': sol.risk,
                        'battery': sol.recharges
                    }
                }
                for sol in solutions
            ],
            'statistics': stats
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResultados guardados en: {args.output}")


def cmd_benchmark(args):
    """Ejecuta el benchmark completo."""
    print("Iniciando benchmark completo...")
    
    if not os.path.exists(args.instances_dir) or not os.listdir(args.instances_dir):
        print("Generando instancias...")
        generate_all_instances(args.instances_dir)
    
    results = run_experiments(
        instances_dir=args.instances_dir,
        results_dir=args.output_dir,
        num_replicas=args.replicas,
        time_limit=args.time_limit
    )
    
    with open(f"{args.output_dir}/resultados_completos.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    generate_tables(results, f"{args.output_dir}/tables")
    generate_graphs(results, f"{args.output_dir}/graphs")
    generate_report(results, f"{args.output_dir}/REPORTE.txt")
    
    print("\nBenchmark completado.")
    print(f"Resultados en: {args.output_dir}/")


def cmd_visualize(args):
    """Visualiza una instancia."""
    instance_path = args.instance
    if not instance_path.endswith('.json'):
        instance_path = f"instances/{instance_path}.json"
    
    if not os.path.exists(instance_path):
        print(f"Error: No se encuentra la instancia '{instance_path}'")
        sys.exit(1)
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    output_path = args.output if args.output else None
    visualize_instance(instance, output_path)
    
    if output_path:
        print(f"Visualización guardada en: {output_path}")


def cmd_compare(args):
    """Compara los tres algoritmos en una instancia."""
    instance_path = args.instance
    if not instance_path.endswith('.json'):
        instance_path = f"instances/{instance_path}.json"
    
    if not os.path.exists(instance_path):
        print(f"Error: No se encuentra la instancia '{instance_path}'")
        sys.exit(1)
    
    print(f"Cargando instancia: {instance_path}")
    graph = load_instance(instance_path)
    
    print(f"  - Destinos: {len(graph.destination_ids)}")
    print(f"  - Zonas no-fly: {len(graph.no_fly_zones)}")
    
    results = {}
    ref_point = (1000.0, 50.0, 10.0)
    
    if len(graph.destination_ids) <= 12:
        print("\n[1/3] Ejecutando Branch & Bound...")
        start = time.time()
        solutions, stats = solve_bb(graph, max_solutions=10, time_limit=30.0)
        elapsed = time.time() - start
        
        results['Branch & Bound'] = {
            'time': elapsed,
            'solutions': len(solutions),
            'hypervolume': hypervolume(solutions, ref_point) if solutions else 0,
            'diversity': diversity(solutions) if solutions else 0
        }
    else:
        print("\n[1/3] Branch & Bound omitido (N > 12)")
        results['Branch & Bound'] = {'skipped': True}
    
    print("\n[2/3] Ejecutando Heurística Geométrica...")
    start = time.time()
    solutions, stats = solve_geo(graph, num_solutions=10)
    elapsed = time.time() - start
    
    results['Geométrico'] = {
        'time': elapsed,
        'solutions': len(solutions),
        'hypervolume': hypervolume(solutions, ref_point) if solutions else 0,
        'diversity': diversity(solutions) if solutions else 0
    }
    
    print("\n[3/3] Ejecutando Simulated Annealing...")
    start = time.time()
    solutions, stats = solve_sa(
        graph, 
        initial_temp=5000.0,
        cooling_rate=0.995,
        max_iterations=5000,
        time_limit=30.0,
        num_runs=3
    )
    elapsed = time.time() - start
    
    results['Simulated Annealing'] = {
        'time': elapsed,
        'solutions': len(solutions),
        'hypervolume': hypervolume(solutions, ref_point) if solutions else 0,
        'diversity': diversity(solutions) if solutions else 0
    }
    
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE ALGORITMOS")
    print(f"{'='*70}")
    print(f"{'Algoritmo':<25} {'Tiempo (s)':<12} {'Soluciones':<12} {'Hipervolumen':<14} {'Diversidad':<12}")
    print("-" * 70)
    
    for algo, data in results.items():
        if data.get('skipped'):
            print(f"{algo:<25} {'OMITIDO':<12} {'-':<12} {'-':<14} {'-':<12}")
        else:
            print(f"{algo:<25} {data['time']:<12.4f} {data['solutions']:<12} "
                  f"{data['hypervolume']:<14.2f} {data['diversity']:<12.2f}")
    
    print("=" * 70)


def main():
    """Función principal con parser de argumentos."""
    parser = argparse.ArgumentParser(
        description='Planificación Multiobjetivo de Rutas de Drones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py generate                      Genera las 4 instancias
  python main.py solve instance_n10 --algo bb  Resuelve con Branch & Bound
  python main.py solve instance_n15 --algo sa  Resuelve con Simulated Annealing
  python main.py compare instance_n10          Compara los 3 algoritmos
  python main.py benchmark                     Ejecuta benchmark completo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    gen_parser = subparsers.add_parser('generate', help='Generar instancias de prueba')
    gen_parser.add_argument('--output-dir', '-o', default='instances',
                           help='Directorio de salida (default: instances)')
    gen_parser.add_argument('--visualize', '-v', action='store_true',
                           help='Generar visualizaciones de las instancias')
    
    solve_parser = subparsers.add_parser('solve', help='Resolver una instancia')
    solve_parser.add_argument('instance', help='Ruta o nombre de la instancia')
    solve_parser.add_argument('--algo', '-a', required=True,
                             choices=['bb', 'geo', 'sa', 'branch_bound', 
                                     'geometric', 'simulated_annealing'],
                             help='Algoritmo a usar')
    solve_parser.add_argument('--max-solutions', '-m', type=int, default=10,
                             help='Máximo de soluciones a encontrar')
    solve_parser.add_argument('--time-limit', '-t', type=float, default=60.0,
                             help='Límite de tiempo en segundos')
    solve_parser.add_argument('--iterations', '-i', type=int, default=10000,
                             help='Iteraciones máximas (para SA)')
    solve_parser.add_argument('--output', '-o', help='Archivo de salida JSON')
    
    bench_parser = subparsers.add_parser('benchmark', help='Ejecutar benchmark completo')
    bench_parser.add_argument('--instances-dir', default='instances',
                             help='Directorio de instancias')
    bench_parser.add_argument('--output-dir', '-o', default='results',
                             help='Directorio de resultados')
    bench_parser.add_argument('--replicas', '-r', type=int, default=5,
                             help='Número de réplicas por experimento')
    bench_parser.add_argument('--time-limit', '-t', type=float, default=60.0,
                             help='Límite de tiempo por algoritmo')
    
    viz_parser = subparsers.add_parser('visualize', help='Visualizar una instancia')
    viz_parser.add_argument('instance', help='Ruta o nombre de la instancia')
    viz_parser.add_argument('--output', '-o', help='Archivo de salida PNG')
    
    cmp_parser = subparsers.add_parser('compare', help='Comparar algoritmos en una instancia')
    cmp_parser.add_argument('instance', help='Ruta o nombre de la instancia')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'solve':
        cmd_solve(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'compare':
        cmd_compare(args)


if __name__ == "__main__":
    main()
