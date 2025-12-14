"""
Módulo de benchmarking para medición de rendimiento.
Mide tiempo de ejecución y uso de memoria de los algoritmos.

Autor: David Valbuena Segura
"""

import time
import tracemalloc
import gc
from typing import Callable, Any, Tuple, Dict
from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    """Resultado de una medición de benchmark."""
    algorithm: str
    instance: str
    execution_time: float
    memory_peak: float
    num_solutions: int
    additional_stats: Dict[str, Any]


def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Mide el tiempo de ejecución de una función.
    
    Returns:
        Tupla (resultado de la función, tiempo en segundos)
    """
    gc.collect()  # Limpiar memoria antes de medir
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    
    return result, end - start


def measure_memory(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Mide el uso máximo de memoria de una función.
    
    Returns:
        Tupla (resultado de la función, memoria pico en MB)
    """
    gc.collect()
    
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, peak / (1024 * 1024)  # Convertir a MB


def benchmark_algorithm(func: Callable,
                       args: tuple,
                       kwargs: dict,
                       algorithm_name: str,
                       instance_name: str,
                       num_replicas: int = 5) -> BenchmarkResult:
    """
    Ejecuta benchmark completo de un algoritmo.
    
    Args:
        func: Función a ejecutar
        args: Argumentos posicionales
        kwargs: Argumentos con nombre
        algorithm_name: Nombre del algoritmo
        instance_name: Nombre de la instancia
        num_replicas: Número de réplicas para estadísticas
    
    Returns:
        BenchmarkResult con estadísticas
    """
    times = []
    memory_peaks = []
    results = []
    stats_list = []
    
    for replica in range(num_replicas):
        gc.collect()
        
        # Medir tiempo
        start = time.perf_counter()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        end = time.perf_counter()
        tracemalloc.stop()
        
        elapsed = end - start
        memory_mb = peak / (1024 * 1024)
        
        times.append(elapsed)
        memory_peaks.append(memory_mb)
        
        # Extraer soluciones y estadísticas
        if isinstance(result, tuple) and len(result) == 2:
            solutions, stats = result
            results.append(solutions)
            stats_list.append(stats)
        else:
            results.append(result)
    
    # Calcular estadísticas
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    mean_memory = statistics.mean(memory_peaks)
    
    # Número de soluciones (del último resultado)
    num_solutions = 0
    if results:
        last_result = results[-1]
        if isinstance(last_result, list):
            num_solutions = len(last_result)
    
    # Agregar estadísticas adicionales
    additional_stats = {
        'time_std': std_time,
        'time_min': min(times),
        'time_max': max(times),
        'memory_max': max(memory_peaks),
        'replicas': num_replicas
    }
    
    # Añadir estadísticas del algoritmo si están disponibles
    if stats_list:
        for key in stats_list[0]:
            if isinstance(stats_list[0][key], (int, float)):
                values = [s.get(key, 0) for s in stats_list]
                additional_stats[f'algo_{key}_mean'] = statistics.mean(values)
    
    return BenchmarkResult(
        algorithm=algorithm_name,
        instance=instance_name,
        execution_time=mean_time,
        memory_peak=mean_memory,
        num_solutions=num_solutions,
        additional_stats=additional_stats
    )


def format_time(seconds: float) -> str:
    """Formatea tiempo de forma legible."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_memory(mb: float) -> str:
    """Formatea memoria de forma legible."""
    if mb < 1:
        return f"{mb * 1024:.1f} KB"
    elif mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb / 1024:.2f} GB"


def print_benchmark_result(result: BenchmarkResult):
    """Imprime resultado de benchmark de forma formateada."""
    print(f"\n{'='*50}")
    print(f"Algoritmo: {result.algorithm}")
    print(f"Instancia: {result.instance}")
    print(f"{'='*50}")
    print(f"Tiempo medio: {format_time(result.execution_time)} "
          f"(± {format_time(result.additional_stats.get('time_std', 0))})")
    print(f"Memoria pico: {format_memory(result.memory_peak)}")
    print(f"Soluciones encontradas: {result.num_solutions}")
    
    if result.additional_stats:
        print("\nEstadísticas adicionales:")
        for key, value in result.additional_stats.items():
            if not key.startswith('time_') and not key.startswith('memory_'):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    # Prueba simple
    import math
    
    def heavy_computation(n):
        """Función de prueba que consume tiempo y memoria."""
        data = [i ** 2 for i in range(n)]
        result = sum(math.sqrt(x) for x in data)
        return result
    
    print("=" * 60)
    print("PRUEBA DEL MÓDULO DE BENCHMARKING")
    print("=" * 60)
    
    # Medir tiempo
    result, elapsed = measure_time(heavy_computation, 100000)
    print(f"\nResultado: {result:.2f}")
    print(f"Tiempo: {format_time(elapsed)}")
    
    # Medir memoria
    result, memory = measure_memory(heavy_computation, 100000)
    print(f"Memoria: {format_memory(memory)}")
    
    print("\n" + "=" * 60)
