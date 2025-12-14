"""
Módulo de experimentación y benchmarking.
"""

from .benchmark import (
    BenchmarkResult, 
    measure_time, 
    measure_memory,
    benchmark_algorithm,
    format_time,
    format_memory
)

from .generate_instances import (
    generate_instance,
    generate_all_instances,
    visualize_instance
)

__all__ = [
    'BenchmarkResult', 'measure_time', 'measure_memory',
    'benchmark_algorithm', 'format_time', 'format_memory',
    'generate_instance', 'generate_all_instances', 'visualize_instance'
]
