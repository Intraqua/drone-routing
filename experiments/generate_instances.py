"""
Generador de instancias para el problema de rutas de drones.
Crea instancias con diferentes tamaños (N=10, 15, 20, 25) y zonas no-fly.

Autor: David Valbuena Segura
"""

import json
import random
import math
import os
from typing import List, Tuple


def generate_polygon(center: Tuple[float, float], 
                    radius: float, 
                    num_vertices: int = 4) -> List[List[float]]:
    """
    Genera un polígono regular centrado en un punto.
    
    Args:
        center: Centro del polígono
        radius: Radio del círculo circunscrito
        num_vertices: Número de vértices (4 = cuadrado, 5 = pentágono, etc.)
    
    Returns:
        Lista de vértices [[x1,y1], [x2,y2], ...]
    """
    vertices = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append([round(x, 2), round(y, 2)])
    return vertices


def point_in_polygon_simple(point: Tuple[float, float], 
                           polygon: List[List[float]]) -> bool:
    """Verifica si un punto está dentro de un polígono (ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def generate_instance(n_destinations: int,
                     n_no_fly_zones: int = 3,
                     map_size: float = 100.0,
                     seed: int = None,
                     name: str = None) -> dict:
    """
    Genera una instancia del problema de rutas de drones.
    
    Args:
        n_destinations: Número de destinos (excluyendo el hub)
        n_no_fly_zones: Número de zonas no-fly
        map_size: Tamaño del mapa (cuadrado de map_size x map_size)
        seed: Semilla para reproducibilidad
        name: Nombre de la instancia
    
    Returns:
        Diccionario con la estructura de la instancia
    """
    if seed is not None:
        random.seed(seed)
    
    if name is None:
        name = f"instance_n{n_destinations}"
    
    # Hub en el centro
    hub = {
        "id": 0,
        "x": map_size / 2,
        "y": map_size / 2
    }
    
    # Generar zonas no-fly primero (para evitar colocar destinos dentro)
    no_fly_zones = []
    margin = map_size * 0.1
    
    for _ in range(n_no_fly_zones):
        attempts = 0
        while attempts < 100:
            # Centro aleatorio
            cx = random.uniform(margin, map_size - margin)
            cy = random.uniform(margin, map_size - margin)
            
            # Evitar zona muy cercana al hub
            dist_to_hub = math.sqrt((cx - hub["x"])**2 + (cy - hub["y"])**2)
            if dist_to_hub < map_size * 0.15:
                attempts += 1
                continue
            
            # Radio aleatorio
            radius = random.uniform(map_size * 0.05, map_size * 0.12)
            
            # Número de vértices (3-6)
            num_vertices = random.randint(3, 6)
            
            polygon = generate_polygon((cx, cy), radius, num_vertices)
            
            # Verificar que no se superpone demasiado con otras zonas
            overlap = False
            for existing in no_fly_zones:
                for v in polygon:
                    if point_in_polygon_simple(tuple(v), existing["vertices"]):
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                no_fly_zones.append({"vertices": polygon})
                break
            
            attempts += 1
    
    # Generar destinos
    destinations = []
    charging_stations = []
    
    # Distribuir algunos destinos como estaciones de recarga
    n_charging = max(1, n_destinations // 5)
    charging_indices = random.sample(range(n_destinations), n_charging)
    
    for i in range(n_destinations):
        attempts = 0
        while attempts < 100:
            # Posición aleatoria
            x = random.uniform(margin, map_size - margin)
            y = random.uniform(margin, map_size - margin)
            
            # Verificar que no está dentro de ninguna zona no-fly
            in_no_fly = False
            for zone in no_fly_zones:
                if point_in_polygon_simple((x, y), zone["vertices"]):
                    in_no_fly = True
                    break
            
            # Verificar distancia mínima a otros destinos
            too_close = False
            min_dist = map_size * 0.05
            
            for dest in destinations:
                dist = math.sqrt((x - dest["x"])**2 + (y - dest["y"])**2)
                if dist < min_dist:
                    too_close = True
                    break
            
            # Verificar distancia al hub
            dist_to_hub = math.sqrt((x - hub["x"])**2 + (y - hub["y"])**2)
            if dist_to_hub < min_dist:
                too_close = True
            
            if not in_no_fly and not too_close:
                node_id = i + 1  # IDs empiezan en 1
                is_charging = i in charging_indices
                
                destinations.append({
                    "id": node_id,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "is_charging": is_charging
                })
                
                if is_charging:
                    charging_stations.append(node_id)
                
                break
            
            attempts += 1
    
    # Construir instancia
    instance = {
        "name": name,
        "description": f"Instancia con {n_destinations} destinos y {n_no_fly_zones} zonas no-fly",
        "map_size": map_size,
        "hub": hub,
        "destinations": destinations,
        "no_fly_zones": no_fly_zones,
        "charging_stations": charging_stations,
        "metadata": {
            "n_destinations": n_destinations,
            "n_no_fly_zones": n_no_fly_zones,
            "n_charging_stations": len(charging_stations),
            "seed": seed
        }
    }
    
    return instance


def generate_all_instances(output_dir: str = "instances"):
    """
    Genera las 4 instancias requeridas por el enunciado.
    
    N = 10, 15, 20, 25 destinos con ≥3 polígonos no-fly.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        {"n": 10, "zones": 3, "seed": 42},
        {"n": 15, "zones": 4, "seed": 43},
        {"n": 20, "zones": 4, "seed": 44},
        {"n": 25, "zones": 5, "seed": 45}
    ]
    
    instances = []
    
    for config in configs:
        instance = generate_instance(
            n_destinations=config["n"],
            n_no_fly_zones=config["zones"],
            seed=config["seed"],
            name=f"instance_n{config['n']}"
        )
        
        filepath = os.path.join(output_dir, f"instance_n{config['n']}.json")
        
        with open(filepath, 'w') as f:
            json.dump(instance, f, indent=2)
        
        instances.append(instance)
        print(f"Generada: {filepath}")
        print(f"  - Destinos: {config['n']}")
        print(f"  - Zonas no-fly: {config['zones']}")
        print(f"  - Estaciones de recarga: {len(instance['charging_stations'])}")
    
    return instances


def visualize_instance(instance: dict, output_path: str = None):
    """
    Genera una visualización de la instancia.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib no disponible para visualización")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    map_size = instance.get("map_size", 100)
    
    # Dibujar zonas no-fly
    for zone in instance["no_fly_zones"]:
        vertices = zone["vertices"]
        polygon = patches.Polygon(
            vertices, 
            closed=True,
            facecolor='red',
            edgecolor='darkred',
            alpha=0.3,
            linewidth=2
        )
        ax.add_patch(polygon)
    
    # Dibujar hub
    hub = instance["hub"]
    ax.plot(hub["x"], hub["y"], 's', markersize=15, 
            color='green', label='Hub', zorder=5)
    ax.annotate('HUB', (hub["x"], hub["y"]), 
                textcoords="offset points", xytext=(0, 10),
                ha='center', fontweight='bold')
    
    # Dibujar destinos
    for dest in instance["destinations"]:
        color = 'blue' if not dest.get("is_charging", False) else 'orange'
        marker = 'o' if not dest.get("is_charging", False) else '^'
        ax.plot(dest["x"], dest["y"], marker, markersize=10, 
                color=color, zorder=4)
        ax.annotate(str(dest["id"]), (dest["x"], dest["y"]),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=8)
    
    # Configuración del gráfico
    ax.set_xlim(-5, map_size + 5)
    ax.set_ylim(-5, map_size + 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Instancia: {instance['name']}\n"
                f"{len(instance['destinations'])} destinos, "
                f"{len(instance['no_fly_zones'])} zonas no-fly")
    
    # Leyenda
    from matplotlib.lines import Line2D
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
        print(f"Visualización guardada: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GENERADOR DE INSTANCIAS")
    print("=" * 60)
    
    # Generar todas las instancias
    instances = generate_all_instances("instances")
    
    print("\n" + "=" * 60)
    print("Generación completada")
    print("=" * 60)
