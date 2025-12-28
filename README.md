# Planificación Multiobjetivo de Rutas de Drones en Entornos Urbanos

**Actividad 2 - Diseño Avanzado de Algoritmos**  
**Universidad Digital Europea (UNIPRO)**  
**Autor:** David Valbuena Segura

## Descripción del Problema

Este proyecto implementa tres algoritmos para resolver el problema de planificación de rutas de drones de reparto en entornos urbanos. El objetivo es encontrar un circuito hamiltoniano óptimo desde un hub central que minimice simultáneamente tres objetivos:

1. **Distancia total** recorrida
2. **Riesgo acumulado** de la ruta
3. **Número de paradas** para recargar batería

El problema considera restricciones de zonas de vuelo prohibidas (no-fly zones) representadas como polígonos que el dron no puede atravesar.

## Estructura del Proyecto
```
drone-routing/
├── main.py               # Script principal de ejecución
├── Makefile              # Automatización de comandos
├── requirements.txt      # Dependencias Python
├── README.md             # Este archivo
│
├── common/               # Módulos compartidos
│   ├── geometry.py       # Geometría computacional
│   ├── graph.py          # Representación de grafos
│   └── pareto.py         # Optimización multiobjetivo
│
├── exact_bb/             # Algoritmo exacto
│   └── branch_bound.py   # Branch & Bound multiobjetivo
│
├── geo_heuristic/        # Heurística geométrica
│   └── visibility_graph.py # Grafo de visibilidad
│
├── metaheuristic/        # Metaheurística
│   └── simulated_annealing.py # Simulated Annealing
│
├── experiments/               # Experimentación
│   ├── generate_instances.py  # Generador de instancias
│   ├── benchmark.py           # Medición de rendimiento
│   └── run_experiments.py     # Script de experimentos
│
├── instances/            # Instancias de prueba (JSON)
│   ├── instance_n10.json
│   ├── instance_n15.json
│   ├── instance_n20.json
│   └── instance_n25.json
│
└── results/              # Resultados experimentales
    ├── tables/           # Tablas CSV
    └── graphs/           # Gráficas PNG
```

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`
- Make (opcional, para usar comandos make)

## Instalación

### Windows (PowerShell)
```powershell
# Clonar el repositorio
git clone https://github.com/Intraqua/drone-routing
cd drone-routing

# Instalar dependencias
pip install -r requirements.txt

# Instalar Make (opcional, requiere Chocolatey)
# Abrir PowerShell como Administrador:
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install make
# Cerrar y abrir PowerShell de nuevo
```

### Linux / Mac
```bash
# Clonar el repositorio
git clone https://github.com/Intraqua/drone-routing
cd drone-routing

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Ejecución Principal (make run)
```bash
# Con Make instalado (Linux/Mac/Windows con Chocolatey)
make run
```

Este comando genera las instancias, ejecuta los 3 algoritmos y produce todos los resultados.

### Windows sin Make

Si no tienes Make instalado, puedes usar el script batch o ejecutar Python directamente:
```powershell
# Opción 1: Usar script batch
.\run.bat run

# Opción 2: Ejecutar Python directamente
pip install -r requirements.txt
python main.py generate
python main.py benchmark --replicas 5 --time-limit 60
```

### Comandos Individuales
```bash
# Generar instancias de prueba
python main.py generate

# Resolver una instancia con un algoritmo específico
python main.py solve instance_n10 --algo bb    # Branch & Bound
python main.py solve instance_n15 --algo geo   # Heurística Geométrica
python main.py solve instance_n20 --algo sa    # Simulated Annealing

# Comparar los tres algoritmos
python main.py compare instance_n10

# Ejecutar benchmark completo
python main.py benchmark --replicas 5 --time-limit 60

# Visualizar una instancia
python main.py visualize instance_n10 --output instancia.png
```

### Comandos Make disponibles
```bash
make install      # Instalar dependencias
make generate     # Generar instancias
make run          # Ejecutar benchmark completo
make test         # Ejecutar pruebas unitarias
make clean        # Limpiar archivos generados
make help         # Mostrar ayuda
```

## Algoritmos Implementados

### 1. Branch & Bound (exact_bb/)

Algoritmo exacto basado en la técnica de ramificación y poda del Tema 4.

- **Estrategia:** Exploración del árbol de permutaciones con poda por cota inferior
- **Cota inferior:** Suma de aristas mínimas desde nodos no visitados
- **Complejidad:** O(N!) en el peor caso, reducida significativamente por la poda
- **Aplicabilidad:** Instancias pequeñas (N ≤ 12)

### 2. Heurística Geométrica (geo_heuristic/)

Algoritmo basado en grafo de visibilidad del Tema 5 y 7.

- **Técnica:** Construcción de grafo de visibilidad evitando polígonos no-fly
- **Heurísticas:** Nearest Neighbor, Insertion, Sweep
- **Complejidad:** O(N² log N)
- **Aplicabilidad:** Todas las instancias

### 3. Simulated Annealing (metaheuristic/)

Metaheurística del Tema 8 para optimización multiobjetivo.

- **Operadores de vecindad:** 2-opt, swap, insert
- **Enfriamiento:** Geométrico (T = T × α)
- **Archivo Pareto:** Mantiene soluciones no dominadas
- **Complejidad:** O(iteraciones × N)
- **Aplicabilidad:** Todas las instancias

## Formato de Instancias

Las instancias se almacenan en formato JSON:
```json
{
  "name": "instance_n10",
  "hub": {"id": 0, "x": 50.0, "y": 50.0},
  "destinations": [
    {"id": 1, "x": 20.0, "y": 30.0, "is_charging": false}
  ],
  "no_fly_zones": [
    {"vertices": [[40, 35], [60, 35], [60, 45], [40, 45]]}
  ],
  "charging_stations": [3, 7]
}
```

## Métricas de Evaluación

- **Tiempo de ejecución:** Media y desviación estándar sobre 5 réplicas
- **Memoria pico:** Uso máximo de memoria durante la ejecución
- **Hipervolumen:** Área dominada por el frente de Pareto
- **Diversidad:** Distancia media entre soluciones del frente

## Resultados

Los resultados se generan en el directorio `results/`:

- `resultados_completos.json`: Datos completos en formato JSON
- `tables/tiempo_ejecucion.csv`: Tiempos por algoritmo e instancia
- `tables/memoria.csv`: Uso de memoria
- `tables/calidad.csv`: Métricas de calidad (hipervolumen, diversidad)
- `graphs/tiempo_vs_n.png`: Gráfica de escalabilidad
- `graphs/hipervolumen.png`: Comparación de hipervolumen
- `graphs/diversidad.png`: Comparación de diversidad
- `REPORTE.txt`: Reporte textual completo

## Referencias

- Tema 4: Optimización Combinatoria - Ramificación y Poda (UNIPRO)
- Tema 5: Optimización Multiobjetivo - Intersección de Segmentos (UNIPRO)
- Tema 7: Algoritmos Geométricos (UNIPRO)
- Tema 8: Algoritmos de Aleatorización - Simulated Annealing (UNIPRO)
- Tema 9: Búsqueda Local y con Candidatos - Fronteras de Pareto (UNIPRO)
- Deb, K. et al. (2002). NSGA-II: A Fast Elitist Non-Dominated Sorting Genetic Algorithm

## Licencia

Este proyecto es parte de una actividad académica para la asignatura de Diseño Avanzado de Algoritmos de la UNIPRO.