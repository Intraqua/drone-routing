# Makefile para Planificación Multiobjetivo de Rutas de Drones
# Actividad 2 - Diseño Avanzado de Algoritmos
# Autor: David Valbuena Segura

PYTHON = python
PIP = pip3

.PHONY: all install generate run benchmark clean help test

# Comando por defecto: ejecutar benchmark completo
all: install generate benchmark

# Instalar dependencias
install:
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt --break-system-packages
	@echo "Dependencias instaladas."

# Generar instancias de prueba
generate:
	@echo "Generando instancias..."
	$(PYTHON) main.py generate --visualize
	@echo "Instancias generadas en instances/"

# Ejecutar benchmark completo (comando principal requerido)
run: generate
	@echo "Ejecutando experimentación completa..."
	$(PYTHON) main.py benchmark --replicas 5 --time-limit 60
	@echo "Resultados en results/"

# Alias para run
benchmark: run

# Ejecutar un algoritmo específico
solve-bb:
	@echo "Ejecutando Branch & Bound en instance_n10..."
	$(PYTHON) main.py solve instance_n10 --algo bb

solve-geo:
	@echo "Ejecutando Heurística Geométrica en instance_n15..."
	$(PYTHON) main.py solve instance_n15 --algo geo

solve-sa:
	@echo "Ejecutando Simulated Annealing en instance_n20..."
	$(PYTHON) main.py solve instance_n20 --algo sa

# Comparar algoritmos en una instancia
compare:
	@echo "Comparando algoritmos en instance_n10..."
	$(PYTHON) main.py compare instance_n10

# Ejecutar pruebas unitarias de los módulos
test:
	@echo "Ejecutando pruebas de geometría..."
	$(PYTHON) -m common.geometry
	@echo ""
	@echo "Ejecutando pruebas de grafos..."
	$(PYTHON) -m common.graph
	@echo ""
	@echo "Ejecutando pruebas de Pareto..."
	$(PYTHON) -m common.pareto
	@echo ""
	@echo "Ejecutando pruebas de Branch & Bound..."
	$(PYTHON) -m exact_bb.branch_bound
	@echo ""
	@echo "Ejecutando pruebas de Heurística Geométrica..."
	$(PYTHON) -m geo_heuristic.visibility_graph
	@echo ""
	@echo "Ejecutando pruebas de Simulated Annealing..."
	$(PYTHON) -m metaheuristic.simulated_annealing
	@echo ""
	@echo "Todas las pruebas completadas."

# Limpiar archivos generados
clean:
	@echo "Limpiando archivos generados..."
	rm -rf instances/*.json
	rm -rf instances/*.png
	rm -rf results/*
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	@echo "Limpieza completada."

# Ayuda
help:
	@echo "Makefile para Planificación Multiobjetivo de Rutas de Drones"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias"
	@echo "  make generate   - Generar instancias de prueba"
	@echo "  make run        - Ejecutar experimentación completa (PRINCIPAL)"
	@echo "  make benchmark  - Alias de 'make run'"
	@echo "  make solve-bb   - Ejecutar Branch & Bound"
	@echo "  make solve-geo  - Ejecutar Heurística Geométrica"
	@echo "  make solve-sa   - Ejecutar Simulated Annealing"
	@echo "  make compare    - Comparar los 3 algoritmos"
	@echo "  make test       - Ejecutar pruebas unitarias"
	@echo "  make clean      - Limpiar archivos generados"
	@echo "  make help       - Mostrar esta ayuda"
	@echo ""
	@echo "Uso típico:"
	@echo "  make install    # Primera vez"
	@echo "  make run        # Ejecutar todo"
