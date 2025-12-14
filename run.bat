@echo off
REM ============================================================
REM Script de ejecución para Windows
REM Planificación Multiobjetivo de Rutas de Drones
REM Actividad 2 - Diseño Avanzado de Algoritmos
REM Autor: David Valbuena Segura
REM ============================================================

echo ============================================================
echo Planificacion Multiobjetivo de Rutas de Drones
echo ============================================================
echo.

REM Verificar si se pasó un argumento
if "%1"=="" goto run
if "%1"=="install" goto install
if "%1"=="generate" goto generate
if "%1"=="run" goto run
if "%1"=="benchmark" goto run
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="help" goto help
goto help

:install
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo Dependencias instaladas.
goto end

:generate
echo Generando instancias...
python main.py generate --visualize
echo.
echo Instancias generadas en instances/
goto end

:run
echo Generando instancias...
python main.py generate
echo.
echo Ejecutando experimentacion completa...
python main.py benchmark --replicas 5 --time-limit 60
echo.
echo ============================================================
echo Experimentacion completada.
echo Resultados en: results/
echo ============================================================
goto end

:test
echo Ejecutando pruebas...
echo.
echo --- Pruebas de geometria ---
python -c "from common import geometry; geometry"
echo.
echo --- Pruebas de Pareto ---
python -c "from common import pareto; pareto"
echo.
echo Pruebas completadas.
goto end

:clean
echo Limpiando archivos generados...
if exist instances\*.json del /q instances\*.json
if exist instances\*.png del /q instances\*.png
if exist results\*.* del /q results\*.*
if exist results\tables\*.* del /q results\tables\*.*
if exist results\graphs\*.* del /q results\graphs\*.*
echo Limpieza completada.
goto end

:help
echo.
echo Uso: run.bat [comando]
echo.
echo Comandos disponibles:
echo   install    - Instalar dependencias
echo   generate   - Generar instancias de prueba
echo   run        - Ejecutar experimentacion completa (PRINCIPAL)
echo   benchmark  - Alias de 'run'
echo   test       - Ejecutar pruebas
echo   clean      - Limpiar archivos generados
echo   help       - Mostrar esta ayuda
echo.
echo Uso tipico:
echo   run.bat install    (primera vez)
echo   run.bat run        (ejecutar todo)
echo.
echo O simplemente doble clic en run.bat para ejecutar todo.
goto end

:end
echo.
pause
