# Manual de desarrollo — ChemLink

---

## 1. Propósito del documento

Este documento tiene como objetivo servir de guía técnica para comprender, mantener, extender y dar continuidad al desarrollo del proyecto. Está dirigido a futuros equipos de trabajo que necesiten familiarizarse rápidamente con la estructura del repositorio, la organización de la solución, los contenedores, los scripts, las variables de entorno y el flujo de trabajo del sistema.

ChemLink es una plataforma de orquestación científica para química computacional. No es una aplicación web ni un servicio con API REST: es una herramienta de línea de comandos (CLI) que coordina herramientas científicas externas (GROMACS, AutoDock-GPU, fpocket, Meeko) sobre un clúster HPC con SLURM. Este contexto define toda la arquitectura y las decisiones técnicas documentadas aquí.

---

## 2. Descripción general del proyecto desde la perspectiva de desarrollo

ChemLink es un orquestador de simulaciones moleculares implementado en Python 3. Su función principal es automatizar dos flujos de trabajo científico:

1. **Pipeline de docking molecular:** detecta el sitio activo de un receptor, prepara estructuras de proteínas y ligandos, genera mapas de afinidad y ejecuta búsquedas conformacionales sobre GPU con AutoDock-GPU.
2. **Pipeline de dinámica molecular:** construye el complejo biomolecular, genera topologías con GROMACS/ACPYPE y ejecuta la cadena completa de simulación (minimización → equilibrado NVT/NPT → producción → análisis de trayectoria).

Ambos pipelines se exponen mediante una CLI unificada y pueden ejecutarse en modo nodo único (directo o con SLURM) o en modo multinodo distribuido mediante Job Arrays de SLURM.

### 2.1 Tecnologías principales

| Categoría | Tecnología |
|---|---|
| Lenguaje principal | Python 3.10 |
| CLI y output | `argparse`, `rich` (tema propio, paneles, tablas, región de progreso única), `prompt_toolkit` (lectura de teclado de la TUI) |
| Cómputo científico | GROMACS 2025.4, AutoDock-GPU, AutoGrid4, fpocket, MGLTools |
| Paralelismo GPU | CUDA 13.0 (sm_86, sm_89, sm_90, sm_120) |
| Paralelismo MPI | OpenMPI 4.x (`gmx_mpi`) |
| Gestor de trabajos | SLURM (Job Arrays, dependencias de trabajos) |
| Entornos Python | Conda (`bio` — único entorno) |
| Contenedores | Docker + NVIDIA Container Toolkit |
| Almacenamiento compartido | NFS v4 sobre OpenMediaVault |
| Monitoreo de clúster | Prometheus + Grafana |
| Bibliotecas Python | `numpy`, `rdkit`, `rich`, `prompt_toolkit`, `biopython`, `pillow` |
| Herramientas de desarrollo | `git`, Bash, Lmod (módulos de entorno) |

### 2.2 Componentes principales

| Componente | Función |
|---|---|
| **CLI** (`cli/main.py`) | Punto de entrada único; interpreta subcomandos (`docking`, `peptide-docking`, `dynamic`, `hpc`, `doctor`, `demo`, `update`, `completion`), valida argumentos, enmarca cada ejecución (cabecera, resultado, errores) y delega en los pipelines |
| **TUI** (`cli/tui/`) | Lanzador a pantalla completa (`chemlink` sin argumentos): menús y formularios generados a partir del propio `argparse`, por lo que nunca se desincronizan de la CLI |
| **Presentación** (`utils/theme.py`, `style.py`, `progress.py`, `logger.py`) | Paleta y consola compartida, bloques de salida, región de progreso única y formato de logs comunes a toda la aplicación |
| **Pipeline de docking** (`pipelines/docking/`) | Orquesta la secuencia receptor → ligando → sitio activo → docking → análisis |
| **Pipeline de dinámica** (`pipelines/dynamics/`) | Orquesta la cadena completa GROMACS: topología → solvatación → iones → minimización → equilibrado → producción → postprocesamiento → análisis |
| **Adaptadores** (`adapters/`) | Wrappers que encapsulan las herramientas científicas externas: AutoDock-GPU, AutoGrid4, fpocket |
| **Capa HPC** (`hpc/`) | Detección de recursos del clúster, generación y envío de scripts SLURM, runner de dinámica multinodo |
| **Almacenamiento** (`storage/`) | Gestión centralizada de rutas, creación de directorios de corrida con marca temporal, búsqueda de archivos |
| **Utilidades** (`utils/`) | Módulo de reintentos con backoff exponencial, procesadores de moléculas y receptores, clasificación de cofactores y la capa de presentación descrita arriba |

---

## 3. Estructura del repositorio

### 3.1 Árbol general del repositorio

```
/nfs/chemlink/chemlink/
├── adapters/
│   ├── autodock_gpu/
│   ├── autogrid/
│   └── fpocket/
├── cli/
│   ├── main.py               ← parser, handlers y marco de ejecución
│   ├── demo.py               ← `chemlink demo` (corrida simulada de la interfaz)
│   ├── banner.py             ← logo para terminal (CLI y TUI)
│   ├── logo_art.py           ← logo en caracteres de bloque (GENERADO)
│   └── tui/
│       ├── app.py            ← bucle principal y traspaso de la terminal
│       ├── keys.py · anim.py · widgets.py · sysinfo.py
│       └── screens/          ← list_screen.py (menús) · form.py (formularios)
├── data/
│   ├── input/
│   │   ├── receptors/
│   │   ├── ligands/
│   │   └── dynamics/
│   └── output/
├── docs/
│   ├── Desarrollo.md         ← este documento
│   ├── Informe.md
│   ├── Instalación.md
│   ├── FEP.md · Entropia.md
│   ├── index.html            ← sitio web del proyecto
│   └── images/
│       ├── brand/            ← logo oficial y variantes con fondo transparente
│       ├── diagrams/         ← diagramas de arquitectura (PNG)
│       └── figures/          ← gráficas de rendimiento (PNG)
├── hpc/
│   ├── cluster/
│   └── slurm/
│       ├── container/
│       ├── native/
│       └── runner/
├── pipelines/
│   ├── docking/
│   │   └── steps/
│   └── dynamics/
│       ├── steps/
│       ├── gmx_optimizer.py
│       ├── md_analysis.py
│       ├── dccm_analysis.py
│       ├── mmpbsa_analysis.py
│       ├── mdrun_runner.py
│       └── utils.py
├── scripts/                  ← utilidades (figuras, exportaciones, generación del logo)
├── storage/
├── tests/                    ← pruebas `unittest` (incluida tests/test_ui.py)
├── utils/
│   ├── theme.py · style.py · progress.py · logger.py   ← capa de presentación
│   └── retry.py · *_processor.py · cofactor_*.py
├── chemlink                  ← lanzador (script ejecutable)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 3.2 Descripción de directorios y archivos relevantes

| Ruta | Descripción |
|---|---|
| `adapters/` | Un subdirectorio por herramienta externa; cada adaptador encapsula la invocación, validación de rutas y normalización de salida del binario correspondiente |
| `cli/main.py` | Único punto de entrada de la CLI; define todos los subcomandos con `argparse`, renderiza `--help` con Rich, extrae las opciones globales (`--plain`, `--no-color`, `-v`, `-q`, `--debug`) en cualquier posición y ejecuta cada comando dentro de `run_command` (cabecera, pie con duración y panel de error) |
| `cli/tui/` | TUI: `app.py` (bucle, redibujo, traspaso de la terminal al comando), `screens/list_screen.py` (menús con panel de detalle), `screens/form.py` (formularios con validación, autocompletado de rutas y vista previa del comando), `widgets.py`, `sysinfo.py` (hardware en segundo plano) |
| `cli/demo.py` | `chemlink demo`: ejercita barras, logs, tarjetas y paneles de error con tiempos simulados; no ejecuta herramientas |
| `cli/banner.py` · `cli/logo_art.py` | Logo para terminal; `logo_art.py` se genera con `scripts/gen_tui_logo.py` y no se edita a mano |
| `data/input/` | Datos de entrada del usuario: receptores PDB, biblioteca de ligandos SDF/PDBQT, complejos de dinámica molecular |
| `data/output/` | Corridas generadas automáticamente con marca temporal (`run_<timestamp>_<uuid>/`) |
| `docs/` | Documentación del proyecto: `Desarrollo.md`, `Informe.md`, `Instalación.md`, `index.html` |
| `hpc/cluster/` | `resource_detector.py` (detecta CPU, RAM, GPU, red) y `network_detector.py` (inspecciona conectividad entre nodos) |
| `hpc/slurm/native/` | Scripts SLURM para ejecución bare-metal sobre NFS; incluye variantes para cada etapa del pipeline y para dinámica molecular (`dynamics.slurm`, `dynamics_mpi.slurm`) |
| `hpc/slurm/container/` | Scripts SLURM equivalentes para ejecución dentro del contenedor Docker |
| `hpc/slurm/runner/dynamics_runner.py` | Orquestador Python que construye y envía la cadena de trabajos SLURM para dinámica multinodo |
| `docs/images/brand/` | Logo oficial (`chemlink-logo.png`) y sus variantes transparentes (`-dark`, `-light`, `chemlink-mark.png`) generadas con `scripts/gen_brand_assets.py` |
| `docs/images/diagrams/` | Diagramas de arquitectura, secuencia e interacción de módulos (PNG) |
| `docs/images/figures/` | Gráficas de rendimiento y uso de recursos de los benchmarks (PNG) |
| `pipelines/docking/steps/` | Un archivo por etapa: `receptor_preparation.py`, `ligand_preparation.py`, `active_site_detection.py`, `docking_execution.py`, `docking_analysis.py` |
| `pipelines/dynamics/steps/` | Un archivo por etapa: `complex_builder.py`, `ligand_topology.py`, `topology.py`, `solvation.py`, `ions.py`, `energy_min.py`, `equilibration.py`, `production.py`, `post_processing.py`, `analysis.py` |
| `pipelines/dynamics/gmx_optimizer.py` | Calcula los parámetros óptimos de `gmx mdrun` (threads, GPUs, PME) según el hardware detectado |
| `pipelines/dynamics/md_analysis.py` | Análisis de trayectorias: RMSD, Rg, contactos, energía de enlace |
| `pipelines/dynamics/dccm_analysis.py` | Análisis de correlación dinámica cruzada (DCCM) sobre la trayectoria producida |
| `pipelines/dynamics/mmpbsa_analysis.py` | Cálculo de energía de unión MM-PBSA (módulo experimental) |
| `pipelines/dynamics/mdrun_runner.py` | Runner dedicado para invocaciones de `gmx mdrun` con manejo de reintentos y configuración dinámica |
| `pipelines/dynamics/utils.py` | Utilidades compartidas entre los pasos del pipeline de dinámica |
| `storage/file_manager.py` | Gestiona creación de directorios de corrida, búsqueda de archivos por patrón y división de bibliotecas multimolécula |
| `utils/retry.py` | Estrategia de reintento con retroceso exponencial y fallback de comandos alternativos |
| `utils/theme.py` | Paleta (derivada del logo), estilos `cl.*`, consola compartida y alias de módulo para que `utils.x` y `chemlink.utils.x` sean el mismo objeto |
| `utils/style.py` | Secciones, tarjetas, bloques clave/valor y de estado, tablas, paneles de error y el formato de línea de log |
| `utils/progress.py` | Región de progreso única (`Display`) con `Tracker` anidables; API histórica (`step_bar`, `job_bar`, `pipeline_bar`, `checklist`, `busy`, `make_dynamics_progress`) y modo texto para SLURM/pipes |
| `utils/logger.py` · `pipelines/dynamics/logger.py` | Manejador de consola y formato de archivo comunes; `extra={"file_only": True}` envía un registro solo al archivo |
| `Dockerfile` | Imagen todo-en-uno: Ubuntu 24.04 + CUDA 13.0 + GROMACS 2025.4 + AutoDock-GPU + AutoGrid4 + fpocket + Conda (con Meeko) |
| `docker-compose.yml` | Servicio `chemlink-gpu` con acceso a todas las GPUs del host |
| `requirements.txt` | Dependencias Python del entorno `bio` (instaladas en el contenedor) |
| `chemlink` | Lanzador: usa `CHEMLINK_HOME` o el repositorio donde vive el script (si no, la copia del NFS) y `CHEMLINK_PYTHON` o el entorno `bio` del clúster (si no, el entorno Conda activo o `python3`); ejecuta `python -m chemlink.cli.main` |

---

## 4. Organización de la solución a nivel de código

### 4.1 Organización por capas

El código está organizado en seis capas con responsabilidades bien delimitadas:

| Capa | Directorio | Responsabilidad |
|---|---|---|
| **Interfaz** | `cli/`, `utils/theme.py`, `style.py`, `progress.py`, `logger.py` | Punto de entrada, TUI, interpretación de comandos, validación de argumentos, progreso, logs y presentación de resultados |
| **Orquestación científica** | `pipelines/` | Secuencia lógica de cada pipeline; decide qué etapas ejecutar y en qué orden; maneja el estado global de la corrida |
| **Pasos especializados** | `pipelines/*/steps/` | Implementación de cada etapa con su propia estrategia de error y recuperación |
| **Adaptadores externos** | `adapters/` | Encapsulación de cada herramienta científica; resuelve rutas, variables de entorno y normaliza la salida |
| **Infraestructura HPC** | `hpc/` | Detección de hardware, generación y envío de scripts SLURM, orquestación multinodo |
| **Soporte transversal** | `storage/`, `utils/` | Sistema de archivos, logging estructurado, reintentos, progreso, procesadores de moléculas |

El acoplamiento entre capas es deliberadamente bajo: los pipelines no invocan binarios directamente (lo hacen los adaptadores) ni manipulan el sistema de archivos directamente (lo hace `storage/`). Esto permite extender o sustituir cualquier herramienta externa modificando únicamente su adaptador.

### 4.2 Relación entre componentes del sistema y código fuente

| Componente del sistema | Implementación en el código |
|---|---|
| CLI unificada con subcomandos | `cli/main.py` — `main()` → `run_command()`; subparsers `docking`, `peptide-docking`, `dynamic`, `hpc`, `doctor`, `demo`, `update`, `completion` |
| Lanzador interactivo (TUI) | `cli/tui/app.py` + `cli/tui/screens/` |
| Progreso y logs en consola | `utils/progress.py` (región única) + `utils/logger.py` (formato común) |
| Docking: detección de sitio activo | `pipelines/docking/steps/active_site_detection.py` + `adapters/fpocket/` |
| Docking: preparación de receptor | `pipelines/docking/steps/receptor_preparation.py` + `utils/receptor_processor.py` (PDBFixer + Meeko) |
| Docking: preparación de ligandos | `pipelines/docking/steps/ligand_preparation.py` + `utils/molecule_processor.py` |
| Docking: mapas de afinidad | `adapters/autogrid/autogrid_adapter.py` |
| Docking: búsqueda conformacional GPU | `adapters/autodock_gpu/autodock_gpu_adapter.py` |
| Docking: ranking y análisis | `pipelines/docking/steps/docking_analysis.py` |
| Dinámica: topología y ACPYPE | `pipelines/dynamics/steps/ligand_topology.py` + `steps/topology.py` |
| Dinámica: carga del ligando | `pipelines/dynamics/ligand_charge.py` — detección, preparación y trazabilidad |
| Dinámica: energía de unión | `pipelines/dynamics/mmpbsa_analysis.py` — muestreo, índice Protein/LIG y MM-GBSA/MM-PBSA |
| Dinámica: cadena GROMACS | `pipelines/dynamics/steps/` (un módulo por fase) |
| Dinámica: optimización de parámetros | `pipelines/dynamics/gmx_optimizer.py` |
| Dinámica: análisis de trayectoria | `pipelines/dynamics/md_analysis.py`, `dccm_analysis.py` |
| Detección de hardware | `hpc/cluster/resource_detector.py`, `network_detector.py` |
| Orquestación SLURM (docking) | `hpc/slurm/native/run_multinode_pipeline.sh` + scripts `*.slurm` |
| Orquestación SLURM (dinámica) | `hpc/slurm/runner/dynamics_runner.py` + `native/dynamics.slurm` |

---

## 5. Contenedores

### 5.1 Contenedores utilizados

El proyecto define un único contenedor que encapsula toda la pila de software científico:

| Contenedor | Función |
|---|---|
| `chemlink-gpu` | Entorno de ejecución completo: CUDA, GROMACS, AutoDock-GPU, AutoGrid4, fpocket, entorno Conda `bio` (con Meeko) y la CLI de ChemLink |

No se usa una arquitectura multi-contenedor porque el proyecto no tiene frontend, base de datos ni servicios de backend independientes. Toda la orquestación ocurre dentro del mismo proceso Python.

### 5.2 Archivos relacionados con contenedores

| Archivo | Descripción |
|---|---|
| `Dockerfile` | Imagen basada en `nvidia/cuda:13.0.0-devel-ubuntu24.04`; compila desde código fuente GROMACS 2025.4, AutoDock-GPU (sm_86/89/90/120), AutoGrid4, fpocket; instala MGLTools y los entornos Conda `bio` y `mgl_legacy` |
| `docker-compose.yml` | Define el servicio `chemlink-gpu` con `deploy.resources.reservations` para acceder a todas las GPUs del host; monta el repositorio en `/app/chemlink` |

### 5.3 Construcción y ejecución de contenedores

```bash
# Construir la imagen (primera vez o tras cambios en el Dockerfile)
docker compose build

# Iniciar el contenedor en segundo plano
docker compose up -d

# Abrir una shell interactiva dentro del contenedor
docker compose exec chemlink-gpu bash

# Ejecutar un comando directamente
docker compose exec chemlink-gpu chemlink doctor
docker compose exec chemlink-gpu chemlink docking full \
  /app/chemlink/data/input/receptors \
  /app/chemlink/data/input/ligands \
  /app/chemlink/data/output/run_test
```

> **Nota:** la construcción completa de la imagen tarda entre 30 y 60 minutos en una máquina con GPU NVIDIA debido a la compilación de GROMACS y AutoDock-GPU desde código fuente. Es recomendable guardar la imagen resultante en un registro local o como tarball (`docker save`) para no repetir la compilación en cada nodo del clúster.

### 5.4 Redes, puertos y volúmenes

| Elemento | Configuración |
|---|---|
| Red | `chemlink-network` (bridge); solo comunicación local entre contenedores si se amplía a multi-servicio |
| Puertos expuestos | Ninguno — ChemLink es una CLI, no expone servicios HTTP |
| Volumen montado | `./ → /app/chemlink` (bind mount en modo escritura/lectura) |
| Acceso a GPU | `NVIDIA_VISIBLE_DEVICES=all`; requiere `nvidia-container-toolkit` instalado en el host |

### 5.5 Recomendaciones para modificar contenedores

- **Cambios en dependencias Python:** modificar `requirements.txt` y reconstruir. Si la dependencia requiere compilación nativa, agregarla en la etapa Conda correspondiente del `Dockerfile`.
- **Nueva versión de GROMACS o AutoDock-GPU:** actualizar la URL de descarga y las architecturas CUDA objetivo (`GMX_CUDA_TARGET_SM`) en el `Dockerfile`. Verificar compatibilidad entre la versión de CUDA de la imagen base y la GPU del host.
- **Agregar una herramienta nueva:** añadir su instalación en una nueva capa RUN del `Dockerfile` antes del paso de `COPY . .` para aprovechar la caché de capas de Docker.
- **No modificar `PATH` o los entornos Conda manualmente** dentro de un contenedor en ejecución: los cambios se perderán al reiniciar. Toda modificación de entorno debe hacerse en el `Dockerfile`.

---

## 6. Scripts y automatizaciones

### 6.1 Scripts principales del CLI

| Comando | Descripción |
|---|---|
| `chemlink` | Abre la TUI: navegar con ↑↓/⏎, filtrar con `/`, completar rutas con Tab y ejecutar con `Ctrl+R` |
| `chemlink doctor [--json]` | Verifica herramientas externas, GPU, RAM, variables y rutas; muestra una barra de salud y las correcciones sugeridas |
| `chemlink demo [docking\|dynamics\|logs] [--fail] [--speed X]` | Corrida simulada para revisar la interfaz en cualquier equipo (sin GPU ni herramientas) |
| `chemlink docking full -ir <receptores> -il <ligandos> [--engine gpu\|cpu\|both]` | Pipeline completo de docking en modo nodo único |
| `chemlink docking receptor/ligand/active-site/run/analyze` | Una etapa individual del pipeline de docking |
| `chemlink hpc docking --receptor-dir ... --ligand-dir ... --nodes ...` | Genera y envía la cadena de trabajos SLURM para docking en modo HPC (nodo único o multinodo) |
| `chemlink dynamic <tipo> -ir ... -il ... -t <ns>` | Pipeline completo de dinámica molecular (`oprotein`, `pligand`, `ppeptide`, `pacid`, `pprotein`, `ppligand`, `oligand`; además `fep`, `resume`, `analyze`) |
| `chemlink update [--check]` · `chemlink completion bash\|zsh\|install` | Actualización del checkout y autocompletado de la shell |

### 6.2 Scripts auxiliares de SLURM

| Script | Ubicación | Descripción |
|---|---|---|
| `run_multinode_pipeline.sh` | `hpc/slurm/native/` | Lanza la secuencia completa de trabajos SLURM encadenados para docking multinodo |
| `run_dynamics_pipeline.sh` | `hpc/slurm/native/` | Lanza el trabajo SLURM de dinámica molecular |

### 6.3 Scripts de marca

| Script | Descripción |
|---|---|
| `scripts/gen_brand_assets.py` | A partir de `docs/images/brand/chemlink-logo.png` genera las versiones con fondo transparente (`chemlink-logo-dark.png`, `chemlink-logo-light.png`) y el símbolo solo (`chemlink-mark.png`); separa el fondo azul marino con transparencia parcial en los bordes |
| `scripts/gen_tui_logo.py` | Convierte el mismo PNG en caracteres de bloque (`cli/logo_art.py`) para la TUI y el banner; requiere Pillow solo al regenerar |

Si cambia el logo, basta con reemplazar `docs/images/brand/chemlink-logo.png` y ejecutar ambos scripts.

### 6.4 Consideraciones para su uso

- Los scripts SLURM de `hpc/slurm/native/` asumen que `REPO_DIR`, `OUTPUT_DIR` y `PYTHON_BIN` están definidos como variables de entorno antes de ser enviados por `sbatch`. El runner Python (`hpc/slurm/runner/dynamics_runner.py`) los inyecta automáticamente.
- Los scripts de `hpc/slurm/container/` requieren que Docker y `nvidia-container-toolkit` estén instalados en todos los nodos del clúster.

---

## 7. Variables de entorno

### 7.1 Variables principales del sistema

| Variable | Valor típico | Descripción |
|---|---|---|
| `REPO_DIR` | `/nfs/chemlink/chemlink` | Raíz del repositorio en el NFS |
| `PYTHON_BIN` | `/nfs/chemlink/miniconda/envs/bio/bin/python` | Intérprete Python del entorno principal |
| `OUTPUT_DIR` | `/nfs/chemlink/runs/<run_id>/output` | Directorio de salida de la corrida actual |
| `MODULEPATH` | `/nfs/chemlink/modules:$MODULEPATH` | Ruta al módulo Lmod de ChemLink |
| `PYTHONPATH` | `/nfs/chemlink:${PYTHONPATH:-}` | Permite importar el paquete `chemlink` desde cualquier nodo |
| `GMXRC` | `/usr/local/gromacs/bin/GMXRC` | Script de activación del entorno GROMACS |

### 7.2 Variables de configuración de ejecución

| Variable | Contexto | Descripción |
|---|---|---|
| `DYN_CONFIG_JSON` | SLURM dinámica | Ruta al archivo JSON con la configuración completa de la simulación de dinámica |
| `DYN_MPI_TASKS` | SLURM dinámica MPI | Número de tareas MPI (`$SLURM_NTASKS`) |
| `DYN_MPI_HOSTS` | SLURM dinámica MPI | Lista CSV de nodos participantes |
| `CUDA_VISIBLE_DEVICES` | Docking por lote | GPU asignada a cada trabajo de docking (e.g., `0`) |
| `NVIDIA_VISIBLE_DEVICES` | Contenedor Docker | `all` para exponer todas las GPUs al contenedor |
| `NVIDIA_DRIVER_CAPABILITIES` | Contenedor Docker | `compute,utility` |

### 7.3 Variables de benchmarks (definidas en `common.sh`)

| Variable | Descripción |
|---|---|
| `OPT_LIGAND_WORKERS` | Número de workers para preparación de ligandos en modo optimizado (24) |
| `MN_NODES` | Lista CSV de nodos multinodo (`manager,worker1,worker2`) |
| `MN_PARTITION` | Partición SLURM para trabajos multinodo (`debug`) |
| `MN_MAX_GPU_CONCURRENCY` | Máximo de trabajos de docking GPU simultáneos (3) |
| `MIN_WORKERS` | Workers para modo mínimo (1) |

### 7.4 Variables de la interfaz

| Variable | Descripción |
|---|---|
| `NO_COLOR` | Desactiva el color (equivale a `--no-color`) |
| `CHEMLINK_PROGRESS` | `plain` fuerza líneas de progreso con marca de tiempo (automático fuera de una terminal: SLURM, pipes); `rich` fuerza las barras animadas |
| `CHEMLINK_LOG_LEVEL` | Umbral de los logs en consola (`DEBUG`, `INFO`, `WARNING`); los archivos de log siempre guardan todo |
| `CHEMLINK_DEBUG` | Muestra el traceback completo de los errores (equivale a `--debug`) |
| `CHEMLINK_HOME` · `CHEMLINK_PYTHON` | Repositorio e intérprete que usa el lanzador `./chemlink` |

### 7.5 Archivos de configuración y manejo seguro de secretos

ChemLink no usa archivos `.env` ni gestores de secretos: toda la configuración se pasa como argumentos de CLI o variables de entorno definidas en los scripts SLURM. No existen credenciales de terceros ni tokens en el repositorio.

**Lo que no debe subirse al repositorio:**
- Claves privadas SSH de los nodos (`~/.ssh/id_rsa`); las claves son locales a cada nodo del clúster.
- Archivos de topología o datos de entrada propietarios en `data/input/`.
- Salidas de corridas en `data/output/` — pueden ser grandes y contienen datos experimentales; están en `.gitignore`.

---

## 8. Flujo de trabajo de desarrollo

### 8.1 Preparación del entorno

```bash
# 1. Clonar el repositorio (desde cualquier nodo con acceso al NFS)
git clone <url-repositorio> /nfs/chemlink/chemlink
cd /nfs/chemlink/chemlink

# 2. Activar el módulo de entorno (modo nativo sobre el clúster)
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# — O bien, usar el contenedor —
docker compose up -d
docker compose exec chemlink-gpu bash

# 3. Verificar que el entorno está íntegro
chemlink doctor
```

Para trabajar en la interfaz (CLI, TUI, progreso, logs) no hace falta el clúster ni
una GPU: basta un entorno virtual con `rich` y `prompt_toolkit`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit
./chemlink demo          # corrida simulada
./chemlink               # TUI
```

Para desarrollo interactivo de los pipelines es recomendable activar el entorno Conda directamente:

```bash
source /nfs/chemlink/miniconda/bin/activate
conda activate bio
export PYTHONPATH=/nfs/chemlink:$PYTHONPATH
python -m chemlink.cli.main doctor
```

### 8.2 Desarrollo de nuevas funcionalidades

La estructura en capas define dónde debe implementarse cada tipo de cambio:

| Tipo de cambio | Dónde implementarlo |
|---|---|
| Nueva herramienta científica externa | Crear `adapters/<nombre>/` con su clase adaptadora; registrar en el pipeline correspondiente |
| Nueva etapa en el pipeline de docking | Crear `pipelines/docking/steps/<nombre>.py`; importar y llamar en `docking_pipeline.py` |
| Nueva etapa en el pipeline de dinámica | Crear `pipelines/dynamics/steps/<nombre>.py`; registrar en `dynamics_pipeline.py` |
| Nuevo subcomando de la CLI | Agregar subparser en `cli/main.py`; implementar la función de despacho correspondiente. La TUI lo muestra automáticamente (menú y formulario) a partir del parser |
| Progreso o mensajes en un paso nuevo | Usar `step_bar`/`step_bar_iter`/`track` de `utils/progress.py` y un logger de `utils/logger.py` o `get_step_logger`; nunca `print()` ni un `rich.progress.Progress` propio |
| Soporte para nuevo tipo de simulación dinámica | Extender `dynamics_pipeline.py` con la nueva rama lógica; puede requerir un nuevo paso en `steps/` |
| Nuevo script SLURM | Crear en `hpc/slurm/native/` y/o `hpc/slurm/container/`; parametrizar con variables de entorno siguiendo la convención existente |

**Convención de ramas:**

```
feature/<nombre-funcionalidad>    ← nueva funcionalidad
fix/<descripción-breve>           ← corrección de errores (hotfixes, bugfixes)
```

### 8.3 Ejecución de pruebas y validaciones

Las pruebas unitarias que no requieren hardware se ejecutan con `unittest`; la
integración científica se valida además sobre hardware real:

```bash
# Suite completa (desde la raíz; el directorio padre debe estar en PYTHONPATH
# porque algunas pruebas importan el paquete como `chemlink`)
PYTHONPATH="$(dirname "$PWD")" python -m unittest discover -s tests -t .

# Solo la capa de interfaz (progreso, logs, ayuda, TUI renderizada fuera de pantalla)
PYTHONPATH="$(dirname "$PWD")" python -m unittest tests.test_ui

# Revisión visual de cambios en la interfaz
chemlink demo
chemlink demo dynamics --fail

# Verificación de integración del entorno
chemlink doctor

# Prueba rápida de docking (10 ligandos, modo local)
chemlink docking full \
  data/input/receptors \
  data/input/ligands \
  /tmp/test_run

# Nombre explícito de la carpeta (sin la opción se conserva run_<fecha>_<id>)
chemlink docking full data/input/receptors data/input/ligands /tmp/test_run \
  --run-name docking_tnks1

# El receptor es rígido por defecto. La flexibilidad de cadenas laterales se
# habilita explícitamente indicando el radio desde el centro de la caja:
chemlink docking full data/input/receptors data/input/ligands /tmp/test_run \
  --flex-radius 10
chemlink docking full data/input/receptors data/input/ligands /tmp/test_run \
  --rigid-receptor

# Prueba rápida de dinámica (0.1 ns, proteína-ligando)
chemlink dynamic pligand -ir protein.pdb -il ligand.sdf -t 0.1 --replicas 1

# Dinámica con entradas estandarizadas y nombre explícito
chemlink dynamic pligand -ir protein.pdb -il ligand.sdf \
  --run-name dinamica_tnks1
```

Antes de integrar cambios, verificar manualmente:
1. `chemlink doctor` sin errores en todos los nodos del clúster.
2. El pipeline afectado con la escala mínima (10 ligandos o simulación de 0.1 ns).
3. Que los archivos de salida tienen la estructura esperada (DLG, CSV de ranking, directorio de corrida con marca temporal).

La preparación flexible se realiza después de fijar el centro de la caja. Meeko
genera `<receptor>_rigid.pdbqt`, `<receptor>_flex.pdbqt` y el GPF rígido; Vina
recibe el segundo mediante `--flex` y AutoDock-GPU mediante `--flexres`. Cada
receptor produce además `dimensions_boxes/<receptor>_flexibility.json` con los
residuos seleccionados, sus distancias y las exclusiones aplicadas. Solo se
flexibilizan cadenas laterales; ALA, GLY, cisteínas en puentes disulfuro y
residuos con código de inserción no representable por Meeko se excluyen.

`--run-name NAME` permite reemplazar únicamente el nombre automático de la
carpeta de resultados. Está disponible en `docking prepare/full/flow`, en los
seis tipos de `dynamic` y en `hpc docking/dynamics`. El nombre admite letras
ASCII, números, puntos, guiones y guiones bajos, pero no rutas. ChemLink nunca
reutiliza una carpeta personalizada existente. Cuando `hpc dynamics` crea más
de una simulación, añade sufijos estables (`NAME_001`, `NAME_002`, ...). Los
archivos `docking_inputs.json`, `run_config.json` y `hpc_config.json` registran
el nombre y si el directorio fue automático o personalizado.

Las dinámicas aceptan la misma interfaz de entradas que docking: `-ir` /
`--input-receptors` para proteínas receptoras y `-il` / `--input-ligands` para
el ligando o partner. `ppligand` requiere dos `-ir`; los demás sistemas
requieren uno. En `ppeptide`, `pacid` y `pprotein`, `-il` representa el péptido,
ácido nucleico o segunda proteína, respectivamente. `hpc dynamics pligand`
admite varios `-il` y crea una simulación por ligando. La interfaz heredada
`-i`, y los aliases HPC `--protein`/`--ligands`, se conservan, pero no pueden
mezclarse con entradas estandarizadas.

`dynamic pligand` y `hpc dynamics pligand` ejecutan por defecto **dos réplicas
independientes** de cada simulación, una tras otra. Para otro número hay que
indicarlo de forma explícita con `--replicas N` (`--replicas 1` conserva la
corrida única con estructura plana de siempre). La topología, la solvatación,
los iones y la minimización se calculan una sola vez en el directorio raíz;
cada réplica se crea en `replica_NN/` a partir de `em.gro` y repite NVT, NPT,
producción, post-proceso y análisis. La NVT usa `gen_seed = -1`, así que cada
réplica parte de velocidades iniciales distintas. Al terminar se escribe
`replicas_summary.md` con el ΔH/ΔG de MM-GBSA/MM-PBSA y el RMSD medio de cada
réplica, junto con su media ± desviación estándar. Cada `replica_NN/` es un
directorio de corrida completo: `dynamic resume` y `dynamic analyze` aceptan
tanto la raíz (recorren todas las réplicas y crean las que falten) como una
réplica concreta. En HPC, cada job ejecuta las réplicas en secuencia, por lo que
`--time-limit` debe cubrir `N × --time`.

### 8.4 Integración de cambios

Los contribuidores trabajan sobre `feature/*` o `fix/*` y abren Pull Request hacia `develop`. Nunca se hace push directo a `main`.

```bash
# Crear rama desde develop
git checkout develop
git checkout -b feature/mi-funcionalidad
# — o para una corrección —
git checkout -b fix/error-en-analisis

# Desarrollar y hacer commits descriptivos
git add adapters/mi_herramienta/
git commit -m "Add: adaptador para <herramienta> con soporte de sharding"

# Abrir Pull Request hacia develop
gh pr create --title "Add: <descripción>" --base develop
```

**Criterios mínimos para integrar a `develop`:**
- `chemlink doctor` pasa sin errores.
- El pipeline afectado completa una prueba con la escala mínima (10 ligandos / 0.1 ns).
- No se introducen rutas absolutas hardcodeadas fuera de las variables de entorno.
- Los nuevos scripts SLURM siguen la convención de parametrización por variables de entorno (sin valores fijos de partición, nodos o rutas en el script).

---

### 8.5 Publicación de releases

Los releases se gestionan con el **sistema de releases de GitHub** sobre tags semánticos. No existen ramas `release/*` permanentes: se crean para preparar la versión y se eliminan tras el merge.

#### Paso 1 — Crear la rama de release desde `develop`

```bash
git checkout develop
git checkout -b release/v1.2.0
```

En esta rama se realizan únicamente:
- Corrección de bugs finales (no funcionalidades nuevas).
- Actualización del changelog.
- Cambio de número de versión (si aplica).
- Pruebas de validación finales.

#### Paso 2 — Merge a `main` y tag

```bash
git checkout main
git merge release/v1.2.0

git tag v1.2.0
git push origin main
git push origin v1.2.0
```

#### Paso 3 — Sincronizar `develop`

```bash
git checkout develop
git merge release/v1.2.0
git push origin develop
```

La rama `release/v1.2.0` puede eliminarse tras el merge.

#### Paso 4 — Publicar en GitHub

En el repositorio: **Releases → Draft a new release**, seleccionar el tag `v1.2.0` y completar:
- Notas de la versión y changelog.
- Archivos compilados o artefactos adjuntos (si aplica).
- Marcar como *latest release*.

#### Esquema de ramas

```
main          ──── v1.0.0 ──────────────── v1.1.0 ──────────────── v1.2.0
                      ↑                       ↑                       ↑
develop       ────────┴── feat/A ── fix/B ───┴── feat/C ── fix/D ───┴──
                                                              ↑
release/vX.X.X                                      release/v1.2.0 (temporal)
```

---

## 9. Dependencias y servicios externos

### 9.1 Herramientas científicas externas integradas

| Herramienta | Versión probada | Función en ChemLink |
|---|---|---|
| **GROMACS** (`gmx` / `gmx_mpi`) | 2025.4 | Motor de dinámica molecular; invocado por todos los pasos del pipeline de dinámica |
| **AutoDock-GPU** | HEAD (compilado con CUDA) | Motor de docking conformacional en GPU |
| **AutoGrid4** | HEAD (compilado desde fuente) | Generador de mapas de afinidad energética |
| **fpocket** | 4.0 | Detección de cavidades de unión en receptores |
| **Meeko** | vía Conda `bio` (Python 3) | Preparación de estructuras receptor y ligando a formato PDBQT, y generación de GPF |
| **ACPYPE** | vía Conda `bio` | Generación de topologías GROMACS para ligandos pequeños |

### 9.2 Requisitos de acceso para nuevos equipos

Para que un equipo nuevo pueda trabajar sobre el proyecto necesita:

1. **Acceso SSH sin contraseña** a todos los nodos del clúster (claves RSA en `~/.ssh/authorized_keys` de cada nodo).
2. **Montaje NFS** del volumen `/nfs/chemlink` con permisos de lectura/escritura.
3. **Cuenta en el clúster SLURM** con acceso a la partición `debug` (o la que corresponda) y al menos una GPU asignada.
4. **Docker y `nvidia-container-toolkit`** instalados si se va a usar el modo contenedor.
5. **Módulo Lmod** de ChemLink registrado en `MODULEPATH` (`/nfs/chemlink/modules`).

### 9.3 Consideraciones de desarrollo y pruebas

- AutoDock-GPU **requiere GPU física**; GROMACS y el docking con `--engine cpu` (AutoDock Vina) funcionan sin GPU, aunque mucho más lento. Para pruebas de lógica de orquestación pura (rutas, parseo de argumentos, generación de scripts SLURM) se puede mockear la ejecución redirigiendo las llamadas a `subprocess.run` a un stub.
- La interfaz completa (TUI, barras, logs, paneles de error) se prueba en cualquier equipo con `chemlink demo` y `tests/test_ui.py`.
- La compilación de AutoDock-GPU con soporte para arquitecturas `sm_120` (Blackwell) requiere CUDA 13.0+; versiones anteriores del toolkit no incluyen el perfil de compilación para esa arquitectura.
- Todo el sistema opera íntegramente con el entorno `bio` (Python 3.10); ya no existe un segundo entorno Python legado para la preparación de estructuras.

---

## 10. Convenciones del proyecto

### 10.1 Convenciones de código

| Aspecto | Convención |
|---|---|
| Estilo Python | PEP 8; nombres de funciones y variables en `snake_case`; clases en `PascalCase` |
| Nombres de archivos | `snake_case`; un módulo por responsabilidad; los pasos del pipeline llevan el nombre de la etapa (`receptor_preparation.py`) |
| Imports | Stdlib primero, luego terceros, luego internos; separados por línea en blanco |
| Logging | `setup_logger` (`utils/logger.py`) o `get_step_logger` (dinámica); diagnósticos extensos con `extra={"file_only": True}`. Nunca usar `print()` en lógica de pipeline; para una línea puntual junto a una barra, `get_console().print(log_line(...))` |
| Progreso | Solo a través de `utils/progress.py`; nunca crear un `rich.live.Live` o `rich.progress.Progress` propio (Rich admite un solo display en vivo por consola) |
| Estilos | Usar los nombres de estilo `cl.*` de `utils/theme.py` (`cl.ok`, `cl.warn`, `cl.err`, `cl.ref`, `cl.muted`…) en lugar de colores literales |
| Subprocess | Siempre a través de `utils/retry.py` o del adaptador correspondiente; nunca llamar `subprocess.run()` directamente desde un pipeline |
| Rutas | Usar `pathlib.Path` o construir desde variables de entorno; no hardcodear rutas absolutas fuera de `common.sh` |
| Comentarios | Solo donde el "por qué" no es obvio; no comentar el "qué" si los nombres son descriptivos |

### 10.2 Convenciones de repositorio

| Aspecto | Convención |
|---|---|
| Ramas permanentes | `main` (producción estable, solo recibe merges de `release/*`), `develop` (integración continua) |
| Ramas de trabajo | `feature/<nombre>` para funcionalidad nueva; `fix/<descripción>` para correcciones |
| Ramas de release | `release/vX.X.X` (temporal, se elimina tras el merge a `main`) |
| Commits | Prefijo descriptivo: `Add:`, `Fix:`, `Refactor:`, `Update:`, `Remove:`; mensaje en inglés; una idea por commit |
| Pull Requests | Siempre hacia `develop`; descripción con contexto del cambio y cómo probarlo |
| Releases | Tags semánticos `vX.X.X` sobre `main`; publicados como GitHub Release (ver §8.5) |
| Archivos ignorados | `data/output/`, `__pycache__/`, `*.egg-info/`, archivos `.pyc` |

### 10.3 Convenciones de documentación

- `README.md`: instalación rápida y comandos esenciales; debe mantenerse actualizado al agregar nuevos subcomandos.
- `docs/Desarrollo.md` (este documento): guía de desarrollo para futuros equipos; actualizar al incorporar nuevas herramientas, cambiar convenciones o identificar nuevos problemas frecuentes.
- `docs/Informe.md`: documento académico completo del proyecto con benchmarks y análisis; no modificar como parte del flujo de desarrollo ordinario.
- `docs/Instalación.md`: guía de instalación y configuración del entorno desde cero.
- Logo: el README y el sitio usan las variantes de `docs/images/brand/`; no se editan a mano, se regeneran con `scripts/gen_brand_assets.py` (ver §6.3).

---

## 11. Problemas frecuentes y recomendaciones

### 11.1 Problemas frecuentes

| Problema | Síntoma | Solución |
|---|---|---|
| **GPU no detectada en SLURM** | `No CUDA-capable device is detected` en el log del trabajo | Verificar que el nodo está configurado con GPUs en `slurm.conf` (`Gres=gpu:1`) y que `slurmd` tiene acceso a los dispositivos `/dev/nvidia*` |
| **GROMACS lento en GPUs Blackwell (RTX 5000)** | Simulaciones 3–5× más lentas de lo esperado; advertencia `PTX JIT compilation` en el log | Agregar `nstlist = 40` al archivo MDP de producción; compilar GROMACS con `GMX_CUDA_TARGET_SM` que incluya `120` explícitamente. Ver memoria del proyecto: `project_dynamics_gpu.md` |
| **`pdb_poses: "0\n0"` en `stats.json`** | JSON inválido en los resultados del benchmark | Causado por `set -euo pipefail` + `find \| wc -l` cuando el directorio no existe; siempre comprobar la existencia del directorio antes del pipe o usar el guard `[[ -d "$dir" ]] && ...` |
| **Colisión de directorios de corrida multinodo** | Tres corridas comparten el mismo `run_<timestamp>` | Ocurre cuando varios procesos llaman a la función de generación de ID en el mismo segundo; agregar un `sleep 1` entre lanzamientos consecutivos o incluir un UUID en el ID |
| **Matplotlib: `Image size ... too large`** | Error al generar figuras del reporte con barras de valor cero | Las anotaciones de texto con offset fijo quedan fuera del rango del eje cuando todos los valores son ≈0; usar `label_offset = ymax * 0.04` relativo y `ax.set_ylim(0, ymax * 1.25)` |
| **`pprotein` excede el límite de SLURM** | Trabajo cancelado antes de completar la producción | El tipo proteína+proteína requiere ≥8h de pared; aumentar `--time-limit 08:00:00` en el script SLURM de dinámica |
| **`Last SLURM job: none`** en el benchmark multinodo | El ID del trabajo de análisis no se captura | La salida de `chemlink hpc docking` usa el formato `analysis : <id>`, no `job <id>`; usar `grep -oP 'analysis\s*:\s*\K[0-9]+'` |
| **Módulo `chemlink/1.0` no encontrado** | `module load chemlink/1.0` falla | Verificar que `MODULEPATH` incluye `/nfs/chemlink/modules` y que el archivo de módulo Lmod existe en esa ruta |
| **AutoDock-GPU falla en sm_120** | `illegal instruction` o error de compilación | Compilar con `TARGETS="120"`; con libstdc++ reciente, cambiar `-std=c++11` por `-std=c++14` en `Makefile.Cuda` |
| **`externally-managed-environment` al instalar con pip** | `pip install` se niega a instalar en el Python del sistema (Ubuntu/Debian recientes, PEP 668) | Usar un entorno virtual (`python3 -m venv .venv`) o los paquetes del sistema (`apt install python3-rich python3-prompt-toolkit`) |
| **Códigos de escape en los logs de SLURM** | Archivos `.out` con secuencias `\x1b[...` | Fuera de una terminal el progreso pasa solo a modo texto; si aun así ocurre, exportar `CHEMLINK_PROGRESS=plain` o pasar `--plain` |
| **`LiveError: Only one live display may be active at once`** | Fallo al anidar barras en código nuevo | El código crea su propio `Live`/`Progress`; usar los helpers de `utils/progress.py`, que comparten una sola región |
| **La TUI no responde en Windows** | El lanzador no lee el teclado | La TUI necesita una terminal Unix; usar WSL. Los comandos no interactivos sí funcionan |

### 11.2 Deuda técnica conocida

- **Sin integración continua.** Existe una suite `unittest` (`tests/`, más de 200 pruebas), pero no se ejecuta automáticamente en cada Pull Request. Algunas pruebas requieren OpenBabel y `tests/dynamics/test_reanalysis.py` importa una función que ya no existe (`_resolve_file`).
- **Colisión de run IDs en ejecuciones simultáneas.** El ID de corrida se genera con marca de segundo; si dos procesos lo generan en el mismo segundo obtienen el mismo ID. Agregar un sufijo UUID corto (ya presente en `data/output/` pero no en los scripts SLURM) resuelve el problema.
- **`mmpbsa_analysis.py` es experimental.** El módulo existe pero no está integrado en el pipeline principal ni validado sobre los seis tipos de simulación. No debe usarse en producción sin validación adicional.
- **`postold.txt` en `pipelines/dynamics/steps/`.** Archivo residual de una versión anterior del paso de postprocesamiento; puede eliminarse sin efecto.
- **Scripts SLURM de contenedor no completamente sincronizados con los nativos.** Las adiciones recientes a los scripts nativos (parámetros MPI, `--max-gpu-concurrency`) no se han replicado en la variante de contenedor.
- **Textos de consola en dos idiomas.** La interfaz está en inglés, mientras que algunos reportes generados (p. ej. `simulation_summary.md`) siguen en español.

### 11.3 Recomendaciones para continuidad

1. **Ejecutar la suite en CI** (GitHub Actions basta para las pruebas sin GPU) y ampliar las pruebas de adaptadores con mocks de subprocess.
2. **Actualizar el módulo Lmod** si se cambia la estructura de directorios o los entornos Conda, ya que define las variables `PATH` y `PYTHONPATH` que todo el sistema usa.
3. **No modificar los entornos Conda en los nodos directamente:** cualquier cambio en dependencias debe reflejarse en el `Dockerfile` (modo contenedor) y en el procedimiento de instalación del NFS (modo nativo), para que todos los nodos sean idénticos.
4. **Antes de agregar soporte para una nueva GPU**, verificar que el `Dockerfile` incluye su arquitectura sm en `GMX_CUDA_TARGET_SM` y que AutoDock-GPU se compiló con el perfil PTX de fallback correspondiente.

---

## 12. Historial de decisiones técnicas relevantes

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| **Python 3 como lenguaje del orquestador** | Bash puro | Bash no permite manejo de errores por etapa, reintentos con lógica exponencial, ni estructuras de datos complejas para la configuración de dinámica |
| **Arquitectura en capas con adaptadores** | Llamadas directas a `subprocess` desde los pipelines | El aislamiento de los binarios externos en adaptadores permite sustituir herramientas sin modificar los pipelines; demostrado durante el cambio de versión de GROMACS |
| ~~Dos entornos Conda separados (`bio` y `mgl_legacy`)~~ *(superada, ver fila siguiente)* | Un solo entorno Python | MGLTools requería Python 2 y dependencias incompatibles con el ecosistema Python 3 de GROMACS/ACPYPE; la separación evitaba conflictos irresolubles |
| **Migración de MGLTools/AutoDockTools a Meeko** | Mantener MGLTools con su entorno Python 2 dedicado | MGLTools no tiene mantenimiento activo; Meeko (mismo laboratorio Forli/CCSB que AutoDock) es Python 3 puro, se apoya en RDKit (ya usado por el proyecto) y permite volver a un único entorno Conda `bio` |
| **NFS como almacenamiento compartido (no object storage)** | S3 / almacenamiento de objetos | Las herramientas científicas (GROMACS, AutoDock-GPU) requieren semántica POSIX; S3 no la ofrece nativamente sin capa de compatibilidad |
| **SLURM Job Arrays para docking multinodo** | Lanzar un trabajo por ligando individualmente | Los Job Arrays reducen la latencia de planificación y la carga sobre el controlador SLURM cuando hay miles de ligandos |
| **`rich` para la CLI** | `click` + salida plana | `rich` permite paneles, tablas y progreso con color sin dependencias adicionales y con mejor experiencia en terminales del clúster |
| **Una sola región de progreso (`Display`) para toda la aplicación** | Un `Progress`/`Live` por barra | Rich solo admite un display en vivo por consola: con Rich < 14.1, anidar la barra del pipeline y la de un paso fallaba con `LiveError`. Una región que apila `Tracker`s permite anidar libremente y deja una línea ✓/✗ permanente por etapa |
| **Progreso en texto plano fuera de una terminal** | Barras animadas también en archivos | En los `.out` de SLURM las barras se convierten en códigos de escape o no aparecen hasta el final; las líneas con marca de tiempo cada 10 % son legibles y `grep`-ables |
| **TUI con `rich` + `prompt_toolkit`** | Textual / curses | Reutiliza el tema y la consola de la CLI, no añade un framework y genera menús y formularios desde `argparse`, sin una segunda definición de comandos |
| **Logo de la TUI como caracteres de bloque generados** | Imágenes en terminal (sixel/kitty) | Funciona en cualquier terminal y por SSH; la palabra usa el color de texto de la terminal, legible con fondo claro u oscuro |
| **Un único contenedor todo-en-uno** | Contenedores separados por herramienta | ChemLink no es un sistema de microservicios; la compilación de GROMACS y AutoDock-GPU con las dependencias exactas de CUDA en un contenedor único simplifica el despliegue |
| **Compilación desde fuente de AutoDock-GPU** | Usar el binario precompilado oficial | El binario oficial no incluye soporte para sm_120 (Blackwell); la compilación propia permite especificar exactamente los targets de arquitectura GPU del clúster |
| **`nstlist = 40` para GPUs Blackwell** | Usar el valor por defecto | El kernel de actualización de la lista de vecinos usa compilación JIT en sm_120, lo que genera penalizaciones cada vez que se actualiza; ampliar el intervalo reduce la frecuencia de recompilación a costa de un ligero incremento en el cómputo de pares |

---

## 13. Referencias relacionadas

| Recurso | Descripción |
|---|---|
| `README.md` | Instalación rápida y referencia de comandos principales |
| `docs/Informe.md` | Documento académico completo del proyecto con benchmarks y análisis |
| `docs/Instalación.md` | Guía de instalación y configuración del entorno desde cero |
| `docs/FEP.md` | Flujo de energía libre de unión (FEP/ABFE) |
| `docs/Entropia.md` | Corrección entrópica IE/C2 de la energía de unión |
| [GROMACS 2025 — Manual de usuario](https://manual.gromacs.org/documentation/current/index.html) | Referencia oficial de parámetros MDP, flags de `mdrun` y configuración GPU |
| [AutoDock-GPU — GitHub](https://github.com/ccsb-scripps/AutoDock-GPU) | Repositorio oficial con opciones de compilación y uso |
| [fpocket — GitHub](https://github.com/Discngine/fpocket) | Repositorio oficial de fpocket con documentación de parámetros |
| [SchedMD SLURM — Job Arrays](https://slurm.schedmd.com/job_array.html) | Documentación oficial de Job Arrays en SLURM |
| [NVIDIA CUDA Toolkit — Arquitecturas](https://developer.nvidia.com/cuda-gpus) | Tabla de correspondencia entre GPU y código de arquitectura sm |
| [Prometheus — Documentación](https://prometheus.io/docs/introduction/overview/) | Referencia del sistema de monitoreo usado en el clúster |
| [Grafana — Documentación](https://grafana.com/docs/grafana/latest/) | Referencia de la plataforma de visualización del clúster |
