<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/chemlink-logo-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="docs/images/brand/chemlink-logo-light.png" />
    <img src="docs/images/brand/chemlink-logo-light.png" alt="ChemLink" width="460" />
  </picture>
</h1>

<div align="center">

![ChemLink](https://img.shields.io/badge/ChemLink-v1.0.0-0193C4?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-En_Desarrollo-orange?style=for-the-badge)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=for-the-badge)
![Plataforma](https://img.shields.io/badge/Plataforma-Linux%20HPC-lightgrey?style=for-the-badge)

</div>

## Resumen

ChemLink es una plataforma de orquestación modular para química computacional desarrollada en el Laboratorio de Química y Biología Computacional de la Universidad del Norte. Surge ante la necesidad de convertir una infraestructura HPC subutilizada —múltiples nodos con GPUs NVIDIA sin coordinación— en un entorno de cómputo científico automatizado, trazable y reproducible.

La plataforma automatiza dos flujos experimentales: el pipeline de **docking molecular**, que coordina la detección de sitios activos (fpocket), la preparación de estructuras (Meeko/AutoGrid4) y la búsqueda conformacional en GPU (AutoDock-GPU); y el pipeline de **dinámica molecular**, que estructura la simulación completa con GROMACS desde la preparación de topologías hasta el análisis de trayectorias, soportando seis tipos de sistemas biológicos. Ambos pipelines operan bajo una CLI unificada en modo nodo único o distribuido mediante Job Arrays de SLURM.

El proyecto también comprende la construcción de la infraestructura HPC sobre la que opera: configuración del clúster con comunicación SSH sin contraseña, almacenamiento compartido NFS sobre OpenMediaVault y monitoreo continuo con Prometheus y Grafana. Los benchmarks de validación demuestran una reducción del tiempo de preparación de experimentos de 45 minutos a menos de 5, una tasa de éxito del 100% en campañas de docking de 1.000 ligandos y un speedup de hasta 2,2× con tres GPUs en modo multinodo.

---

## Documentación del repositorio

| Documento | Descripción |
|---|---|
| [Informe.md](docs/Informe.md) | Documento principal del proyecto |
| [Instalación.md](docs/Instalación.md) | Guía de instalación, desarrollo y despliegue |
| [Desarrollo.md](docs/Desarrollo.md) | Detalles técnicos del desarrollo |
| [Entropia.md](docs/Entropia.md) | Energía de unión con corrección entrópica IE/C2 |
| [FEP.md](docs/FEP.md) | Energía libre de unión por perturbación (FEP/ABFE) |

---

## Instalación

### Instalación rápida

```bash
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash
```

Instala ChemLink en `/opt/chemlink` junto con el entorno Conda `bio`.

### Con herramientas científicas compiladas

```bash
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash -s -- --full
```

Compila e instala: **fpocket**, **AutoGrid4**, **AutoDock4**, **AutoDock-GPU** y **GROMACS 2025.4**.
Requiere: CUDA Toolkit 12.x, OpenMPI, cmake. Duración estimada: 45–90 min.

### Opciones del instalador

| Opción | Descripción |
|---|---|
| *(sin opciones)* | Instala la capa Python + entornos Conda |
| `--full` | Compila todas las herramientas científicas + GROMACS |
| `--with-gromacs` | Compila solo GROMACS |
| `--dir PATH` | Directorio de instalación (default: `/opt/chemlink`) |
| `--version TAG` | Rama/tag de Git a instalar (default: `main`) |
| `--skip-conda` | Omite la creación de entornos Conda |

### Primer uso

```bash
chemlink                 # lanzador interactivo (TUI)
chemlink doctor          # verifica entorno, GPU y dependencias
chemlink docking --help  # opciones del pipeline de docking
chemlink dynamic --help  # opciones del pipeline de dinámica
```

### Probar la interfaz sin GPU ni herramientas científicas

La interfaz (TUI, barras de progreso, logs y reportes) se puede probar en cualquier
equipo: solo necesita Python ≥ 3.9 con `rich` y `prompt_toolkit`. En Ubuntu/Debian
recientes `pip` no instala paquetes en el Python del sistema, así que se usa un
entorno virtual (o `sudo apt install python3-rich python3-prompt-toolkit`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit
./chemlink demo                    # corrida simulada de docking + dinámica (no ejecuta nada)
./chemlink demo dynamics --fail    # incluye un fallo para ver el reporte de errores
./chemlink demo --plain            # cómo se ve en un archivo de log de SLURM
./chemlink                         # lanzador interactivo (TUI)
./chemlink doctor                  # diagnóstico de herramientas, GPU y rutas
```

Opciones globales (en cualquier posición): `--plain` (líneas de log en vez de barras
animadas; se activa sola fuera de una terminal), `--no-color`, `-v`/`-q`, `--debug`.

### Interfaz de terminal

| Elemento | Qué ofrece |
|---|---|
| **TUI** (`chemlink` sin argumentos) | Menú de comandos agrupado con panel de detalles, corridas recientes y hardware detectado; formularios generados desde la propia CLI, con valores por defecto, validación, autocompletado de rutas (Tab) y vista previa del comando. `Ctrl+R` ejecuta. |
| **Barras de progreso** | Una única región en vivo que apila todas las barras activas (pipeline, paso, mdrun) con porcentaje, conteo, velocidad y ETA; GROMACS muestra ns simulados y ns/día. Cada etapa deja una línea `✓`/`✗` permanente. |
| **Logs** | Formato común en consola (`hora · nivel · componente · mensaje`, salidas largas recortadas) y archivos (`simulation.log`, etc.) alineados, completos y con cabecera de sesión. |
| **Errores** | Paneles con la causa, el detalle y la acción sugerida (p. ej. «¿quisiste decir `--engine`?»); `--debug` muestra el traceback. |
| **SLURM / pipes** | Fuera de una terminal, el progreso se escribe como líneas con marca de tiempo (cada 10 % y un latido cada 10 min), sin códigos de escape. |

Variables de entorno: `NO_COLOR=1`, `CHEMLINK_PROGRESS=plain|rich`, `CHEMLINK_LOG_LEVEL`,
`CHEMLINK_DEBUG=1`, `CHEMLINK_HOME` y `CHEMLINK_PYTHON` (repositorio e intérprete que usa
el lanzador `./chemlink`).

> Las dinámicas proteína-ligando (`chemlink dynamic pligand`) ejecutan **2 réplicas
> independientes por defecto**, en secuencia. Usa `--replicas N` para cambiar ese
> número (`--replicas 1` para una sola simulación). Ver `docs/Desarrollo.md`.

---

## Estudiantes

| Nombre | GitHub |
|---|---|
| Samuel Matiz García | [@MatizS27](https://github.com/MatizS27) |
| Juan Felipe Santos Rodríguez | [@PipeJF9](https://github.com/PipeJF9) |
| Camilo Andrés Navarro Navarro | [@NavarroCamilo](https://github.com/NavarroCamilo) |

## Tutores

| Nombre |
|---|
| Daniel José Romero Martinez — [@djromerom](https://github.com/djromerom) |
| Augusto Salazar Silva — [@augustosalazar](https://github.com/augustosalazar) |

## Créditos

| Nombre | Rol |
|---|---|
| Edgar Alexander Márquez Brazon | Profesor a cargo del Laboratorio de Química Computacional |
| Oscar Andrés Saurith Coronell — [@ousak20](https://github.com/ousak20) | Desarrollador de scripts |
| José Elías Samur Benitez — [@jesamur](https://github.com/jesamur) | Colaborador en el desarrollo de scripts |
| Francesco Genaro Rosa Chedraui — [@franzeskorch](https://github.com/franzeskorch) | Colaborador en el desarrollo de scripts |

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=PipeJF9/chemlink&type=Date)](https://star-history.com/#PipeJF9/chemlink&Date)

---

## Support

If ChemLink has been useful for your research, consider buying us a coffee to support continued development.

[![Buy us a coffee](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=2GJFPY6AMBXX2)
