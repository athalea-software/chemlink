# ChemLink

<div align="center">
  <img src="docs/chemlink-banner.png" alt="ChemLink Banner" width="480" />
</div>

<div align="center">

![ChemLink](https://img.shields.io/badge/ChemLink-v1.0.0-black?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-En_Desarrollo-orange?style=for-the-badge)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=for-the-badge)
![Plataforma](https://img.shields.io/badge/Plataforma-Linux%20HPC-lightgrey?style=for-the-badge)

</div>

## Resumen

ChemLink es una plataforma de orquestación modular para química computacional desarrollada en el laboratorio Chemlab de la Universidad del Norte. Surge ante la necesidad de convertir una infraestructura HPC subutilizada —múltiples nodos con GPUs NVIDIA sin coordinación— en un entorno de cómputo científico automatizado, trazable y reproducible.

La plataforma automatiza dos flujos experimentales: el pipeline de **docking molecular**, que coordina la detección de sitios activos (fpocket), la preparación de estructuras (Meeko/AutoGrid4) y la búsqueda conformacional en GPU (AutoDock-GPU); y el pipeline de **dinámica molecular**, que estructura la simulación completa con GROMACS desde la preparación de topologías hasta el análisis de trayectorias, soportando seis tipos de sistemas biológicos. Ambos pipelines operan bajo una CLI unificada en modo nodo único o distribuido mediante Job Arrays de SLURM.

El proyecto también comprende la construcción de la infraestructura HPC sobre la que opera: configuración del clúster con comunicación SSH sin contraseña, almacenamiento compartido NFS sobre OpenMediaVault y monitoreo continuo con Prometheus y Grafana. Los benchmarks de validación demuestran una reducción del tiempo de preparación de experimentos de 45 minutos a menos de 5, una tasa de éxito del 100% en campañas de docking de 1.000 ligandos y un speedup de hasta 2,2× con tres GPUs en modo multinodo.

---

## Documentación del repositorio

| Documento | Descripción |
|---|---|
| [Informe.md](docs/Informe.md) | Documento principal del proyecto |
| [Instalación.md](docs/Instalación.md) | Guía de instalación, desarrollo y despliegue |
| [Desarrollo.md](docs/Desarrollo.md) | Detalles técnicos del desarrollo |

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
chemlink doctor          # verifica entorno, GPU y dependencias
chemlink docking --help  # opciones del pipeline de docking
chemlink dynamics --help # opciones del pipeline de dinámica
```

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
