<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/chemlink-logo-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="docs/images/brand/chemlink-logo-light.png" />
    <img src="docs/images/brand/chemlink-logo-light.png" alt="ChemLink" width="460" />
  </picture>
</h1>

<div align="center">

![ChemLink](https://img.shields.io/badge/ChemLink-v1.0.0-0193C4?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-AGPL--3.0%20%2B%20commercial-blue?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Linux%20HPC-lightgrey?style=for-the-badge)

</div>

## Overview

ChemLink is a modular orchestration platform for computational chemistry, developed at the Computational Chemistry and Biology Laboratory of Universidad del Norte. It grew out of the need to turn an underused HPC infrastructure (several nodes with NVIDIA GPUs and no coordination between them) into an automated, traceable and reproducible scientific computing environment.

The platform automates two experimental workflows. The **molecular docking** pipeline coordinates active-site detection (fpocket), structure preparation (Meeko/AutoGrid4) and GPU conformational search (AutoDock-GPU). The **molecular dynamics** pipeline runs a complete GROMACS simulation, from topology preparation to trajectory analysis, for six types of biological systems. Both pipelines share a single CLI and run on a single node or distributed through SLURM job arrays.

The project also covers the HPC infrastructure it runs on: a cluster with passwordless SSH between nodes, NFS shared storage on OpenMediaVault, and continuous monitoring with Prometheus and Grafana. Validation benchmarks show experiment setup time dropping from 45 minutes to under 5, a 100% success rate on 1,000-ligand docking campaigns, and a speedup of up to 2.2× with three GPUs in multi-node mode.

---

## Documentation

| Document | Description |
|---|---|
| [Report.md](docs/Report.md) | Main project report |
| [Installation.md](docs/Installation.md) | Installation, development and deployment guide |
| [Development.md](docs/Development.md) | Technical development details |
| [Entropy.md](docs/Entropy.md) | Binding energy with IE/C2 entropy correction |

---

## Installation

### Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash
```

Installs ChemLink in `/opt/chemlink` together with the `bio` Conda environment.

### With compiled scientific tools

```bash
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash -s -- --full
```

Builds and installs **fpocket**, **AutoGrid4**, **AutoDock4**, **AutoDock-GPU** and **GROMACS 2025.4**.
Requires CUDA Toolkit 12.x, OpenMPI and cmake. Expected time: 45–90 min.

### Installer options

| Option | Description |
|---|---|
| *(no options)* | Installs the Python layer and Conda environments |
| `--full` | Builds all scientific tools, including GROMACS |
| `--with-gromacs` | Builds GROMACS only |
| `--dir PATH` | Installation directory (default: `/opt/chemlink`) |
| `--version TAG` | Git branch/tag to install (default: `main`) |
| `--skip-conda` | Skips creating the Conda environments |

### First steps

```bash
chemlink                 # interactive launcher (TUI)
chemlink doctor          # checks the environment, GPU and dependencies
chemlink docking --help  # docking pipeline options
chemlink dynamic --help  # dynamics pipeline options
chemlink docking peptide --help  # peptide-protein docking (LightDock + PRODIGY)
```

### Automatic paths

No path option is required. Each command reads its inputs from the directory it
runs in and writes its results there:

| Input | Looked up in |
|---|---|
| Receptors | `./receptors/`, otherwise the `*.pdb` files in the directory |
| Ligands | `./ligands/`, otherwise the `*.sdf/*.mol2/*.mol/*.pdbqt` files in the directory |
| Peptides | `./peptides/`, otherwise the `*.sdf` files in the directory |
| Dynamics | protein, peptide, nucleic acid or ligand, detected from file content |
| Results | `./results/<pipeline>/` (`-o DIR` for another base) |

```bash
cd my_project               # 1hsg.pdb, ligand.sdf
chemlink docking full        # → results/docking/run_…
chemlink dynamic pligand     # → results/dynamics/run_pligand_…
chemlink dynamic analyze     # latest run in results/dynamics
```

Executable paths are set with environment variables (`FPOCKET_BIN`,
`AUTOGRID4_BIN`, `AUTODOCK_GPU_BIN`, `VINA_BIN`, `PRODIGY_BIN`) or found on `PATH`; `chemlink doctor` shows which ones it found.

### Trying the interface without a GPU or scientific tools

The interface (TUI, progress bars, logs and reports) runs on any machine: it only
needs Python ≥ 3.9 with `rich` and `prompt_toolkit`. Recent Ubuntu/Debian releases
don't let `pip` install into the system Python, so use a virtual environment
(or `sudo apt install python3-rich python3-prompt-toolkit`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit
./chemlink demo                    # simulated docking + dynamics run (runs nothing)
./chemlink demo dynamics --fail    # includes a failure to show the error report
./chemlink demo --plain            # what it looks like in a SLURM log file
./chemlink                         # interactive launcher (TUI)
./chemlink doctor                  # tools, GPU and paths diagnostics
```

Global options (accepted anywhere on the command line): `--plain` (log lines instead
of animated bars; turned on automatically outside a terminal), `--no-color`, `-v`/`-q`, `--debug`.

### Terminal interface

| Element | What it does |
|---|---|
| **TUI** (`chemlink` with no arguments) | Grouped command menu with a details panel, recent runs and detected hardware; forms generated from the CLI itself, with defaults, validation, path completion (Tab) and a command preview. `Ctrl+R` runs. |
| **Progress bars** | A single live region that stacks every active bar (pipeline, step, mdrun) with percentage, count, rate and ETA; GROMACS shows simulated ns and ns/day. Each stage leaves a permanent `✓`/`✗` line. |
| **Logs** | One console format (`time · level · component · message`, long output truncated) and aligned, complete log files (`simulation.log`, etc.) with a session header. |
| **Errors** | Panels with the cause, the details and a suggested fix (e.g. "did you mean `--engine`?"); `--debug` shows the traceback. |
| **SLURM / pipes** | Outside a terminal, progress is written as timestamped lines (every 10% plus a heartbeat every 10 min), with no escape codes. |

Environment variables: `NO_COLOR=1`, `CHEMLINK_PROGRESS=plain|rich`, `CHEMLINK_LOG_LEVEL`,
`CHEMLINK_DEBUG=1`, `CHEMLINK_HOME` and `CHEMLINK_PYTHON` (repository and interpreter used
by the `./chemlink` launcher).

> Protein-ligand dynamics (`chemlink dynamic pligand`) run **2 independent replicas
> by default**, one after the other. Use `--replicas N` to change that number
> (`--replicas 1` for a single simulation). Protein-only dynamics
> (`chemlink dynamic oprotein`) run one simulation unless you pass `--replicas N`.
> See `docs/Development.md`.

---

## Students

| Name | GitHub |
|---|---|
| Samuel Matiz García | [@MatizS27](https://github.com/MatizS27) |
| Juan Felipe Santos Rodríguez | [@PipeJF9](https://github.com/PipeJF9) |
| Camilo Andrés Navarro Navarro | [@NavarroCamilo](https://github.com/NavarroCamilo) |

## Advisors

| Name |
|---|
| Daniel José Romero Martinez — [@djromerom](https://github.com/djromerom) |
| Augusto Salazar Silva — [@augustosalazar](https://github.com/augustosalazar) |

## Credits

| Name | Role |
|---|---|
| Edgar Alexander Márquez Brazon | Professor in charge of the Computational Chemistry Laboratory |
| Oscar Andrés Saurith Coronell — [@ousak20](https://github.com/ousak20) | Script developer |
| José Elías Samur Benitez — [@jesamur](https://github.com/jesamur) | Script development contributor |
| Francesco Genaro Rosa Chedraui — [@franzeskorch](https://github.com/franzeskorch) | Script development contributor |

## License

ChemLink is available under [AGPL-3.0-only](LICENSE) or under a commercial
license for those who cannot comply with the AGPL. See [LICENSING.md](LICENSING.md);
the licenses of third-party libraries and programs are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). To contribute, see
[CONTRIBUTING.md](CONTRIBUTING.md) and the [contributor agreement](CLA.md).

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=PipeJF9/chemlink&type=Date)](https://star-history.com/#PipeJF9/chemlink&Date)

---

## Support

If ChemLink has been useful for your research, consider buying us a coffee to support continued development.

[![Buy us a coffee](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=2GJFPY6AMBXX2)
