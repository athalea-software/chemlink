# Development Manual — ChemLink

---

## 1. Purpose of this document

This document is a technical guide for understanding, maintaining, extending and continuing the development of the project. It is aimed at future teams who need to get familiar quickly with the repository structure, the organization of the solution, the containers, the scripts, the environment variables and the system's workflow.

ChemLink is a scientific orchestration platform for computational chemistry. It is not a web application or a service with a REST API: it is a command-line tool (CLI) that coordinates external scientific tools (GROMACS, AutoDock-GPU, fpocket, Meeko) on an HPC cluster with SLURM. This context drives the whole architecture and the technical decisions documented here.

---

## 2. Project overview from a development perspective

ChemLink is a molecular simulation orchestrator written in Python 3. Its main job is to automate two scientific workflows:

1. **Molecular docking pipeline:** detects a receptor's active site, prepares protein and ligand structures, generates affinity maps and runs GPU conformational searches with AutoDock-GPU.
2. **Molecular dynamics pipeline:** builds the biomolecular complex, generates topologies with GROMACS/ACPYPE and runs the full simulation chain (minimization → NVT/NPT equilibration → production → trajectory analysis).

Both pipelines are exposed through a unified CLI and can run in single-node mode (directly or with SLURM) or in distributed multi-node mode through SLURM job arrays.

### 2.1 Main technologies

| Category | Technology |
|---|---|
| Main language | Python 3.10 |
| CLI and output | `argparse`, `rich` (custom theme, panels, tables, single progress region), `prompt_toolkit` (TUI keyboard input) |
| Scientific computing | GROMACS 2025.4, AutoDock-GPU, AutoDock Vina, AutoGrid4, fpocket |
| GPU parallelism | CUDA 13.0 (sm_86, sm_89, sm_90, sm_120) |
| MPI parallelism | OpenMPI 4.x (`gmx_mpi`) |
| Job scheduler | SLURM (job arrays, job dependencies) |
| Python environments | Conda (`bio` — single environment) |
| Containers | Docker + NVIDIA Container Toolkit |
| Shared storage | NFS v4 on OpenMediaVault |
| Cluster monitoring | Prometheus + Grafana |
| Python libraries | `numpy`, `rdkit`, `rich`, `prompt_toolkit`, `biopython`, `pillow` |
| Development tools | `git`, Bash, Lmod (environment modules) |

### 2.2 Main components

| Component | Role |
|---|---|
| **CLI** (`cli/main.py`) | Single entry point; parses subcommands (`docking`, including `docking peptide`, `dynamic`, `hpc`, `doctor`, `demo`, `update`, `completion`), validates arguments, frames each run (header, result, errors) and delegates to the pipelines |
| **TUI** (`cli/tui/`) | Full-screen launcher (`chemlink` with no arguments): menus and forms generated from the `argparse` parser itself, so they never drift from the CLI |
| **Presentation** (`utils/theme.py`, `style.py`, `progress.py`, `logger.py`) | Shared palette and console, output blocks, single progress region and log format common to the whole application |
| **Docking pipeline** (`pipelines/docking/`) | Orchestrates the receptor → ligand → active site → docking → analysis sequence |
| **Peptide docking pipeline** (`pipelines/docking/peptide_docking/`) | Peptide preparation → LightDock → ranking; the robust variant adds gate → audit → PRODIGY → HADDOCK3 → PyRosetta/FlexPepDock → combined ranking |
| **Dynamics pipeline** (`pipelines/dynamics/`) | Orchestrates the full GROMACS chain: topology → solvation → ions → minimization → equilibration → production → post-processing → analysis |
| **Adapters** (`adapters/`) | Wrappers around the external scientific tools: AutoDock-GPU, AutoGrid4, fpocket, LightDock, PRODIGY, HADDOCK3, PyRosetta |
| **HPC layer** (`hpc/`) | Cluster resource detection, SLURM script generation and submission, multi-node dynamics runner |
| **Storage** (`storage/`) | Centralized path management, creation of timestamped run directories, file lookup |
| **Utilities** (`utils/`) | Retry module with exponential backoff, molecule and receptor processors, cofactor classification and the presentation layer described above |

---

## 3. Repository structure

### 3.1 Repository tree

```
/nfs/chemlink/chemlink/
├── adapters/
│   ├── autodock_gpu/
│   ├── autogrid/
│   ├── fpocket/
│   ├── haddock3/
│   ├── lightdock/
│   ├── prodigy/
│   └── pyrosetta/
├── cli/
│   ├── main.py               ← parser, handlers and run framing
│   ├── demo.py               ← `chemlink demo` (simulated run of the interface)
│   ├── banner.py             ← terminal logo (CLI and TUI)
│   ├── logo_art.py           ← logo in block characters (GENERATED)
│   └── tui/
│       ├── app.py            ← main loop and terminal hand-off
│       ├── keys.py · anim.py · widgets.py · sysinfo.py
│       └── screens/          ← list_screen.py (menus) · form.py (forms)
├── data/
│   ├── input/
│   │   ├── receptors/
│   │   ├── ligands/
│   │   └── dynamics/
│   └── output/
├── docs/
│   ├── Development.md        ← this document
│   ├── Report.md
│   ├── Installation.md
│   ├── FEP.md · Entropy.md
│   ├── index.html            ← project website
│   └── images/
│       ├── brand/            ← official logo and transparent-background variants
│       ├── diagrams/         ← architecture diagrams (PNG)
│       └── figures/          ← performance charts (PNG)
├── hpc/
│   ├── cluster/
│   └── slurm/
│       ├── container/
│       ├── native/
│       └── runner/
├── pipelines/
│   ├── docking/
│   │   ├── steps/
│   │   └── peptide_docking/  ← LightDock + robustness pipeline (steps/)
│   └── dynamics/
│       ├── steps/
│       ├── gmx_optimizer.py
│       ├── md_analysis.py
│       ├── dccm_analysis.py
│       ├── mmpbsa_analysis.py
│       ├── mdrun_runner.py
│       └── utils.py
├── scripts/                  ← utilities (figures, exports, logo generation)
├── storage/
├── tests/                    ← `unittest` tests (including tests/test_ui.py)
├── utils/
│   ├── theme.py · style.py · progress.py · logger.py   ← presentation layer
│   └── retry.py · *_processor.py · cofactor_*.py
├── chemlink                  ← launcher (executable script)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 3.2 Relevant directories and files

| Path | Description |
|---|---|
| `adapters/` | One subdirectory per external tool; each adapter wraps the invocation, path validation and output normalization of the corresponding binary |
| `cli/main.py` | Single CLI entry point; defines every subcommand with `argparse`, renders `--help` with Rich, extracts the global options (`--plain`, `--no-color`, `-v`, `-q`, `--debug`) from any position and runs each command inside `run_command` (header, footer with duration and error panel) |
| `cli/tui/` | TUI: `app.py` (loop, redraw, terminal hand-off to the command), `screens/list_screen.py` (menus with a details panel), `screens/form.py` (forms with validation, path completion and a command preview), `widgets.py`, `sysinfo.py` (hardware, in the background) |
| `cli/demo.py` | `chemlink demo`: exercises bars, logs, cards and error panels with simulated timings; runs no tools |
| `cli/banner.py` · `cli/logo_art.py` | Terminal logo; `logo_art.py` is generated by `scripts/gen_tui_logo.py` and is not edited by hand |
| `data/input/` | User input data: PDB receptors, SDF/PDBQT ligand library, molecular dynamics complexes |
| `data/output/` | Automatically generated timestamped runs (`run_<timestamp>_<uuid>/`) |
| `docs/` | Project documentation: `Development.md`, `Report.md`, `Installation.md`, `FEP.md`, `Entropy.md`, `index.html` |
| `hpc/cluster/` | `resource_detector.py` (detects CPU, RAM, GPU, network) |
| `hpc/slurm/native/` | SLURM scripts for bare-metal execution on NFS; includes variants for each pipeline stage and for molecular dynamics (`dynamics.slurm`) |
| `hpc/slurm/container/` | Equivalent SLURM scripts for execution inside the Docker container |
| `hpc/slurm/runner/dynamics_runner.py` | Python orchestrator that builds and submits the SLURM job chain for multi-node dynamics |
| `docs/images/brand/` | Official logo (`chemlink-logo.png`) and its transparent variants (`-dark`, `-light`, `chemlink-mark.png`) generated with `scripts/gen_brand_assets.py` |
| `docs/images/diagrams/` | Architecture, sequence and module interaction diagrams (PNG) |
| `docs/images/figures/` | Benchmark performance and resource usage charts (PNG) |
| `pipelines/docking/peptide_docking/` | `peptide_docking_pipeline.py` (preparation + LightDock + ranking) and `robust_pipeline.py` (gate, audit, PRODIGY, HADDOCK3, PyRosetta/FlexPepDock, combined ranking, resumable through `robust_config.json`); `steps/` holds one file per stage |
| `utils/peptide_processor.py` | Peptide SDF → PDB reconstruction (residue classification, chain walk, standardization of non-canonical residues, deduplication) |
| `pipelines/docking/steps/` | One file per stage: `receptor_preparation.py`, `ligand_preparation.py`, `active_site_detection.py`, `docking_execution.py`, `docking_analysis.py` |
| `pipelines/dynamics/steps/` | One file per stage: `complex_builder.py`, `ligand_topology.py`, `topology.py`, `solvation.py`, `ions.py`, `energy_min.py`, `equilibration.py`, `production.py`, `post_processing.py`, `analysis.py` |
| `pipelines/dynamics/gmx_optimizer.py` | Computes the best `gmx mdrun` parameters (threads, GPUs, PME) for the detected hardware |
| `pipelines/dynamics/md_analysis.py` | Trajectory analysis: RMSD, Rg, contacts, binding energy |
| `pipelines/dynamics/dccm_analysis.py` | Dynamic cross-correlation (DCCM) analysis of the produced trajectory |
| `pipelines/dynamics/fep_pipeline.py` · `fep_execution.py` · `fep_analysis.py` | FEP preparation (ABFE/RBFE, Boresch restraints), λ-window execution and ΔG analysis (alchemlyb/pymbar) |
| `pipelines/dynamics/mmpbsa_analysis.py` | MM-PBSA binding energy calculation (experimental module) |
| `pipelines/dynamics/mdrun_runner.py` | Dedicated runner for `gmx mdrun` invocations with retry handling and dynamic configuration |
| `pipelines/dynamics/utils.py` | Utilities shared by the dynamics pipeline steps |
| `storage/file_manager.py` | Handles run directory creation, file lookup by pattern and splitting of multi-molecule libraries |
| `utils/retry.py` | Retry strategy with exponential backoff and fallback to alternative commands |
| `utils/theme.py` | Palette (derived from the logo), `cl.*` styles, shared console and module aliasing so that `utils.x` and `chemlink.utils.x` are the same object |
| `utils/style.py` | Sections, cards, key/value and status blocks, tables, error panels and the log line format |
| `utils/progress.py` | Single progress region (`Display`) with nestable `Tracker`s; legacy API (`step_bar`, `job_bar`, `pipeline_bar`, `checklist`, `busy`, `make_dynamics_progress`) and a text mode for SLURM/pipes |
| `utils/logger.py` · `pipelines/dynamics/logger.py` | Shared console handler and file format; `extra={"file_only": True}` sends a record to the file only |
| `Dockerfile` | Two-stage all-in-one image: Ubuntu 24.04 + CUDA 13.0 runtime + GROMACS 2025.4 + AutoDock-GPU + AutoGrid4 + fpocket + Miniforge (with Meeko), with license notices and third-party source code |
| `docker-compose.yml` | `chemlink-gpu` service with access to every host GPU |
| `requirements.txt` | Python dependencies for pip installs (`venv`); the `bio` Conda environment gets them from conda-forge |
| `chemlink` | Launcher: uses `CHEMLINK_HOME` or the repository the script lives in (otherwise the NFS copy) and `CHEMLINK_PYTHON` or the cluster's `bio` environment (otherwise the active Conda environment or `python3`); runs `python -m chemlink.cli.main` |

---

## 4. Code organization

### 4.1 Layers

The code is organized in six layers with clearly separated responsibilities:

| Layer | Directory | Responsibility |
|---|---|---|
| **Interface** | `cli/`, `utils/theme.py`, `style.py`, `progress.py`, `logger.py` | Entry point, TUI, command parsing, argument validation, progress, logs and presentation of results |
| **Scientific orchestration** | `pipelines/` | Logical sequence of each pipeline; decides which stages run and in what order; holds the global state of the run |
| **Specialized steps** | `pipelines/*/steps/` | Implementation of each stage with its own error and recovery strategy |
| **External adapters** | `adapters/` | Wrapping of each scientific tool; resolves paths and environment variables and normalizes the output |
| **HPC infrastructure** | `hpc/` | Hardware detection, SLURM script generation and submission, multi-node orchestration |
| **Cross-cutting support** | `storage/`, `utils/` | File system, structured logging, retries, progress, molecule processors |

Coupling between layers is deliberately low: pipelines do not call binaries directly (the adapters do) nor manipulate the file system directly (`storage/` does). This makes it possible to extend or replace any external tool by changing only its adapter.

### 4.2 System components and source code

| System component | Implementation |
|---|---|
| Unified CLI with subcommands | `cli/main.py` — `main()` → `run_command()`; subparsers `docking`, `dynamic`, `hpc`, `doctor`, `demo`, `update`, `completion` |
| Interactive launcher (TUI) | `cli/tui/app.py` + `cli/tui/screens/` |
| Console progress and logs | `utils/progress.py` (single region) + `utils/logger.py` (shared format) |
| Docking: active-site detection | `pipelines/docking/steps/active_site_detection.py` + `adapters/fpocket/` |
| Docking: receptor preparation | `pipelines/docking/steps/receptor_preparation.py` + `utils/receptor_processor.py` (PDBFixer + Meeko) |
| Docking: ligand preparation | `pipelines/docking/steps/ligand_preparation.py` + `utils/molecule_processor.py` |
| Docking: affinity maps | `adapters/autogrid/autogrid_adapter.py` |
| Docking: GPU conformational search | `adapters/autodock_gpu/autodock_gpu_adapter.py` |
| Docking: ranking and analysis | `pipelines/docking/steps/docking_analysis.py` |
| Dynamics: topology and ACPYPE | `pipelines/dynamics/steps/ligand_topology.py` + `steps/topology.py` |
| Dynamics: ligand charge | `pipelines/dynamics/ligand_charge.py` — detection, preparation and traceability |
| Dynamics: binding energy | `pipelines/dynamics/mmpbsa_analysis.py` — sampling, Protein/LIG index and MM-GBSA/MM-PBSA |
| Dynamics: GROMACS chain | `pipelines/dynamics/steps/` (one module per phase) |
| Dynamics: parameter tuning | `pipelines/dynamics/gmx_optimizer.py` |
| Dynamics: trajectory analysis | `pipelines/dynamics/md_analysis.py`, `dccm_analysis.py` |
| Hardware detection | `hpc/cluster/resource_detector.py` |
| SLURM orchestration (docking) | `hpc/slurm/native/run_multinode_pipeline.sh` + `*.slurm` scripts |
| SLURM orchestration (dynamics) | `hpc/slurm/runner/dynamics_runner.py` + `native/dynamics.slurm` |

---

## 5. Containers

### 5.1 Containers used

The project defines a single container that wraps the whole scientific software stack:

| Container | Role |
|---|---|
| `chemlink-gpu` | Full runtime environment: CUDA, GROMACS, AutoDock-GPU, AutoGrid4, fpocket, the `bio` Conda environment (with Meeko) and the ChemLink CLI |

There is no multi-container architecture because the project has no frontend, database or separate backend services. All orchestration happens inside the same Python process.

### 5.2 Container-related files

| File | Description |
|---|---|
| `Dockerfile` | Builds GROMACS 2025.4, AutoDock-GPU (sm_86/89/90/120), AutoGrid4 and AutoDock4 (pinned commits) on `nvidia/cuda:13.0.0-devel-ubuntu24.04`; installs AutoDock Vina, CHARMM36m and the `bio` and `gmx_mmpbsa` Conda environments (Miniforge); the final image starts from `nvidia/cuda:13.0.0-runtime-ubuntu24.04` |
| `docker-compose.yml` | Defines the `chemlink-gpu` service with `deploy.resources.reservations` to access every host GPU; mounts the repository at `/app/chemlink` |

### 5.3 Building and running containers

```bash
# Build the image (first time or after Dockerfile changes)
docker compose build

# Start the container in the background
docker compose up -d

# Open an interactive shell in the container
docker compose exec chemlink-gpu bash

# Run a command directly
docker compose exec chemlink-gpu chemlink doctor
docker compose exec chemlink-gpu chemlink docking full \
  -ir /app/chemlink/data/test/receptors \
  -il /app/chemlink/data/test/ligands \
  -o  /app/chemlink/results/docking
```

> **Note:** a full image build takes 30 to 60 minutes on a machine with an NVIDIA GPU because GROMACS and AutoDock-GPU are compiled from source. It is a good idea to keep the resulting image in a local registry or as a tarball (`docker save`) so the build is not repeated on every cluster node.

### 5.4 Networks, ports and volumes

| Element | Configuration |
|---|---|
| Network | `chemlink-network` (bridge); only local communication between containers if it grows into multiple services |
| Exposed ports | None — ChemLink is a CLI and exposes no HTTP services |
| Mounted volume | `./ → /app/chemlink` (read/write bind mount) |
| GPU access | `NVIDIA_VISIBLE_DEVICES=all`; requires `nvidia-container-toolkit` installed on the host |

### 5.5 Guidelines for changing the containers

- **Python dependency changes:** add them in the three places that install the environment: the `bio` `conda create` in the `Dockerfile` and in `install.sh`, and `requirements.txt` (pip installs). Then rebuild the image. Check the license first and record it in `THIRD_PARTY_LICENSES.md` (see `CONTRIBUTING.md`).
- **New GROMACS or AutoDock-GPU version:** update the download URL and the target CUDA architectures (`GMX_CUDA_TARGET_SM`) in the `Dockerfile`. Check that the base image's CUDA version is compatible with the host GPU.
- **Adding a new tool:** add its installation in a new RUN layer of the `Dockerfile` before the `COPY . .` step to take advantage of Docker's layer cache.
- **Do not change `PATH` or the Conda environments by hand** inside a running container: the changes are lost on restart. Every environment change belongs in the `Dockerfile`.

---

## 6. Scripts and automation

### 6.1 Main CLI commands

| Command | Description |
|---|---|
| `chemlink` | Opens the TUI: navigate with ↑↓/⏎, filter with `/`, complete paths with Tab and run with `Ctrl+R` |
| `chemlink doctor [--json]` | Checks external tools, GPU, RAM, variables and paths; shows a health bar and the suggested fixes |
| `chemlink demo [docking\|dynamics\|logs] [--fail] [--speed X]` | Simulated run to review the interface on any machine (no GPU or tools needed) |
| `chemlink docking full [-ir <receptors>] [-il <ligands>] [--engine gpu\|cpu\|both]` | Full docking pipeline in single-node mode; without paths it uses `./receptors`, `./ligands` or the files in the directory |
| `chemlink docking receptor/ligand/active-site/run/analyze` | A single stage of the docking pipeline |
| `chemlink docking peptide full` · `chemlink docking peptide robust lightdock/audit/prodigy/haddock3/rosetta/combine/all` | Peptide-protein docking with LightDock, and its robustness pipeline |
| `chemlink hpc docking [-ir ...] [-il ...] [--nodes ...]` | Generates and submits the SLURM job chain for docking in HPC mode (single node or multi-node) |
| `chemlink dynamic <type> [-ir ...] [-il ...] -t <ns>` | Full molecular dynamics pipeline (inputs detected in the directory if omitted) (`oprotein`, `pligand`, `ppeptide`, `pacid`, `pprotein`, `ppligand`, `oligand`; plus `fep`, `resume`, `analyze`) |
| `chemlink update [--check]` · `chemlink completion bash\|zsh\|install` | Checkout update and shell completion |

### 6.2 SLURM helper scripts

| Script | Location | Description |
|---|---|---|
| `run_multinode_pipeline.sh` | `hpc/slurm/native/` | Launches the full sequence of chained SLURM jobs for multi-node docking |
| `run_dynamics_pipeline.sh` | `hpc/slurm/native/` | Launches the molecular dynamics SLURM job |

### 6.3 Brand scripts

| Script | Description |
|---|---|
| `scripts/gen_brand_assets.py` | From `docs/images/brand/chemlink-logo.png` it generates the transparent-background versions (`chemlink-logo-dark.png`, `chemlink-logo-light.png`) and the mark alone (`chemlink-mark.png`); it separates the navy background with partial transparency at the edges |
| `scripts/gen_tui_logo.py` | Converts the same PNG into block characters (`cli/logo_art.py`) for the TUI and the banner; needs Pillow only when regenerating |

If the logo changes, replace `docs/images/brand/chemlink-logo.png` and run both scripts.

### 6.4 Usage notes

- The SLURM scripts in `hpc/slurm/native/` assume that `REPO_DIR`, `OUTPUT_DIR` and `PYTHON_BIN` are defined as environment variables before they are submitted with `sbatch`. The Python runner (`hpc/slurm/runner/dynamics_runner.py`) injects them automatically.
- The scripts in `hpc/slurm/container/` require Docker and `nvidia-container-toolkit` to be installed on every cluster node.

---

## 7. Environment variables

### 7.1 Main system variables

| Variable | Typical value | Description |
|---|---|---|
| `REPO_DIR` | `/nfs/chemlink/chemlink` | Repository root on NFS |
| `PYTHON_BIN` | `/nfs/chemlink/miniconda/envs/bio/bin/python` | Python interpreter of the main environment |
| `OUTPUT_DIR` | `/nfs/chemlink/runs/<run_id>/output` | Output directory of the current run |
| `MODULEPATH` | `/nfs/chemlink/modules:$MODULEPATH` | Path to the ChemLink Lmod module |
| `PYTHONPATH` | `/nfs/chemlink:${PYTHONPATH:-}` | Lets any node import the `chemlink` package |
| `GMXRC` | `/usr/local/gromacs/bin/GMXRC` | GROMACS environment activation script |

### 7.2 Run configuration variables

| Variable | Context | Description |
|---|---|---|
| `DYN_CONFIG_JSON` | SLURM dynamics | Path to the JSON file with the full dynamics simulation configuration |
| `DYN_MPI_TASKS` | SLURM MPI dynamics | Number of MPI tasks (`$SLURM_NTASKS`) |
| `DYN_MPI_HOSTS` | SLURM MPI dynamics | CSV list of participating nodes |
| `CUDA_VISIBLE_DEVICES` | Batch docking | GPU assigned to each docking job (e.g., `0`) |
| `NVIDIA_VISIBLE_DEVICES` | Docker container | `all` to expose every GPU to the container |
| `NVIDIA_DRIVER_CAPABILITIES` | Docker container | `compute,utility` |

### 7.3 Benchmark variables (defined in `common.sh`)

| Variable | Description |
|---|---|
| `OPT_LIGAND_WORKERS` | Number of workers for ligand preparation in optimized mode (24) |
| `MN_NODES` | CSV list of multi-node nodes (`manager,worker1,worker2`) |
| `MN_PARTITION` | SLURM partition for multi-node jobs (`debug`) |
| `MN_MAX_GPU_CONCURRENCY` | Maximum simultaneous GPU docking jobs (3) |
| `MIN_WORKERS` | Workers for minimal mode (1) |

### 7.4 Interface variables

| Variable | Description |
|---|---|
| `NO_COLOR` | Turns colour off (same as `--no-color`) |
| `CHEMLINK_PROGRESS` | `plain` forces timestamped progress lines (automatic outside a terminal: SLURM, pipes); `rich` forces the animated bars |
| `CHEMLINK_LOG_LEVEL` | Console log threshold (`DEBUG`, `INFO`, `WARNING`); log files always keep everything |
| `CHEMLINK_DEBUG` | Shows the full traceback of errors (same as `--debug`) |
| `CHEMLINK_HOME` · `CHEMLINK_PYTHON` | Repository and interpreter used by the `./chemlink` launcher |

### 7.5 Configuration files and handling secrets

ChemLink does not use `.env` files or secret managers: all configuration is passed as CLI arguments or environment variables defined in the SLURM scripts. There are no third-party credentials or tokens in the repository.

**What must not be pushed to the repository:**
- The nodes' private SSH keys (`~/.ssh/id_rsa`); keys are local to each cluster node.
- Proprietary topology files or input data in `data/input/`.
- Run outputs in `data/output/` — they can be large and contain experimental data; they are in `.gitignore`.

---

## 8. Development workflow

### 8.1 Setting up the environment

```bash
# 1. Clone the repository (from any node with NFS access)
git clone <repository-url> /nfs/chemlink/chemlink
cd /nfs/chemlink/chemlink

# 2. Load the environment module (native mode on the cluster)
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# — Or use the container —
docker compose up -d
docker compose exec chemlink-gpu bash

# 3. Check that the environment is intact
chemlink doctor
```

Working on the interface (CLI, TUI, progress, logs) needs neither the cluster nor
a GPU: a virtual environment with `rich` and `prompt_toolkit` is enough.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit
./chemlink demo          # simulated run
./chemlink               # TUI
```

For interactive pipeline development, activating the Conda environment directly is recommended:

```bash
source /nfs/chemlink/miniconda/bin/activate
conda activate bio
export PYTHONPATH=/nfs/chemlink:$PYTHONPATH
python -m chemlink.cli.main doctor
```

### 8.2 Developing new features

The layered structure defines where each kind of change belongs:

| Kind of change | Where to implement it |
|---|---|
| New external scientific tool | Create `adapters/<name>/` with its adapter class; register it in the corresponding pipeline |
| New docking pipeline stage | Create `pipelines/docking/steps/<name>.py`; import and call it in `docking_pipeline.py` |
| New dynamics pipeline stage | Create `pipelines/dynamics/steps/<name>.py`; register it in `dynamics_pipeline.py` |
| New CLI subcommand | Add a subparser in `cli/main.py`; implement the corresponding dispatch function. The TUI shows it automatically (menu and form) from the parser |
| Progress or messages in a new step | Use `step_bar`/`step_bar_iter`/`track` from `utils/progress.py` and a logger from `utils/logger.py` or `get_step_logger`; never `print()` or a custom `rich.progress.Progress` |
| Support for a new dynamics simulation type | Extend `dynamics_pipeline.py` with the new logic branch; it may need a new step in `steps/` |
| New SLURM script | Create it in `hpc/slurm/native/` and/or `hpc/slurm/container/`; parametrize it with environment variables following the existing convention |

**Branch naming:**

```
feature/<feature-name>    ← new feature
fix/<short-description>   ← bug fixes (hotfixes, bugfixes)
```

### 8.3 Running tests and checks

Unit tests that need no hardware run with `unittest`; the scientific
integration is also validated on real hardware:

```bash
# Full suite (from the repository root; the parent directory must be on PYTHONPATH
# because some tests import the package as `chemlink`)
PYTHONPATH="$(dirname "$PWD")" python -m unittest discover -s tests -t .

# Interface layer only (progress, logs, help, TUI rendered off-screen)
PYTHONPATH="$(dirname "$PWD")" python -m unittest tests.test_ui

# Visual review of interface changes
chemlink demo
chemlink demo dynamics --fail

# Environment integration check
chemlink doctor

# Quick docking test (10 ligands, local mode)
chemlink docking full -ir data/test/receptors -il data/test/ligands -o /tmp/test_run

# Explicit folder name (without the option, run_<date>_<id> is kept)
chemlink docking full -ir data/test/receptors -il data/test/ligands -o /tmp/test_run \
  --run-name docking_tnks1

# The receptor is rigid by default (--flex-radius 0). Side-chain flexibility
# is turned on by giving the radius from the box center:
chemlink docking full -ir data/test/receptors -il data/test/ligands -o /tmp/test_run \
  --flex-radius 10

# Quick dynamics test (0.1 ns, protein-ligand)
chemlink dynamic pligand -ir protein.pdb -il ligand.sdf -t 0.1 --replicas 1

# Dynamics with standardized inputs and an explicit name
chemlink dynamic pligand -ir protein.pdb -il ligand.sdf \
  --run-name dynamics_tnks1
```

Before merging changes, check by hand:
1. `chemlink doctor` with no errors on every cluster node.
2. The affected pipeline at the smallest scale (10 ligands or a 0.1 ns simulation).
3. That the output files have the expected structure (DLG, ranking CSV, timestamped run directory).

Flexible preparation happens after the box center is fixed. Meeko
generates `<receptor>_rigid.pdbqt`, `<receptor>_flex.pdbqt` and the rigid GPF; Vina
receives the second one through `--flex` and AutoDock-GPU through `--flexres`. Each
receptor also produces `dimensions_boxes/<receptor>_flexibility.json` with the
selected residues, their distances and the exclusions applied. Only
side chains are made flexible; ALA, GLY, cysteines in disulfide bridges and
residues with an insertion code Meeko cannot represent are excluded.

`--run-name NAME` replaces only the automatic name of the
results folder. It is available in `docking prepare/full/flow`, in all
`dynamic` types and in `hpc docking/dynamics`. The name accepts ASCII
letters, digits, dots, hyphens and underscores, but not paths. ChemLink never
reuses an existing custom folder. When `hpc dynamics` creates more
than one simulation, it adds stable suffixes (`NAME_001`, `NAME_002`, ...). The
`docking_inputs.json`, `run_config.json` and `hpc_config.json` files record
the name and whether the directory was automatic or custom.

Dynamics accept the same input interface as docking: `-ir` /
`--input-receptors` for receptor proteins and `-il` / `--input-ligands` for
the ligand or partner. `ppligand` requires two `-ir`; the other systems
require one. In `ppeptide`, `pacid` and `pprotein`, `-il` is the peptide,
nucleic acid or second protein, respectively. `hpc dynamics pligand`
accepts several `-il` and creates one simulation per ligand. The legacy
`-i` interface and the HPC aliases `--protein`/`--ligands` are kept, but cannot
be mixed with the standardized inputs.

`dynamic pligand` and `hpc dynamics pligand` run **two independent
replicas** of each simulation by default, one after the other. Any other number has to be
given explicitly with `--replicas N` (`--replicas 1` keeps the usual
single run with a flat structure). `dynamic oprotein` and `hpc dynamics oprotein`
accept the same option but default to a single run. Topology, solvation,
ions and minimization are computed once in the root directory;
each replica is created in `replica_NN/` from `em.gro` and repeats NVT, NPT,
production, post-processing and analysis. NVT uses `gen_seed = -1`, so each
replica starts from different initial velocities. At the end,
`replicas_summary.md` is written with the MM-GBSA/MM-PBSA ΔH/ΔG (when computed) and the mean RMSD of each
replica, together with their mean ± standard deviation. Each `replica_NN/` is a
complete run directory: `dynamic resume` and `dynamic analyze` accept
either the root (they go through every replica and create the missing ones) or a
specific replica. On HPC, each job runs its replicas in sequence, so
`--time-limit` must cover `N × --time`.

### 8.4 Merging changes

Contributors work on `feature/*` or `fix/*` and open pull requests against `develop`. Nobody pushes directly to `main`.

```bash
# Create a branch from develop
git checkout develop
git checkout -b feature/my-feature
# — or for a fix —
git checkout -b fix/analysis-error

# Develop and make descriptive commits
git add adapters/my_tool/
git commit -m "Add: adapter for <tool> with sharding support"

# Open a pull request against develop
gh pr create --title "Add: <description>" --base develop
```

**Minimum criteria for merging into `develop`:**
- `chemlink doctor` passes with no errors.
- The affected pipeline completes a test at the smallest scale (10 ligands / 0.1 ns).
- No hardcoded absolute paths outside the environment variables.
- New SLURM scripts follow the convention of parametrizing through environment variables (no fixed partition, node or path values in the script).

---

### 8.5 Publishing releases

Releases are managed with **GitHub's release system** on semantic tags. There are no permanent `release/*` branches: they are created to prepare the version and deleted after the merge.

#### Step 1 — Create the release branch from `develop`

```bash
git checkout develop
git checkout -b release/v1.2.0
```

Only the following happens on this branch:
- Final bug fixes (no new features).
- Changelog update.
- Version number change (if applicable).
- Final validation tests.

#### Step 2 — Merge into `main` and tag

```bash
git checkout main
git merge release/v1.2.0

git tag v1.2.0
git push origin main
git push origin v1.2.0
```

#### Step 3 — Sync `develop`

```bash
git checkout develop
git merge release/v1.2.0
git push origin develop
```

The `release/v1.2.0` branch can be deleted after the merge.

#### Step 4 — Publish on GitHub

In the repository: **Releases → Draft a new release**, select the `v1.2.0` tag and fill in:
- Release notes and changelog.
- Compiled files or attached artifacts (if applicable).
- Mark it as *latest release*.

#### Branch diagram

```
main          ──── v1.0.0 ──────────────── v1.1.0 ──────────────── v1.2.0
                      ↑                       ↑                       ↑
develop       ────────┴── feat/A ── fix/B ───┴── feat/C ── fix/D ───┴──
                                                              ↑
release/vX.X.X                                      release/v1.2.0 (temporary)
```

---

## 9. Dependencies and external services

### 9.1 Integrated external scientific tools

| Tool | Tested version | Role in ChemLink |
|---|---|---|
| **GROMACS** (`gmx` / `gmx_mpi`) | 2025.4 | Molecular dynamics engine; called by every step of the dynamics pipeline |
| **AutoDock-GPU** | HEAD (built with CUDA) | GPU conformational docking engine |
| **AutoGrid4** | HEAD (built from source) | Energy affinity map generator |
| **fpocket** | 4.0 | Detection of binding cavities in receptors |
| **Meeko** | via Conda `bio` (Python 3) | Preparation of receptor and ligand structures in PDBQT format, and GPF generation |
| **ACPYPE** | via Conda `bio` | Generation of GROMACS topologies for small ligands |
| **LightDock** | 0.9.4 (pip, `bio`) | Peptide-protein docking (GSO with DFIRE scoring); needs Python < 3.12 |
| **PRODIGY** | 2.4 (pip `prodigy-prot`, `bio`) | Contact-based binding-affinity cross-check of peptide poses |
| **HADDOCK3** | pip, `haddock3` env (optional) | Refinement and convergence of peptide poses (bundles CNS) |
| **PyRosetta** | pip installer, `pyrosetta` env (optional) | FlexPepDock refinement and interface ΔG of peptide poses |

### 9.2 Access requirements for new teams

For a new team to work on the project, it needs:

1. **Passwordless SSH access** to every cluster node (RSA keys in each node's `~/.ssh/authorized_keys`).
2. **An NFS mount** of the `/nfs/chemlink` volume with read/write permissions.
3. **An account on the SLURM cluster** with access to the `debug` partition (or the relevant one) and at least one assigned GPU.
4. **Docker and `nvidia-container-toolkit`** installed if container mode will be used.
5. **The ChemLink Lmod module** registered in `MODULEPATH` (`/nfs/chemlink/modules`).

### 9.3 Development and testing notes

- AutoDock-GPU **requires a physical GPU**; GROMACS and docking with `--engine cpu` (AutoDock Vina) work without a GPU, although much more slowly. For tests of pure orchestration logic (paths, argument parsing, SLURM script generation) execution can be mocked by redirecting the `subprocess.run` calls to a stub.
- The whole interface (TUI, bars, logs, error panels) can be tested on any machine with `chemlink demo` and `tests/test_ui.py`.
- Building AutoDock-GPU with support for `sm_120` (Blackwell) requires CUDA 13.0+; earlier toolkit versions do not include the compilation profile for that architecture.
- The whole system runs entirely on the `bio` environment (Python 3.10); there is no longer a second, legacy Python environment for structure preparation.

---

## 10. Project conventions

### 10.1 Code conventions

| Aspect | Convention |
|---|---|
| Python style | PEP 8; function and variable names in `snake_case`; classes in `PascalCase` |
| File names | `snake_case`; one module per responsibility; pipeline steps are named after the stage (`receptor_preparation.py`) |
| Imports | Stdlib first, then third-party, then internal; separated by a blank line |
| Logging | `setup_logger` (`utils/logger.py`) or `get_step_logger` (dynamics); long diagnostics with `extra={"file_only": True}`. Never use `print()` in pipeline logic; for a one-off line next to a bar, use `get_console().print(log_line(...))` |
| Progress | Only through `utils/progress.py`; never create a custom `rich.live.Live` or `rich.progress.Progress` (Rich supports a single live display per console) |
| Styles | Use the `cl.*` style names from `utils/theme.py` (`cl.ok`, `cl.warn`, `cl.err`, `cl.ref`, `cl.muted`…) instead of literal colours |
| Subprocess | Always through `utils/retry.py` or the corresponding adapter; never call `subprocess.run()` directly from a pipeline |
| Paths | Use `pathlib.Path` or build them from environment variables; do not hardcode absolute paths outside `common.sh` |
| Comments | Only where the "why" is not obvious; do not comment the "what" if the names are descriptive |
| Language | Code, comments, messages, generated reports and documentation are written in English |

### 10.2 Repository conventions

| Aspect | Convention |
|---|---|
| Permanent branches | `main` (stable production, only receives merges from `release/*`), `develop` (continuous integration) |
| Work branches | `feature/<name>` for new features; `fix/<description>` for fixes |
| Release branches | `release/vX.X.X` (temporary, deleted after the merge into `main`) |
| Commits | Descriptive prefix: `Add:`, `Fix:`, `Refactor:`, `Update:`, `Remove:`; message in English; one idea per commit |
| Pull requests | Always against `develop`; description with the context of the change and how to test it |
| Releases | Semantic `vX.X.X` tags on `main`; published as a GitHub Release (see §8.5) |
| Ignored files | `data/output/`, `__pycache__/`, `*.egg-info/`, `.pyc` files |

### 10.3 Documentation conventions

- `README.md`: quick install and essential commands; must be kept up to date when new subcommands are added.
- `docs/Development.md` (this document): development guide for future teams; update it when adding new tools, changing conventions or finding new common problems.
- `docs/Report.md`: full academic document of the project with benchmarks and analysis; do not change it as part of the ordinary development workflow.
- `docs/Installation.md`: guide to installing and configuring the environment from scratch.
- Logo: the README and the website use the variants in `docs/images/brand/`; they are not edited by hand but regenerated with `scripts/gen_brand_assets.py` (see §6.3).

---

## 11. Common problems and recommendations

### 11.1 Common problems

| Problem | Symptom | Fix |
|---|---|---|
| **GPU not detected under SLURM** | `No CUDA-capable device is detected` in the job log | Check that the node is configured with GPUs in `slurm.conf` (`Gres=gpu:1`) and that `slurmd` can access the `/dev/nvidia*` devices |
| **Slow GROMACS on Blackwell GPUs (RTX 5000)** | Simulations 3–5× slower than expected; `PTX JIT compilation` warning in the log | Add `nstlist = 40` to the production MDP file; build GROMACS with a `GMX_CUDA_TARGET_SM` that explicitly includes `120`. See the project notes: `project_dynamics_gpu.md` |
| **`pdb_poses: "0\n0"` in `stats.json`** | Invalid JSON in the benchmark results | Caused by `set -euo pipefail` + `find \| wc -l` when the directory does not exist; always check the directory exists before the pipe or use the guard `[[ -d "$dir" ]] && ...` |
| **Multi-node run directory collision** | Three runs share the same `run_<timestamp>` | Happens when several processes call the ID generation function in the same second; add a `sleep 1` between consecutive launches or include a UUID in the ID |
| **Matplotlib: `Image size ... too large`** | Error when generating report figures with zero-value bars | Text annotations with a fixed offset fall outside the axis range when all values are ≈0; use a relative `label_offset = ymax * 0.04` and `ax.set_ylim(0, ymax * 1.25)` |
| **`pprotein` exceeds the SLURM limit** | Job cancelled before production finishes | The protein+protein type needs ≥8 h of wall time; raise `--time-limit 08:00:00` in the dynamics SLURM script |
| **`Last SLURM job: none`** in the multi-node benchmark | The analysis job ID is not captured | The output of `chemlink hpc docking` uses the `analysis : <id>` format, not `job <id>`; use `grep -oP 'analysis\s*:\s*\K[0-9]+'` |
| **Module `chemlink/1.0` not found** | `module load chemlink/1.0` fails | Check that `MODULEPATH` includes `/nfs/chemlink/modules` and that the Lmod module file exists at that path |
| **AutoDock-GPU fails on sm_120** | `illegal instruction` or a build error | Build with `TARGETS="120"`; with a recent libstdc++, change `-std=c++11` to `-std=c++14` in `Makefile.Cuda` |
| **`externally-managed-environment` when installing with pip** | `pip install` refuses to install into the system Python (recent Ubuntu/Debian, PEP 668) | Use a virtual environment (`python3 -m venv .venv`) or the system packages (`apt install python3-rich python3-prompt-toolkit`) |
| **Escape codes in SLURM logs** | `.out` files with `\x1b[...` sequences | Outside a terminal, progress switches to text mode on its own; if it still happens, export `CHEMLINK_PROGRESS=plain` or pass `--plain` |
| **`LiveError: Only one live display may be active at once`** | Failure when nesting bars in new code | The code creates its own `Live`/`Progress`; use the helpers in `utils/progress.py`, which share a single region |
| **The TUI does not respond on Windows** | The launcher does not read the keyboard | The TUI needs a Unix terminal; use WSL. The non-interactive commands do work |

### 11.2 Known technical debt

- **No continuous integration.** There is a `unittest` suite (`tests/`, more than 200 tests), but it does not run automatically on each pull request. The ligand loading tests need the `obabel` executable (3.2.1 or later) on `PATH`, and `tests/dynamics/test_reanalysis.py` imports a function that no longer exists (`_resolve_file`).
- **Run ID collisions in simultaneous runs.** The run ID is generated with a seconds timestamp; if two processes generate it in the same second they get the same ID. Adding a short UUID suffix (already present in `data/output/` but not in the SLURM scripts) solves the problem.
- **`mmpbsa_analysis.py` is experimental.** The module exists but is not integrated into the main pipeline nor validated on the six simulation types. It must not be used in production without further validation.
- **Container SLURM scripts not fully in sync with the native ones.** Recent additions to the native scripts (MPI parameters, `--max-gpu-concurrency`) have not been replicated in the container variant.
- **The legacy docking statistics CSV keeps its original Spanish header** (`Molécula`, `StanDesv`) for compatibility with the reference script it mirrors.

### 11.3 Recommendations for continuity

1. **Run the suite in CI** (GitHub Actions is enough for the tests without a GPU) and extend the adapter tests with subprocess mocks.
2. **Update the Lmod module** if the directory structure or the Conda environments change, since it defines the `PATH` and `PYTHONPATH` variables the whole system uses.
3. **Do not change the Conda environments on the nodes directly:** any dependency change must be reflected in the `Dockerfile` (container mode) and in the NFS installation procedure (native mode), so that every node is identical.
4. **Before adding support for a new GPU**, check that the `Dockerfile` includes its sm architecture in `GMX_CUDA_TARGET_SM` and that AutoDock-GPU was built with the corresponding fallback PTX profile.

---

## 12. History of relevant technical decisions

| Decision | Rejected alternative | Reason |
|---|---|---|
| **Python 3 as the orchestrator language** | Plain Bash | Bash allows neither per-stage error handling, retries with exponential logic, nor complex data structures for the dynamics configuration |
| **Layered architecture with adapters** | Direct `subprocess` calls from the pipelines | Isolating the external binaries in adapters makes it possible to replace tools without changing the pipelines; proven during the GROMACS version change |
| ~~Two separate Conda environments (`bio` and `mgl_legacy`)~~ *(superseded, see next row)* | A single Python environment | MGLTools required Python 2 and dependencies incompatible with the Python 3 ecosystem of GROMACS/ACPYPE; the separation avoided unsolvable conflicts |
| **Migration from MGLTools/AutoDockTools to Meeko** | Keep MGLTools with its dedicated Python 2 environment | MGLTools is not actively maintained; Meeko (from the same Forli/CCSB lab as AutoDock) is pure Python 3, builds on RDKit (already used by the project) and allows going back to a single `bio` Conda environment |
| **NFS as shared storage (not object storage)** | S3 / object storage | The scientific tools (GROMACS, AutoDock-GPU) need POSIX semantics; S3 does not offer them natively without a compatibility layer |
| **SLURM job arrays for multi-node docking** | Launching one job per ligand individually | Job arrays reduce scheduling latency and the load on the SLURM controller when there are thousands of ligands |
| **`rich` for the CLI** | `click` + plain output | `rich` provides panels, tables and progress with colour, without extra dependencies and with a better experience on the cluster terminals |
| **A single progress region (`Display`) for the whole application** | One `Progress`/`Live` per bar | Rich supports only one live display per console: with Rich < 14.1, nesting the pipeline bar and a step bar failed with `LiveError`. A region that stacks `Tracker`s allows free nesting and leaves a permanent ✓/✗ line per stage |
| **Plain-text progress outside a terminal** | Animated bars in files too | In SLURM `.out` files the bars turn into escape codes or only appear at the end; timestamped lines every 10% are readable and `grep`-able |
| **TUI with `rich` + `prompt_toolkit`** | Textual / curses | Reuses the CLI's theme and console, adds no framework and generates menus and forms from `argparse`, without a second command definition |
| **TUI logo as generated block characters** | Images in the terminal (sixel/kitty) | Works in any terminal and over SSH; the wordmark uses the terminal's text colour, readable on light or dark backgrounds |
| **A single all-in-one container** | Separate containers per tool | ChemLink is not a microservice system; building GROMACS and AutoDock-GPU with the exact CUDA dependencies in a single container simplifies deployment |
| **Building AutoDock-GPU from source** | Using the official prebuilt binary | The official binary has no sm_120 (Blackwell) support; building it ourselves lets us specify exactly the cluster's GPU architecture targets |
| **`nstlist = 40` for Blackwell GPUs** | Using the default value | The neighbour-list update kernel uses JIT compilation on sm_120, which adds a penalty on every update; widening the interval reduces how often it recompiles at the cost of a slight increase in pair computation |

---

## 13. Related references

| Resource | Description |
|---|---|
| `README.md` | Quick install and main command reference |
| `docs/Report.md` | Full academic document of the project with benchmarks and analysis |
| `docs/Installation.md` | Guide to installing and configuring the environment from scratch |
| `docs/Entropy.md` | IE/C2 entropy correction of the binding energy |
| `docs/FEP.md` | Binding free energy workflow (FEP/ABFE) |
| [GROMACS 2025 — User guide](https://manual.gromacs.org/documentation/current/index.html) | Official reference for MDP parameters, `mdrun` flags and GPU configuration |
| [AutoDock-GPU — GitHub](https://github.com/ccsb-scripps/AutoDock-GPU) | Official repository with build and usage options |
| [fpocket — GitHub](https://github.com/Discngine/fpocket) | Official fpocket repository with parameter documentation |
| [SchedMD SLURM — Job Arrays](https://slurm.schedmd.com/job_array.html) | Official SLURM job array documentation |
| [NVIDIA CUDA Toolkit — Architectures](https://developer.nvidia.com/cuda-gpus) | Table mapping GPUs to sm architecture codes |
| [Prometheus — Documentation](https://prometheus.io/docs/introduction/overview/) | Reference for the monitoring system used on the cluster |
| [Grafana — Documentation](https://grafana.com/docs/grafana/latest/) | Reference for the cluster's visualization platform |
