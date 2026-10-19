# Installation and Deployment — ChemLink

---

## 1. Solution overview

ChemLink is a scientific orchestration platform for **molecular docking** and **molecular dynamics** workflows in HPC environments. It sits between the researcher and the computational chemistry tools (AutoDock-GPU, GROMACS, fpocket, Meeko), exposing a unified CLI that hides the complexity of cluster management, GPU allocation and stage coordination.

The platform can run in three configurations:

| Mode | Description |
|---|---|
| **Local single node** | CLI run directly on a GPU workstation, without SLURM |
| **Native HPC** | Automatic generation of SLURM scripts on shared NFS storage |
| **Container** | Self-contained Docker/Apptainer image, runnable from SLURM or locally |

---

### 1.1 Languages and technologies

| Layer | Technology | Version |
|---|---|---|
| Main language | Python | 3.10 |
| CLI, TUI and output | Rich, prompt_toolkit, argparse | ≥ 13.0 / ≥ 3.0 |
| Computational chemistry | RDKit, PDBFixer, OpenMM | conda-forge |
| Protonation of PDB/PDBQT ligands | Open Babel (`obabel` command, run as a separate program) | conda-forge |
| Structure preparation | Meeko | conda-forge |
| Topology preparation | ACPYPE / AmberTools | conda-forge |
| Active-site detection | fpocket | 4.x |
| Affinity maps | AutoGrid4 | 4.2 |
| Molecular docking | AutoDock-GPU | HEAD (CUDA) |
| Molecular docking (CPU) | AutoDock Vina | 1.2.5 |
| Peptide-protein docking | LightDock + PRODIGY (run as separate programs) | PyPI |
| Molecular dynamics | GROMACS | 2025.4 |
| GPU acceleration | NVIDIA CUDA | 12.x |
| MPI parallelism | OpenMPI | ≥ 4.1 |
| Environment management | Miniforge / Conda (conda-forge and bioconda only) | 26.7.2-0 |
| Job scheduler | SLURM | ≥ 23.x |
| Environment modules | Lmod | ≥ 8.x |
| Network storage | NFS v4 / OpenMediaVault | — |
| Containers | Docker / Apptainer (Singularity) | ≥ 24 / ≥ 1.3 |
| Monitoring | Prometheus + Grafana | — |

---

### 1.2 Solution components

```
chemlink/
├── cli/                  # Unified entry point (chemlink docking / dynamic / hpc / doctor / demo) and TUI (cli/tui/)
├── pipelines/
│   ├── docking/          # Orchestrator for the full docking pipeline
│   └── dynamics/         # Orchestrator for the full molecular dynamics pipeline
├── adapters/             # Wrappers around external binaries (fpocket, AutoGrid4, AutoDock-GPU, GROMACS)
├── hpc/
│   ├── cluster/          # Resource detection (CPU, GPU, network, SLURM)
│   └── slurm/
│       ├── native/       # SLURM scripts for native execution on NFS
│       └── container/    # SLURM scripts for execution with Docker/Apptainer
├── storage/              # Path, run directory and NAS management
├── utils/                # Theme, logs, progress, exponential retry, molecular processors
├── data/
│   ├── input/            # Input receptors and ligands
│   └── output/           # Results per run (run_<timestamp>_<id>/)
├── Dockerfile            # Self-contained image with CUDA + GROMACS + AutoDock-GPU
├── docker-compose.yml    # Compose file with GPU support for local development
├── chemlink              # Launcher (uses the checkout it lives in, or the NFS one)
└── install.sh            # curl-pipe-bash installer
```

---

## 2. Prerequisites

### 2.1 Required software

#### Operating system

- **Ubuntu 24.04 LTS** (recommended; tested in production)
- Ubuntu 22.04 LTS (works, with warnings)
- Arch Linux (works with a native install; CUDA is detected in `/opt/cuda`)
- Architecture: `x86_64`

#### GPU and drivers

> ChemLink can run on CPU, but AutoDock-GPU and GROMACS offloading require an NVIDIA GPU.

| Requirement | Minimum | Recommended |
|---|---|---|
| NVIDIA driver | 535.x | 570.x |
| CUDA Toolkit | 12.0 | 12.4 |
| VRAM per node | 8 GB (RTX 3060) | 16 GB (RTX 4080 / A4000) |
| CUDA architecture | sm_86 (Ampere) | sm_89/sm_90 (Ada / Hopper) |

Check the installed driver:

```bash
nvidia-smi
nvcc --version
```

#### System dependencies (without Docker)

```bash
sudo apt-get update && sudo apt-get install -y \
  git curl wget build-essential cmake autoconf automake libtool \
  openmpi-bin libopenmpi-dev libfftw3-dev libfftw3-mpi-dev \
  libeigen3-dev libxml2-dev libx11-dev libxt-dev libqhull-dev \
  libnetcdf-dev zlib1g-dev python3 python3-pip python2
```

On Arch Linux:

```bash
sudo pacman -S --needed \
  base-devel git curl wget cmake autoconf automake libtool \
  openmpi fftw eigen libxml2 libx11 libxt qhull netcdf zlib \
  python python-pip cuda
```

#### Docker (for container installs)

- Docker Engine ≥ 24.0 with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2 (`docker compose version`)

Install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check GPU access from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi
```

#### SLURM (for cluster deployment)

- `slurmctld` running on the manager node
- `slurmd` running on every worker node
- Nodes declared in `/etc/slurm/slurm.conf`
- Lmod installed in `/usr/share/lmod/` for environment modules

---

### 2.2 Environment variables

The following variables control pipeline behaviour. They can be exported before running `chemlink` or set in the SLURM scripts.

#### Main paths

`chemlink hpc docking` sets these variables from the directory it runs in;
the values in the table only apply when the SLURM scripts are launched by hand.

| Variable | Default | Description |
|---|---|---|
| `REPO_DIR` | `/nfs/chemlink/chemlink` | Source code root (`hpc docking`: the checkout running it) |
| `RUN_DIR` | `/nfs/chemlink/runs/<RUN_ID>` | Directory of the active run (`hpc docking`: `./results/docking/<RUN_ID>`) |
| `INPUT_RECEPTORS_DIR` | `$REPO_DIR/data/input/receptors` | Input receptor PDB files (`hpc docking`: `-ir` or automatic detection) |
| `INPUT_LIGANDS_DIR` | `$REPO_DIR/data/input/ligands` | Ligand Mol2/SDF/PDBQT files (`hpc docking`: `-il` or automatic detection) |
| `OUTPUT_DIR` | `$RUN_DIR/output` | Generated artifacts |
| `PYTHONPATH` | `/nfs/chemlink:$PYTHONPATH` | To import the package |

#### External tools

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `/nfs/chemlink/miniconda/envs/bio/bin/python` | `bio` environment interpreter |
| `FPOCKET_PATH` | `/usr/local/bin/fpocket` | fpocket executable (the scripts export it as `FPOCKET_BIN`, which is what ChemLink reads) |
| `AUTOGRID_EXECUTABLE` | `/usr/local/bin/autogrid4` | AutoGrid4 executable (exported as `AUTOGRID4_BIN`) |
| `AUTODOCK_GPU_EXECUTABLE` | `/usr/local/bin/autodock-gpu` | AutoDock-GPU executable (exported as `AUTODOCK_GPU_BIN`) |
| `GMXRC` | `/usr/local/gromacs/bin/GMXRC` | GROMACS initialization script |

#### GPU and parallelism

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Accessible GPUs (e.g. `0,1` to limit them) |
| `CUDA_VISIBLE_DEVICES` | — | Per-task override in a SLURM array |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Driver capabilities inside containers |
| `PYTHONUNBUFFERED` | `1` | Real-time logs in containers |

#### Terminal interface

| Variable | Default | Description |
|---|---|---|
| `CHEMLINK_PROGRESS` | *(automatic)* | `plain`: timestamped progress lines (used without a terminal, e.g. under SLURM); `rich`: animated bars |
| `CHEMLINK_LOG_LEVEL` | `INFO` | Console log threshold; log files always keep everything |
| `NO_COLOR` | — | Turns colour off |
| `CHEMLINK_DEBUG` | — | Shows the full traceback of errors |
| `CHEMLINK_HOME` | *(launcher's checkout)* | Repository used by `./chemlink` |
| `CHEMLINK_PYTHON` | *(`bio` environment or `python3`)* | Interpreter used by `./chemlink` |

#### SLURM (distributed pipelines)

| Variable | Default | Description |
|---|---|---|
| `SLURM_PARTITION` | *(default)* | Cluster partition |
| `SLURM_NODELIST` | — | Node list (e.g. `worker01,worker02`) |
| `SLURM_ACCOUNT` | — | SLURM project account |
| `SLURM_QOS` | — | Quality of service |
| `MAX_GPU_CONCURRENCY` | `6` | Concurrent SLURM tasks using a GPU |
| `BATCH_SIZE` | `200` | Ligands per batch in the job array |
| `DYN_TIME_LIMIT` | `72:00:00` | Wall time for SLURM dynamics |
| `DYN_TYPE` | `pligand` | Dynamics simulation type |
| `DYN_NS_TIME` | `100` | Production nanoseconds |
| `DYN_CHARGE` | *(empty)* | Optional override of the ligand net charge; it is normally resolved per simulation |

---

## 3. Development installation

### 3.1 Development without containers

#### 3.1.1 Clone the repository

```bash
git clone https://github.com/PipeJF9/chemlink.git
cd chemlink
```

Or with the automatic installer:

```bash
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash
```

#### 3.1.2 Install dependencies

**Miniforge** (if no conda is installed). Miniforge is used instead of Miniconda
because Anaconda's terms of service restrict commercial use of its channels;
ChemLink only needs `conda-forge` and `bioconda`. It is installed in
`/opt/miniconda` because that is where ChemLink and its scripts look for the environments:

```bash
wget https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh -O /tmp/miniforge.sh
echo "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05  /tmp/miniforge.sh" | sha256sum -c
bash /tmp/miniforge.sh -b -p /opt/miniconda
export PATH=/opt/miniconda/bin:$PATH
```

**Main `bio` environment** (Python 3.10 + scientific stack):

```bash
conda create -n bio -y --override-channels -c conda-forge \
  python=3.10 pip \
  ambertools acpype "openbabel>=3.2.1" pdbfixer openmm rdkit meeko scipy gemmi \
  matplotlib tqdm pandas numpy rich prompt-toolkit biopython
conda activate bio
pip install "lightdock>=0.9" "prodigy-prot>=2.4"   # peptide docking
```

LightDock 0.9 does not run on Python 3.12 or later (it uses a `ConfigParser`
method removed in 3.12), so keep the `bio` environment on Python 3.10 or 3.11.

**Scientific tools** (fpocket, AutoGrid4, AutoDock4):

```bash
# fpocket
git clone --depth 1 --branch 4.2.3 \
  https://github.com/Discngine/fpocket.git /tmp/fpocket
cd /tmp/fpocket
sed -i 's/-Werror//g' makefile
# Compatibility with GCC 14+ (including Arch Linux's GCC 16)
sed -i \
  -e 's/char \*residue_string\[/char residue_string[/' \
  -e 's/strcpy(&residue_string, pt);/strcpy(residue_string, pt);/' \
  src/fparams.c
make -j$(nproc) && sudo find bin -type f -executable -exec cp {} /usr/local/bin/ \;

# AutoGrid4 + AutoDock4
git clone https://github.com/ccsb-scripps/AutoGrid.git /tmp/autogrid
cd /tmp/autogrid && autoreconf -i && ./configure --prefix=/usr/local
make -j$(nproc) && sudo make install
sudo mv /usr/local/bin/autogrid /usr/local/bin/autogrid4 2>/dev/null || true

git clone --depth 1 https://github.com/ccsb-scripps/AutoDock4.git /tmp/autodock4
cd /tmp/autodock4 && autoreconf -i && ./configure --prefix=/usr/local
make -j$(nproc) && sudo make install
```

**AutoDock-GPU** (requires CUDA):

> The target architectures cover the lab's GPUs:
> `sm_86` = RTX 3050/3060/3080 (Ampere) | `sm_89` = RTX 4070/4090 (Ada Lovelace) | `sm_90` = H100 (Hopper) | `sm_120` = RTX 5000 (Blackwell)

```bash
git clone https://github.com/ccsb-scripps/AutoDock-GPU.git /tmp/autodock-gpu
cd /tmp/autodock-gpu
sed -i 's/-std=c++11/-std=c++14/g' Makefile.Cuda
make DEVICE=CUDA NUMWI=128 TARGETS="86 89 90 120" \
  CUDA_PATH=/usr/local/cuda \
  GPU_INCLUDE_PATH=/usr/local/cuda/include \
  GPU_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -j$(nproc)
sudo cp bin/autodock_gpu_128wi /usr/local/bin/autodock-gpu
sudo chmod +x /usr/local/bin/autodock-gpu
```

**GROMACS 2025.4** (requires CUDA + OpenMPI, ~30–60 min):

```bash
wget https://ftp.gromacs.org/gromacs/gromacs-2025.4.tar.gz -O /tmp/gromacs.tar.gz
tar -xzf /tmp/gromacs.tar.gz -C /tmp && cd /tmp/gromacs-2025.4 && mkdir build && cd build
cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DGMX_MPI=ON \
  -DGMX_SIMD=AVX2_256 \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGMX_CUDA_TARGET_SM="75;80;86;89;90;100;120" \
  -DGMX_CUDA_TARGET_COMPUTE="90"
make -j$(nproc) && sudo make install
echo "source /usr/local/gromacs/bin/GMXRC" >> ~/.bashrc
```

On Arch Linux, CUDA 12.x uses the versioned GCC 14 package. Add to the
configuration above:

```bash
-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
```

#### 3.1.3 Configure environment variables

Create a `~/.chemlink_env` file (or add to `.bashrc`):

```bash
# ChemLink environment
export REPO_DIR=/opt/chemlink
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export FPOCKET_PATH=/usr/local/bin/fpocket
export AUTOGRID_EXECUTABLE=/usr/local/bin/autogrid4
export AUTODOCK_GPU_EXECUTABLE=/usr/local/bin/autodock-gpu
source /usr/local/gromacs/bin/GMXRC
```

#### 3.1.4 Start the required services

For multi-node development, check that NFS is mounted on every node:

```bash
# On the manager (OpenMediaVault already configured)
showmount -e localhost

# On each worker
sudo mount -t nfs4 <manager-ip>:/nfs/chemlink /nfs/chemlink
# Add to /etc/fstab to make it persistent:
# <manager-ip>:/nfs/chemlink  /nfs/chemlink  nfs4  defaults  0  0
```

Check SLURM (multi-node development):

```bash
sinfo          # partition status
squeue         # queued jobs
```

#### 3.1.5 Start the application

```bash
conda activate bio
export PYTHONPATH=/opt/chemlink:${PYTHONPATH:-}

# Interactive launcher (TUI): menus, forms and Ctrl+R to run
chemlink

# Environment diagnostics
chemlink doctor

# Docking pipeline (single node) from the project folder: reads ./receptors
# and ./ligands (or loose *.pdb and *.sdf files) and writes to ./results/docking.
# --engine cpu uses AutoDock Vina without a GPU
cd my_project
chemlink docking full --engine gpu

# Protein-ligand molecular dynamics pipeline (protein and ligand detected
# in the directory; -ir/-il pick them explicitly)
chemlink dynamic pligand --time 100
```

No path option is required: inputs are looked up in `./receptors`,
`./ligands` and `./peptides` and, if those don't exist, among the files in the current
directory. Results are written to `./results/<pipeline>/` (`-o DIR` changes the
base). For dynamics, each structure is classified by its content (protein,
peptide, nucleic acid or ligand); if detection is ambiguous, the command says so
and asks for `-ir`/`-il`. `dynamic resume`, `dynamic analyze` and `docking analyze`
use the current run (if run inside it) or the most recent one.

Global options, valid anywhere on the command line: `--plain` (progress as text
lines), `--no-color`, `-v`/`--verbose`, `-q`/`--quiet` and `--debug`. Each command
shows a header, one `✓`/`✗` line per stage and a final summary; errors
appear in a panel with the cause and the suggested fix.

#### Trying the interface without a GPU

The interface can be checked on any machine, without a GPU or scientific
tools. Recent Ubuntu/Debian releases don't let `pip` install into the system Python
(`externally-managed-environment` error), so a virtual environment is used:

```bash
git clone https://github.com/PipeJF9/chemlink.git && cd chemlink   # the directory must be named chemlink
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit

./chemlink demo                    # simulated docking + dynamics run
./chemlink demo dynamics --fail    # includes a failure to show the error report
./chemlink demo --plain            # what it looks like in a SLURM log
./chemlink                         # TUI
```

#### Peptide-protein docking

`chemlink docking peptide` docks a batch of peptides (one SDF file each) against
a single receptor with LightDock and ranks them by their best DFIRE score:

```bash
cd my_peptide_project      # peptides/*.sdf, receptors/receptor.pdb, restraints.txt
chemlink docking peptide full
```

`docking peptide robust` adds cross-validation against a known reference peptide
from the same batch (for example an approved drug):

1. `lightdock`: preparation + LightDock docking (triage), then the **gate**: only
   candidates that beat the reference on both DFIRE score and distance to the
   restrained site go on.
2. `audit`: flags candidates whose non-canonical or modified residues had to be
   standardized to ALA/GLY for docking.
3. `prodigy`: independent binding-affinity estimate (PRODIGY) on each best pose.
4. `combine`: merges every signal into `combined_ranking.csv` (robust z-score and
   mean rank), with the reference as one more row.

```bash
chemlink docking peptide robust all --reference semaglutide
# or stage by stage, each one reading the run created by 'lightdock':
chemlink docking peptide robust lightdock --reference semaglutide --run-name glp1
chemlink docking peptide robust prodigy results/peptide-docking/glp1
```

Input conventions:

- The receptor PDB must be a single, clean chain labelled **`R`**: peptides are
  written as chain `A`, and the site distance and PRODIGY use those two chain IDs.
- The restraints file lists one binding-site residue per line as
  `R <chain>.<resname>.<resnum>`, e.g. `R R.LYS.197`.
- `--standardize` docks peptides with non-canonical residues by mutating those
  positions to ALA/GLY (the original molecule is kept under `original_peptides_pdb/`
  for dynamics); without it, such peptides are excluded and reported.
- With restraints, LightDock places 20 swarms per restrained residue and ignores
  `--swarms`.

#### Automatic ligand validation

Before preparing a ligand for docking or dynamics, ChemLink validates its
chemical integrity with RDKit. The check requires a readable structure,
valid valences and bond orders, and a single connected fragment. For
structures with 3D coordinates it also checks bond lengths against
covalent radii, angles consistent with hybridization, and severe clashes
between non-bonded atoms. Small rings get specific tolerances so that valid
strained molecules are not rejected.

2D files are validated topologically and their geometry is generated and optimized
during preparation. PDBQT is validated without inferring angles, because that format
does not keep reliable bond orders. A disconnected or broken ligand, or one with an
impossible 3D geometry, is rejected before docking, ACPYPE or dynamics start.
For `oligand`, `pligand` and `ppligand` the diagnosis is saved as
`ligand_validation.json` inside the run directory.

#### Ligand-only dynamics (`oligand`)

`chemlink dynamic oligand` simulates a single ligand in explicit water, with no
protein. It parametrizes with GAFF2/ACPYPE (topology in `topol.top` and `ligand.itp`),
solvates, neutralizes and runs minimization, NVT, NPT and production. The analysis
includes RMSD, radius of gyration, SASA, per-atom RMSF and ligand–water contacts
(< 0.35 nm); no binding energy is computed. The ligand is detected in the
directory or given with `-il`; it also accepts `-c`, `--charge-mode` and `--ligand-ph`.

```bash
chemlink dynamic oligand -il Ligand.sdf -t 10 --run-name LIG-WATER
```

#### Automatic ligand charge resolution

For `oligand`, `pligand` and `ppligand`, ChemLink reads the formal charge from the
V2000/V3000 records of SDF or MOL files. For MOL2 and PDBQT it sums the partial charges and
only accepts the nearest integer when the difference is 0.15 or less.
An SDF/MOL without an explicit formal charge is treated as undetermined during
detection. In `auto` mode, if the other methods cannot resolve the charge either,
ChemLink uses a low-confidence neutral estimate (`0`) and shows a
warning. PDB files that include the optional formal-charge column are also
recognised directly.

PDB/PDBQT ligands can be protonated with Open Babel at the pH given by
`--ligand-ph` (7.4 by default), and the prepared SDF is also used as the
ACPYPE input. The resolution and where it came from are recorded in
`ligand_charge.json` inside the simulation directory.

- `--charge-mode auto` (default, also for HPC): uses the best available charge; if it cannot resolve it, it warns, records the uncertainty and continues with a low-confidence neutral estimate.
- `--charge-mode prompt`: shows the result and asks for confirmation or a manual charge.
- `--charge-mode strict`: only accepts a high-confidence formal charge or an explicit value.
- `-c/--charge Q`: sets the charge manually and takes priority over any detection.

For non-interactive runs with a known charge:

```bash
chemlink dynamic pligand -ir protein.pdb -il ligand.mol2 -c -1 --time 100
```

#### Force field selection for protein-only and protein complexes

Protein-only simulations can use any force field available in GROMACS.
If none is given, `oprotein` keeps its historical behaviour and uses `amber03`:

```bash
chemlink dynamic oprotein \
  -ir protein.pdb -t 100 \
  --force-field amber99sb-ildn \
  --water-model tip3p
```

You can also choose, for example, `amber03`, `amber99sb-ildn` or
`charmm36-jul2022`, as long as the corresponding directory is installed.
The name is written without the `.ff` suffix.

`oprotein` can also run several independent replicas from the same minimized
system with `--replicas N` (default 1), just like `pligand`.

The same selection is available for protein–peptide complexes. If no
force field is given, `ppeptide` keeps `amber99sb-ildn`:

```bash
chemlink dynamic ppeptide \
  -ir protein.pdb -il peptide.pdb -t 100 \
  --force-field charmm36-jul2022 \
  --water-model tip3p
```

The `pprotein` module keeps `amber99sb-ildn` as ChemLink's historical
behaviour, but lets you select the ChemFusion protocol, which uses
CHARMM36m with TIP3P water:

```bash
chemlink dynamic pprotein \
  -ir protein1.pdb -il protein2.pdb -t 100 \
  --force-field charmm36-jul2022
```

For CHARMM36m, ChemLink normalizes the partners as chains `A` and `B` and
explicitly selects charged protein termini (`NH3+`/`COO-`, with the
glycine- and proline-specific patches). This prevents `pdb2gmx` from mistaking an
N-terminal methionine for the `MET1` carbohydrate patch. The decisions and the
diagnosis are saved in `topology_preflight.json` inside the run.

The repository does not include CHARMM36m, because its files do not state any
redistribution terms. The installer downloads the official
`charmm36-jul2022.ff.tgz` distribution, verifies its SHA-256 checksum and installs it in
`CHEMLINK_DIR/forcefields/charmm36-jul2022.ff` (the Docker image, in
`/opt/forcefields`). So no copy of ChemFusion and no machine-specific
path are needed. The value of
`--force-field` is the actual GROMACS directory name without the `.ff` suffix.
ChemLink first looks for the package bundled with its installation and then in
`CHEMLINK_FORCEFIELD_DIR`, `GMXLIB` and the standard GROMACS paths.

If the force field is managed externally, its path can be given:

```bash
chemlink dynamic pprotein \
  -ir protein1.pdb -il protein2.pdb \
  --force-field charmm36-jul2022 \
  --force-field-dir /path/to/forcefields \
  --water-model tip3p
```

The directory must contain, for example,
`charmm36-jul2022.ff/forcefield.itp`. ChemLink records the resolved name and directory
in `run_config.json`, so the same force field is kept when
resuming from a checkpoint.

Installing the force field can be skipped explicitly with
`./install.sh --without-charmm36`; it must then be provided through
`--force-field-dir`, `CHEMLINK_FORCEFIELD_DIR` or `GMXLIB`.

#### MM-GBSA/MM-PBSA binding energy

The `pprotein` and `ppeptide` dynamics run the ChemFusion protocol for
macromolecular complexes: ChemLink builds an isolated index with both
partners (`Protein` and `Other` groups), uses `-cg 0 1` without a MOL2 topology and applies
MM-GBSA (`igb=5`, 0.150 M). If the standard index does not contain both groups, the
two main chains are detected automatically from `md.pdb`. PB is skipped for these
systems, following the ChemFusion protocol.

The number of MPI processes for the energy calculation is chosen manually for
the types with a binding energy (`pligand`, `ppeptide`, `pprotein`, `ppligand`)
with `--mmpbsa-cores N`. Each type only accepts the `--mmpbsa-*` options it uses:
`ppeptide`/`pprotein` fix the method to GB and `ppligand` accepts `--mmpbsa-cores`
and `--mmpbsa-entropy`. When `N` is greater than one,
ChemLink runs `mpirun -np N gmx_MMPBSA MPI`; the setting is also honoured
in `hpc dynamics` runs. `gmx_MMPBSA` needs the serial `gmx` executable
for parallel calculations; an installation that only has `gmx_mpi` can run
the analysis with `--mmpbsa-cores 1` until `gmx` is also installed. If
several processes are requested in that situation, ChemLink automatically falls back
to one process and records the reason in `analysis_metadata.json`.

```bash
chemlink dynamic pprotein -ir protein1.pdb -il protein2.pdb \
  --mmpbsa-cores 12
```

`pligand` dynamics run `gmx_MMPBSA` by default after the structural
analysis. ChemLink discards the first 20%, spreads up to 1000 frames over
the remaining trajectory, creates an isolated index with the `Protein` and
`LIG` groups, and reuses the GAFF2 MOL2 generated by ACPYPE.

```bash
chemlink dynamic pligand -ir protein.pdb -il ligand.sdf \
  --mmpbsa-method both --mmpbsa-frames 1000 \
  --mmpbsa-discard 20 --mmpbsa-cores 6
```

Results are saved in
`binding_energy_analysis/gmx_MMPBSA/`. A failure of this analysis is recorded in
`analysis_metadata.json` and does not invalidate the trajectory already completed. To
skip it, use `--no-binding-energy`.

The isolated environment avoids AmberTools conflicts with the main environment:

```bash
./install.sh --with-mmpbsa
```

External executables can also be given through
`CHEMLINK_MMPBSA_BIN` and `CHEMLINK_MMPBSA_MPIRUN`. The path of the GROMACS
serial/thread-MPI executable can be set with `CHEMLINK_GMX_BIN`; for the
MPI variant used by the dynamics, `CHEMLINK_GMX_MPI_BIN` is accepted.
ChemLink also looks for both executables automatically in `PATH`, `GMXBIN`,
the active Conda environment, the Python prefix, the usual GROMACS paths
and the directory where the other variant is installed.

If the installed copy of ChemLink has local changes or its remote requires
authentication, updating it can be skipped when installing GROMACS:

```bash
./install.sh --with-gromacs --skip-update
```

#### Resuming an interrupted dynamics run

ChemLink keeps the configuration of new runs in `run_config.json`
and can continue from the GROMACS checkpoints without repeating the system
preparation:

```bash
# Check which stage would be resumed, without running any calculation
chemlink dynamic resume results/dynamics/run_pligand_YYYYMMDD_HHMMSS --dry-run

# Automatically resume NVT, NPT, production, post-processing or analysis;
# without a path it uses the current run or the latest one in ./results/dynamics
chemlink dynamic resume
```

For older runs without `run_config.json`, the type and duration are inferred
when the files are unambiguous. If that is not possible, they can be given:

```bash
chemlink dynamic resume PATH --type pligand --time 100
```

#### Re-running the analysis of a finished dynamics run

The analyses can be regenerated from the results directory without repeating
preparation or molecular production. `--only` runs the
general structural analysis, the binding energy, or both:

```bash
# RMSD, RMSF, radius of gyration, SASA and the other general analyses
chemlink dynamic analyze results/dynamics/run_pprotein_YYYYMMDD_HHMMSS \
  --only general

# MM-GBSA/MM-PBSA only, choosing the MPI processes manually
chemlink dynamic analyze results/dynamics/run_pprotein_YYYYMMDD_HHMMSS \
  --only binding --mmpbsa-cores 12

# Repeat both analyses for the latest run
chemlink dynamic analyze
```

The command loads `run_config.json` (or `hpc_config.json`) from the given path,
or from the current or latest run if it is omitted.
For legacy runs without a configuration, `--type` and
`--time` can be given. The `--mmpbsa-method`, `--mmpbsa-frames`,
`--mmpbsa-discard`, `--mmpbsa-cores` and `--mmpbsa-salt` parameters override the saved
values only for the re-processing.

Continuing `nvt`, `npt` and `md` uses the matching `.cpt`/`.tpr` pair
with `-append`; GROMACS verifies the checksums before appending data.
Finished stages are detected from the final structure and the
`Finished mdrun` mark in the log. If the trajectory has already finished, ChemLink only
runs the post-processing or analysis that is still pending.

---

### 3.2 Development with containers

#### 3.2.1 Building the containers

The image includes the whole stack: CUDA, Miniforge, GROMACS, AutoDock-GPU, fpocket and Meeko.
It is built in two stages: everything is compiled on `nvidia/cuda:13.0.0-devel` and
the final image starts from `nvidia/cuda:13.0.0-runtime`. The final image keeps the
licenses, the source code of what is compiled and the exact package list in
`/usr/share/chemlink/third-party/`. For an image that will be handed to
third parties, build with `--build-arg WITH_CHARMM36=false` (see
`THIRD_PARTY_LICENSES.md`).

```bash
# Build the image (first time: ~60–90 min because of the compilations)
docker build -t chemlink:latest .

# Tag with a version
docker build -t chemlink:1.0.0 .
```

> **GPU considerations for the image:**
> The base image is `nvidia/cuda:13.0.0-devel-ubuntu24.04`. AutoDock-GPU is built with support for `sm_86`, `sm_89`, `sm_90` and `sm_120`. GROMACS is built with support for `sm_75`–`sm_120`. The host driver must be compatible with the image's CUDA version.

#### 3.2.2 Running the environment

```bash
# Start the service with GPU
docker compose up -d

# Open a shell in the container
docker compose exec chemlink-gpu bash

# Inside the container: check the environment
chemlink doctor
```

Running directly without `docker compose`:

```bash
docker run --rm --gpus all \
  -v "$(pwd)":/app/chemlink \
  -e NVIDIA_VISIBLE_DEVICES=all \
  chemlink:latest \
  chemlink doctor
```

#### 3.2.3 Available services

| Service | Description | Internal network |
|---|---|---|
| `chemlink-gpu` | Main container with CUDA + all tools | `chemlink-network` |

The `./:/app/chemlink` volume mounts the source code live, so code changes show up immediately without rebuilding the image.

#### 3.2.4 Shutting down the environment

```bash
docker compose down
# To also remove the built image:
docker compose down --rmi local
```

---

## 4. Deployment

### 4.1 Deployment architecture

![ChemLink deployment architecture](images/diagrams/architecture_diagram_visualizing.png)

The NFS volume `/nfs/chemlink` contains:
- `/nfs/chemlink/chemlink/` — source code and Lmod module
- `/nfs/chemlink/miniconda/` — shared Conda environments
- `/nfs/chemlink/runs/` — runs and results
- `/nfs/chemlink/software/` — compiled binaries (optional)
- `/nfs/chemlink/modules/` — Lmod modules

---

### 4.2 Update process

```bash
# Update ChemLink (without touching the Conda environments or data)
cd /nfs/chemlink/chemlink
git pull origin main

# Or with the installer (detects an existing install and pulls):
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | bash
```

---

### 4.3 Deployment without containers (native on NFS + SLURM)

#### 4.3.1 Server preparation

On the **manager node**, run once:

```bash
# 1. Install ChemLink with all tools on NFS
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | \
  bash -s -- --full --dir /nfs/chemlink/chemlink

# 2. Check that NFS is exported
cat /etc/exports
# Must include: /nfs/chemlink  10.0.0.0/24(rw,sync,no_subtree_check)

# 3. Check SLURM
sinfo -l
```

On each **worker node**:

```bash
# Mount NFS (if it is not in /etc/fstab)
sudo mount -t nfs4 10.0.0.1:/nfs/chemlink /nfs/chemlink

# Check that slurmd is running
sudo systemctl status slurmd
```

#### 4.3.2 Installing dependencies

When installing with `--full --dir /nfs/chemlink/chemlink`, the script compiles every binary on the manager host. For the workers to use the same binaries:

**Option A** — binaries on NFS (recommended):

```bash
# The compiled binaries are copied to the NFS volume
sudo mkdir -p /nfs/chemlink/software/bin
sudo cp /usr/local/bin/{fpocket,autogrid4,autodock4,autodock-gpu} \
        /nfs/chemlink/software/bin/
# Export on each node:
export PATH=/nfs/chemlink/software/bin:$PATH
```

**Option B** — compile on each worker:

```bash
# Re-run the installer on each worker
curl -fsSL https://raw.githubusercontent.com/PipeJF9/chemlink/main/install.sh | \
  bash -s -- --full --skip-conda
```

#### 4.3.3 Application configuration

Register the Lmod module (run on the manager; takes effect immediately on every node through NFS):

```bash
sudo mkdir -p /nfs/chemlink/modules/chemlink
sudo tee /nfs/chemlink/modules/chemlink/1.0.lua <<'EOF'
help([[ ChemLink v1.0 — Molecular Orchestration Platform ]])
whatis("Name: ChemLink")
whatis("Version: 1.0")

prepend_path("PATH",       "/nfs/chemlink/software/bin:/nfs/chemlink/chemlink")
prepend_path("PYTHONPATH", "/nfs/chemlink")
setenv("CHEMLINK_HOME",    "/nfs/chemlink/chemlink")
setenv("PYTHON_BIN",       "/nfs/chemlink/miniconda/envs/bio/bin/python")
EOF
```

#### 4.3.4 Running the application

**Single-node mode** (without SLURM):

```bash
module load chemlink/1.0
cd /nfs/chemlink/projects/my_project     # receptors/ and ligands/ (or loose files)
chemlink docking full -d 4
```

**HPC mode with SLURM** (multi-node docking), from a project folder on NFS
so the compute nodes can see the inputs and `results/`:

```bash
module load chemlink/1.0
cd /nfs/chemlink/projects/my_project
chemlink hpc docking --partition gpu --batch-size 200 --max-gpu-concurrency 6
chemlink hpc dynamics pligand -t 100 --partition gpu   # one job per ligand
```

The SLURM scripts can also be launched by hand with the variables from
section 2.2:

```bash
module load chemlink/1.0
export DYN_TYPE=pligand
export DYN_NS_TIME=100
export DYN_CHARGE=0
export DYN_TIME_LIMIT=08:00:00
export DYN_CONFIGS_JSON=/nfs/chemlink/runs/sim_batch/configs.json

bash /nfs/chemlink/chemlink/hpc/slurm/native/run_dynamics_pipeline.sh
```

#### 4.3.5 Updating versions

```bash
cd /nfs/chemlink/chemlink
git fetch origin
git checkout main && git pull
# The workers see the change immediately through NFS — no further action needed.
```

---

### 4.4 Deployment with containers

#### 4.4.1 Building images

```bash
cd /nfs/chemlink/chemlink

# Local build
docker build -t chemlink:1.0.0 .
docker tag chemlink:1.0.0 chemlink:latest

# Build with BuildKit (faster, per-layer cache)
DOCKER_BUILDKIT=1 docker build \
  --cache-from chemlink:latest \
  -t chemlink:1.0.0 .
```

> **Note on GPU support in the image:**
> The image builds AutoDock-GPU with the `sm_86`, `sm_89`, `sm_90` and `sm_120` targets, including the Ampere family.
> GROMACS is built with the `sm_75;80;86;89;90;100;120` targets for maximum compatibility.

**For Apptainer/Singularity** (HPC without Docker privileges):

```bash
# Convert the Docker image to SIF
apptainer build chemlink.sif docker-daemon://chemlink:latest

# Or build directly from a registry image
apptainer build chemlink.sif docker://chemlink:latest
```

#### 4.4.2 Running on a server

**Docker Compose** (development and single server):

```bash
docker compose up -d
docker compose exec chemlink-gpu bash
# Or run directly:
docker compose run --rm chemlink-gpu chemlink doctor
```

**Plain Docker** with an explicit GPU:

```bash
# All GPUs
docker run --rm --gpus all \
  -v /nfs/chemlink:/nfs/chemlink \
  -w /nfs/chemlink/chemlink \
  -e PYTHONPATH=/nfs/chemlink \
  chemlink:latest \
  chemlink docking full -ir /nfs/chemlink/projects/my_project/receptors \
                       -il /nfs/chemlink/projects/my_project/ligands \
                       -o  /nfs/chemlink/projects/my_project/results/docking

# A specific GPU (index 0)
docker run --rm --gpus '"device=0"' \
  -v /nfs/chemlink:/nfs/chemlink \
  chemlink:latest chemlink doctor
```

**Apptainer on SLURM nodes** (without root privileges):

```bash
# Interactive run
apptainer exec --nv \
  --bind /nfs/chemlink:/nfs/chemlink \
  /nfs/chemlink/chemlink.sif \
  chemlink doctor

# From a SLURM script — use the CONTAINER_IMAGE variable:
export CONTAINER_IMAGE=/nfs/chemlink/chemlink.sif
export CONTAINER_RUNTIME=apptainer
bash /nfs/chemlink/chemlink/hpc/slurm/native/run_multinode_pipeline.sh
```

The SLURM scripts detect `CONTAINER_RUNTIME=apptainer|singularity|docker` automatically and adapt the invocation command.

#### 4.4.3 Environment variables and secrets

Pass variables to the container with `-e` or an `.env` file:

```bash
# .env file (do not commit)
cat > .env <<EOF
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
PYTHONUNBUFFERED=1
REPO_DIR=/nfs/chemlink/chemlink
PYTHON_BIN=/opt/miniconda/envs/bio/bin/python
EOF

docker run --rm --gpus all --env-file .env \
  -v /nfs/chemlink:/nfs/chemlink \
  chemlink:latest chemlink docking full ...
```

With Docker Compose, add to the service:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - PYTHONUNBUFFERED=1
  - REPO_DIR=/app/chemlink
```

#### 4.4.4 Persistence and networking

The container does not persist data by default. Always mount the NFS volume or the data directory:

```yaml
# docker-compose.yml
services:
  chemlink-gpu:
    volumes:
      - ./:/app/chemlink:z           # source code (development)
      - /nfs/chemlink/runs:/nfs/chemlink/runs   # persistent results
      - /nfs/chemlink/data:/nfs/chemlink/data   # input data
```

For cluster networking with several containers:

```yaml
networks:
  chemlink-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

#### 4.4.5 Updating the deployment

```bash
# 1. Stop the service
docker compose down

# 2. Update the code
git -C /nfs/chemlink/chemlink pull origin main

# 3. Rebuild the image (only if the Dockerfile or dependencies changed)
docker build -t chemlink:latest .

# 4. Start the service
docker compose up -d

# Update the SIF image for Apptainer
apptainer build --force chemlink.sif docker-daemon://chemlink:latest
```

---

## 5. Checking that it works

```bash
# Full environment diagnostics
chemlink doctor
```

The output reports the status of each component:

| Check | Manual check command |
|---|---|
| Python 3.10 in `bio` | `conda run -n bio python --version` |
| Meeko | `mk_prepare_receptor.py --help 2>&1 \| head -1` |
| fpocket | `fpocket --help 2>&1 \| head -1` |
| AutoGrid4 | `autogrid4 -h 2>&1 \| head -1` |
| AutoDock-GPU | `autodock-gpu --version` |
| GROMACS (MPI) | `gmx_mpi --version \| head -5` |
| CUDA available | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| NFS mounted | `mount \| grep nfs` |
| SLURM running | `sinfo -l` |
| GPU container | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu24.04 nvidia-smi` |

`chemlink doctor` groups the checks (software, hardware, configuration and
paths), shows how many are ready in each group, a health bar and a numbered
list of suggested fixes. With `--json` it produces the same information in
machine-readable form (useful in CI). It exits with code 1 only if there are blocking
errors; warnings (for example, no GPU) do not prevent using ChemLink.

Interface test (no GPU or scientific tools needed):

```bash
chemlink demo
```

Minimal functional docking test without a GPU (requires Meeko, RDKit, PDBFixer,
OpenMM, fpocket and AutoDock Vina), using the repository's test structures:

```bash
chemlink docking receptor -ir data/test/receptors -o /tmp/test_receptor
chemlink docking full -ir data/test/receptors -il data/test/ligands \
  -o /tmp/test_docking --engine cpu
```

Functional dynamics test without a GPU (small system; GROMACS uses the CPU
automatically when it detects no GPU):

```bash
chemlink dynamic oprotein -ir data/test/receptors/3ptb.pdb -t 0.01
```

---

## 6. Troubleshooting

### GPU not detected in Docker

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Fix:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Check:
docker info | grep -i runtime
```

### AutoDock-GPU: `illegal instruction` or `CUDA error: no kernel image`

The host GPU has a different architecture from the ones compiled into the image.

**Fix:** rebuild with the right `sm_`:
```bash
# Check the host's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Edit the Dockerfile: add -gencode arch=compute_XX,code=sm_XX
DOCKER_BUILDKIT=1 docker build -t chemlink:latest .
```

### GROMACS: slow `JIT PTX compilation` (RTX 5000 / sm_120)

The binary has no native code for sm_120 and compiles it at run time.

**Fix:** make sure the build included `sm_120`, or use the one built in the Dockerfile (which does include it). See also the project report's section on mitigating it with `nstlist=40`.

### NFS: `mount: connection timed out`

```bash
# Check that the NFS server is running
sudo systemctl status nfs-kernel-server   # on the manager
sudo exportfs -v                          # check the exports
# Check the firewall
sudo ufw status
sudo ufw allow from 10.0.0.0/24 to any port 2049
```

### SLURM: `job cancelled — time limit`

Increase the time limit for complex dynamics:
```bash
export DYN_TIME_LIMIT=12:00:00   # for pprotein or ppligand
bash hpc/slurm/native/run_dynamics_pipeline.sh
```

### `ModuleNotFoundError: No module named 'chemlink'`

The package is imported as `chemlink`, so `PYTHONPATH` must contain the
**parent** directory of the repository (and the repository directory must be
named `chemlink`). The `./chemlink` launcher sets this up on its own:

```bash
./chemlink doctor
# or manually:
export PYTHONPATH=/nfs/chemlink:/nfs/chemlink/chemlink:${PYTHONPATH:-}
# or load the module:
module load chemlink/1.0
```

### `externally-managed-environment` when running `pip install`

Recent Ubuntu/Debian releases do not allow installing packages into the system Python
(PEP 668). Use a virtual environment or the system packages:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "rich>=13" prompt_toolkit
# or:
sudo apt install python3-rich python3-prompt-toolkit
```

### Escape codes or broken bars in SLURM logs

Outside a terminal, ChemLink writes progress as timestamped
lines (every 10% plus a heartbeat every 10 min). If an environment presents itself as a
terminal and `\x1b[...` sequences still show up, force text mode:

```bash
export CHEMLINK_PROGRESS=plain    # or add --plain to the command
```

### The TUI does not open or does not respond to the keyboard

- `chemlink` with no arguments only opens the TUI if both input and output are a
  terminal; inside a pipe or with `--plain` it shows the help.
- The TUI does not work on native Windows; use WSL.
- If the logo looks misaligned, the terminal font does not draw the
  block characters (`▀▄▌▐▙▟`) properly; use a modern monospaced font
  (DejaVu Sans Mono, JetBrains Mono, Cascadia, Ubuntu Mono).

### Ligand not processed by AutoDock-GPU (abnormal geometry)

The system records the failure in `runs/<id>/logs/` and continues. To inspect it:
```bash
grep "ERROR\|FAILED" /nfs/chemlink/runs/<run_id>/logs/*.err
```

---

## 7. Maintenance and updates

### Updating ChemLink

```bash
# Without Docker
git -C /nfs/chemlink/chemlink pull origin main

# With Docker (rebuild only if the Dockerfile or requirements changed)
git -C /nfs/chemlink/chemlink pull origin main
docker build -t chemlink:latest /nfs/chemlink/chemlink
docker compose up -d
```

### Updating Conda environments

```bash
conda activate bio
conda update --all -c conda-forge
```

In the `bio` Conda environment the dependencies come from conda-forge; do not install
`requirements.txt` on top of it with pip, because it would replace Conda's RDKit.
`requirements.txt` is for pip-only installs (for example a `venv`):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Cleaning up old runs

```bash
# Delete runs older than 30 days
find /nfs/chemlink/runs -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
```

### Cluster monitoring

- **Grafana:** `http://<manager-ip>:3000` — per-node CPU, GPU, RAM and network dashboards
- **Prometheus:** `http://<manager-ip>:9090` — raw metrics and alerts
- **SLURM:** `squeue -u $USER` — active jobs; `sacct -j <jobid>` — history

### Backups

Results in `/nfs/chemlink/runs/` are backed up automatically by OpenMediaVault snapshots (SMB/CIFS). For a manual backup:

```bash
rsync -av /nfs/chemlink/runs/ /backup/chemlink-runs/
```

---

## 8. Related references

| Resource | Description |
|---|---|
| [README.md](../README.md) | Project overview and quick start |
| [docs/Installation.md](Installation.md) | This document |
| [Report.md](Report.md) | Full technical report of the project |
| [hpc/slurm/native/](../hpc/slurm/native/) | SLURM scripts for native deployment |
| [hpc/slurm/container/](../hpc/slurm/container/) | SLURM scripts for container deployment |
| [Dockerfile](../Dockerfile) | Self-contained Docker image |
| [docker-compose.yml](../docker-compose.yml) | Compose file for GPU development |
| [install.sh](../install.sh) | curl-pipe-bash installer |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Official guide for GPUs in Docker |
| [SLURM Job Arrays](https://slurm.schedmd.com/job_array.html) | Official SLURM documentation |
| [GROMACS GPU offloading](https://manual.gromacs.org/current/user-guide/mdrun-performance.html) | Configuring GROMACS with a GPU |
| [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) | Official repository |
