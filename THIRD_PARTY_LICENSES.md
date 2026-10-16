# Third-party licenses

ChemLink is distributed under AGPL-3.0-only or under a commercial license (see
[LICENSING.md](LICENSING.md)). This file lists the third-party software and
data that ChemLink uses or that are installed alongside it, with their license
and how ChemLink relates to each one.

None of these licenses applies to ChemLink's own code: each component keeps its
own, and the ChemLink commercial license grants no rights over them. The full
text of each license ships with the corresponding package (`*.dist-info/` with
pip, `pkgs/<package>/info/licenses/` with Conda) or is in the repository listed.
The Docker image collects them in `/usr/share/chemlink/third-party/` (see
section 5).

**How each component is used**

- **Imported**: loaded inside the ChemLink process (`import`).
- **Separate program**: ChemLink runs it as an independent program
  (`subprocess`) and communicates with it through files and arguments. It does
  not form a combined work with ChemLink.

No imported component is GPL-licensed, and none has restrictions on commercial
use.

## 1. Imported Python libraries

Installed with `requirements.txt` or in the `bio` Conda environment.

| Component | Version | License | Source |
|---|---|---|---|
| NumPy | ≥ 1.24 | BSD-3-Clause (includes 0BSD, MIT, Zlib, CC0-1.0 parts) | https://numpy.org |
| SciPy | ≥ 1.10 | BSD-3-Clause | https://scipy.org |
| RDKit | 2025.9.6 | BSD-3-Clause | https://github.com/rdkit/rdkit |
| Meeko | 0.7.1 | LGPL-2.1-or-later | https://github.com/forlilab/Meeko |
| gemmi | 0.7.5 | MPL-2.0 | https://github.com/project-gemmi/gemmi |
| PDBFixer | ≥ 1.9 | MIT | https://github.com/openmm/pdbfixer |
| OpenMM | ≥ 8.0 | MIT (API, Reference and CPU platforms); LGPL-2.1 (CUDA and OpenCL platforms) | https://github.com/openmm/openmm |
| NetworkX | ≥ 2.8 | BSD-3-Clause | https://github.com/networkx/networkx |
| Matplotlib | ≥ 3.5 | Matplotlib License (PSF-based) | https://matplotlib.org |
| Rich | ≥ 13.0 | MIT | https://github.com/Textualize/rich |
| prompt_toolkit | ≥ 3.0 | BSD-3-Clause | https://github.com/prompt-toolkit/python-prompt-toolkit |

The `bio` environment also includes pandas (BSD-3-Clause), tqdm (MIT and
MPL-2.0) and Biopython (Biopython License Agreement / BSD-3-Clause), which
ChemLink does not import directly.

**LGPL and MPL libraries.** Meeko, gemmi and the OpenMM GPU platforms are used
unmodified and as independent packages that the user can replace with another
version, as the LGPL and the MPL allow. Anyone distributing a modified version
of one of them must publish that modification under the same license.

## 2. External programs

ChemLink runs these as separate programs. The installer (`install.sh`) or the
`Dockerfile` downloads or builds them; none of them is in this repository.

| Program | Version | License | Source |
|---|---|---|---|
| GROMACS | 2025.4 | LGPL-2.1 | https://gitlab.com/gromacs/gromacs |
| AutoDock-GPU | commit `be06a135` | GPL-2.0 | https://github.com/ccsb-scripps/AutoDock-GPU |
| AutoGrid4 | commit `6d2847be` | GPL-2.0 | https://github.com/ccsb-scripps/AutoGrid |
| AutoDock4 | commit `192ecda0` | GPL-2.0 | https://github.com/ccsb-scripps/AutoDock4 |
| AutoDock Vina | 1.2.5 | Apache-2.0 | https://github.com/ccsb-scripps/AutoDock-Vina |
| fpocket | 4.2.3 | MIT | https://github.com/Discngine/fpocket |
| Open Babel (`obabel`) | ≥ 3.2.1 | GPL-2.0 | https://github.com/openbabel/openbabel |
| ACPYPE | conda-forge | GPL-3.0 | https://github.com/alanwilter/acpype |
| AmberTools | conda-forge | GPL-3.0-or-later, LGPL-3.0-or-later, BSD-3-Clause and MIT depending on the component | https://ambermd.org/AmberTools.php |
| gmx_MMPBSA | PyPI | GPL-3.0 | https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA |
| LightDock (`lightdock3_setup.py`, `lightdock3.py`, `lgd_*.py`) | ≥ 0.9 | GPL-3.0 | https://github.com/lightdock/lightdock |
| PRODIGY (`prodigy`, package `prodigy-prot`) | ≥ 2.4 | Apache-2.0 | https://github.com/haddocking/prodigy |
| Open MPI | conda-forge / Ubuntu | BSD-3-Clause | https://github.com/open-mpi/ompi |
| mpi4py | conda-forge | BSD-3-Clause | https://github.com/mpi4py/mpi4py |

LightDock and PRODIGY are installed with pip into the `bio` environment, together
with their dependencies ProDy (MIT), freesasa (MIT) and Cython (Apache-2.0).
ChemLink runs them as separate programs for peptide docking (`docking peptide`)
and never imports them.

gmx_MMPBSA, AmberTools, Open MPI and mpi4py live in the separate
`gmx_mmpbsa` Conda environment (`config/environments/gmx_mmpbsa.yml`), which
also includes pandas, Matplotlib, seaborn (BSD-3-Clause) and tqdm.

## 3. Installation and environment

| Component | License | Notes |
|---|---|---|
| Miniforge 26.7.2-0 | BSD-3-Clause | `install.sh` and the `Dockerfile` install it when no conda is found. It only uses the `conda-forge` and `bioconda` channels; ChemLink does not use the Anaconda channels or accept their terms of service. https://github.com/conda-forge/miniforge |
| conda | BSD-3-Clause | https://github.com/conda/conda |
| CMake 3.29.6 | BSD-3-Clause | Only in the `Dockerfile` build stage; not present in the final image. |
| `nvidia/cuda:13.0.0-devel-ubuntu24.04` image | NVIDIA licenses | Only in the build stage; not present in the final image. |
| `nvidia/cuda:13.0.0-runtime-ubuntu24.04` image | NVIDIA and Ubuntu package licenses | Base of the final image. It only contains the CUDA runtime libraries that NVIDIA distributes for that purpose; the compiled binaries link cuFFT and cuRAND from that image. |

The Conda libraries use FreeType under the FreeType License: *Portions of
this software are copyright © The FreeType Project (www.freetype.org). All
rights reserved.*

## 4. Data

| Content | Origin | License | Location |
|---|---|---|---|
| CHARMM36m (`charmm36-jul2022.ff`) | MacKerell group, University of Maryland; GROMACS conversion by charmm2gmx (A. Wacha) | No redistribution terms stated | **Not included in the repository.** `install.sh` and the `Dockerfile` download it from https://mackerell.umaryland.edu and verify its SHA-256 (see [forcefields/README.md](forcefields/README.md)). |
| `data/test/receptors/`, `data/test/cofactors/` | RCSB PDB (1HSG, 3PTB, 6LU7, 1TQN) | CC0-1.0 (wwPDB data usage policy) | Repository |
| `data/test/ligands/` | PubChem (NCBI) | Public domain; NCBI places no restrictions on use | Repository |

A Docker image built from the `Dockerfile` does contain CHARMM36m in
`/opt/forcefields`. Until the MacKerell group confirms it may be
redistributed, build any image you will hand to third parties with
`--build-arg WITH_CHARMM36=false`; each user then downloads it into
`/opt/forcefields` with the commands in [forcefields/README.md](forcefields/README.md).

## 5. Docker image: notices and source code

The repository publishes the `Dockerfile` as a recipe: whoever builds the image
downloads each component from its original source. For anyone handing a built
image to third parties, the image includes what is needed to comply with the
licenses of the components it contains:

| Path in the image | Contents |
|---|---|
| `/usr/share/chemlink/third-party/THIRD_PARTY_LICENSES.md`, `LICENSE`, `LICENSING.md` | This file and the ChemLink licenses |
| `/usr/share/chemlink/third-party/licenses/` | Licenses of AutoGrid4, AutoDock4, AutoDock-GPU, GROMACS and AutoDock Vina |
| `/usr/share/chemlink/third-party/licenses/conda/` | Licenses of every installed Conda package |
| `/usr/share/chemlink/third-party/src/` | Exact source code of AutoGrid4, AutoDock4, AutoDock-GPU and GROMACS, as built |
| `/usr/share/chemlink/third-party/manifests/` | Exact list of Conda, pip and Ubuntu packages (version, license and URL) |

**Source code offer.** The GPL and LGPL Conda and Ubuntu packages that are not
built in the `Dockerfile` are installed as binaries. Anyone handing over a
built image offers, for at least three years from delivery, the corresponding
source code of those packages to anyone who requests it from
**[contact to be defined]**, for a charge no greater than the cost of making
the copy. The image manifests identify each exact version, and the source code
of each one can be obtained from its conda-forge or bioconda recipe and from
the Ubuntu source archive.

## 6. Components ChemLink no longer uses

These components are no longer part of ChemLink on this branch: Open Babel as
an imported library (GPL-2.0), MGLTools (custom license with restrictions),
PyRosetta (non-commercial use only), HADDOCK3/CNS, alchemlyb, pymbar and
Miniconda (Anaconda terms of service).
