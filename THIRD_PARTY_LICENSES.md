# Licencias de terceros

ChemLink se distribuye bajo AGPL-3.0-only o bajo una licencia comercial (ver
[LICENSING.md](LICENSING.md)). Este archivo lista el software y los datos de
terceros que ChemLink usa o que se instalan junto con él, con su licencia y la
forma en que ChemLink se relaciona con cada uno.

Ninguna de estas licencias se aplica al código de ChemLink: cada componente
conserva la suya, y la licencia comercial de ChemLink no otorga derechos sobre
ellos. El texto completo de cada licencia acompaña al paquete correspondiente
(`*.dist-info/` con pip, `pkgs/<paquete>/info/licenses/` con Conda) o está en el
repositorio indicado. La imagen Docker los reúne en
`/usr/share/chemlink/third-party/` (ver la sección 5).

**Cómo se usa cada componente**

- **Importado**: se carga dentro del proceso de ChemLink (`import`).
- **Programa aparte**: ChemLink lo ejecuta como un programa independiente
  (`subprocess`) y se comunica con él mediante archivos y argumentos. No forma
  una obra combinada con ChemLink.

Ningún componente importado tiene licencia GPL, y ninguno tiene restricciones
de uso comercial.

## 1. Bibliotecas Python importadas

Se instalan con `requirements.txt` o en el entorno Conda `bio`.

| Componente | Versión | Licencia | Fuente |
|---|---|---|---|
| NumPy | ≥ 1.24 | BSD-3-Clause (incluye partes 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | ≥ 1.10 | BSD-3-Clause | https://scipy.org |
| RDKit | 2025.9.6 | BSD-3-Clause | https://github.com/rdkit/rdkit |
| Meeko | 0.7.1 | LGPL-2.1-or-later | https://github.com/forlilab/Meeko |
| gemmi | 0.7.5 | MPL-2.0 | https://github.com/project-gemmi/gemmi |
| PDBFixer | ≥ 1.9 | MIT | https://github.com/openmm/pdbfixer |
| OpenMM | ≥ 8.0 | MIT (API, plataformas Reference y CPU); LGPL-2.1 (plataformas CUDA y OpenCL) | https://github.com/openmm/openmm |
| NetworkX | ≥ 2.8 | BSD-3-Clause | https://github.com/networkx/networkx |
| Matplotlib | ≥ 3.5 | Matplotlib License (tipo PSF) | https://matplotlib.org |
| Rich | ≥ 13.0 | MIT | https://github.com/Textualize/rich |
| prompt_toolkit | ≥ 3.0 | BSD-3-Clause | https://github.com/prompt-toolkit/python-prompt-toolkit |

El entorno `bio` también incluye pandas (BSD-3-Clause), tqdm (MIT y MPL-2.0)
y Biopython (Biopython License Agreement / BSD-3-Clause), que ChemLink no
importa directamente.

**Bibliotecas LGPL y MPL.** Meeko, gemmi y las plataformas GPU de OpenMM se
usan sin modificar y como paquetes independientes que el usuario puede
reemplazar por otra versión, como permiten la LGPL y la MPL. Quien distribuya
una versión modificada de alguno debe publicar esa modificación bajo su misma
licencia.

## 2. Programas externos

ChemLink los ejecuta como programas aparte. El instalador (`install.sh`) o el
`Dockerfile` los descargan o los compilan; ninguno está en este repositorio.

| Programa | Versión | Licencia | Fuente |
|---|---|---|---|
| GROMACS | 2025.4 | LGPL-2.1 | https://gitlab.com/gromacs/gromacs |
| AutoDock-GPU | commit `be06a135` | GPL-2.0 | https://github.com/ccsb-scripps/AutoDock-GPU |
| AutoGrid4 | commit `6d2847be` | GPL-2.0 | https://github.com/ccsb-scripps/AutoGrid |
| AutoDock4 | commit `192ecda0` | GPL-2.0 | https://github.com/ccsb-scripps/AutoDock4 |
| AutoDock Vina | 1.2.5 | Apache-2.0 | https://github.com/ccsb-scripps/AutoDock-Vina |
| fpocket | 4.2.3 | MIT | https://github.com/Discngine/fpocket |
| Open Babel (`obabel`) | ≥ 3.2.1 | GPL-2.0 | https://github.com/openbabel/openbabel |
| ACPYPE | conda-forge | GPL-3.0 | https://github.com/alanwilter/acpype |
| AmberTools | conda-forge | GPL-3.0-or-later, LGPL-3.0-or-later, BSD-3-Clause y MIT según el componente | https://ambermd.org/AmberTools.php |
| gmx_MMPBSA | PyPI | GPL-3.0 | https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA |
| Open MPI | conda-forge / Ubuntu | BSD-3-Clause | https://github.com/open-mpi/ompi |
| mpi4py | conda-forge | BSD-3-Clause | https://github.com/mpi4py/mpi4py |

gmx_MMPBSA, AmberTools, Open MPI y mpi4py viven en el entorno Conda aparte
`gmx_mmpbsa` (`config/environments/gmx_mmpbsa.yml`), que también incluye
pandas, Matplotlib, seaborn (BSD-3-Clause) y tqdm.

## 3. Instalación y entorno

| Componente | Licencia | Notas |
|---|---|---|
| Miniforge 26.7.2-0 | BSD-3-Clause | `install.sh` y el `Dockerfile` lo instalan cuando no hay conda. Solo usa los canales `conda-forge` y `bioconda`; ChemLink no usa los canales de Anaconda ni acepta sus términos de servicio. https://github.com/conda-forge/miniforge |
| conda | BSD-3-Clause | https://github.com/conda/conda |
| CMake 3.29.6 | BSD-3-Clause | Solo en la etapa de compilación del `Dockerfile`; no queda en la imagen final. |
| Imagen `nvidia/cuda:13.0.0-devel-ubuntu24.04` | Licencias de NVIDIA | Solo en la etapa de compilación; no queda en la imagen final. |
| Imagen `nvidia/cuda:13.0.0-runtime-ubuntu24.04` | Licencias de NVIDIA y de los paquetes de Ubuntu | Base de la imagen final. Contiene solo las bibliotecas de ejecución de CUDA que NVIDIA distribuye para ese fin; los binarios compilados enlazan cuFFT y cuRAND de esa imagen. |

Las bibliotecas de Conda usan FreeType bajo la FreeType License: *Portions of
this software are copyright © The FreeType Project (www.freetype.org). All
rights reserved.*

## 4. Datos

| Contenido | Origen | Licencia | Dónde está |
|---|---|---|---|
| CHARMM36m (`charmm36-jul2022.ff`) | Grupo MacKerell, Universidad de Maryland; conversión a GROMACS de charmm2gmx (A. Wacha) | Sin condiciones de redistribución declaradas | **No se incluye en el repositorio.** `install.sh` y el `Dockerfile` lo descargan de https://mackerell.umaryland.edu y verifican su SHA-256 (ver [forcefields/README.md](forcefields/README.md)). |
| `data/test/receptors/`, `data/test/cofactors/` | RCSB PDB (1HSG, 3PTB, 6LU7, 1TQN) | CC0-1.0 (política de uso de datos del wwPDB) | Repositorio |
| `data/test/ligands/` | PubChem (NCBI) | Dominio público; NCBI no impone restricciones de uso | Repositorio |

Una imagen Docker construida con el `Dockerfile` sí contiene CHARMM36m en
`/opt/forcefields`. Mientras el grupo MacKerell no confirme que puede
redistribuirse, construya las imágenes que vaya a entregar a terceros con
`--build-arg WITH_CHARMM36=false`; cada usuario lo descarga después en
`/opt/forcefields` con los comandos de [forcefields/README.md](forcefields/README.md).

## 5. Imagen Docker: avisos y código fuente

El repositorio publica el `Dockerfile` como receta: quien construye la imagen
descarga cada componente de su fuente original. Para quien entregue una imagen
ya construida a terceros, la imagen incluye lo necesario para cumplir las
licencias de los componentes que contiene:

| Ruta en la imagen | Contenido |
|---|---|
| `/usr/share/chemlink/third-party/THIRD_PARTY_LICENSES.md`, `LICENSE`, `LICENSING.md` | Este archivo y las licencias de ChemLink |
| `/usr/share/chemlink/third-party/licenses/` | Licencias de AutoGrid4, AutoDock4, AutoDock-GPU, GROMACS y AutoDock Vina |
| `/usr/share/chemlink/third-party/licenses/conda/` | Licencias de cada paquete Conda instalado |
| `/usr/share/chemlink/third-party/src/` | Código fuente exacto de AutoGrid4, AutoDock4, AutoDock-GPU y GROMACS, tal como se compilaron |
| `/usr/share/chemlink/third-party/manifests/` | Lista exacta de paquetes Conda, pip y de Ubuntu (versión, licencia y URL) |

**Oferta de código fuente.** Los paquetes GPL y LGPL de Conda y de Ubuntu que
no se compilan en el `Dockerfile` se instalan como binarios. Quien entregue una
imagen construida ofrece, durante al menos tres años desde la entrega, el
código fuente correspondiente de esos paquetes a cualquier persona que lo
solicite a **[contacto por definir]**, por un costo no mayor que el de
realizar la copia. Los manifiestos de la imagen identifican cada versión
exacta, y el código fuente de cada una puede obtenerse de su receta en
conda-forge o bioconda y del archivo de fuentes de Ubuntu.

## 6. Componentes que ChemLink ya no usa

Estos componentes ya no forman parte de ChemLink en esta rama: Open Babel como
biblioteca importada (GPL-2.0), MGLTools (licencia propia con restricciones),
PyRosetta (solo uso no comercial), HADDOCK3/CNS, PRODIGY, LightDock (GPL-3.0),
alchemlyb, pymbar y Miniconda (términos de servicio de Anaconda).
