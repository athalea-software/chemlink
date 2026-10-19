#!/usr/bin/env bash
# ChemLink Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/athalea-software/chemlink/main/install.sh | bash
# Or:    curl -fsSL https://raw.githubusercontent.com/athalea-software/chemlink/main/install.sh | bash -s -- --full
set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; YELLOW=''; GREEN=''; CYAN=''; BOLD=''; RESET=''
fi

# ─── Defaults ─────────────────────────────────────────────────────────────────
INSTALL_DIR="${CHEMLINK_DIR:-/opt/chemlink}"
REPO_URL="https://github.com/athalea-software/chemlink.git"
INSTALL_TOOLS=false   # --full enables scientific tools compilation
INSTALL_GROMACS=false
INSTALL_MMPBSA=false
INSTALL_CHARMM36=true
SKIP_CONDA=false
SKIP_UPDATE=false
VERSION="main"
FPOCKET_VERSION="${FPOCKET_VERSION:-4.2.3}"
AUTODOCK_GPU_TARGETS="${AUTODOCK_GPU_TARGETS:-86 89 90 120}"
CHARMM36_URL="${CHARMM36_URL:-https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files%2Fcharmm36-jul2022.ff.tgz}"
CHARMM36_SHA256="${CHARMM36_SHA256:-9f31db8502852bc8fcf36497c27db14f1b6509b33701e7cafc2187840b7d2f09}"
# Miniforge (BSD-3-Clause, conda-forge only) rather than Miniconda, whose
# Anaconda terms of service restrict commercial use of its channels.
MINIFORGE_VERSION="${MINIFORGE_VERSION:-26.7.2-0}"
MINIFORGE_SHA256="${MINIFORGE_SHA256:-281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05}"
# Exact upstream commits, so a build can always be traced to its source.
AUTOGRID_COMMIT="${AUTOGRID_COMMIT:-6d2847beaeac8ff43ca99094707fd74e3ca1ff37}"
AUTODOCK4_COMMIT="${AUTODOCK4_COMMIT:-192ecda05d7c566161046f0a1d604f3336e0cf3a}"
AUTODOCK_GPU_COMMIT="${AUTODOCK_GPU_COMMIT:-be06a135181178c8631653e1e32d9ece86b5b4d8}"

# ─── Argument parsing ─────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}ChemLink Installer${RESET}

Usage: install.sh [options]

Options:
  --full            Install and compile all scientific tools
                    (fpocket, AutoGrid4, AutoDock4, AutoDock-GPU, Vina, GROMACS)
  --with-gromacs    Also compile GROMACS (long — requires CUDA + MPI)
  --with-mmpbsa     Create the dedicated gmx_MMPBSA/AmberTools environment
  --dir PATH        Installation directory (default: /opt/chemlink)
  --version TAG     Git branch/tag to install (default: main)
  --skip-conda      Skip Conda environment setup
  --skip-update     Keep an existing ChemLink checkout unchanged
  --without-charmm36
                    Do not install the ChemFusion-compatible CHARMM36m field
  --help            Show this help message

Environment variables:
  CHEMLINK_DIR      Override installation directory
  FPOCKET_VERSION   fpocket tag to build (default: 4.2.3)
  AUTODOCK_GPU_TARGETS
                    Space-separated CUDA targets (default: 86 89 90 120)
  CHARMM36_URL       Official CHARMM36m archive URL
  CHARMM36_SHA256    Expected archive SHA-256 checksum
  MINIFORGE_VERSION, MINIFORGE_SHA256
                    Miniforge release installed when no conda is found
  AUTOGRID_COMMIT, AUTODOCK4_COMMIT, AUTODOCK_GPU_COMMIT
                    Upstream commits to build (defaults are pinned)

Examples:
  curl -fsSL https://raw.githubusercontent.com/athalea-software/chemlink/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/athalea-software/chemlink/main/install.sh | bash -s -- --full
  curl -fsSL https://raw.githubusercontent.com/athalea-software/chemlink/main/install.sh | bash -s -- --dir ~/chemlink
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)         INSTALL_TOOLS=true; INSTALL_GROMACS=true; INSTALL_MMPBSA=true ;;
    --with-gromacs) INSTALL_GROMACS=true ;;
    --with-mmpbsa)  INSTALL_MMPBSA=true ;;
    --dir)          INSTALL_DIR="$2"; shift ;;
    --version)      VERSION="$2"; shift ;;
    --skip-conda)   SKIP_CONDA=true ;;
    --skip-update)  SKIP_UPDATE=true ;;
    --without-charmm36) INSTALL_CHARMM36=false ;;
    --help|-h)      usage ;;
    *) echo -e "${RED}Unknown option: $1${RESET}" >&2; usage ;;
  esac
  shift
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}  •${RESET} $*"; }
success() { echo -e "${GREEN}  ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}  !${RESET} $*"; }
die()     { echo -e "${RED}  ✗ ERROR:${RESET} $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" &>/dev/null || die "Required command not found: $1 — install it and re-run."
}

have_cmd() { command -v "$1" &>/dev/null; }

# Return the real prefix of a named Conda environment.  Conda may place named
# environments under ~/.conda/envs when the base installation (for example
# /opt/conda on Arch Linux) is read-only for the current user.
conda_env_prefix() {
  local conda_bin="$1" env_name="$2" base_python
  base_python="$(dirname "$conda_bin")/python"
  [[ -x "$base_python" ]] || return 1
  "$conda_bin" env list --json | "$base_python" -c '
import json, os, sys
name = sys.argv[1]
for prefix in json.load(sys.stdin).get("envs", []):
    if os.path.basename(prefix.rstrip(os.sep)) == name:
        print(prefix)
        break
' "$env_name"
}

# Honour an explicit CUDA_PATH, otherwise derive the toolkit root from nvcc.
# Ubuntu commonly uses /usr/local/cuda; Arch Linux packages CUDA in /opt/cuda.
detect_cuda_path() {
  if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    printf '%s\n' "$CUDA_PATH"
    return 0
  fi
  if have_cmd nvcc; then
    local nvcc_bin
    nvcc_bin="$(readlink -f "$(command -v nvcc)")"
    dirname "$(dirname "$nvcc_bin")"
    return 0
  fi
  for candidate in /usr/local/cuda /opt/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' /usr/local/cuda
}

# CUDA packages on rolling distributions can require a versioned host
# compiler even when the system default GCC is newer.  Respect an explicit
# override, prefer GCC 14 when available (CUDA 12.x on Arch), then fall back to
# the default C++ compiler used on Ubuntu and other supported systems.
detect_cuda_host_compiler() {
  if [[ -n "${CUDA_HOST_COMPILER:-}" && -x "${CUDA_HOST_COMPILER}" ]]; then
    printf '%s\n' "$CUDA_HOST_COMPILER"
    return 0
  fi
  local candidate
  for candidate in g++-14 g++-13 g++; do
    if have_cmd "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# ─── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
cat <<'EOF'
   _____ _                    _     _       _
  / ____| |                  | |   (_)     | |
 | |    | |__   ___ _ __ ___ | |    _ _ __ | | __
 | |    | '_ \ / _ \ '_ ` _ \| |   | | '_ \| |/ /
 | |____| | | |  __/ | | | | | |___| | | | |   <
  \_____|_| |_|\___|_| |_| |_|_____|_|_| |_|_|\_\

EOF
echo -e "${RESET}${BOLD}  Molecular Docking & Dynamics Orchestration Platform${RESET}"
echo -e "  ${CYAN}https://github.com/athalea-software/chemlink${RESET}"
echo ""

# ─── 1. System checks ─────────────────────────────────────────────────────────
echo -e "${BOLD}[1/6] Checking system requirements${RESET}"

# OS
if [[ "$(uname -s)" != "Linux" ]]; then
  die "ChemLink requires Linux (Ubuntu 24.04 LTS recommended)."
fi

# Architecture
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  die "ChemLink requires x86_64 architecture (detected: $ARCH)."
fi
success "Linux x86_64"

# Ubuntu version (warn but don't block)
if have_cmd lsb_release; then
  DISTRO="$(lsb_release -is 2>/dev/null || true)"
  DISTRO_VER="$(lsb_release -rs 2>/dev/null || true)"
  if [[ "$DISTRO" == "Ubuntu" && "$DISTRO_VER" == "24.04" ]]; then
    success "Ubuntu $DISTRO_VER"
  elif [[ "$DISTRO" == "Arch" || "$DISTRO" == "Arch Linux" ]]; then
    success "Arch Linux ${DISTRO_VER:-rolling}"
  else
    warn "Ubuntu 24.04 LTS is recommended. Detected: ${DISTRO:-unknown} ${DISTRO_VER:-unknown}"
  fi
fi

# Required system commands
for cmd in git curl wget bash; do
  need_cmd "$cmd"
done
success "Core tools (git, curl, wget)"

# NVIDIA GPU
if have_cmd nvidia-smi; then
  GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  success "NVIDIA GPU: ${GPU_INFO:-detected}"
else
  warn "nvidia-smi not found — GPU acceleration will not be available."
  warn "Install NVIDIA drivers and CUDA Toolkit to enable GPU pipelines."
fi

# CUDA
if have_cmd nvcc; then
  CUDA_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || true)"
  success "CUDA ${CUDA_VER:-detected}"
else
  warn "CUDA Toolkit not found. AutoDock-GPU and GROMACS GPU offloading require CUDA 12.x."
fi

echo ""

# ─── 2. Installation directory ────────────────────────────────────────────────
echo -e "${BOLD}[2/6] Setting up installation directory${RESET}"

# Determine if we need sudo
USE_SUDO=false
if [[ "$INSTALL_DIR" == /opt/* || "$INSTALL_DIR" == /usr/* ]]; then
  if [[ "$EUID" -ne 0 ]]; then
    if have_cmd sudo; then
      USE_SUDO=true
      info "Requesting sudo for system-wide install at $INSTALL_DIR"
    else
      warn "Cannot write to $INSTALL_DIR without root. Falling back to ~/.local/chemlink"
      INSTALL_DIR="$HOME/.local/chemlink"
    fi
  fi
fi

run_as_root() {
  if $USE_SUDO; then sudo "$@"; else "$@"; fi
}

# /opt and /usr/local need root even when INSTALL_DIR is user-owned (--dir ~/...).
as_system() {
  if [[ "$EUID" -ne 0 ]] && have_cmd sudo; then sudo "$@"; else "$@"; fi
}

if [[ -d "$INSTALL_DIR/.git" ]]; then
  if $SKIP_UPDATE; then
    success "Existing installation kept unchanged at $INSTALL_DIR (--skip-update)"
  elif [[ -n "$(git -C "$INSTALL_DIR" status --porcelain 2>/dev/null)" ]]; then
    warn "Existing installation at $INSTALL_DIR has local changes; keeping it unchanged."
    warn "Dependency installation will continue. Commit/stash those changes before a future update."
  else
    info "Existing installation found at $INSTALL_DIR — updating..."
    # Never allow a system installer to block on an interactive GitHub prompt.
    # A private HTTPS remote can require credentials even when dependencies can
    # still be installed successfully from the existing checkout.
    if GIT_TERMINAL_PROMPT=0 git -C "$INSTALL_DIR" fetch --quiet origin; then
      git -C "$INSTALL_DIR" checkout --quiet "$VERSION"
      GIT_TERMINAL_PROMPT=0 git -C "$INSTALL_DIR" pull --quiet origin "$VERSION"
      success "Updated to latest $VERSION"
    else
      warn "Could not update $INSTALL_DIR non-interactively; keeping the existing checkout."
      warn "Dependency installation will continue. Configure SSH/token access to update later."
    fi
  fi
else
  info "Cloning ChemLink into $INSTALL_DIR ..."
  run_as_root mkdir -p "$INSTALL_DIR"
  if [[ "$EUID" -ne 0 ]] && $USE_SUDO; then
    sudo chown "$USER":"$(id -gn)" "$INSTALL_DIR"
  fi
  git clone --quiet --branch "$VERSION" --depth 1 "$REPO_URL" "$INSTALL_DIR"
  success "Cloned ChemLink $VERSION"
fi

# Install the same CHARMM36m GROMACS port used by ChemFusion inside ChemLink's
# own prefix. This keeps protein-protein runs independent of another checkout.
if $INSTALL_CHARMM36; then
  CHARMM36_TARGET="$INSTALL_DIR/forcefields/charmm36-jul2022.ff"
  if [[ -s "$CHARMM36_TARGET/forcefield.itp" ]]; then
    success "CHARMM36m force field already installed"
  else
    info "Installing ChemFusion-compatible CHARMM36m force field..."
    CHARMM36_TMP="$(mktemp -d)"
    curl -fL --retry 3 -o "$CHARMM36_TMP/charmm36-jul2022.ff.tgz" "$CHARMM36_URL"
    printf '%s  %s\n' "$CHARMM36_SHA256" "$CHARMM36_TMP/charmm36-jul2022.ff.tgz" \
      | sha256sum --check --status \
      || die "CHARMM36m checksum verification failed."
    run_as_root mkdir -p "$INSTALL_DIR/forcefields"
    run_as_root tar -xzf "$CHARMM36_TMP/charmm36-jul2022.ff.tgz" -C "$INSTALL_DIR/forcefields"
    rm -rf "$CHARMM36_TMP"
    [[ -s "$CHARMM36_TARGET/forcefield.itp" ]] \
      || die "CHARMM36m archive did not contain the expected forcefield.itp."
    success "CHARMM36m installed at $CHARMM36_TARGET"
  fi
else
  warn "--without-charmm36: ChemFusion-compatible force field not installed."
fi

echo ""

# ─── 3. Conda setup ───────────────────────────────────────────────────────────
echo -e "${BOLD}[3/6] Setting up Conda environments${RESET}"

if $SKIP_CONDA; then
  warn "--skip-conda: skipping environment setup."
else
  # Find an existing conda, or install Miniforge
  CONDA_BASE=""
  for candidate in \
      "${CONDA_PREFIX:-}" \
      "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/miniconda" \
      "/opt/miniconda" "/opt/miniconda3" "/opt/miniforge3" \
      "/opt/conda" "$HOME/anaconda3"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      CONDA_BASE="$candidate"
      break
    fi
  done

  if [[ -z "$CONDA_BASE" ]]; then
    # /opt/miniconda is where ChemLink's tools look for the gmx_mmpbsa env.
    info "Conda not found — installing Miniforge $MINIFORGE_VERSION to /opt/miniconda ..."
    MINIFORGE_SH="$(mktemp /tmp/miniforge.XXXXXX.sh)"
    wget -q "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
         -O "$MINIFORGE_SH"
    printf '%s  %s\n' "$MINIFORGE_SHA256" "$MINIFORGE_SH" | sha256sum -c --status \
      || die "Miniforge installer checksum verification failed."
    as_system bash "$MINIFORGE_SH" -b -p /opt/miniconda
    rm -f "$MINIFORGE_SH"
    CONDA_BASE="/opt/miniconda"
    success "Miniforge installed at $CONDA_BASE"
  else
    success "Conda found at $CONDA_BASE"
  fi

  CONDA="$CONDA_BASE/bin/conda"
  # ChemLink's environments only use conda-forge and bioconda; Anaconda's
  # 'defaults' channels carry their own terms of service.
  if "$CONDA" config --show channels 2>/dev/null | grep -qx '  - defaults'; then
    warn "This conda lists Anaconda's 'defaults' channel, whose terms of service restrict"
    warn "commercial use. ChemLink does not need it; Miniforge avoids it entirely."
  fi

  # ── bio environment (Python 3.10 + scientific stack) ──────────────────────
  if "$CONDA" env list | grep -q "^bio "; then
    info "Conda env 'bio' already exists — skipping creation."
  else
    info "Creating conda env 'bio' (Python 3.10, MD and docking stack…)"
    # openbabel provides the `obabel` command ChemLink runs as a separate
    # program (pH protonation, format conversion); ChemLink never imports it.
    # 3.1.1 does not add hydrogens to PDBQT input with -h.
    "$CONDA" create -n bio -y --quiet --override-channels -c conda-forge \
      python=3.10 pip \
      ambertools acpype "openbabel>=3.2.1" pdbfixer openmm rdkit meeko scipy gemmi \
      matplotlib networkx tqdm pandas numpy rich prompt-toolkit biopython
    success "Conda env 'bio' ready"
  fi

  BIO_PREFIX="$(conda_env_prefix "$CONDA" bio || true)"
  [[ -n "$BIO_PREFIX" ]] || die "Conda env 'bio' was created but its prefix could not be resolved."
  BIO_PY="$BIO_PREFIX/bin/python"

  # Install ChemLink Python deps inside bio
  info "Installing ChemLink Python requirements into 'bio'…"
  "$BIO_PY" -m pip install --quiet --no-cache-dir \
    "numpy>=1.24" "rich>=13.0" "prompt_toolkit>=3.0" "tqdm>=4.60" "rdkit" 2>/dev/null || true
  # An existing 'bio' env may predate packages ChemLink now imports.
  if ! "$BIO_PY" -c "import rdkit, meeko, scipy, gemmi, pdbfixer, openmm, matplotlib, networkx" 2>/dev/null; then
    info "Adding missing packages to 'bio'…"
    "$CONDA" install -n bio -y --quiet --override-channels -c conda-forge \
      pdbfixer openmm rdkit meeko scipy gemmi matplotlib networkx
  fi
  "$BIO_PY" -c "import rdkit, meeko, scipy, gemmi, pdbfixer, openmm, matplotlib, networkx" \
    || die "ChemLink's Python stack could not be imported in the bio environment."
  OBABEL_VERSION="$("$BIO_PREFIX/bin/obabel" -V 2>/dev/null | awk '{print $3}' || true)"
  if [[ -n "$OBABEL_VERSION" ]] && ! printf '3.2.1\n%s\n' "$OBABEL_VERSION" | sort -V -C; then
    info "Upgrading Open Babel $OBABEL_VERSION in 'bio' (3.2.1+ needed for PDBQT ligands)…"
    "$CONDA" install -n bio -y --quiet --override-channels -c conda-forge "openbabel>=3.2.1" \
      || warn "Open Babel could not be upgraded: PDBQT ligands may lose their hydrogens."
  fi
  [[ -x "$BIO_PREFIX/bin/obabel" ]] \
    || warn "obabel not found in 'bio': PDB/PDBQT ligands cannot be protonated for dynamics."
  # Peptide docking: LightDock (GPL-3.0) and PRODIGY (Apache-2.0) are only
  # published on PyPI. ChemLink runs both as separate programs.
  if [[ ! -x "$BIO_PREFIX/bin/lightdock3_setup.py" || ! -x "$BIO_PREFIX/bin/prodigy" ]]; then
    info "Installing LightDock and PRODIGY into 'bio' (peptide docking)…"
    "$BIO_PY" -m pip install --quiet --no-cache-dir "lightdock>=0.9" "prodigy-prot>=2.4" \
      || warn "LightDock/PRODIGY could not be installed: peptide docking will be unavailable."
  fi
  success "Python dependencies installed"

  # Keep AmberTools/gmx_MMPBSA isolated because it requires Python <3.12 and
  # NumPy <2, which may conflict with the main ChemLink environment.
  if $INSTALL_MMPBSA; then
    # The spec only uses conda-forge/bioconda ('nodefaults'). Anaconda's
    # terms are not accepted on the user's behalf; a Miniconda base that
    # insists on them fails here with conda's own instructions.
    MMPBSA_PREFIX="$(conda_env_prefix "$CONDA" gmx_mmpbsa || true)"
    if [[ -n "$MMPBSA_PREFIX" ]]; then
      info "Conda env 'gmx_mmpbsa' already exists — updating from specification."
      "$CONDA" env update -p "$MMPBSA_PREFIX" -f "$INSTALL_DIR/config/environments/gmx_mmpbsa.yml" --prune
    else
      info "Creating dedicated gmx_MMPBSA environment…"
      if [[ -d "$CONDA_BASE/envs" && -w "$CONDA_BASE/envs" ]]; then
        MMPBSA_PREFIX="$CONDA_BASE/envs/gmx_mmpbsa"
      else
        MMPBSA_PREFIX="$HOME/.conda/envs/gmx_mmpbsa"
        mkdir -p "$(dirname "$MMPBSA_PREFIX")"
      fi
      "$CONDA" env create -p "$MMPBSA_PREFIX" -f "$INSTALL_DIR/config/environments/gmx_mmpbsa.yml"
    fi
    [[ -x "$MMPBSA_PREFIX/bin/python" ]] || die "The gmx_mmpbsa environment could not be created."
    # gmx-MMPBSA 1.6.x pins an old mpi4py release when dependencies are
    # resolved by pip. Use the compatible mpi4py supplied by Conda instead.
    "$MMPBSA_PREFIX/bin/python" -m pip install --quiet --no-deps gmx-MMPBSA
    success "gmx_MMPBSA environment ready"
  else
    info "Skipping optional gmx_MMPBSA environment (use --with-mmpbsa to install)."
  fi
fi

echo ""

# ─── 4. Scientific tools (--full only) ────────────────────────────────────────
echo -e "${BOLD}[4/6] Scientific tools${RESET}"

if ! $INSTALL_TOOLS; then
  warn "Skipping scientific tool installation (fpocket, AutoGrid4, AutoDock-GPU, Vina)."
  warn "Re-run with ${BOLD}--full${RESET} to compile all tools, or install them manually."
  warn "See: $INSTALL_DIR/docs/Installation.md"
else
  need_cmd make
  need_cmd gcc
  need_cmd g++

  # ── fpocket ────────────────────────────────────────────────────────────────
  if have_cmd fpocket; then
    success "fpocket already installed ($(fpocket --version 2>&1 | head -1 || echo 'found'))"
  else
    info "Building fpocket $FPOCKET_VERSION…"
    TMP_DIR="$(mktemp -d)"
    git clone --quiet --depth 1 --branch "$FPOCKET_VERSION" \
      https://github.com/Discngine/fpocket.git "$TMP_DIR/fpocket"
    (cd "$TMP_DIR/fpocket" \
      && sed -i 's/-Werror//g' makefile \
      && sed -i \
        -e 's/char \*residue_string\[/char residue_string[/' \
        -e 's/strcpy(&residue_string, pt);/strcpy(residue_string, pt);/' \
        src/fparams.c \
      && make -j"$(nproc)" \
      && as_system find bin -type f -executable -exec cp {} /usr/local/bin/ \;)
    rm -rf "$TMP_DIR"
    success "fpocket installed"
  fi

  # ── AutoGrid4 + AutoDock4 ──────────────────────────────────────────────────
  if have_cmd autogrid4 && have_cmd autodock4; then
    success "AutoGrid4/AutoDock4 already installed"
  else
    need_cmd autoconf
    info "Building AutoGrid4…"
    TMP_DIR="$(mktemp -d)"
    git clone --quiet https://github.com/ccsb-scripps/AutoGrid.git "$TMP_DIR/autogrid"
    git -C "$TMP_DIR/autogrid" checkout --quiet "$AUTOGRID_COMMIT"
    (cd "$TMP_DIR/autogrid" && autoreconf -i && ./configure --prefix=/usr/local \
      && make -j"$(nproc)" && as_system make install)
    if have_cmd autogrid && ! have_cmd autogrid4; then
      as_system mv /usr/local/bin/autogrid /usr/local/bin/autogrid4
    fi
    rm -rf "$TMP_DIR"

    info "Building AutoDock4…"
    TMP_DIR="$(mktemp -d)"
    git clone --quiet https://github.com/ccsb-scripps/AutoDock4.git "$TMP_DIR/autodock4"
    git -C "$TMP_DIR/autodock4" checkout --quiet "$AUTODOCK4_COMMIT"
    (cd "$TMP_DIR/autodock4" && autoreconf -i && ./configure --prefix=/usr/local \
      && make -j"$(nproc)" && as_system make install)
    rm -rf "$TMP_DIR"
    success "AutoGrid4 + AutoDock4 installed"
  fi

  # ── AutoDock-GPU ───────────────────────────────────────────────────────────
  if have_cmd autodock-gpu; then
    success "AutoDock-GPU already installed"
  elif ! have_cmd nvcc; then
    warn "nvcc not found — skipping AutoDock-GPU. Install CUDA Toolkit and re-run with --full."
  else
    info "Building AutoDock-GPU (sm_86 / sm_89 / sm_90 / sm_120)…"
    CUDA_PATH="$(detect_cuda_path)"
    info "Using CUDA toolkit at $CUDA_PATH"
    TMP_DIR="$(mktemp -d)"
    git clone --quiet https://github.com/ccsb-scripps/AutoDock-GPU.git "$TMP_DIR/autodock-gpu"
    git -C "$TMP_DIR/autodock-gpu" checkout --quiet "$AUTODOCK_GPU_COMMIT"
    (cd "$TMP_DIR/autodock-gpu" && \
      sed -i 's/-std=c++11/-std=c++14/g' Makefile.Cuda && \
      make DEVICE=CUDA NUMWI=128 \
        TARGETS="$AUTODOCK_GPU_TARGETS" \
        CUDA_PATH="$CUDA_PATH" \
        CUDA_INC_PATH="$CUDA_PATH/include" \
        CUDA_LIB_PATH="$CUDA_PATH/lib64" \
        GPU_INCLUDE_PATH="$CUDA_PATH/include" \
        GPU_LIBRARY_PATH="$CUDA_PATH/lib64" \
        -j"$(nproc)" && \
      as_system cp bin/autodock_gpu_128wi /usr/local/bin/autodock-gpu && \
      as_system chmod +x /usr/local/bin/autodock-gpu)
    rm -rf "$TMP_DIR"
    success "AutoDock-GPU installed"
  fi

  # ── AutoDock Vina (prebuilt static binaries; CPU engine) ─────────────────
  if have_cmd vina; then
    success "AutoDock Vina already installed"
  else
    VINA_VERSION=1.2.5
    VINA_URL="https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}"
    TMP_DIR="$(mktemp -d)"
    if wget -q "$VINA_URL/vina_${VINA_VERSION}_linux_x86_64" -O "$TMP_DIR/vina" \
      && wget -q "$VINA_URL/vina_split_${VINA_VERSION}_linux_x86_64" -O "$TMP_DIR/vina_split"; then
      as_system install -m 755 "$TMP_DIR/vina" "$TMP_DIR/vina_split" /usr/local/bin/
      success "AutoDock Vina $VINA_VERSION installed"
    else
      warn "Could not download AutoDock Vina; the CPU docking engine will be unavailable."
    fi
    rm -rf "$TMP_DIR"
  fi
fi

# ── GROMACS (only with --with-gromacs or --full) ───────────────────────────
if $INSTALL_GROMACS; then
  if have_cmd gmx; then
    success "Serial/thread-MPI GROMACS already installed"
  else
    if have_cmd gmx_mpi; then
      warn "Only gmx_mpi was found; building the gmx executable required by parallel gmx_MMPBSA."
    fi
    need_cmd cmake || die "cmake is required to build GROMACS. Install with: apt install cmake"
    info "Building GROMACS 2025.4 with the gmx executable (this may take 30–60 minutes)…"
    CUDA_PATH="$(detect_cuda_path)"
    CUDA_HOST_COMPILER="$(detect_cuda_host_compiler)" \
      || die "No compatible C++ compiler found for CUDA."
    info "Using CUDA toolkit at $CUDA_PATH"
    info "Using CUDA host compiler at $CUDA_HOST_COMPILER"
    TMP_DIR="$(mktemp -d)"
    wget -q https://ftp.gromacs.org/gromacs/gromacs-2025.4.tar.gz -O "$TMP_DIR/gromacs.tar.gz"
    tar -xzf "$TMP_DIR/gromacs.tar.gz" -C "$TMP_DIR"
    mkdir -p "$TMP_DIR/gromacs-2025.4/build"
    (cd "$TMP_DIR/gromacs-2025.4/build" && \
      cmake .. \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DREGRESSIONTEST_DOWNLOAD=OFF \
        -DGMX_GPU=CUDA \
        -DGMX_MPI=OFF \
        -DGMX_SIMD=AVX2_256 \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER" \
        -DGMX_CUDA_TARGET_SM="75;80;86;89;90;100;120" \
        -DGMX_CUDA_TARGET_COMPUTE="90" && \
      make -j"$(nproc)" && as_system make install)
    as_system bash -c \
      'echo "source /usr/local/gromacs/bin/GMXRC" >> /etc/bash.bashrc' || true
    rm -rf "$TMP_DIR"
    success "GROMACS 2025.4 installed at /usr/local/gromacs"
  fi
fi

echo ""

# ─── 5. CLI entry point ────────────────────────────────────────────────────────
echo -e "${BOLD}[5/6] Installing chemlink command${RESET}"

# Resolve conda base for the entry point
if $SKIP_CONDA; then
  PYTHON_BIN="$(command -v python3 || echo python3)"
else
  CONDA_BASE="${CONDA_BASE:-}"
  if [[ -z "$CONDA_BASE" ]]; then
    for candidate in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/miniconda" "/opt/miniconda" "/opt/miniforge3" "/opt/conda"; do
      [[ -x "${candidate}/bin/conda" ]] && CONDA_BASE="$candidate" && break
    done
  fi
  CONDA="${CONDA_BASE:+$CONDA_BASE/bin/conda}"
  BIO_PREFIX=""
  if [[ -n "$CONDA" && -x "$CONDA" ]]; then
    BIO_PREFIX="$(conda_env_prefix "$CONDA" bio || true)"
  fi
  PYTHON_BIN="${BIO_PREFIX:+$BIO_PREFIX/bin/python}"
  [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || echo python3)"
fi

GMXRC_LINE='[ -f /usr/local/gromacs/bin/GMXRC ] && source /usr/local/gromacs/bin/GMXRC'

PARENT_DIR="$(dirname "${INSTALL_DIR}")"
RUNTIME_BIN_PATH=""
if [[ -n "${BIO_PREFIX:-}" ]]; then
  RUNTIME_BIN_PATH+="${BIO_PREFIX}/bin:"
fi
# gmx_MMPBSA is searched in a few fixed prefixes; point at the real env when
# conda lives elsewhere (an exported value still wins).
MMPBSA_EXPORTS=""
if [[ -n "${CONDA:-}" && -x "${CONDA:-}" ]]; then
  MMPBSA_PREFIX="$(conda_env_prefix "$CONDA" gmx_mmpbsa || true)"
  if [[ -n "$MMPBSA_PREFIX" && -x "$MMPBSA_PREFIX/bin/gmx_MMPBSA" ]]; then
    MMPBSA_EXPORTS="export CHEMLINK_MMPBSA_BIN=\"\${CHEMLINK_MMPBSA_BIN:-${MMPBSA_PREFIX}/bin/gmx_MMPBSA}\"
"
    [[ -x "$MMPBSA_PREFIX/bin/mpirun" ]] && MMPBSA_EXPORTS+="export CHEMLINK_MMPBSA_MPIRUN=\"\${CHEMLINK_MMPBSA_MPIRUN:-${MMPBSA_PREFIX}/bin/mpirun}\"
"
  fi
fi
ENTRY_POINT_CONTENT="#!/usr/bin/env bash
export PYTHONPATH=\"${PARENT_DIR}:${INSTALL_DIR}:\${PYTHONPATH:-}\"
${MMPBSA_EXPORTS}export PATH=\"${RUNTIME_BIN_PATH}\${PATH}\"
set +u; ${GMXRC_LINE}; set -u
exec \"${PYTHON_BIN}\" -m chemlink.cli.main \"\$@\""

# Choose bin dir: /usr/local/bin (system) or ~/.local/bin (user)
if [[ "$EUID" -eq 0 ]] || $USE_SUDO; then
  BIN_DIR="/usr/local/bin"
  ENTRY="/usr/local/bin/chemlink"
  echo "$ENTRY_POINT_CONTENT" | run_as_root tee "$ENTRY" > /dev/null
  run_as_root chmod +x "$ENTRY"
else
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
  ENTRY="$BIN_DIR/chemlink"
  echo "$ENTRY_POINT_CONTENT" > "$ENTRY"
  chmod +x "$ENTRY"
fi

success "chemlink command installed at $ENTRY"

# Ensure ~/.local/bin is on PATH (user installs)
if [[ "$BIN_DIR" == "$HOME/.local/bin" ]] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  warn "Add $HOME/.local/bin to your PATH to use chemlink anywhere:"
  warn "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
fi

echo ""

# ─── 6. Lmod module (optional, if SLURM cluster) ──────────────────────────────
echo -e "${BOLD}[6/6] Cluster module (optional)${RESET}"

LMOD_MODULEPATH="${MODULEPATH:-}"
if have_cmd module && [[ -n "$LMOD_MODULEPATH" ]]; then
  LMOD_DIR="$(echo "$LMOD_MODULEPATH" | tr ':' '\n' | head -1)/chemlink"
  info "Lmod detected — writing module file to $LMOD_DIR/1.0.lua"
  run_as_root mkdir -p "$LMOD_DIR" 2>/dev/null || mkdir -p "$LMOD_DIR" 2>/dev/null || true
  if [[ -d "$LMOD_DIR" ]]; then
    cat > "$LMOD_DIR/1.0.lua" <<LUAEOF
help([[
ChemLink v1.0 — Molecular Docking & Dynamics Orchestration Platform
]])
whatis("Name:    ChemLink")
whatis("Version: 1.0")
whatis("URL:     https://github.com/athalea-software/chemlink")

prepend_path("PATH",            "${BIN_DIR}")
prepend_path("PYTHONPATH",      "${INSTALL_DIR}")
setenv("CHEMLINK_HOME",         "${INSTALL_DIR}")
LUAEOF
    success "Lmod module chemlink/1.0 installed"
  fi
elif have_cmd sinfo || have_cmd sbatch; then
  warn "SLURM detected but Lmod not found. To use chemlink on all nodes, add $INSTALL_DIR"
  warn "to your Conda/module setup or run: module load chemlink/1.0 (after Lmod setup)."
else
  info "Not a SLURM cluster — skipping Lmod module."
fi

echo ""

# ─── Done ─────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}  ChemLink installed successfully!${RESET}"
echo ""
echo -e "  ${BOLD}Quick start:${RESET}"
echo -e "    ${CYAN}chemlink doctor${RESET}          verify environment and dependencies"
echo -e "    ${CYAN}chemlink docking --help${RESET}  docking pipeline options"
echo -e "    ${CYAN}chemlink dynamics --help${RESET} dynamics pipeline options"
echo ""
echo -e "  ${BOLD}Install location:${RESET} $INSTALL_DIR"
echo -e "  ${BOLD}Docs:${RESET}             $INSTALL_DIR/docs/"
echo -e "  ${BOLD}GitHub:${RESET}           https://github.com/athalea-software/chemlink"
echo ""
if ! $INSTALL_TOOLS; then
  echo -e "  ${YELLOW}Tip:${RESET} scientific tools (fpocket, AutoDock-GPU) were not compiled."
  echo -e "       Re-run with ${BOLD}--full${RESET} to build them, or install pre-built packages manually."
  echo ""
fi
