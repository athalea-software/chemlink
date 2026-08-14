#!/bin/bash
#SBATCH --job-name=chemlink_active_site
#SBATCH --output=logs/active_site_%j.out
#SBATCH --error=logs/active_site_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

# --- Lmod ---
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# --- Rutas ---
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/chemlink/runs/test_run/output}"
FPOCKET_PATH="${FPOCKET_PATH:-/nfs/chemlink/software/fpocket-4.2.3/bin/fpocket}"
ACTIVE_SITE_WORKERS="${ACTIVE_SITE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/chemlink/miniconda/envs/bio/bin/python}"
MANUAL_CENTER="${MANUAL_CENTER:-}"
MANUAL_NPTS="${MANUAL_NPTS:-}"
BOX_CONFIG_PATH="${BOX_CONFIG_PATH:-}"
FLEX_RADIUS="${FLEX_RADIUS:-0.0}"

mkdir -p logs

RECEPTOR_DIR="${OUTPUT_DIR}/prepared_receptors_pdbqt"
LIGAND_DIR="${OUTPUT_DIR}/prepared_ligands_pdbqt"

cd "${REPO_DIR}"
export PYTHONPATH="/nfs/chemlink:${PYTHONPATH:-}"

# Build optional manual-box / box-config arguments
MANUAL_ARGS=()
if [[ -n "${MANUAL_CENTER}" ]]; then
  read -ra _mc <<< "${MANUAL_CENTER}"
  MANUAL_ARGS=(--manual-center "${_mc[@]}")
fi
if [[ -n "${MANUAL_NPTS}" ]]; then
  read -ra _mn <<< "${MANUAL_NPTS}"
  MANUAL_ARGS+=(--manual-npts "${_mn[@]}")
fi
if [[ -n "${BOX_CONFIG_PATH}" ]]; then
  MANUAL_ARGS+=(--box-config "${BOX_CONFIG_PATH}")
fi
MANUAL_ARGS+=(--flex-radius "${FLEX_RADIUS}")

"${PYTHON_BIN}" -m chemlink.cli.main docking active-site \
  "${RECEPTOR_DIR}" \
  "${LIGAND_DIR}" \
  "${OUTPUT_DIR}" \
  --fpocket-path "${FPOCKET_PATH}" \
  --workers "${ACTIVE_SITE_WORKERS}" \
  "${MANUAL_ARGS[@]}"

echo "[$(date)] Active-site completed"
