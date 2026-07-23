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

REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/chemlink/runs/test_run/output}"
FPOCKET_PATH="${FPOCKET_PATH:-/usr/bin/fpocket}"
ACTIVE_SITE_WORKERS="${ACTIVE_SITE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT:-/nfs/chemlink}"

mkdir -p logs

RECEPTOR_DIR="${OUTPUT_DIR}/prepared_receptors_pdbqt"
LIGAND_DIR="${OUTPUT_DIR}/prepared_ligands_pdbqt"

cd "${REPO_DIR}"

if [[ -n "${CONTAINER_IMAGE}" ]]; then
  if [[ "${CONTAINER_RUNTIME}" == "apptainer" || "${CONTAINER_RUNTIME}" == "singularity" ]]; then
    "${CONTAINER_RUNTIME}" exec \
      --bind "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      --env "PYTHONPATH=/nfs/chemlink:${PYTHONPATH:-}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" -m chemlink.cli.main docking active-site \
      "${RECEPTOR_DIR}" \
      "${LIGAND_DIR}" \
      "${OUTPUT_DIR}" \
      --fpocket-path "${FPOCKET_PATH}" \
      --workers "${ACTIVE_SITE_WORKERS}"
  else
    "${CONTAINER_RUNTIME}" run --rm \
      --user "$(id -u):$(id -g)" \
      -v "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      -w "${REPO_DIR}" \
      -e "PYTHONPATH=/nfs/chemlink:${PYTHONPATH:-}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" -m chemlink.cli.main docking active-site \
      "${RECEPTOR_DIR}" \
      "${LIGAND_DIR}" \
      "${OUTPUT_DIR}" \
      --fpocket-path "${FPOCKET_PATH}" \
      --workers "${ACTIVE_SITE_WORKERS}"
  fi
else
  export PYTHONPATH="/nfs/chemlink:${PYTHONPATH:-}"
  "${PYTHON_BIN}" -m chemlink.cli.main docking active-site \
    "${RECEPTOR_DIR}" \
    "${LIGAND_DIR}" \
    "${OUTPUT_DIR}" \
    --fpocket-path "${FPOCKET_PATH}" \
    --workers "${ACTIVE_SITE_WORKERS}"
fi

echo "[$(date)] Active-site completed"
