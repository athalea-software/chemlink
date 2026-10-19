#!/bin/bash
#SBATCH --job-name=chemlink_ligand_prep
#SBATCH --output=logs/ligand_prep_%A_%a.out
#SBATCH --error=logs/ligand_prep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-5

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/input/ligands}"
OUTPUT_DIR="${OUTPUT_DIR:-data/input}"
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT:-/nfs/chemlink}"

mkdir -p logs

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Array task: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "[$(date)] Input: ${INPUT_DIR}"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "[$(date)] Local workers: ${N_WORKERS}"

cd "${REPO_DIR}"

if [[ -n "${CONTAINER_IMAGE}" ]]; then
  if [[ "${CONTAINER_RUNTIME}" == "apptainer" || "${CONTAINER_RUNTIME}" == "singularity" ]]; then
    "${CONTAINER_RUNTIME}" exec \
      --bind "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      --env "PYTHONPATH=/nfs/chemlink:${PYTHONPATH:-}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" -m chemlink.cli.main docking ligand \
      -il "${INPUT_DIR}" \
      -o "${OUTPUT_DIR}" \
      --workers "${N_WORKERS}"
  else
    "${CONTAINER_RUNTIME}" run --rm \
      --user "$(id -u):$(id -g)" \
      -v "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      -w "${REPO_DIR}" \
      -e "PYTHONPATH=/nfs/chemlink:${PYTHONPATH:-}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" -m chemlink.cli.main docking ligand \
      -il "${INPUT_DIR}" \
      -o "${OUTPUT_DIR}" \
      --workers "${N_WORKERS}"
  fi
else
  export PYTHONPATH="/nfs/chemlink:${PYTHONPATH:-}"
  "${PYTHON_BIN}" -m chemlink.cli.main docking ligand \
    -il "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    --workers "${N_WORKERS}"
fi

echo "[$(date)] Task completed"
