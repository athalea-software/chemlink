#!/bin/bash
#SBATCH --job-name=chemlink_docking_array
#SBATCH --output=logs/docking_%A_%a.out
#SBATCH --error=logs/docking_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
RUN_DIR="${RUN_DIR:-/nfs/chemlink/runs/test_run}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-1}"
AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE:-/usr/local/bin/autogrid4}"
AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE:-/usr/local/bin/autodock-gpu}"
export AUTOGRID4_BIN="${AUTOGRID_EXECUTABLE}" AUTODOCK_GPU_BIN="${AUTODOCK_GPU_EXECUTABLE}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT:-/nfs/chemlink}"

PREPARED_RECEPTORS_DIR="${PREPARED_RECEPTORS_DIR:-${RUN_DIR}/output/prepared_receptors_pdbqt}"
BATCH_ROOT="${RUN_DIR}/ligand_batches"
TASK_OUT_ROOT="${RUN_DIR}/docking_tasks"

BATCH_DIR="${BATCH_ROOT}/batch_${SLURM_ARRAY_TASK_ID}"
TASK_OUT_DIR="${TASK_OUT_ROOT}/task_${SLURM_ARRAY_TASK_ID}"

mkdir -p logs "${TASK_OUT_DIR}"

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "[$(date)] No batch assigned for task ${SLURM_ARRAY_TASK_ID}; skipping"
  exit 0
fi

if ! find "${BATCH_DIR}" -maxdepth 1 \( -type f -o -type l \) -name "*.pdbqt" | grep -q .; then
  echo "[$(date)] Empty batch dir ${BATCH_DIR}; skipping"
  exit 0
fi

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Task: ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Batch: ${BATCH_DIR}"
echo "[$(date)] Receptors: ${PREPARED_RECEPTORS_DIR}"
echo "[$(date)] Output: ${TASK_OUT_DIR}"

cd "${REPO_DIR}"
if [[ -n "${CONTAINER_IMAGE}" ]]; then
  if [[ "${CONTAINER_RUNTIME}" == "apptainer" || "${CONTAINER_RUNTIME}" == "singularity" ]]; then
    "${CONTAINER_RUNTIME}" exec --nv \
      --bind "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" cli/main.py docking run \
      -ir "${PREPARED_RECEPTORS_DIR}" \
      -il "${BATCH_DIR}" \
      -o "${TASK_OUT_DIR}" \
      --workers "${WORKERS}"
  else
    "${CONTAINER_RUNTIME}" run --rm --gpus all \
      -e "AUTOGRID4_BIN=${AUTOGRID4_BIN}" -e "AUTODOCK_GPU_BIN=${AUTODOCK_GPU_BIN}" \
      --user "$(id -u):$(id -g)" \
      -v "${CONTAINER_MOUNT_ROOT}:${CONTAINER_MOUNT_ROOT}" \
      -w "${REPO_DIR}" \
      "${CONTAINER_IMAGE}" \
      "${PYTHON_BIN}" cli/main.py docking run \
      -ir "${PREPARED_RECEPTORS_DIR}" \
      -il "${BATCH_DIR}" \
      -o "${TASK_OUT_DIR}" \
      --workers "${WORKERS}"
  fi
else
  "${PYTHON_BIN}" cli/main.py docking run \
    -ir "${PREPARED_RECEPTORS_DIR}" \
    -il "${BATCH_DIR}" \
    -o "${TASK_OUT_DIR}" \
    --workers "${WORKERS}"
fi

echo "[$(date)] Docking task ${SLURM_ARRAY_TASK_ID} completed"
