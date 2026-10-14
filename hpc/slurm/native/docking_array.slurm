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

# --- Lmod ---
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# --- Rutas ---
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
RUN_DIR="${RUN_DIR:-/nfs/chemlink/runs/test_run}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/chemlink/miniconda/envs/bio/bin/python}"
WORKERS="${WORKERS:-1}"
AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE:-/nfs/chemlink/software/autogrid4/bin/autogrid4}"
AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE:-/nfs/chemlink/software/autodock-gpu/bin/autodock-gpu}"
export AUTOGRID4_BIN="${AUTOGRID_EXECUTABLE}" AUTODOCK_GPU_BIN="${AUTODOCK_GPU_EXECUTABLE}"

PREPARED_RECEPTORS_DIR="${PREPARED_RECEPTORS_DIR:-${RUN_DIR}/output/prepared_receptors_pdbqt}"
BATCH_ROOT="${RUN_DIR}/ligand_batches"
TASK_OUT_ROOT="${RUN_DIR}/docking_tasks"

BATCH_DIR="${BATCH_ROOT}/batch_${SLURM_ARRAY_TASK_ID}"
BATCH_READY="${BATCH_ROOT}/ready_${SLURM_ARRAY_TASK_ID}"
TASK_OUT_DIR="${TASK_OUT_ROOT}/task_${SLURM_ARRAY_TASK_ID}"

mkdir -p logs "${TASK_OUT_DIR}"

# Wait for the per-batch sentinel file written by prepare_ligand_batches.slurm.
# The sentinel is written AFTER all symlinks for this batch are in place, so
# seeing it guarantees the batch directory is fully populated.  Critically, this
# prevents the task from ever accessing BATCH_DIR before it is complete — which
# would cause the NFS client to cache stale empty directory entries (observed on
# NFSv4.2 with kernel 6.17 at batch sizes ≥ 334 where directory creation and
# filling span ~1s and the attribute cache mtime doesn't advance visibly to the
# remote node within acdirmax).
_WAIT_MAX=300
_waited=0
while [[ ! -f "${BATCH_READY}" ]]; do
  if (( _waited >= _WAIT_MAX )); then
    echo "[$(date)] No batch sentinel for task ${SLURM_ARRAY_TASK_ID} after ${_WAIT_MAX}s; skipping"
    exit 0
  fi
  sleep 5; _waited=$((_waited + 5))
done

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "[$(date)] Batch dir ${BATCH_DIR} missing after sentinel appeared; skipping"
  exit 0
fi

# Short fallback loop (≤90s) for any residual NFS attribute cache (acdirmax≤60s).
# The sentinel guarantees all symlinks are written; in practice this loop exits
# immediately on the first iteration since BATCH_DIR is read fresh.
_waited=0
while ! find "${BATCH_DIR}" -maxdepth 1 \( -type f -o -type l \) -name "*.pdbqt" | grep -q .; do
  if (( _waited >= 90 )); then
    echo "[$(date)] Empty batch dir ${BATCH_DIR} after 90s (sentinel existed); skipping"
    exit 0
  fi
  sleep 5; _waited=$((_waited + 5))
done

# AutoGrid4 generates .map files and writes .maps.fld last as a completion marker.
# All docking array tasks start simultaneously, but only task 0 should trigger
# AutoGrid4 (if maps are not yet built). Tasks 1+ wait for .maps.fld to appear,
# guaranteeing all .map files are fully written before they start reading them.
_MAPS_FLD=$(find "${PREPARED_RECEPTORS_DIR}" -maxdepth 1 -name "*.maps.fld" 2>/dev/null | head -1 || true)
if [[ -z "${_MAPS_FLD}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
  _MAP_WAIT=600
  _waited=0
  echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: waiting for task 0 to build receptor maps..."
  while ! find "${PREPARED_RECEPTORS_DIR}" -maxdepth 1 -name "*.maps.fld" -size +0c | grep -q .; do
    if (( _waited >= _MAP_WAIT )); then
      echo "[$(date)] ERROR: receptor maps.fld not found after ${_MAP_WAIT}s; task 0 may have failed"
      exit 1
    fi
    sleep 5; _waited=$((_waited + 5))
  done
  echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: receptor maps ready after ${_waited}s"
fi

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Task: ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Batch: ${BATCH_DIR}"
echo "[$(date)] Receptors: ${PREPARED_RECEPTORS_DIR}"
echo "[$(date)] Output: ${TASK_OUT_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="/nfs/chemlink:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m chemlink.cli.main docking run \
  -ir "${PREPARED_RECEPTORS_DIR}" \
  -il "${BATCH_DIR}" \
  -o "${TASK_OUT_DIR}" \
  --workers "${WORKERS}"

echo "[$(date)] Docking task ${SLURM_ARRAY_TASK_ID} completed"
