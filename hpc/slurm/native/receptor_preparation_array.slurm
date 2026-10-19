#!/bin/bash
#SBATCH --job-name=chemlink_receptor_prep
#SBATCH --output=logs/receptor_prep_%A_%a.out
#SBATCH --error=logs/receptor_prep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-5

set -euo pipefail

# --- Lmod ---
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# --- Rutas ---
INPUT_DIR="${INPUT_DIR:-/nfs/chemlink/chemlink/data/input/receptors}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/chemlink/chemlink/data/input}"
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/chemlink/miniconda/envs/bio/bin/python}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p logs

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Array task: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "[$(date)] Input: ${INPUT_DIR}"
echo "[$(date)] Output: ${OUTPUT_DIR}"
echo "[$(date)] Local workers: ${N_WORKERS}"

cd "${REPO_DIR}"
export PYTHONPATH="/nfs/chemlink:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m chemlink.cli.main docking receptor \
  -ir "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  --workers "${N_WORKERS}"

echo "[$(date)] Task completed"
