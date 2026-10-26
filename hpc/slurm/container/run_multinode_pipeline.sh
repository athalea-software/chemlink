#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-/nfs/chemlink/runs/${RUN_ID}}"
RUN_NAME_CUSTOM="${RUN_NAME_CUSTOM:-0}"

INPUT_RECEPTORS_DIR="${INPUT_RECEPTORS_DIR:-/nfs/chemlink/data/input/receptors}"
INPUT_LIGANDS_DIR="${INPUT_LIGANDS_DIR:-/nfs/chemlink/data/input/ligands}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/output}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

FPOCKET_PATH="${FPOCKET_PATH:-/usr/local/bin/fpocket}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PREP_ARRAY_RANGE="${PREP_ARRAY_RANGE:-0-5}"
RECEPTOR_WORKERS="${RECEPTOR_WORKERS:-4}"
LIGAND_WORKERS="${LIGAND_WORKERS:-8}"
ACTIVE_SITE_WORKERS="${ACTIVE_SITE_WORKERS:-4}"

RECEPTOR_CPUS_PER_TASK="${RECEPTOR_CPUS_PER_TASK:-4}"
LIGAND_CPUS_PER_TASK="${LIGAND_CPUS_PER_TASK:-8}"
ACTIVE_SITE_CPUS_PER_TASK="${ACTIVE_SITE_CPUS_PER_TASK:-4}"
BATCH_CPUS_PER_TASK="${BATCH_CPUS_PER_TASK:-1}"
DOCKING_CPUS_PER_TASK="${DOCKING_CPUS_PER_TASK:-4}"
MERGE_CPUS_PER_TASK="${MERGE_CPUS_PER_TASK:-2}"
ANALYSIS_CPUS_PER_TASK="${ANALYSIS_CPUS_PER_TASK:-2}"

RECEPTOR_MEM="${RECEPTOR_MEM:-4G}"
LIGAND_MEM="${LIGAND_MEM:-8G}"
ACTIVE_SITE_MEM="${ACTIVE_SITE_MEM:-8G}"
BATCH_MEM="${BATCH_MEM:-2G}"
DOCKING_MEM="${DOCKING_MEM:-16G}"
MERGE_MEM="${MERGE_MEM:-4G}"
ANALYSIS_MEM="${ANALYSIS_MEM:-4G}"
SLURM_MEM_OVERRIDE="${SLURM_MEM_OVERRIDE:-}"

BATCH_SIZE="${BATCH_SIZE:-200}"
MAX_GPU_CONCURRENCY="${MAX_GPU_CONCURRENCY:-6}"
DOCKING_WORKERS="${DOCKING_WORKERS:-1}"
AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE:-/usr/local/bin/autogrid4}"
AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE:-/usr/local/bin/autodock-gpu}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT:-/nfs/chemlink}"
DOCKING_GRES="${DOCKING_GRES:-}"
ALLOW_HOST_PYTHON="${ALLOW_HOST_PYTHON:-0}"

# Optional manual grid box (bypasses fpocket when both are set)
MANUAL_CENTER="${MANUAL_CENTER:-}"
MANUAL_NPTS="${MANUAL_NPTS:-}"
# Optional per-receptor grid box config (JSON on shared storage; overrides
# MANUAL_CENTER/MANUAL_NPTS and fpocket for the receptors it lists)
BOX_CONFIG_PATH="${BOX_CONFIG_PATH:-}"
FLEX_RADIUS="${FLEX_RADIUS:-0.0}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_NODELIST="${SLURM_NODELIST:-}"

if [[ -n "${SLURM_MEM_OVERRIDE}" ]]; then
  RECEPTOR_MEM="${SLURM_MEM_OVERRIDE}"
  LIGAND_MEM="${SLURM_MEM_OVERRIDE}"
  ACTIVE_SITE_MEM="${SLURM_MEM_OVERRIDE}"
  BATCH_MEM="${SLURM_MEM_OVERRIDE}"
  DOCKING_MEM="${SLURM_MEM_OVERRIDE}"
  MERGE_MEM="${SLURM_MEM_OVERRIDE}"
  ANALYSIS_MEM="${SLURM_MEM_OVERRIDE}"
fi

if [[ -z "${CONTAINER_IMAGE}" && "${ALLOW_HOST_PYTHON}" != "1" ]]; then
  echo "[ERROR] CONTAINER_IMAGE is empty. This pipeline expects containerized execution." >&2
  echo "        Set CONTAINER_IMAGE (e.g. registry.example.org:5000/chemlink:2.0.0)" >&2
  echo "        or set ALLOW_HOST_PYTHON=1 to run with host Python environment." >&2
  exit 1
fi

if [[ "${RUN_NAME_CUSTOM}" == "1" && -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Run directory already exists: ${RUN_DIR}. Choose another --run-name." >&2
  exit 1
fi
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${REPO_DIR}"

COMMON_SBATCH_ARGS=()
if [[ -n "${SLURM_PARTITION}" ]]; then
  COMMON_SBATCH_ARGS+=("--partition=${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_ACCOUNT}" ]]; then
  COMMON_SBATCH_ARGS+=("--account=${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_QOS}" ]]; then
  COMMON_SBATCH_ARGS+=("--qos=${SLURM_QOS}")
fi
if [[ -n "${SLURM_CONSTRAINT}" ]]; then
  COMMON_SBATCH_ARGS+=("--constraint=${SLURM_CONSTRAINT}")
fi
if [[ -n "${SLURM_EXCLUDE}" ]]; then
  COMMON_SBATCH_ARGS+=("--exclude=${SLURM_EXCLUDE}")
fi
if [[ -n "${SLURM_NODELIST}" ]]; then
  COMMON_SBATCH_ARGS+=("--nodelist=${SLURM_NODELIST}")
fi

DOCKING_SBATCH_ARGS=()
if [[ -n "${DOCKING_GRES}" && "${DOCKING_GRES}" != "none" ]]; then
  DOCKING_SBATCH_ARGS+=("--gres=${DOCKING_GRES}")
fi

mapfile -t ligand_candidates < <(find "${INPUT_LIGANDS_DIR}" -maxdepth 1 -type f \( -name "*.sdf" -o -name "*.mol2" -o -name "*.pdb" -o -name "*.mol" -o -name "*.pdbqt" \) | sort)
ligand_count=${#ligand_candidates[@]}
if [[ "${ligand_count}" -eq 0 ]]; then
  echo "[ERROR] No ligands found in ${INPUT_LIGANDS_DIR}" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "[ERROR] BATCH_SIZE must be > 0" >&2
  exit 1
fi

total_batches=$(( (ligand_count + BATCH_SIZE - 1) / BATCH_SIZE ))
array_expr="0-$((total_batches - 1))%${MAX_GPU_CONCURRENCY}"

echo "Run ID: ${RUN_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Ligands detected: ${ligand_count}"
echo "Batch size: ${BATCH_SIZE}"
echo "Total docking batches: ${total_batches}"
echo "Docking array: ${array_expr}"
echo "SLURM partition: ${SLURM_PARTITION:-<default>}"
if [[ -n "${DOCKING_GRES}" && "${DOCKING_GRES}" != "none" ]]; then
  echo "DOCKING GRES: ${DOCKING_GRES}"
else
  echo "DOCKING GRES: <none>"
fi
if [[ -n "${CONTAINER_IMAGE}" ]]; then
  echo "CONTAINER IMAGE: ${CONTAINER_IMAGE}"
else
  echo "CONTAINER IMAGE: <none> (host python mode)"
fi
echo "Manual grid center: ${MANUAL_CENTER:-<fpocket>}"
echo "Manual grid npts: ${MANUAL_NPTS:-<fpocket>}"
echo "Box config: ${BOX_CONFIG_PATH:-<none>}"
echo "Flexible radius: ${FLEX_RADIUS} A"
echo "CPUs/task [receptor,ligand,active,batch,docking,merge,analysis]: ${RECEPTOR_CPUS_PER_TASK},${LIGAND_CPUS_PER_TASK},${ACTIVE_SITE_CPUS_PER_TASK},${BATCH_CPUS_PER_TASK},${DOCKING_CPUS_PER_TASK},${MERGE_CPUS_PER_TASK},${ANALYSIS_CPUS_PER_TASK}"
echo "Mem [receptor,ligand,active,batch,docking,merge,analysis]: ${RECEPTOR_MEM},${LIGAND_MEM},${ACTIVE_SITE_MEM},${BATCH_MEM},${DOCKING_MEM},${MERGE_MEM},${ANALYSIS_MEM}"

jid_receptor=$(INPUT_DIR="${INPUT_RECEPTORS_DIR}" \
  REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  N_WORKERS="${RECEPTOR_WORKERS}" \
  CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME}" \
  CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --array="${PREP_ARRAY_RANGE}" \
  --cpus-per-task="${RECEPTOR_CPUS_PER_TASK}" \
  --mem="${RECEPTOR_MEM}" \
  hpc/slurm/container/receptor_preparation_array.slurm)

jid_ligand=$(INPUT_DIR="${INPUT_LIGANDS_DIR}" \
  REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  N_WORKERS="${LIGAND_WORKERS}" \
  CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME}" \
  CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --array="${PREP_ARRAY_RANGE}" \
  --cpus-per-task="${LIGAND_CPUS_PER_TASK}" \
  --mem="${LIGAND_MEM}" \
  hpc/slurm/container/ligand_preparation_array.slurm)

jid_active=$(REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  FPOCKET_PATH="${FPOCKET_PATH}" \
  ACTIVE_SITE_WORKERS="${ACTIVE_SITE_WORKERS}" \
  MANUAL_CENTER="${MANUAL_CENTER}" \
  MANUAL_NPTS="${MANUAL_NPTS}" \
  BOX_CONFIG_PATH="${BOX_CONFIG_PATH}" \
  FLEX_RADIUS="${FLEX_RADIUS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME}" \
  CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_receptor}:${jid_ligand}" \
  --cpus-per-task="${ACTIVE_SITE_CPUS_PER_TASK}" \
  --mem="${ACTIVE_SITE_MEM}" \
  --output="${LOG_DIR}/active_site_%j.out" \
  --error="${LOG_DIR}/active_site_%j.err" \
  hpc/slurm/container/active_site.slurm)

jid_batch=$(RUN_DIR="${RUN_DIR}" \
  PREPARED_LIGANDS_DIR="${OUTPUT_DIR}/prepared_ligands_pdbqt" \
  BATCH_SIZE="${BATCH_SIZE}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_ligand}" \
  --cpus-per-task="${BATCH_CPUS_PER_TASK}" \
  --mem="${BATCH_MEM}" \
  --output="${LOG_DIR}/batch_ligands_%j.out" \
  --error="${LOG_DIR}/batch_ligands_%j.err" \
  hpc/slurm/container/prepare_ligand_batches.slurm)

jid_docking=$(REPO_DIR="${REPO_DIR}" \
  RUN_DIR="${RUN_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  WORKERS="${DOCKING_WORKERS}" \
  PREPARED_RECEPTORS_DIR="${OUTPUT_DIR}/prepared_receptors_pdbqt" \
  AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE}" \
  AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE}" \
  CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME}" \
  CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  "${DOCKING_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_active}:${jid_batch}" \
  --array="${array_expr}" \
  --cpus-per-task="${DOCKING_CPUS_PER_TASK}" \
  --mem="${DOCKING_MEM}" \
  --output="${LOG_DIR}/docking_%A_%a.out" \
  --error="${LOG_DIR}/docking_%A_%a.err" \
  hpc/slurm/container/docking_array.slurm)

jid_merge=$(RUN_DIR="${RUN_DIR}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_docking}" \
  --cpus-per-task="${MERGE_CPUS_PER_TASK}" \
  --mem="${MERGE_MEM}" \
  --output="${LOG_DIR}/merge_%j.out" \
  --error="${LOG_DIR}/merge_%j.err" \
  hpc/slurm/container/merge_docking_results.slurm)

jid_analysis=$(REPO_DIR="${REPO_DIR}" \
  ANALYSIS_OUTPUT_DIR="${RUN_DIR}/merged_output" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME}" \
  CONTAINER_MOUNT_ROOT="${CONTAINER_MOUNT_ROOT}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_merge}" \
  --cpus-per-task="${ANALYSIS_CPUS_PER_TASK}" \
  --mem="${ANALYSIS_MEM}" \
  --output="${LOG_DIR}/analysis_%j.out" \
  --error="${LOG_DIR}/analysis_%j.err" \
  hpc/slurm/container/analysis.slurm)

echo
echo "Submitted jobs:"
echo "  receptor prep : ${jid_receptor}"
echo "  ligand prep   : ${jid_ligand}"
echo "  active site   : ${jid_active}"
echo "  make batches  : ${jid_batch}"
echo "  docking array : ${jid_docking}"
echo "  merge results : ${jid_merge}"
echo "  analysis      : ${jid_analysis}"
echo
echo "Monitor with: squeue -u $USER"
echo "Logs in: ${LOG_DIR}"
