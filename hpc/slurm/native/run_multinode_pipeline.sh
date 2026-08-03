#!/usr/bin/env bash
set -euo pipefail

# --- Lmod ---
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# --- Rutas base ---
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-/nfs/chemlink/runs/${RUN_ID}}"

INPUT_RECEPTORS_DIR="${INPUT_RECEPTORS_DIR:-/nfs/chemlink/chemlink/data/input/receptors}"
INPUT_LIGANDS_DIR="${INPUT_LIGANDS_DIR:-/nfs/chemlink/chemlink/data/input/ligands}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/output}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

# --- Herramientas (rutas NFS) ---
FPOCKET_PATH="${FPOCKET_PATH:-/nfs/chemlink/software/fpocket-4.2.3/bin/fpocket}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/chemlink/miniconda/envs/bio/bin/python}"
AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE:-/nfs/chemlink/software/autogrid4/bin/autogrid4}"
AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE:-/nfs/chemlink/software/autodock-gpu/bin/autodock-gpu}"

# --- Recursos SLURM ---
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
DOCKING_GRES="${DOCKING_GRES:-}"

# Optional manual grid box (bypasses fpocket when both are set)
MANUAL_CENTER="${MANUAL_CENTER:-}"
MANUAL_NPTS="${MANUAL_NPTS:-}"
# Optional per-receptor grid box config (JSON on shared storage; overrides
# MANUAL_CENTER/MANUAL_NPTS and fpocket for the receptors it lists)
BOX_CONFIG_PATH="${BOX_CONFIG_PATH:-}"

# --- Opciones SLURM opcionales ---
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_NODELIST="${SLURM_NODELIST:-}"

# --- Override de memoria global ---
if [[ -n "${SLURM_MEM_OVERRIDE}" ]]; then
  RECEPTOR_MEM="${SLURM_MEM_OVERRIDE}"
  LIGAND_MEM="${SLURM_MEM_OVERRIDE}"
  ACTIVE_SITE_MEM="${SLURM_MEM_OVERRIDE}"
  BATCH_MEM="${SLURM_MEM_OVERRIDE}"
  DOCKING_MEM="${SLURM_MEM_OVERRIDE}"
  MERGE_MEM="${SLURM_MEM_OVERRIDE}"
  ANALYSIS_MEM="${SLURM_MEM_OVERRIDE}"
fi

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${REPO_DIR}"

# --- Argumentos SLURM comunes ---
COMMON_SBATCH_ARGS=()
[[ -n "${SLURM_PARTITION}" ]]  && COMMON_SBATCH_ARGS+=("--partition=${SLURM_PARTITION}")
[[ -n "${SLURM_ACCOUNT}" ]]    && COMMON_SBATCH_ARGS+=("--account=${SLURM_ACCOUNT}")
[[ -n "${SLURM_QOS}" ]]        && COMMON_SBATCH_ARGS+=("--qos=${SLURM_QOS}")
[[ -n "${SLURM_CONSTRAINT}" ]] && COMMON_SBATCH_ARGS+=("--constraint=${SLURM_CONSTRAINT}")
[[ -n "${SLURM_EXCLUDE}" ]]    && COMMON_SBATCH_ARGS+=("--exclude=${SLURM_EXCLUDE}")
[[ -n "${SLURM_NODELIST}" ]]   && COMMON_SBATCH_ARGS+=("--nodelist=${SLURM_NODELIST}")

DOCKING_SBATCH_ARGS=()
[[ -n "${DOCKING_GRES}" && "${DOCKING_GRES}" != "none" ]] && \
  DOCKING_SBATCH_ARGS+=("--gres=${DOCKING_GRES}")

# --- Calcular batches ---
mapfile -t ligand_candidates < <(find "${INPUT_LIGANDS_DIR}" -maxdepth 1 -type f \
  \( -name "*.sdf" -o -name "*.mol2" -o -name "*.pdb" -o -name "*.mol" -o -name "*.pdbqt" \) | sort)
ligand_count=${#ligand_candidates[@]}

if [[ "${ligand_count}" -eq 0 ]]; then
  echo "[ERROR] No ligands found in ${INPUT_LIGANDS_DIR}" >&2
  exit 1
fi

total_batches=$(( (ligand_count + BATCH_SIZE - 1) / BATCH_SIZE ))
array_expr="0-$((total_batches - 1))%${MAX_GPU_CONCURRENCY}"

echo "Run ID              : ${RUN_ID}"
echo "Run dir             : ${RUN_DIR}"
echo "Ligands detectados  : ${ligand_count}"
echo "Batch size          : ${BATCH_SIZE}"
echo "Total batches       : ${total_batches}"
echo "Docking array       : ${array_expr}"
echo "SLURM partition     : ${SLURM_PARTITION:-<default>}"
echo "Docking GRES        : ${DOCKING_GRES:-<none>}"
echo "Python              : ${PYTHON_BIN}"
echo "Manual grid center  : ${MANUAL_CENTER:-<fpocket>}"
echo "Manual grid npts    : ${MANUAL_NPTS:-<fpocket>}"
echo "Box config          : ${BOX_CONFIG_PATH:-<none>}"
echo "AutoDock-GPU        : ${AUTODOCK_GPU_EXECUTABLE}"
echo "AutoGrid4           : ${AUTOGRID_EXECUTABLE}"
echo "fpocket             : ${FPOCKET_PATH}"

# --- Step 1: Preparación de receptores ---
jid_receptor=$(INPUT_DIR="${INPUT_RECEPTORS_DIR}" \
  REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  N_WORKERS="${RECEPTOR_WORKERS}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --array="${PREP_ARRAY_RANGE}" \
  --cpus-per-task="${RECEPTOR_CPUS_PER_TASK}" \
  --mem="${RECEPTOR_MEM}" \
  hpc/slurm/native/receptor_preparation_array.slurm)

# --- Step 2: Preparación de ligandos ---
jid_ligand=$(INPUT_DIR="${INPUT_LIGANDS_DIR}" \
  REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  N_WORKERS="${LIGAND_WORKERS}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --array="${PREP_ARRAY_RANGE}" \
  --cpus-per-task="${LIGAND_CPUS_PER_TASK}" \
  --mem="${LIGAND_MEM}" \
  hpc/slurm/native/ligand_preparation_array.slurm)

# --- Step 3: Detección de sitios activos (depende de receptor + ligando) ---
jid_active=$(REPO_DIR="${REPO_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  FPOCKET_PATH="${FPOCKET_PATH}" \
  ACTIVE_SITE_WORKERS="${ACTIVE_SITE_WORKERS}" \
  MANUAL_CENTER="${MANUAL_CENTER}" \
  MANUAL_NPTS="${MANUAL_NPTS}" \
  BOX_CONFIG_PATH="${BOX_CONFIG_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_receptor}:${jid_ligand}" \
  --cpus-per-task="${ACTIVE_SITE_CPUS_PER_TASK}" \
  --mem="${ACTIVE_SITE_MEM}" \
  --output="${LOG_DIR}/active_site_%j.out" \
  --error="${LOG_DIR}/active_site_%j.err" \
  hpc/slurm/native/active_site.slurm)

# --- Step 4: Crear batches de ligandos (depende de ligando) ---
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
  hpc/slurm/native/prepare_ligand_batches.slurm)

# --- Step 5: Docking array (depende de active_site + batches) ---
jid_docking=$(REPO_DIR="${REPO_DIR}" \
  RUN_DIR="${RUN_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  WORKERS="${DOCKING_WORKERS}" \
  PREPARED_RECEPTORS_DIR="${OUTPUT_DIR}/prepared_receptors_pdbqt" \
  AUTOGRID_EXECUTABLE="${AUTOGRID_EXECUTABLE}" \
  AUTODOCK_GPU_EXECUTABLE="${AUTODOCK_GPU_EXECUTABLE}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  "${DOCKING_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_active}:${jid_batch}" \
  --array="${array_expr}" \
  --cpus-per-task="${DOCKING_CPUS_PER_TASK}" \
  --mem="${DOCKING_MEM}" \
  --output="${LOG_DIR}/docking_%A_%a.out" \
  --error="${LOG_DIR}/docking_%A_%a.err" \
  hpc/slurm/native/docking_array.slurm)

# --- Step 6: Merge de resultados (depende de docking) ---
jid_merge=$(RUN_DIR="${RUN_DIR}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_docking}" \
  --cpus-per-task="${MERGE_CPUS_PER_TASK}" \
  --mem="${MERGE_MEM}" \
  --output="${LOG_DIR}/merge_%j.out" \
  --error="${LOG_DIR}/merge_%j.err" \
  hpc/slurm/native/merge_docking_results.slurm)

# --- Step 7: Análisis final (depende de merge) ---
jid_analysis=$(REPO_DIR="${REPO_DIR}" \
  ANALYSIS_OUTPUT_DIR="${RUN_DIR}/merged_output" \
  PYTHON_BIN="${PYTHON_BIN}" \
  sbatch --parsable \
  "${COMMON_SBATCH_ARGS[@]}" \
  --dependency="afterok:${jid_merge}" \
  --cpus-per-task="${ANALYSIS_CPUS_PER_TASK}" \
  --mem="${ANALYSIS_MEM}" \
  --output="${LOG_DIR}/analysis_%j.out" \
  --error="${LOG_DIR}/analysis_%j.err" \
  hpc/slurm/native/analysis.slurm)

echo
echo "Jobs enviados:"
echo "  receptor prep : ${jid_receptor}"
echo "  ligand prep   : ${jid_ligand}"
echo "  active site   : ${jid_active}"
echo "  make batches  : ${jid_batch}"
echo "  docking array : ${jid_docking}"
echo "  merge results : ${jid_merge}"
echo "  analysis      : ${jid_analysis}"
echo
echo "Monitor: squeue -u $USER"
echo "Logs en: ${LOG_DIR}"
