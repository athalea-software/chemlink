#!/usr/bin/env bash
set -euo pipefail

# --- Lmod ---
source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/nfs/chemlink/modules:$MODULEPATH
module load chemlink/1.0

# --- Base paths ---
REPO_DIR="${REPO_DIR:-/nfs/chemlink/chemlink}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/chemlink/miniconda/envs/bio/bin/python}"

# --- Simulation parameters ---
DYN_TYPE="${DYN_TYPE:-pligand}"
DYN_NS_TIME="${DYN_NS_TIME:-100}"
DYN_CHARGE="${DYN_CHARGE:-}"

# --- Config list ---
DYN_CONFIGS_JSON="${DYN_CONFIGS_JSON:-}"

# --- SLURM resource overrides ---
DYN_TIME_LIMIT="${DYN_TIME_LIMIT:-72:00:00}"
DYN_MEM="${DYN_MEM:-32G}"
DYN_CPUS="${DYN_CPUS:-8}"

# --- Optional SLURM args ---
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_NODELIST="${SLURM_NODELIST:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"

if [[ -z "${DYN_CONFIGS_JSON}" ]]; then
  echo "[ERROR] DYN_CONFIGS_JSON is not set" >&2
  exit 1
fi

if [[ ! -f "${DYN_CONFIGS_JSON}" ]]; then
  echo "[ERROR] DYN_CONFIGS_JSON file not found: ${DYN_CONFIGS_JSON}" >&2
  exit 1
fi

cd "${REPO_DIR}"

# --- Parse config list ---
mapfile -t config_files < <(
  "${PYTHON_BIN}" -c "
import json, sys
with open('${DYN_CONFIGS_JSON}') as f:
    configs = json.load(f)
for c in configs:
    print(c['_config_path'])
"
)

n_sims=${#config_files[@]}
if [[ "${n_sims}" -eq 0 ]]; then
  echo "[ERROR] No simulation configs found in ${DYN_CONFIGS_JSON}" >&2
  exit 1
fi

# --- Parse node list ---
mapfile -t node_list < <(
  if [[ -n "${SLURM_NODELIST}" ]]; then
    scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null || echo "${SLURM_NODELIST}"
  fi
)
n_nodes=${#node_list[@]}

# --- Build common sbatch args ---
COMMON_SBATCH_ARGS=()
[[ -n "${SLURM_PARTITION}" ]] && COMMON_SBATCH_ARGS+=("--partition=${SLURM_PARTITION}")
[[ -n "${SLURM_ACCOUNT}" ]]   && COMMON_SBATCH_ARGS+=("--account=${SLURM_ACCOUNT}")
[[ -n "${SLURM_QOS}" ]]       && COMMON_SBATCH_ARGS+=("--qos=${SLURM_QOS}")

echo "Dynamics type       : ${DYN_TYPE}"
echo "Simulation time     : ${DYN_NS_TIME} ns"
echo "Ligand charge       : ${DYN_CHARGE:-<resolved per simulation config>}"
echo "Simulations         : ${n_sims}"
echo "Config list         : ${DYN_CONFIGS_JSON}"
echo "Time limit          : ${DYN_TIME_LIMIT}"
echo "Memory              : ${DYN_MEM}"
echo "CPUs per task       : ${DYN_CPUS}"
echo "SLURM partition     : ${SLURM_PARTITION:-<default>}"
echo "Node list           : ${SLURM_NODELIST:-<any>}"
echo "Python              : ${PYTHON_BIN}"
echo

job_ids=()
for i in "${!config_files[@]}"; do
  cfg_path="${config_files[$i]}"
  work_dir=$(dirname "${cfg_path}")

  NODE_ARGS=()
  if [[ "${n_nodes}" -gt 0 ]]; then
    node_idx=$(( i % n_nodes ))
    NODE_ARGS+=("--nodelist=${node_list[$node_idx]}")
  fi

  jid=$(DYN_CONFIG_JSON="${cfg_path}" \
    DYN_WORK_DIR="${work_dir}" \
    DYN_TIME_LIMIT="${DYN_TIME_LIMIT}" \
    REPO_DIR="${REPO_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    sbatch --parsable \
    "${COMMON_SBATCH_ARGS[@]}" \
    "${NODE_ARGS[@]}" \
    --cpus-per-task="${DYN_CPUS}" \
    --mem="${DYN_MEM}" \
    --time="${DYN_TIME_LIMIT}" \
    --output="${work_dir}/slurm_%j.out" \
    --error="${work_dir}/slurm_%j.err" \
    "${REPO_DIR}/hpc/slurm/native/dynamics.slurm")

  job_ids+=("${jid}")
  echo "Sim $((i + 1))/${n_sims}  →  job ${jid}  (${work_dir})"
done

echo
echo "Jobs submitted:"
for jid in "${job_ids[@]}"; do
  echo "  ${jid}"
done
echo
echo "Monitor: squeue -u $USER"
