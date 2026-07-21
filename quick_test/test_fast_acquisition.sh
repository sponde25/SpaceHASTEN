#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || ("$1" != "lcb" && "$1" != "ei") ]]; then
    echo "Usage: $0 {lcb|ei} [run_dir] [run_name]" >&2
    exit 2
fi

acquisition="$1"
CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "${CWD}/.." && pwd -P)"

source /wrk/setup_conda.sh
conda activate spacehasten-quick

spacehasten_exe="$(command -v spacehasten)"
config="${REPO_DIR}/svdkl-small.ini"
run_name="${3:-test_fast_auto_${acquisition}}"
RUN_DIR="${2:-/wrk/${USER}/${run_name}}"
ROUNDS="${ROUNDS:-3}"
LCB_BETA="${LCB_BETA:-1.0}"
EI_HIT_THRESHOLD="${EI_HIT_THRESHOLD:--5.0}"
EI_XI="${EI_XI:-0.0}"
CLUSTER_LAMBDA="${CLUSTER_LAMBDA:-0.5}"
SIMSEARCH_JOBS="${SIMSEARCH_JOBS:-10}"

mkdir "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/quick_test_${acquisition}.log") 2>&1

echo "Running ${acquisition^^} quick test in ${RUN_DIR}"
echo "Parameters: rounds=${ROUNDS} simsearch_jobs=${SIMSEARCH_JOBS} beta=${LCB_BETA} hit_threshold=${EI_HIT_THRESHOLD} xi=${EI_XI} cluster_lambda=${CLUSTER_LAMBDA}"

"${spacehasten_exe}" --config "${config}" init "${RUN_DIR}" \
    --name "${run_name}" \
    --dock-grid "${CWD}/grid-test_dock.zip" \
    --dock-params "${CWD}/test_dock.in"

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" seed-training \
    --smi "${CWD}/100seeds.smi" \
    --dock-cpus 2

acquisition_args=(
    --dock-acquisition "${acquisition}"
    --cluster-lambda "${CLUSTER_LAMBDA}"
)
if [[ "${acquisition}" == "lcb" ]]; then
    acquisition_args+=(--lcb-beta "${LCB_BETA}")
else
    acquisition_args+=(
        --ei-hit-threshold "${EI_HIT_THRESHOLD}"
        --ei-xi "${EI_XI}"
    )
fi

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" screening-cycle \
    --simsearch-top-n 10 \
    --simsearch-jobs "${SIMSEARCH_JOBS}" \
    --nnn 10 \
    --dock-top-n 10 \
    --dock-cpus 2 \
    --rounds "${ROUNDS}" \
    --strategy greedy \
    "${acquisition_args[@]}"

db_path="${RUN_DIR}/$(basename "${RUN_DIR}").dbsh"
python3 "${CWD}/verify_uncertainty_db.py" "${db_path}"
python3 "${CWD}/verify_acquisition.py" \
    "${RUN_DIR}" \
    --method "${acquisition}" \
    --expected-rounds "${ROUNDS}" \
    --expected-batch-size 10 \
    --cluster-lambda "${CLUSTER_LAMBDA}" \
    --lcb-beta "${LCB_BETA}" \
    --ei-hit-threshold "${EI_HIT_THRESHOLD}" \
    --ei-xi "${EI_XI}"

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" status

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" export csv \
    --cutoff -5 \
    --output "${RUN_DIR}/hits.csv"
