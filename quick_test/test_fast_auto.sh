#!/bin/bash

set -euo pipefail

# Absolute path to this quick-test directory.
CWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "${CWD}/.." && pwd -P)"

source /wrk/setup_conda.sh
conda activate spacehasten-quick

spacehasten_exe="$(command -v spacehasten)"
config="${REPO_DIR}/svdkl-small.ini"

run_name="${2:-test_fast_auto}"
RUN_DIR="${1:-/wrk/${USER}/${run_name}}"

# If you already ran a test with the same name, uncomment the next two lines.
# rm -rf "${RUN_DIR}"
# rm -rf "/data/${USER}/SPACEHASTEN/${run_name}"

mkdir "${RUN_DIR}"

"${spacehasten_exe}" --config "${config}" init "${RUN_DIR}" \
    --name "${run_name}" \
    --dock-grid "${CWD}/grid-test_dock.zip" \
    --dock-params "${CWD}/test_dock.in"

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" seed-training \
    --smi "${CWD}/100seeds.smi" \
    --dock-cpus 2

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" screening-cycle \
    --simsearch-top-n 10 \
    --simsearch-jobs 100 \
    --nnn 10 \
    --dock-top-n 10 \
    --dock-cpus 2 \
    --rounds 3 \
    --strategy greedy

db_path="${RUN_DIR}/$(basename "${RUN_DIR}").dbsh"
python3 "${CWD}/verify_uncertainty_db.py" "${db_path}"

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" status

"${spacehasten_exe}" --config "${config}" -w "${RUN_DIR}" export csv \
    --cutoff -5 \
    --output "${RUN_DIR}/hits.csv"
