#!/bin/bash
# Reproduce the SimpleStream recent-window OVO baseline before SSD training.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

DATA_DIR="${1:-${PROJECT_DIR}/data/ovo_bench}"
RESULTS_DIR="${2:-${PROJECT_DIR}/results}"
MODEL_ID="${3:-Qwen/Qwen3-VL-8B-Instruct}"

mkdir -p "${RESULTS_DIR}"

python "${PROJECT_DIR}/eval/eval_ovo_bench.py" \
    --config "${PROJECT_DIR}/configs/eval_ovo_base.yaml" \
    --model_path "${MODEL_ID}" \
    --data_path "${DATA_DIR}" \
    --output_file "${RESULTS_DIR}/ovo_base_simplestream_recent4.json"

echo "Baseline results: ${RESULTS_DIR}/ovo_base_simplestream_recent4.json"
