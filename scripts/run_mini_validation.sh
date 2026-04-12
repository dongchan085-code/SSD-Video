#!/bin/bash
# End-to-end mini validation of the SSD-VLM pipeline.
#
# Runs on Mac Mini M4 (CPU, 16GB RAM) with:
#   - Qwen3-VL-2B-Instruct (~4GB download, ~8GB RAM in float32)
#   - Perception Test sample split (~215MB) or synthetic fallback
#   - OVO-Bench annotation subset + synthetic videos
#
# Expected runtime: ~30-40 minutes total
#
# Usage:
#   bash scripts/run_mini_validation.sh [--synthetic-only]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "${PROJECT_DIR}"

PT_DIR="./data/perception_test_mini"
OVO_DIR="./data/ovo_bench_mini"
OUTPUT_DIR="./outputs/mini"
RESULTS_DIR="./results/mini"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"

echo "==========================================="
echo "SSD-VLM Mini Validation Pipeline"
echo "==========================================="
echo "Model:   ${MODEL_ID}"
echo "Device:  CPU (float32)"
echo "PT data: ${PT_DIR}"
echo "OVO data: ${OVO_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

# ── Step 0: Data ─────────────────────────────────────────────────────
echo ""
echo "Step 0: Preparing data"
echo "======================"

if [ "$1" = "--synthetic-only" ]; then
    echo "Using synthetic data only (--synthetic-only flag)."
    mkdir -p "${PT_DIR}" "${OVO_DIR}"
    python3 scripts/prepare_mini_data.py \
        --pt_dir "${PT_DIR}" \
        --ovo_dir "${OVO_DIR}" \
        --max_pt_samples 10 \
        --max_ovo_samples 10
else
    bash scripts/download_mini_data.sh
fi

echo "Data ready."

# ── Step 1: SSD Sample Generation ────────────────────────────────────
echo ""
echo "Step 1: Generating SSD samples"
echo "=============================="
echo "Model: ${MODEL_ID} | Temp: 1.5 | Top-k: 10 | Frames: 4"

if [ ! -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ]; then
    python3 ssd_vlm/sampling/generate_samples.py \
        --config configs/mini/sample_generation_mini.yaml \
        --output_dir "${OUTPUT_DIR}/ssd_samples" \
        --data_path "${PT_DIR}" \
        --num_samples 10
    echo "SSD samples generated."
else
    echo "SSD samples already exist, skipping."
fi

SAMPLE_COUNT=$(wc -l < "${OUTPUT_DIR}/ssd_samples/samples.jsonl" 2>/dev/null || echo "0")
echo "Sample count: ${SAMPLE_COUNT}"

# ── Step 2: LoRA Fine-tuning ─────────────────────────────────────────
echo ""
echo "Step 2: LoRA fine-tuning"
echo "========================"
echo "LoRA rank: 8, alpha: 16 | Epochs: 1 | Device: CPU"

if [ ! -d "${OUTPUT_DIR}/lora_checkpoint/merged" ]; then
    python3 ssd_vlm/training/train_lora.py \
        --config configs/mini/train_lora_mini.yaml \
        --samples_path "${OUTPUT_DIR}/ssd_samples/samples.jsonl" \
        --output_dir "${OUTPUT_DIR}/lora_checkpoint"
    echo "LoRA training complete."
else
    echo "LoRA checkpoint already exists, skipping."
fi

# ── Step 3: Evaluation (base model) ─────────────────────────────────
echo ""
echo "Step 3: Evaluating base model"
echo "============================="

python3 eval/eval_ovo_bench.py \
    --config configs/mini/eval_ovo_mini.yaml \
    --model_path "${MODEL_ID}" \
    --data_path "${OVO_DIR}" \
    --output_file "${RESULTS_DIR}/ovo_base.json" \
    || echo "[WARN] Base model evaluation failed (may need model download)"

# ── Step 4: Evaluation (SSD-VLM) ────────────────────────────────────
echo ""
echo "Step 4: Evaluating SSD-VLM (LoRA merged)"
echo "=========================================="

if [ -d "${OUTPUT_DIR}/lora_checkpoint/merged" ]; then
    python3 eval/eval_ovo_bench.py \
        --config configs/mini/eval_ovo_mini.yaml \
        --model_path "${OUTPUT_DIR}/lora_checkpoint/merged" \
        --data_path "${OVO_DIR}" \
        --output_file "${RESULTS_DIR}/ovo_ssd.json" \
        || echo "[WARN] SSD evaluation failed"
else
    echo "Skipping SSD eval (merged checkpoint not found)"
fi

# ── Step 5: Entropy analysis ────────────────────────────────────────
echo ""
echo "Step 5: Entropy analysis"
echo "========================"

python3 eval/compute_entropy.py \
    --model_path "${MODEL_ID}" \
    --data_dir "${OVO_DIR}" \
    --output "${RESULTS_DIR}/entropy_base.json" \
    --dtype float32 \
    || echo "[WARN] Entropy analysis failed"

# ── Step 6: Figures ───────────────────────────────────────────────────
echo ""
echo "Step 6: Skipping figure generation"
echo "=================================="
echo "Real-data plotting is not yet wired into the local validation script."

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "Mini Validation Complete!"
echo "==========================================="
echo ""
echo "Output structure:"
echo "  ${OUTPUT_DIR}/ssd_samples/samples.jsonl"
echo "  ${OUTPUT_DIR}/lora_checkpoint/merged/"
echo "  ${RESULTS_DIR}/ovo_base.json"
echo "  ${RESULTS_DIR}/ovo_ssd.json"
echo "  ${RESULTS_DIR}/entropy_base.json"
echo ""

# Quick verification
echo "Verification:"
[ -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ] && echo "  [OK] SSD samples exist" || echo "  [FAIL] SSD samples missing"
[ -d "${OUTPUT_DIR}/lora_checkpoint" ] && echo "  [OK] LoRA checkpoint exists" || echo "  [FAIL] LoRA checkpoint missing"
[ -f "${RESULTS_DIR}/ovo_base.json" ] && echo "  [OK] Base eval results exist" || echo "  [FAIL] Base eval results missing"
[ -f "${RESULTS_DIR}/ovo_ssd.json" ] && echo "  [OK] SSD eval results exist" || echo "  [SKIP] SSD eval results missing"
echo ""
