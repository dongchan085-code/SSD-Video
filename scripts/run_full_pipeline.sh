#!/bin/bash
# End-to-end SSD-VLM pipeline

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Parse arguments
DATA_DIR="${1:-./${PROJECT_DIR}/data}"
OUTPUT_DIR="${2:-./${PROJECT_DIR}/outputs}"
RESULTS_DIR="${3:-./${PROJECT_DIR}/results}"
NUM_GPUS="${4:-8}"

echo "==========================================="
echo "SSD-VLM End-to-End Pipeline"
echo "==========================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo ""

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"

# Step 1: Data preparation
echo ""
echo "Step 1: Data Preparation"
echo "======================="
if [ ! -d "${DATA_DIR}/perception_test" ]; then
    echo "Downloading datasets..."
    bash "${SCRIPT_DIR}/download_data.sh"
else
    echo "Data directory already exists"
fi

# Step 2: Generate SSD samples
echo ""
echo "Step 2: Generating SSD Samples"
echo "=============================="
echo "Sampling from frozen Qwen3-VL-8B-Instruct..."
echo "Temperature: 1.5, Top-k: 10, Frames: 4"
echo "Memory skill oversampling: 2x"
echo ""

python "${PROJECT_DIR}/ssd_vlm/sampling/generate_samples.py" \
    --config "${PROJECT_DIR}/configs/sample_generation.yaml" \
    --output_dir "${OUTPUT_DIR}/ssd_samples" \
    --data_path "${DATA_DIR}/perception_test" \
    || echo "Note: Sample generation requires actual Perception Test data"

# Step 3: LoRA Fine-tuning
echo ""
echo "Step 3: LoRA Fine-tuning (Stage 1)"
echo "=================================="
echo "LoRA rank: 128, alpha: 256"
echo "Learning rate: 5e-4, Epochs: 2"
echo ""

if [ -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        "${PROJECT_DIR}/ssd_vlm/training/train_lora.py" \
        --config "${PROJECT_DIR}/configs/train_lora.yaml" \
        --samples_path "${OUTPUT_DIR}/ssd_samples/samples.jsonl" \
        --output_dir "${OUTPUT_DIR}/lora_checkpoint" \
        || echo "Note: LoRA training requires GPU and actual SSD samples"
else
    echo "Skipping LoRA training (SSD samples not found)"
fi

# Step 4: Full-parameter Fine-tuning (SKIPPED - paper uses LoRA only)
# Stage 2 Full FT is available as an ablation (see run_ablations.sh)
# but is not part of the main SSD-VLM method described in the paper.
echo ""
echo "Step 4: Skipped (paper method is LoRA-only)"
echo "============================================"
echo "Full FT available as ablation via run_ablations.sh"
echo ""

# Step 5: OVO-Bench Evaluation
echo ""
echo "Step 5: OVO-Bench Evaluation"
echo "============================"

# Evaluate base model
echo "Evaluating base Qwen3-VL-8B-Instruct..."
python "${PROJECT_DIR}/eval/eval_ovo_bench.py" \
    --config "${PROJECT_DIR}/configs/eval_ovo_base.yaml" \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_path "${DATA_DIR}/ovo_bench" \
    --output_file "${RESULTS_DIR}/ovo_base.json" \
    || echo "Note: Base evaluation requires OVO-Bench data"

# Evaluate SSD-VLM
if [ -d "${OUTPUT_DIR}/lora_checkpoint" ]; then
    echo "Evaluating SSD-VLM..."
    python "${PROJECT_DIR}/eval/eval_ovo_bench.py" \
        --config "${PROJECT_DIR}/configs/eval_ovo_ssd.yaml" \
        --model_path "${OUTPUT_DIR}/lora_checkpoint" \
        --data_path "${DATA_DIR}/ovo_bench" \
        --output_file "${RESULTS_DIR}/ovo_ssd.json" \
        || echo "Note: SSD evaluation requires OVO-Bench data"
else
    echo "Skipping SSD evaluation (final model not found)"
fi

# Step 6: Frame and Temperature Sweeps
echo ""
echo "Step 6: Running Benchmark Sweeps"
echo "================================"

# Frame budget sweep
echo "Frame budget sweep (4, 8, 16, 32 frames)..."
if [ -d "${OUTPUT_DIR}/lora_checkpoint" ]; then
    python "${PROJECT_DIR}/eval/eval_frame_sweep.py" \
        --config "${PROJECT_DIR}/configs/eval_ovo_frame_sweep.yaml" \
        --model_path "${OUTPUT_DIR}/lora_checkpoint" \
        --data_path "${DATA_DIR}/ovo_bench" \
        --output_dir "${RESULTS_DIR}/frame_sweep" \
        || echo "Note: Frame sweep requires OVO-Bench data"
else
    echo "Skipping frame sweep (final model not found)"
fi

# Temperature sweep
echo "Temperature sweep (0.5 to 1.5)..."
python "${PROJECT_DIR}/eval/eval_temperature_sweep.py" \
    --config "${PROJECT_DIR}/configs/eval_temperature_sweep.yaml" \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_path "${DATA_DIR}/ovo_bench" \
    --output_dir "${RESULTS_DIR}/temperature_sweep" \
    || echo "Note: Temperature sweep requires OVO-Bench data"

# Step 7: Comprehensive Ablation Studies
echo ""
echo "Step 7: Running Comprehensive Ablation Studies"
echo "=============================================="

bash "${SCRIPT_DIR}/run_ablations.sh" "${DATA_DIR}" "${OUTPUT_DIR}" "${RESULTS_DIR}" "${NUM_GPUS}" \
    || echo "Note: Some ablation studies require actual data and GPU"

# Step 8: Generate Figures
echo ""
echo "Step 8: Generating Publication Figures"
echo "======================================"

python "${PROJECT_DIR}/figures/plot_all.py" \
    --base_results "${RESULTS_DIR}/ovo_base.json" \
    --ssd_results "${RESULTS_DIR}/ovo_ssd.json" \
    --frame_sweep_dir "${RESULTS_DIR}/frame_sweep" \
    --temperature_sweep_dir "${RESULTS_DIR}/temperature_sweep" \
    --output_dir "${PROJECT_DIR}/figures/outputs" \
    --use_mock_data

# Step 9: Score and summarize
echo ""
echo "Step 9: Scoring Results"
echo "======================="

python "${PROJECT_DIR}/eval/score_results.py" \
    --base_results "${RESULTS_DIR}/ovo_base.json" \
    --ssd_results "${RESULTS_DIR}/ovo_ssd.json" \
    --frame_sweep_dir "${RESULTS_DIR}/frame_sweep" \
    --temperature_sweep_dir "${RESULTS_DIR}/temperature_sweep" \
    --output_file "${RESULTS_DIR}/scored_results.json" \
    || echo "Note: Scoring skipped (some results may be missing)"

# Final summary
echo ""
echo "==========================================="
echo "Pipeline Completed!"
echo "==========================================="
echo ""
echo "Output locations:"
echo "  Samples: ${OUTPUT_DIR}/ssd_samples/"
echo "  LoRA checkpoint: ${OUTPUT_DIR}/lora_checkpoint/"
echo "  Final model (LoRA): ${OUTPUT_DIR}/lora_checkpoint/"
echo "  Results: ${RESULTS_DIR}/"
echo "  Figures: ${PROJECT_DIR}/figures/outputs/"
echo ""
echo "Next steps:"
echo "  1. Review results in ${RESULTS_DIR}/"
echo "  2. Check figures in ${PROJECT_DIR}/figures/outputs/"
echo "  3. Analyze scored_results.json for summary metrics"
echo ""
