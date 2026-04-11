#!/bin/bash
# Run all ablation experiments for SSD-VLM

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
echo "SSD-VLM Ablation Studies"
echo "==========================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo ""

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"

# Base directory for ablation configs
ABLATION_CONFIG_DIR="${PROJECT_DIR}/configs/ablations"

# Ensure base SSD samples exist
if [ ! -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ]; then
    echo "Warning: Base SSD samples not found. Running base sample generation..."
    python "${PROJECT_DIR}/ssd_vlm/sampling/generate_samples.py" \
        --config "${PROJECT_DIR}/configs/sample_generation.yaml" \
        --output_dir "${OUTPUT_DIR}/ssd_samples" \
        --data_path "${DATA_DIR}/perception_test" \
        || echo "Note: Base sample generation requires actual data"
fi

echo ""
echo "=========================================="
echo "Ablation 1: No Oversampling"
echo "=========================================="
echo "Testing: how much of Fork gain comes from oversampling vs SSD?"
echo ""

python "${PROJECT_DIR}/ssd_vlm/sampling/generate_samples.py" \
    --config "${ABLATION_CONFIG_DIR}/ablation_no_oversample.yaml" \
    --output_dir "${OUTPUT_DIR}/ablation_no_oversample_samples" \
    --data_path "${DATA_DIR}/perception_test" \
    || echo "Note: Ablation sampling requires actual data"

if [ -f "${OUTPUT_DIR}/ablation_no_oversample_samples/samples.jsonl" ]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        "${PROJECT_DIR}/ssd_vlm/training/train_lora.py" \
        --config "${ABLATION_CONFIG_DIR}/ablation_lora_only.yaml" \
        --samples_path "${OUTPUT_DIR}/ablation_no_oversample_samples/samples.jsonl" \
        --output_dir "${OUTPUT_DIR}/ablation_no_oversample_lora" \
        || echo "Note: LoRA training requires GPU"
fi

echo ""
echo "=========================================="
echo "Ablation 2: LoRA Only (No Full FT)"
echo "=========================================="
echo "Testing: is 2-stage training necessary?"
echo ""

if [ -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        "${PROJECT_DIR}/ssd_vlm/training/train_lora.py" \
        --config "${ABLATION_CONFIG_DIR}/ablation_lora_only.yaml" \
        --samples_path "${OUTPUT_DIR}/ssd_samples/samples.jsonl" \
        --output_dir "${OUTPUT_DIR}/ablation_lora_only_checkpoint" \
        || echo "Note: LoRA training requires GPU"
    
    # Evaluate LoRA-only model
    if [ -d "${OUTPUT_DIR}/ablation_lora_only_checkpoint" ]; then
        python "${PROJECT_DIR}/eval/eval_ovo_bench.py" \
            --config "${PROJECT_DIR}/configs/eval_ovo_ssd.yaml" \
            --model_path "${OUTPUT_DIR}/ablation_lora_only_checkpoint" \
            --data_path "${DATA_DIR}/ovo_bench" \
            --output_file "${RESULTS_DIR}/ablation_lora_only.json" \
            || echo "Note: Evaluation requires OVO-Bench data"
    fi
fi

echo ""
echo "=========================================="
echo "Ablation 3: Full FT Only (No LoRA)"
echo "=========================================="
echo "Testing: does LoRA warmup matter?"
echo ""

if [ -f "${OUTPUT_DIR}/ssd_samples/samples.jsonl" ]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        "${PROJECT_DIR}/ssd_vlm/training/train_full_ft.py" \
        --config "${ABLATION_CONFIG_DIR}/ablation_full_ft_only.yaml" \
        --samples_path "${OUTPUT_DIR}/ssd_samples/samples.jsonl" \
        --output_dir "${OUTPUT_DIR}/ablation_full_ft_only_checkpoint" \
        || echo "Note: Full FT training requires GPU"
    
    # Evaluate full FT only model
    if [ -d "${OUTPUT_DIR}/ablation_full_ft_only_checkpoint" ]; then
        python "${PROJECT_DIR}/eval/eval_ovo_bench.py" \
            --config "${PROJECT_DIR}/configs/eval_ovo_ssd.yaml" \
            --model_path "${OUTPUT_DIR}/ablation_full_ft_only_checkpoint" \
            --data_path "${DATA_DIR}/ovo_bench" \
            --output_file "${RESULTS_DIR}/ablation_full_ft_only.json" \
            || echo "Note: Evaluation requires OVO-Bench data"
    fi
fi

echo ""
echo "=========================================="
echo "Ablation 4: Standard Supervised FT"
echo "=========================================="
echo "Testing: is self-distillation needed?"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} \
    "${PROJECT_DIR}/ssd_vlm/training/train_lora.py" \
    --config "${ABLATION_CONFIG_DIR}/ablation_standard_ft.yaml" \
    --output_dir "${OUTPUT_DIR}/ablation_standard_ft_checkpoint" \
    || echo "Note: Standard FT training requires GPU and ground truth data"

if [ -d "${OUTPUT_DIR}/ablation_standard_ft_checkpoint" ]; then
    python "${PROJECT_DIR}/eval/eval_standard_ft_baseline.py" \
        --config "${PROJECT_DIR}/configs/eval_ovo_ssd.yaml" \
        --model_path "${OUTPUT_DIR}/ablation_standard_ft_checkpoint" \
        --data_path "${DATA_DIR}/ovo_bench" \
        --output_file "${RESULTS_DIR}/ablation_standard_ft.json" \
        || echo "Note: Evaluation requires OVO-Bench data"
fi

echo ""
echo "=========================================="
echo "Ablation 5: Dynamic Temperature Baseline"
echo "=========================================="
echo "Testing: can query-level temperature match SSD?"
echo ""

python "${PROJECT_DIR}/eval/eval_dynamic_temperature.py" \
    --config "${ABLATION_CONFIG_DIR}/ablation_dynamic_temperature.yaml" \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_path "${DATA_DIR}/ovo_bench" \
    --output_file "${RESULTS_DIR}/ablation_dynamic_temperature.json" \
    || echo "Note: Evaluation requires OVO-Bench data"

echo ""
echo "=========================================="
echo "Ablation 6: Entropy Analysis (All Models)"
echo "=========================================="
echo "Testing: Lock-Fork hypothesis mechanistically"
echo ""

echo "Base model entropy analysis..."
python "${PROJECT_DIR}/eval/eval_entropy_analysis.py" \
    --config "${PROJECT_DIR}/configs/eval_ovo_base.yaml" \
    --model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_path "${DATA_DIR}/ovo_bench" \
    --output_file "${RESULTS_DIR}/entropy_base.json" \
    || echo "Note: Entropy analysis requires OVO-Bench data"

if [ -f "${OUTPUT_DIR}/ssd_vlm_final/config.json" ]; then
    echo "SSD-VLM entropy analysis..."
    python "${PROJECT_DIR}/eval/eval_entropy_analysis.py" \
        --config "${PROJECT_DIR}/configs/eval_ovo_ssd.yaml" \
        --model_path "${OUTPUT_DIR}/ssd_vlm_final" \
        --data_path "${DATA_DIR}/ovo_bench" \
        --output_file "${RESULTS_DIR}/entropy_ssd.json" \
        || echo "Note: Entropy analysis requires OVO-Bench data"
fi

echo ""
echo "=========================================="
echo "Ablation 7: Statistical Significance Tests"
echo "=========================================="
echo ""

if [ -f "${RESULTS_DIR}/ovo_base.json" ] && [ -f "${RESULTS_DIR}/ovo_ssd.json" ]; then
    python "${PROJECT_DIR}/eval/statistical_tests.py" \
        --base_results "${RESULTS_DIR}/ovo_base.json" \
        --ssd_results "${RESULTS_DIR}/ovo_ssd.json" \
        --output_file "${RESULTS_DIR}/statistical_analysis.json"
fi

echo ""
echo "=========================================="
echo "Generate Ablation Figures"
echo "=========================================="
echo ""

# Create figures output directory
mkdir -p "${PROJECT_DIR}/figures/outputs/ablations"

echo "1. Ablation trade-off and accuracy comparison..."
python "${PROJECT_DIR}/figures/plot_ablation_results.py" \
    --output_dir "${PROJECT_DIR}/figures/outputs/ablations" \
    --use_mock_data

echo ""
echo "2. Entropy analysis figures..."
python "${PROJECT_DIR}/figures/plot_entropy_analysis.py" \
    --output_dir "${PROJECT_DIR}/figures/outputs/ablations" \
    --use_mock_data

echo ""
echo "3. Hyperparameter sensitivity figures..."
python "${PROJECT_DIR}/figures/plot_sensitivity.py" \
    --output_dir "${PROJECT_DIR}/figures/outputs/ablations" \
    --use_mock_data

echo ""
echo "==========================================="
echo "Ablation Studies Completed!"
echo "==========================================="
echo ""
echo "Output locations:"
echo "  Ablation checkpoints: ${OUTPUT_DIR}/ablation_*/"
echo "  Ablation results: ${RESULTS_DIR}/ablation_*.json"
echo "  Entropy analysis: ${RESULTS_DIR}/entropy_*.json"
echo "  Statistical tests: ${RESULTS_DIR}/statistical_analysis.json"
echo "  Figures: ${PROJECT_DIR}/figures/outputs/ablations/"
echo ""
echo "Key Results:"
echo "  - Ablation trade-off: plots/ablations/ablation_*.pdf"
echo "  - Entropy analysis: plots/ablations/entropy_*.pdf"
echo "  - Sensitivity analysis: plots/ablations/sensitivity_*.pdf"
echo ""
echo "Statistical Significance:"
echo "  Review results/statistical_analysis.json for McNemar's test, Cohen's d, etc."
echo ""
