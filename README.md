# SSD-VLM: Simple Self-Distillation for Vision Language Models

**Applying Apple's Simple Self-Distillation (SSD) to Qwen3-VL-8B-Instruct for Streaming Video Understanding**

This project implements Simple Self-Distillation for vision language models, enabling efficient streaming video understanding through self-generated sample fine-tuning. The approach uses no reward models, no verification, and no ground truth labels—only raw self-generated completions from a frozen teacher model.

## Project Overview

### Key Innovation
- **Label-free fine-tuning**: Sample completions from a frozen Qwen3-VL-8B-Instruct model at high temperature (1.5), then fine-tune on raw samples without any external rewards or verification
- **Streaming efficiency**: Optimize for a 4-frame budget in the SimpleStream inference protocol
- **Two-stage training**: LoRA fine-tuning (rank 128, alpha 256) followed by full-parameter fine-tuning
- **Comprehensive evaluation**: Test on OVO-Bench with detailed per-task breakdown (Lock vs Fork task categories)

### Training Data
- **Perception Test** training split (multiple-choice VQA videos)
- Memory skill category oversampled 2x to improve semantic memory understanding

### Evaluation Protocol
- **OVO-Bench** benchmark with SimpleStream's 4-frame inference budget
- Per-task performance breakdown showing category-specific improvements
- Lock tasks: OCR, ATR, OJR, STU
- Fork tasks: EPM, ASI

## Setup

### Requirements
- GPU: A100 or H100 (recommend 8-GPU setup for training)
- Python 3.10+
- CUDA 12.1+
- ~500GB storage for Perception Test dataset + model checkpoints

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the ssd_vlm package
pip install -e .
```

### Data Preparation

```bash
# Download Perception Test and OVO-Bench datasets
bash scripts/download_data.sh

# Expected structure after download:
# ./data/
# ├── perception_test/
# │   ├── videos/
# │   ├── train_split.json
# │   └── train_annotations.json
# └── ovo_bench/
#     ├── test_split.json
#     └── task_definitions.json
```

## Pipeline

### 1. Generate SSD Samples (Label-free Sampling)

Sample high-temperature completions from the frozen base model:

```bash
python ssd_vlm/sampling/generate_samples.py \
  --config configs/sample_generation.yaml \
  --output_dir ./outputs/ssd_samples
```

**Config parameters**:
- `model_id`: Qwen3-VL-8B-Instruct
- `temperature`: 1.5
- `top_k`: 10
- `num_frames`: 4 (uniform sampling from video)
- `batch_size`: 32
- `num_gpus`: 8 (for efficient sampling)
- `memory_skill_oversample_ratio`: 2.0

**Output**: JSONL file with raw model samples (no filtering/verification)

### 2. LoRA Fine-tuning

Fine-tune with LoRA adapters on generated samples:

```bash
torchrun --nproc_per_node=8 ssd_vlm/training/train_lora.py \
  --config configs/train_lora.yaml \
  --samples_path ./outputs/ssd_samples/samples.jsonl \
  --output_dir ./outputs/lora_checkpoint
```

**Config parameters**:
- `lora_rank`: 128
- `lora_alpha`: 256
- `target_modules`: ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
- `learning_rate`: 5e-4
- `batch_size`: 2 (per GPU)
- `gradient_accumulation_steps`: 8
- `num_epochs`: 3
- `use_deepspeed`: true
- `deepspeed_config`: deepspeed_zero2.json

**Output**: LoRA checkpoint (can be merged or used with base model)

### 3. Full-parameter Fine-tuning

Merge LoRA and perform full-parameter fine-tuning:

```bash
torchrun --nproc_per_node=8 ssd_vlm/training/train_full_ft.py \
  --config configs/train_full_ft.yaml \
  --lora_checkpoint ./outputs/lora_checkpoint \
  --samples_path ./outputs/ssd_samples/samples.jsonl \
  --output_dir ./outputs/ssd_vlm_final
```

**Config parameters**:
- `learning_rate`: 2e-5
- `batch_size`: 1 (per GPU)
- `gradient_accumulation_steps`: 16
- `num_epochs`: 1
- `use_deepspeed`: true
- `deepspeed_config`: deepspeed_zero3.json
- `save_safetensors`: true

**Output**: Full SSD-VLM checkpoint ready for evaluation

### 4. Evaluation on OVO-Bench

Evaluate both base model and SSD-VLM:

```bash
# Evaluate base model
python eval/eval_ovo_bench.py \
  --config configs/eval_ovo_base.yaml \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --output_file ./results/ovo_base.json

# Evaluate SSD-VLM
python eval/eval_ovo_bench.py \
  --config configs/eval_ovo_ssd.yaml \
  --model_path ./outputs/ssd_vlm_final \
  --output_file ./results/ovo_ssd.json
```

### 5. Detailed Analysis

Run ablation studies and sweeps:

```bash
# Frame budget sweep (4, 8, 16, 32)
python eval/eval_frame_sweep.py \
  --config configs/eval_ovo_frame_sweep.yaml \
  --model_path ./outputs/ssd_vlm_final \
  --output_dir ./results/frame_sweep

# Temperature sweep (base model)
python eval/eval_temperature_sweep.py \
  --config configs/eval_temperature_sweep.yaml \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --output_dir ./results/temperature_sweep
```

### 6. Visualization

Generate publication-quality figures:

```bash
python figures/plot_all.py \
  --base_results ./results/ovo_base.json \
  --ssd_results ./results/ovo_ssd.json \
  --frame_sweep_dir ./results/frame_sweep \
  --temperature_sweep_dir ./results/temperature_sweep \
  --output_dir ./figures/outputs
```

## Running the Full Pipeline

Execute end-to-end training and evaluation:

```bash
bash scripts/run_full_pipeline.sh \
  --data_dir ./data \
  --output_dir ./outputs \
  --results_dir ./results
```

This script will:
1. Download data (if needed)
2. Generate SSD samples from frozen model
3. LoRA fine-tuning
4. Full-parameter fine-tuning
5. OVO-Bench evaluation (base + SSD)
6. Frame and temperature sweeps
7. Generate all figures
8. Create a comprehensive results report

## Project Structure

```
ssd-vlm/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── setup.py                          # Package installation
├── configs/                          # Configuration files (YAML)
│   ├── sample_generation.yaml       # SSD sampling config
│   ├── train_lora.yaml              # LoRA fine-tuning config
│   ├── train_full_ft.yaml           # Full FT config
│   ├── eval_ovo_base.yaml           # Base model eval
│   ├── eval_ovo_ssd.yaml            # SSD model eval
│   ├── eval_ovo_frame_sweep.yaml    # Frame budget sweep
│   └── eval_temperature_sweep.yaml  # Temperature sweep
├── ssd_vlm/                         # Main package
│   ├── __init__.py
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   ├── perception_test_dataset.py
│   │   └── ssd_sample_dataset.py
│   ├── sampling/                    # SSD sample generation
│   │   ├── __init__.py
│   │   └── generate_samples.py
│   └── training/                    # Training pipelines
│       ├── __init__.py
│       ├── train_lora.py
│       ├── train_full_ft.py
│       └── utils.py
├── eval/                            # Evaluation (SimpleStream adapted)
│   ├── __init__.py
│   ├── eval_ovo_bench.py
│   ├── eval_temperature_sweep.py
│   ├── eval_frame_sweep.py
│   ├── score_results.py
│   └── run_eval.sh
├── scripts/                         # Utility scripts
│   ├── run_full_pipeline.sh
│   └── download_data.sh
└── figures/                         # Visualization scripts
    ├── plot_perception_memory_tradeoff.py
    ├── plot_lock_fork_asymmetry.py
    ├── plot_temperature_plateau.py
    ├── plot_all.py
    └── style.py
```

## Key Parameters

### SSD Sampling
- **Temperature**: 1.5 (high temperature for diverse samples)
- **Top-k**: 10 (restrict to top 10 tokens)
- **Frames**: 4 per video (uniform sampling)
- **Memory oversample**: 2x

### LoRA Fine-tuning
- **Rank**: 128
- **Alpha**: 256 (scaling factor)
- **Target modules**: All linear layers
- **Learning rate**: 5e-4
- **Epochs**: 3
- **Batch size**: 2 per GPU (16 with gradient accumulation)

### Full-parameter Fine-tuning
- **Learning rate**: 2e-5 (much lower)
- **Epochs**: 1
- **Batch size**: 1 per GPU (16 with gradient accumulation)
- **DeepSpeed**: ZeRO-3 for memory efficiency

### Evaluation
- **Frame budget**: 4 (primary), also test 8, 16, 32
- **Task categories**:
  - Lock: OCR, ATR, OJR, STU (perception-heavy)
  - Fork: EPM, ASI (reasoning-heavy)

## Expected Results

SSD-VLM typically achieves:
- **~3-5% improvement** on Lock tasks (OCR-heavy)
- **~1-2% improvement** on Fork tasks (reasoning-heavy)
- **Minimal regression** on unseen task categories
- **Favorable memory-speed tradeoff** vs. larger models

The "green zone" in the perception-memory tradeoff space represents this sweet spot of improved performance with reasonable compute efficiency.

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Enable gradient checkpointing: `use_gradient_checkpointing: true`
- Use DeepSpeed ZeRO-3 instead of ZeRO-2

### Slow data loading
- Use SSD disk for video cache
- Pre-extract frames to local storage
- Increase `num_workers` in DataLoader config

### Training divergence
- Reduce learning rate by 2x
- Check sample quality (may need to adjust sampling temperature)
- Verify batch normalization is disabled in LoRA training

## Citation

```bibtex
@article{ssd-vlm-2025,
  title={Simple Self-Distillation for Efficient Vision Language Models in Streaming Settings},
  author={Your Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please ensure:
- All code follows PEP 8 style guidelines
- Tests pass: `pytest tests/`
- Configuration is documented in README
- New features include logging and error handling

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
