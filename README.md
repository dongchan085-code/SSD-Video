# SSD-VLM: Simple Self-Distillation for Vision Language Models

**Applying Apple's Simple Self-Distillation (SSD) to Qwen3-VL-8B-Instruct for Streaming Video Understanding**

This project implements Simple Self-Distillation for vision language models, enabling efficient streaming video understanding through self-generated sample fine-tuning. The approach uses no reward models, no verification, and no ground truth labels—only raw self-generated completions from a frozen teacher model.

## Project Overview

### Key Innovation
- **Label-free fine-tuning**: Sample completions from a frozen Qwen3-VL-8B-Instruct model at high temperature (1.5), then fine-tune on raw samples without any external rewards or verification
- **Streaming efficiency**: Optimize for a 4-frame budget in the SimpleStream inference protocol
- **LoRA fine-tuning**: rank 128, alpha 256 — the paper method (full-parameter FT available as ablation only)
- **Comprehensive evaluation**: Test on OVO-Bench with detailed per-task breakdown (Lock vs Fork task categories)

### Training Data
- **Perception Test** training split (multiple-choice VQA videos)
- Memory skill category oversampled 2x to improve semantic memory understanding

### Evaluation Protocol
- **OVO-Bench** benchmark with SimpleStream's 4-frame inference budget
- Per-task performance breakdown showing category-specific improvements
- Lock tasks (real-time perception): OCR, ATR, OJR, STU, ACR, FPD
- Fork tasks (retrospective memory): EPM, ASI, HLD

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
- `target_modules`: ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
- `learning_rate`: 5e-4
- `batch_size`: 2 (per GPU)
- `gradient_accumulation_steps`: 8
- `num_epochs`: 2
- `use_deepspeed`: true
- `deepspeed_config`: deepspeed_zero2.json

**Output**: LoRA checkpoint (can be merged or used with base model)

### 3. Evaluation on OVO-Bench

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

### 4. Detailed Analysis

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

### 5. Visualization

Generate publication-quality figures:

```bash
python figures/plot_all.py \
  --base_results ./results/ovo_base.json \
  --ssd_results ./results/ovo_ssd.json \
  --frame_sweep_dir ./results/frame_sweep \
  --temperature_sweep_dir ./results/temperature_sweep \
  --output_dir ./figures/outputs
```

## OVO-Bench Benchmark Scripts (Single-GPU T4)

Tooling for running the OVO-Bench evaluation against `Qwen3-VL-8B-Instruct` on a
single 16 GB GPU (e.g. Tesla T4). The pipeline downloads the public dataset,
extracts a stratified subset, chunks the videos with ffmpeg stream-copy, runs
inference with NF4 quantization + SDPA + video temporal-pack, and provides
helpers to compare against the published OVO-Bench paper numbers.

### Pre-requisites

```bash
# Conda env with torch, transformers, qwen-vl-utils, bitsandbytes, imageio-ffmpeg, etc.
conda activate env_ssd_simplestream     # name used in the snippets below

# All large artifacts (OVO videos, model weights) go to D:\ to keep C:\ free.
# Set HF_HOME so the model lands on D:\ not the default user cache.
export HF_HOME=D:/hf_cache
export HUGGINGFACE_HUB_CACHE=D:/hf_cache/hub
export PYTHONPATH=.                     # the scripts import from ssd_vlm/
export PYTHONIOENCODING=utf-8           # avoids cp949 errors on Windows
```

### 1. Download OVO-Bench source tar parts (~43 GB)

```bash
python -u scripts/download_ovo_sources.py \
  --data_root D:/ssd_video_data \
  --parts_dir D:/ssd_video_data/ovo_src_parts \
  --anno_path D:/ssd_video_data/ovo_bench_new.json \
  --max_gb 100
```

Pulls `src_videos.tar.part*` from the HF dataset `JoeLeelyf/OVO-Bench` plus the
GitHub `ovo_bench_new.json` annotation file. Use `--skip_parts` to only refresh
the annotation when iterating.

### 2. Stratified subset manifest

```bash
# 10% stratified-per-task sample (recommended for quick local benchmarking)
python -u scripts/prepare_ovo_subset.py \
  --anno_path D:/ssd_video_data/ovo_bench_new.json \
  --output_dir D:/ssd_video_data/ovo_subset_10pct \
  --ratio 0.10 --seed 42 --min_per_task 1
```

Writes `ovo_bench_subset.json`, `required_sources.txt`, `required_chunks.txt`,
and `subset_report.json` (task / split / query-unit counts). Pass `--ratio 1.0`
for the full benchmark, or filter the annotation upstream for a single-task
sweep (see `tests/compare_paper.py` for an HLD-only example).

### 3. Extract only the required source videos from the tar stream

```bash
python -u scripts/extract_ovo_src_subset.py \
  --parts_dir D:/ssd_video_data/ovo_src_parts \
  --required_sources D:/ssd_video_data/ovo_subset_10pct/required_sources.txt \
  --output_dir D:/ssd_video_data/ovo_subset_10pct/src_videos
```

Streams the multi-part tar once (~170 MB/s end to end). The script reports
progress every GB and verifies every required filename was found.

### 4. Chunk videos to per-question end-times (ffmpeg stream-copy)

```bash
python -u scripts/chunk_ovo_subset.py \
  --anno_path D:/ssd_video_data/ovo_subset_10pct/ovo_bench_subset.json \
  --src_dir D:/ssd_video_data/ovo_subset_10pct/src_videos \
  --output_dir D:/ssd_video_data/ovo_subset_10pct/chunked_videos
```

Default path uses the `imageio-ffmpeg` bundled binary with `-c copy` — ~50x
faster than re-encoding (1.5 min for 301 chunks vs ~70 min with OpenCV). Pass
`--reencode` to force the OpenCV path when stream-copy is incompatible with the
source codec.

### 5. Run the benchmark

```bash
mkdir -p results/ovo_10pct

python -u eval/eval_ovo_bench.py \
  --config configs/eval_ovo_base_10pct_t4.yaml \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path D:/ssd_video_data/ovo_subset_10pct \
  --output_file ./results/ovo_10pct/qwen3vl8b_nf4_t4.json
```

The T4-targeted config sets `dtype: float16`, `device_map: cuda`,
`load_in_4bit: true`, `attn_implementation: sdpa`, `max_pixels: 200704`,
`max_new_tokens: 64`, `batch_size: 1`. Peak VRAM is ~6.6 GB on 16 GB T4, no
CPU offload. The script writes both the aggregate JSON and a per-sample
JSONL (`.partial_predictions.jsonl`) so an interrupted run can be resumed —
delete that file when changing the dataset loader or prompts.

Related configs:

| File | Purpose |
|---|---|
| `configs/eval_ovo_base_10pct_t4.yaml` | 10% subset eval (recommended) |
| `configs/eval_ovo_base_1pct_t4.yaml`  | 1% smoke-test eval |
| `configs/eval_ovo_hld_full_t4.yaml`   | Single-task full HLD eval (186 samples) |

### 6. Analysis helpers

```bash
# 1% vs 10% (Qwen2VL-2B baseline vs Qwen3VL-8B NF4)
python tests/compare_subsets.py

# Per-task delta vs the OVO-Bench paper Table 1 leaderboard
PYTHONIOENCODING=utf-8 python tests/compare_paper.py

# Wilson 95% CI per task — distinguishes sampling noise from true gaps
PYTHONIOENCODING=utf-8 python tests/analyze_variance.py
```

`tests/analyze_variance.py` reports per-task Wilson 95% CIs and tells you
whether the paper number lies inside the CI (i.e. is the gap noise or
real?). On the 10% subset only HLD remains statistically distinguishable
from the paper Qwen2-VL-7B; the rest are noise-equivalent at N=11–73.

### GPU smoke test (no full eval)

```bash
python tests/smoke_oom.py --quant nf4 --gen_tokens 16 --budget_gb 14.5
```

Loads the NF4 model, runs a text-only generate, then a 4-frame synthetic
video generate, and aborts if peak VRAM exceeds the budget.

### Known caveats on a single T4

- `Qwen2-VL-7B` paper numbers used **bf16 + 64 frames**; this pipeline uses
  **NF4 + 4 frames** so the cross-comparison is only fair for the lock and
  forward macro-averages. HLD specifically is ~33 pp below paper on the
  full 186-sample run — currently attributed to NF4 logit noise (see
  `tests/analyze_variance.py` for the CI proof that this gap is real and
  not a sampling artifact).
- D:\ on Azure VMs is a **temporary disk**. After a VM stop/start the OVO
  tar parts and the model cache are gone and step 1 has to be repeated.

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
3. LoRA fine-tuning (paper method)
4. OVO-Bench evaluation (base + SSD)
5. Frame and temperature sweeps
6. Generate all figures
7. Create a comprehensive results report

## Project Structure

```
ssd-vlm/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── setup.py                          # Package installation
├── configs/                          # Configuration files (YAML)
│   ├── sample_generation.yaml       # SSD sampling config
│   ├── train_lora.yaml              # LoRA fine-tuning config
│   ├── train_full_ft.yaml           # Full FT config (ablation only)
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
│       ├── train_full_ft.py          # Ablation only
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
- **Epochs**: 2 (with early stopping)
- **Batch size**: 2 per GPU (16 with gradient accumulation)

### Evaluation
- **Frame budget**: 4 (primary), also test 8, 16, 32
- **Task categories**:
  - Lock (real-time perception): OCR, ATR, OJR, STU, ACR, FPD
  - Fork (retrospective memory): EPM, ASI, HLD

## Expected Results

SSD-VLM targets the **Pareto expansion** region ($\Delta$RT $\geq 0$, $\Delta$Mem $> 0$):
- **Fork (memory) tasks**: primary improvement target — SSD selectively reshapes distributions at temporally ambiguous positions
- **Lock (perception) tasks**: accuracy preserved or marginally improved (zero inference overhead from LoRA merge)
- **Zero additional latency**: merged LoRA adds no parameters at inference time vs. base SimpleStream

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
