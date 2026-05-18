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
# Conda env with torch, Qwen3 official transformers/accelerate pins,
# qwen-vl-utils, bitsandbytes, imageio-ffmpeg, etc.
conda activate D:/conda_envs/env_ssd_simplestream_officialdeps

# If rebuilding the env, install a CUDA-compatible torch build first, then:
python -m pip install -r requirements-qwen3-officialdeps.txt
python -m pip install -e .

# All large artifacts (OVO videos, model weights) go to D:\ to keep C:\ free.
# Set HF_HOME so the model lands on D:\ not the default user cache.
export HF_HOME=D:/hf_cache
export HUGGINGFACE_HUB_CACHE=D:/hf_cache/hub
export PYTHONPATH=.                     # the scripts import from ssd_vlm/
export PYTHONIOENCODING=utf-8           # avoids cp949 errors on Windows
```

### 1. Pull OVO-Bench data from HuggingFace

Two HF archives are relevant:

| Archive | Size | Contents |
|---|---|---|
| `chunked_videos.tar.part{aa..ao}` | **152 GB** (15 parts) | Pre-cut chunk videos — what the eval actually reads |
| `src_videos.tar.part{aa..ae}` | 43 GB (5 parts) | Raw source clips — only needed if you want to re-chunk locally |

**Recommended path: skip `src_videos.tar` entirely and stream-download the
pre-chunked archive.** Doing the chunking locally with `chunk_ovo_subset.py`
produced ~100 GB of output (ffmpeg stream-copy preserves source quality), and
the resulting chunks are still functionally identical to the HF ones at the
4-frame eval budget.

```bash
# Stream-download + extract chunked_videos.tar.part{aa..ao} (~152 GB).
# Each tar part is deleted as soon as the extractor finishes reading it,
# so peak disk usage stays near the extracted-size + 1-2 parts.
python -u scripts/download_extract_chunked.py \
  --repo_id JoeLeelyf/OVO-Bench \
  --tar_glob "chunked_videos.tar.part*" \
  --parts_dir D:/ssd_video_data/_chunked_parts \
  --output_dir D:/ssd_video_data/chunked_videos

# Pull the annotation file (small, no streaming needed)
python -u scripts/download_ovo_sources.py \
  --data_root D:/ssd_video_data \
  --anno_path D:/ssd_video_data/ovo_bench_new.json \
  --skip_parts
```

The `chunked_videos/` directory it writes is the only thing the eval needs.
**On a 176 GB D:\ disk you must clear any prior `ovo_src_parts/`, `src_videos/`,
and `chunked_videos/` first**, otherwise the 152 GB extraction will overflow:

```bash
rm -rf D:/ssd_video_data/ovo_src_parts D:/ssd_video_data/src_videos D:/ssd_video_data/chunked_videos
```

(Legacy path — local chunking — still works via the original
`scripts/download_ovo_sources.py` + `extract_ovo_src_subset.py` +
`chunk_ovo_subset.py` chain; see git history for an example session.)

### 2. Fullset manifest (one-time)

```bash
# Generate manifests + enriched annotation for the entire benchmark
python -u scripts/prepare_ovo_subset.py \
  --anno_path D:/ssd_video_data/ovo_bench_new.json \
  --output_dir D:/ssd_video_data \
  --ratio 1.0 --seed 42 --min_per_task 1
```

Writes `ovo_bench_full.json` (enriched annotation), `required_sources.txt`
(~644 unique source video paths), `required_chunks.txt` (~3035 chunk
filenames), and `subset_report.json` (task / split / query-unit counts).

**Subset selection happens at load time** — set `data.sample_ratio: 0.25` in
the eval config or pass `--sample_ratio 0.25 --sample_seed 42` on the CLI.
`OVOBenchDataset` does stratified-by-task, grouped-by-source-video sampling
deterministically, so all chunks of the same forward-task source stay together.
No per-ratio chunked-videos directory needed.

### 3. Extract all source videos from the tar stream (one-time)

```bash
python -u scripts/extract_ovo_src_subset.py \
  --parts_dir D:/ssd_video_data/ovo_src_parts \
  --required_sources D:/ssd_video_data/required_sources.txt \
  --output_dir D:/ssd_video_data/src_videos
```

Streams the multi-part tar once (~170 MB/s end to end, ~5 min total).

### 4. Chunk all videos to per-question end-times (ffmpeg stream-copy)

```bash
python -u scripts/chunk_ovo_subset.py \
  --anno_path D:/ssd_video_data/ovo_bench_full.json \
  --src_dir D:/ssd_video_data/src_videos \
  --output_dir D:/ssd_video_data/chunked_videos
```

Default path uses the `imageio-ffmpeg` bundled binary with `-c copy` — ~50x
faster than re-encoding (~14 min for 3035 chunks vs hours with OpenCV). Pass
`--reencode` to force the OpenCV path when stream-copy is incompatible with the
source codec.

### 5. Run the benchmark

```bash
mkdir -p results/ovo_simplestream

# Required env: HF_HOME on D:\, decord backend, utf-8 stdout
HF_HOME=D:/hf_cache HUGGINGFACE_HUB_CACHE=D:/hf_cache/hub \
FORCE_QWENVL_VIDEO_READER=decord PYTHONIOENCODING=utf-8 \
python -u eval/eval_ovo_bench.py \
  --config configs/eval_ovo_simplestream_10pct_t4.yaml \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path D:/ssd_video_data \
  --output_file ./results/ovo_simplestream/qwen3vl8b_int8_t4.json \
  --sample_ratio 0.25 --sample_seed 42
```

The SimpleStream-aligned config sets `dtype: float16`, `device_map: cuda`,
`load_in_8bit: true`, `attn_implementation: sdpa`, image-list frame encoding,
`use_simplestream_decode: true`, `max_new_tokens: 256`, `batch_size: 1`.
Peak VRAM is ~15 GB on 16 GB T4, no CPU offload. The script writes both the
aggregate JSON and a per-sample JSONL (`.partial_predictions.jsonl`) so an
interrupted run can be resumed — delete that file when changing the dataset
loader or prompts.

Related configs:

| File | Purpose |
|---|---|
| `configs/eval_ovo_simplestream_10pct_t4.yaml`     | Reproduces SimpleStream Qwen3-VL 4f (int8 + sdpa + image-list + simplestream decode + max_new_tokens=256). 10% subset by default; set `data.sample_ratio` to change |
| `configs/eval_ovo_simplestream_10pct_t4_nf4.yaml` | Same setup with NF4 instead of int8 — faster, ~7pp HLD accuracy cost |
| `configs/eval_ovo_base_10pct_t4.yaml`             | OVO-Bench official prompts (older Qwen3-VL setup before SimpleStream alignment) |
| `configs/eval_ovo_base_1pct_t4.yaml`              | 1% smoke-test eval |
| `configs/eval_ovo_hld_full_t4.yaml`               | Single-task full HLD eval (186 samples) |

### 6. Analysis helpers

```bash
# 1% vs 10% (Qwen2VL-2B baseline vs Qwen3VL-8B NF4)
python tests/compare_subsets.py

# Per-task delta vs the OVO-Bench paper Table 1 leaderboard
PYTHONIOENCODING=utf-8 python tests/compare_paper.py

# Per-task delta vs SimpleStream Qwen3-VL 4f published numbers
PYTHONIOENCODING=utf-8 python tests/compare_simplestream.py

# Wilson 95% CI per task — distinguishes sampling noise from true gaps
PYTHONIOENCODING=utf-8 python tests/analyze_variance.py        # vs OVO paper
PYTHONIOENCODING=utf-8 python tests/analyze_simplestream_ci.py # vs SimpleStream
```

`analyze_*.py` reports per-task Wilson 95% CIs and tells you whether the
reference number lies inside the CI (i.e. is the gap noise or real?). At the
10% subset size SimpleStream Qwen3-VL 4f numbers all sit inside our 95%
CIs — gaps that look large per-task (e.g. OCR 100% vs SimpleStream 94%) are
small-N artifacts, not real differences.

### GPU smoke test (no full eval)

```bash
python tests/smoke_oom.py --quant nf4 --gen_tokens 16 --budget_gb 14.5
```

Loads the NF4 model, runs a text-only generate, then a 4-frame synthetic
video generate, and aborts if peak VRAM exceeds the budget.

### Known caveats on a single T4

- The reproduction target is now **SimpleStream Qwen3-VL 4f**, not the
  OVO-Bench paper baseline. SimpleStream evaluates with `bf16 + flash_attention_2 +
  image-list encoding + qwen_vl_utils.fetch_video decode + max_new_tokens=256`
  on multi-GPU. The two T4 concessions in this repo are bf16 → int8 (or NF4)
  and FA2 → SDPA; everything else (prompts, scoring, frame encoding, decode
  pipeline) is byte-for-byte aligned with `EvolvingLMMs-Lab/SimpleStream`.
- HLD's apparent -33pp gap (NF4 + video temporal-pack + OVO-Bench prompts)
  was a setup confound, not a real model weakness. After SimpleStream
  alignment the gap drops to +0.5pp.
- D:\ on Azure VMs is a **temporary disk**. After a VM stop/start the OVO
  tar parts, source extraction, chunk videos, and the HF model cache are
  all gone and step 1 has to be repeated.

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
