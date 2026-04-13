# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SSD-VLM** implements Simple Self-Distillation for Vision Language Models — a label-free technique applied to Qwen3-VL-8B-Instruct for streaming video understanding. The core idea: sample from a frozen teacher at high temperature (T=1.5), then LoRA fine-tune the same model on those raw completions to resolve the perception-memory trade-off.

Evaluation benchmark: **OVO-Bench**, measuring Lock tasks (perception: OCR, ATR, OJR, STU, ACR, FPD) vs Fork tasks (memory: EPM, ASI, HLD).

## Git Workflow

- 별도 브랜치 생성 금지 — 항상 `main`에서 직접 작업한다.
- 테스트 통과 후 확인 질문 없이 `git push origin main`을 바로 실행한다.

## Commands

**Install** (torch/torchvision/torchaudio는 서버 사전 설치 버전 사용):
```bash
pip install -r requirements.txt
pip install -e .
```

**Run the full pipeline** (8 GPUs):
```bash
bash scripts/run_full_pipeline.sh ./data ./outputs ./results 8
```

**Individual pipeline stages**:
```bash
# Stage 1: Generate teacher samples
torchrun --nproc_per_node=8 ssd_vlm/sampling/generate_samples.py --config configs/sample_generation.yaml

# Stage 2: LoRA fine-tuning (NOT full FT — that's ablation-only)
torchrun --nproc_per_node=8 ssd_vlm/training/train_lora.py --config configs/train_lora.yaml

# Stage 3: Evaluate on OVO-Bench
python eval/eval_ovo_bench.py --config configs/eval_ovo_ssd.yaml
python eval/score_results.py --base results/ovo_base.json --ssd results/ovo_ssd.json
```

**Ablations and sweeps**:
```bash
bash scripts/run_ablations.sh
python eval/eval_frame_sweep.py     # 4/8/16/32 frames
python eval/eval_temperature_sweep.py  # T=0.5–1.5 on base model
```

**Visualization** (publication-quality PDF figures):
```bash
python figures/plot_all.py  # Orchestrates all 3 figures
```

**Smoke tests** (no GPU required):
```bash
pytest tests/smoke_test.py
```

**Code quality**:
```bash
black ssd_vlm/ eval/ figures/
isort ssd_vlm/ eval/ figures/
flake8 ssd_vlm/ eval/ figures/
```

**Mini validation** (small dataset subset, for fast iteration):
```bash
bash scripts/run_mini_validation.sh
```

## Architecture

The pipeline has four sequential stages:

### Stage 1 — Sample Generation (`ssd_vlm/sampling/generate_samples.py`)
- Loads Qwen3-VL-8B-Instruct **frozen** (`device_map="auto"`)
- Sources: **Perception Test** train split via `ssd_vlm/data/perception_test_dataset.py`
- 4 frames/video uniform sampled; memory skill items oversampled 2×
- Generates at T=1.5, top-k=10, max 512 tokens — **no filtering, no reward**
- Saves incrementally to `outputs/ssd_samples/samples.jsonl`

### Stage 2 — LoRA Fine-Tuning (`ssd_vlm/training/train_lora.py`)
- Base model: Qwen3-VL-8B-Instruct
- LoRA: rank=128, α=256, target modules: `q_proj, v_proj, o_proj, up_proj, down_proj, gate_proj`
- Optimizer: AdamW (8-bit), cosine schedule with 10% warmup
- Effective batch size: 4/GPU × 4 grad-accum × 8 GPUs = 128
- DeepSpeed ZeRO-2 (`configs/deepspeed_zero2.json`), optimizer states offloaded to CPU
- Training data: `ssd_vlm/data/ssd_sample_dataset.py` — multimodal replay (frames + JSONL completions)
- `train_full_ft.py` exists but is **ablation-only** — the paper method is LoRA

### Stage 3 — Evaluation (`eval/eval_ovo_bench.py`)
- OVO-Bench test split via `ssd_vlm/data/ovo_bench_dataset.py`
- 4-frame budget for all evals
- Produces per-task accuracy → aggregated into Lock (perception) vs Fork (memory) scores
- `eval/score_results.py` computes ΔLock and ΔFork between base and SSD-VLM

### Stage 4 — Visualization (`figures/`)
- `plot_pareto.py` → Perception-Memory Pareto frontier
- `plot_lock_fork_asymmetry.py` → per-task bar charts
- `plot_entropy.py` → output distribution entropy
- `figures/style.py` defines Nature-journal publication styling

### Key shared utilities
- `ssd_vlm/training/utils/model_loading.py` — abstracts loading base vs LoRA-merged checkpoints
- `ssd_vlm/data/video_utils.py` — frame extraction, resolution handling, frame cache (memory-mapped `.npy`)
- `configs/skill_categories.json` — defines Lock vs Fork task groupings

## Configuration

All hyperparameters live in `configs/*.yaml`. Key files:

| Config | Purpose |
|--------|---------|
| `sample_generation.yaml` | T=1.5, top-k=10, 4 frames, 2× memory oversample |
| `train_lora.yaml` | LoRA rank/alpha, LR, batch size, DeepSpeed path |
| `train_full_ft.yaml` | Full FT ablation (ZeRO-3) |
| `eval_ovo_base.yaml` / `eval_ovo_ssd.yaml` | Eval configs for base vs SSD-VLM |
| `deepspeed_zero2.json` | ZeRO-2 for LoRA training |
| `deepspeed_zero3.json` | ZeRO-3 for full FT ablation |

## Important Constraints

- **LoRA is the paper method**; `train_full_ft.py` is ablation-only — do not conflate them
- Frame budget is fixed at **4 frames** for all reported results (sweeps test other budgets separately)
- Teacher sampling uses **no labels or rewards** — raw completions only; this is the label-free claim
- `outputs/ssd_samples/samples.jsonl` is written incrementally (checkpoint every 100 batches) to survive interruptions
- Hardware target: 8× A100 80GB; single-GPU runs require reducing batch size and may require ZeRO-3
