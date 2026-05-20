# SSD-VLM

Simple Self-Distillation for **Qwen3-VL-8B-Instruct** on streaming video understanding.
Sample at high temperature from the frozen teacher → LoRA-finetune on those raw completions → evaluate on **OVO-Bench**.
No reward model, no verification, no labels.

> **Where things live**: `wiki.md` (file map + public APIs) · `CLAUDE.md` (rationale, constraints, paper method) · `REFACTORING.md` (open cleanups). Read those for *why*; this README is *how to run*.

---

## TL;DR — common workflows

| I want to… | Run this |
|---|---|
| Reproduce SimpleStream Qwen3-VL 4-frame OVO numbers on a single T4 | [Single-GPU T4 eval](#single-gpu-t4-eval-most-common-today) |
| Train SSD-VLM end-to-end (8× A100) | [Full pipeline](#full-pipeline-8gpu) |
| Smoke-test on 1% of OVO before committing GPU hours | `--sample_ratio 0.01 --sample_seed 42` (see [Subsetting](#subsetting-the-eval-set)) |
| Compare a run vs. SimpleStream / OVO-Bench paper | [Analysis helpers](#analysis-helpers) |

---

## Install

```bash
# Torch/torchvision/torchaudio: use what's pre-installed on the server
pip install -r requirements.txt
pip install -e .
```

Qwen3 SimpleStream reproduction uses a pinned env (`transformers==4.57.6`, `accelerate==1.12.0`):

```bash
conda activate D:/conda_envs/env_ssd_simplestream_officialdeps
pip install -r requirements-qwen3-officialdeps.txt
pip install -e .
```

Required env vars before any eval/training run (on Windows / Azure T4):

```powershell
$env:HF_HOME            = "D:/hf_cache"            # keep model weights off C:\
$env:HUGGINGFACE_HUB_CACHE = "D:/hf_cache/hub"
$env:FORCE_QWENVL_VIDEO_READER = "decord"          # torchvision PyAV reader is broken here
$env:PYTHONIOENCODING   = "utf-8"                  # avoid cp949 stdout errors
```

(Linux/macOS: same names, `export VAR=...` syntax.)

---

## Single-GPU T4 eval (most common today)

The repo's main reproduction target is **SimpleStream Qwen3-VL 4-frame OVO-Bench** on a 16 GB T4.

### 1. Bootstrap OVO data on `D:\`

D:\ on Azure VMs is a **temporary disk** — wiped on stop/start. Re-run this whenever D:\ is empty.

```bash
# Stream-download HF's pre-chunked archive (152 GB, 15 parts) and delete
# each tar part as soon as the extractor reads it — peak disk stays low.
python -u scripts/download_extract_chunked.py \
  --repo_id JoeLeelyf/OVO-Bench \
  --tar_glob "chunked_videos.tar.part*" \
  --parts_dir D:/ssd_video_data/_chunked_parts \
  --output_dir D:/ssd_video_data/chunked_videos

# Pull the small annotation file
python -u scripts/download_ovo_sources.py \
  --data_root D:/ssd_video_data \
  --anno_path D:/ssd_video_data/ovo_bench_new.json \
  --skip_parts
```

For long evals on a small disk, extract last-32 frames as PNG and free the mp4s:

```bash
python scripts/extract_chunk_frames.py \
  --input_dir D:/ssd_video_data/chunked_videos \
  --output_dir D:/ssd_video_data/chunked_frames \
  --delete_source
```

Then flip `data.use_precomputed_frames: true` in the eval config.

### 2. Run the benchmark

```bash
python -u eval/eval_ovo_bench.py \
  --config configs/eval_ovo_simplestream_10pct_t4.yaml \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path D:/ssd_video_data \
  --output_file ./results/ovo_simplestream/qwen3vl8b_int8_t4.json \
  --sample_ratio 0.25 --sample_seed 42
```

The config sets `int8 + sdpa + image-list + use_simplestream_decode + max_new_tokens=256`.
Peak VRAM ≈ 15 GB; the evaluator writes a `.partial_predictions.jsonl` next to `--output_file` so an interrupted run resumes on the next launch (delete that file if you change the dataset or prompts).

### 3. Single-task / quick smoke variants

```bash
# HLD-only full eval (186 samples)
python -u eval/eval_ovo_bench.py --config configs/eval_ovo_hld_full_t4.yaml ...

# Faster but ~7pp HLD accuracy cost vs int8
python -u eval/eval_ovo_bench.py --config configs/eval_ovo_simplestream_10pct_t4_nf4.yaml ...
```

### Subsetting the eval set

`OVOBenchDataset` does stratified-by-task, grouped-by-source-video sampling deterministically. **No per-ratio directory needed**:

```bash
# CLI override:
--sample_ratio 0.01 --sample_seed 42 --sample_min_per_task 1
# or set in YAML: data.sample_ratio: 0.10
```

Forward tasks (REC/SSR/CRR) are sampled at the **annotation** level — every chunk of a selected annotation is included so per-source accuracy stays well-defined.

---

## Full pipeline (8 GPU)

End-to-end: teacher sampling → LoRA → eval → figures.

```bash
bash scripts/run_full_pipeline.sh ./data ./outputs ./results 8
```

Or run the four stages individually:

```bash
# 1. Sample from frozen teacher (T=1.5, top-k=10, 4 frames, no filtering)
torchrun --nproc_per_node=8 ssd_vlm/sampling/generate_samples.py \
  --config configs/sample_generation.yaml \
  --output_dir ./outputs/ssd_samples

# 2. LoRA fine-tune (paper method — rank 128, alpha 256)
torchrun --nproc_per_node=8 ssd_vlm/training/train_lora.py \
  --config configs/train_lora.yaml \
  --samples_path ./outputs/ssd_samples/samples.jsonl \
  --output_dir ./outputs/lora_checkpoint

# 3. Evaluate base + SSD on OVO-Bench
python eval/eval_ovo_bench.py --config configs/eval_ovo_base.yaml --output_file ./results/ovo_base.json
python eval/eval_ovo_bench.py --config configs/eval_ovo_ssd.yaml  --output_file ./results/ovo_ssd.json
python eval/score_results.py --base ./results/ovo_base.json --ssd ./results/ovo_ssd.json

# 4. Figures
python figures/plot_all.py
```

`train_full_ft.py` exists as an **ablation only** — the paper method is LoRA.

---

## Configs at a glance

`configs/` uses `extends:` deep-merge — leaf files are typically <15 lines and only override what differs from the base.

| File | Use it when… |
|---|---|
| `eval_ovo_base.yaml` / `eval_ovo_ssd.yaml` | Standard 8-GPU eval, base vs LoRA |
| `_t4_nf4_sdpa.yaml` | Shared T4 quantization profile (NF4 + SDPA + pixel cap) — others extend this |
| `eval_ovo_simplestream_10pct_t4.yaml` | T4, int8, SimpleStream Qwen3-VL parity (best HLD accuracy on T4) |
| `eval_ovo_simplestream_10pct_t4_nf4.yaml` | T4, NF4 — faster, ~7pp HLD cost |
| `eval_ovo_hld_full_t4.yaml` | HLD-only full eval (186 samples) |
| `eval_ovo_base_1pct_t4.yaml` | 1% smoke-test |
| `sample_generation.yaml` | Stage 1 — teacher sampling |
| `train_lora.yaml` | Stage 2 — LoRA |
| `train_full_ft.yaml` | Stage 2 ablation — full FT (ZeRO-3) |
| `ablations/ablation_*.yaml` | Hyperparameter sweeps (driven by `scripts/run_ablations.sh`) |
| `mini/*.yaml` | Tiny configs for `scripts/run_mini_validation.sh` |

To compose a custom config:

```yaml
# configs/eval_ovo_my_run.yaml
extends: eval_ovo_simplestream_10pct_t4.yaml
data:
  sample_ratio: 0.50
inference:
  max_new_tokens: 128
```

---

## Tests

GPU-free smoke tests run quickly and are safe to invoke at any point:

```bash
pytest tests/smoke_test.py tests/test_simplestream.py tests/test_seed.py
```

GPU smoke (loads Qwen3-VL-8B at NF4 and checks peak VRAM under 14.5 GB):

```bash
python tests/smoke_oom.py --quant nf4 --gen_tokens 16 --budget_gb 14.5
```

---

## Analysis helpers

After an eval run, compare against published numbers:

```bash
# Per-task delta vs SimpleStream Qwen3-VL 4-frame
python tests/compare_simplestream.py

# Per-task delta vs the OVO-Bench paper Table 1
python tests/compare_paper.py

# Wilson 95% CI per task — flags whether gaps are noise vs significant
python tests/analyze_variance.py          # vs OVO paper
python tests/analyze_simplestream_ci.py   # vs SimpleStream paper
```

At a 10% subset the SimpleStream numbers usually sit inside our 95% CIs — large-looking per-task gaps are typically small-N artifacts.

---

## Troubleshooting

**OOM on T4 eval** → switch to `eval_ovo_simplestream_10pct_t4_nf4.yaml` (NF4 instead of int8), or reduce `inference.max_new_tokens`. Verify peak VRAM with `tests/smoke_oom.py`.

**Forward tasks (REC/SSR/CRR) score ~0** → make sure you're on the canonical `eval/eval_ovo_bench.py`. The sweep / dynamic-temperature scripts now route through `ssd_vlm.simplestream.score_prediction` too, but if you wrote a custom evaluator, do the same — don't call `extract_choice` directly on a forward task.

**Slow data loading** → use the PNG cache path: run `scripts/extract_chunk_frames.py --delete_source` once, then flip `data.use_precomputed_frames: true`.

**Reproduction drifts run-to-run** → confirm `seed` is set in the YAML (defaults to 42). `set_global_seed()` is called at the top of every entry point and the DataLoader worker_init_fn is wired in, so identical configs should be bit-identical.

**`cp949` codec error on Windows** → set `PYTHONIOENCODING=utf-8` before running.

**D:\ wiped after VM stop/start** → that's expected, D:\ is the Azure temp disk. Re-run the [Bootstrap](#1-bootstrap-ovo-data-on-d) step.

---

## Citation

```bibtex
@article{ssd-vlm-2026,
  title  = {Simple Self-Distillation for Efficient Vision Language Models in Streaming Settings},
  year   = {2026}
}
```

MIT-licensed. Contributions welcome — please run `pytest tests/` and `black ssd_vlm/ eval/ figures/` before opening a PR.
