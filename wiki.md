# SSD-VLM Wiki

Compact codebase reference. Loaded alongside `CLAUDE.md` to skip re-exploration on planning tasks. `CLAUDE.md` covers *what* and *why*; this file covers *where*.

Last refreshed: 2026-05-13. If file moves/renames invalidate paths below, refresh by re-running the `/plan` exploration.

---

## Module map (`ssd_vlm/`)

```
ssd_vlm/
├── __init__.py                      # lazy exports: PerceptionTestDataset, SSDSampleDataset
├── model_loading.py                 # load_vlm_processor_and_model(), quant+attn+pixel-cap logic
├── simplestream.py                  # OVO prompt formatting, scoring, task groups (LOCK/FORK)
├── eval_metrics.py                  # summarize_ovo_predictions() — aggregation logic split out of eval_ovo_bench.py
├── utils/
│   ├── __init__.py                  # re-exports load_config
│   └── config.py                    # load_config() with extends: deep-merge
├── sampling/
│   └── generate_samples.py          # Stage 1 entry — SSDSampleGenerator class
├── training/
│   ├── train_lora.py                # Stage 2 entry — LoRA (paper method)
│   ├── train_full_ft.py             # Stage 2 ablation — full FT (ZeRO-3)
│   └── utils.py                     # CosineWarmupScheduler, save_checkpoint, log_model_info
└── data/
    ├── __init__.py                  # exports OVOBenchDataset, SSDSampleDataset, create_*_dataloader(s)
    ├── perception_test_dataset.py   # train-split loader, 4-frame uniform, 2× memory oversample
    ├── ssd_sample_dataset.py        # JSONL replay loader for LoRA training
    ├── ovo_bench_dataset.py         # OVO-Bench test loader with frame cache
    └── video_utils.py               # load_video_frames_dual() / load_video_frames() / load_video_frame_images() / build_frame_transform()
```

**Important path correction**: CLAUDE.md mentions `ssd_vlm/training/utils/model_loading.py` — actual location is `ssd_vlm/model_loading.py` (top-level), and `ssd_vlm/training/utils.py` is a single file (not a package).

### Public APIs

| Symbol | Defined in | Used by |
|---|---|---|
| `load_vlm_processor_and_model(...)` | `ssd_vlm/model_loading.py` | sampling, training, all eval scripts |
| `is_peft_adapter_path(...)` | `ssd_vlm/model_loading.py` | model loading (LoRA vs merged) |
| `format_ovo_prompt(task_type, question, options)` | `ssd_vlm/simplestream.py` | eval, sampling, entropy analysis |
| `extract_choice(text, options)` / `score_prediction(...)` | `ssd_vlm/simplestream.py` | eval_ovo_bench, sweeps |
| `task_group(task_type)`, `LOCK_TASKS`/`FORK_TASKS` | `ssd_vlm/simplestream.py` | eval, datasets |
| `CosineWarmupScheduler`, `save_checkpoint`, `log_model_info` | `ssd_vlm/training/utils.py` | both trainers |
| `load_video_frames_dual` (one-pass tensor+PIL), `load_video_frames`, `load_video_frame_images`, `build_frame_transform` | `ssd_vlm/data/video_utils.py` | all three dataset classes |
| `create_ssd_sample_dataloader(s)` | `ssd_vlm/data/ssd_sample_dataset.py` | `train_lora.py`, `train_full_ft.py` |
| `apply_style()` | `figures/style.py` | every plot script |

Task sets (canonical in `simplestream.py`): `BACKWARD_TASKS=[EPM, ASI, HLD]` (Fork/memory), `REAL_TIME_TASKS=[OCR, ACR, ATR, STU, FPD, OJR]` (Lock/perception), `FORWARD_TASKS=[REC, SSR, CRR]`. `MC_DIRECTIVE`/`REC_DIRECTIVE`/`YES_NO_DIRECTIVE` constants drive prompt formatting.

---

## Dataflow

```
Perception Test ──► Stage 1 ──► samples.jsonl ──► Stage 2 ──► lora_ckpt/merged ──► Stage 3 ──► ovo_*.json ──► Stage 4 ──► *.pdf
   (train split)    sampling    {video_id,          training      (HF format)        eval        (per-task,     figures
                   T=1.5,k=10    question,                                          OVO test     Lock/Fork)
                   4 frames      options,                                           4 frames)
                   2× mem        completion,
                                 metadata}
```

JSONL is written incrementally (every 100 batches) — survives interruption. Frame cache: memory-mapped `.npy` files keyed by `(video_id, num_frames, resolution)`.

---

## Entry points

| Script | Stage | Config |
|---|---|---|
| `scripts/run_full_pipeline.sh DATA OUT RESULTS NGPU` | orchestrator | — |
| `scripts/run_mini_validation.sh` | mini end-to-end | `configs/mini/*.yaml` |
| `scripts/run_ablations.sh` | ablation suite | `configs/ablations/ablation_*.yaml` |
| `scripts/run_simplestream_baseline.sh` (+`.ps1`) | baseline (no SSD) | — |
| `ssd_vlm/sampling/generate_samples.py --config …` | 1 | `sample_generation.yaml` |
| `ssd_vlm/training/train_lora.py --config …` | 2 | `train_lora.yaml` |
| `ssd_vlm/training/train_full_ft.py --config …` | 2 (ablation) | `train_full_ft.yaml` |
| `eval/eval_ovo_bench.py --config …` | 3 | `eval_ovo_{base,ssd,base_1pct_t4,…}.yaml` |
| `eval/eval_frame_sweep.py` | 3 sweep | `eval_ovo_frame_sweep.yaml` |
| `eval/eval_temperature_sweep.py` | 3 sweep | `eval_temperature_sweep.yaml` |
| `eval/eval_dynamic_temperature.py` | 3 ablation | (config-driven) |
| `eval/eval_standard_ft_baseline.py` | 3 baseline | (config-driven) |
| `eval/eval_entropy_analysis.py`, `eval/compute_entropy.py`, `eval/compare_entropy.py` | analysis | — |
| `eval/score_results.py --base … --ssd …` | aggregation | — |
| `eval/statistical_tests.py` | significance | — |
| `figures/plot_all.py` | viz orchestrator | — |
| `figures/plot_*.py` (8 scripts) | viz | — |
| `scripts/prepare_mini_data.py` / `prepare_ovo_subset.py` / `chunk_ovo_subset.py` / `extract_ovo_src_subset.py` / `download_ovo_sources.py` | data prep | — |
| `scripts/smoke_qwen8b_load.py` | preflight | — |

PowerShell variants: `scripts/prepare_ovo_subset_pipeline.ps1`, `scripts/run_simplestream_baseline.ps1`.

---

## Configs (`configs/`)

**Main pipeline**: `sample_generation.yaml`, `train_lora.yaml`, `eval_ovo_base.yaml`, `eval_ovo_ssd.yaml`.

**Ablations** (`configs/ablations/`): `ablation_{sampling_temp, topk, no_oversample, oversample_ratio, dynamic_temperature, lora_only, full_ft_only, standard_ft, greedy_ssd, random_lora}.yaml`. **Not wired into `run_ablations.sh` yet** — orphaned until pipeline runner is extended.

**Sweeps**: `eval_ovo_frame_sweep.yaml`, `eval_temperature_sweep.yaml`.

**Low-resource subsets**: `eval_ovo_subset_{1,10}pct_{base,qwen2vl2b,qwen25vl3b}.yaml`, `eval_ovo_base_{1,10}pct_t4.yaml`.

**Mini**: `configs/mini/{sample_generation_mini, train_lora_mini, eval_ovo_mini}.yaml`.

**DeepSpeed**: `deepspeed_zero2.json` (LoRA), `deepspeed_zero3.json` (full FT).

**Shared data**: `configs/skill_categories.json` defines Lock/Fork groupings and oversample ratios — consumed by `perception_test_dataset.py`.

---

## Tests

| File | Coverage |
|---|---|
| `tests/smoke_test.py` (~900 lines) | imports, datasets, schedulers, checkpointing, scoring, dataloader iteration, prompt formatting — **mock-only, no GPU** |
| `tests/test_simplestream.py` | task groups, `format_ovo_prompt`, `score_prediction` |
| `tests/smoke_oom.py` | real GPU load test for Qwen3-VL-8B at NF4/int8/SDPA, peak VRAM measurement |
| `tests/compare_1pct.py` | untracked one-off — compares two result JSONs (hardcoded paths under `results/ovo_1pct/`) |

No `pytest.ini`/`conftest.py`/`pyproject.toml`. Coverage gaps: **no test exercises `train_lora.py`, `train_full_ft.py`, or `video_utils.py`**; choice-extraction is tested via re-implementation in `smoke_test.py` (lines 649-780) rather than direct import.

---

## Packaging

`setup.py`: package `ssd-vlm` v0.1.0. Deps: `transformers≥4.51`, `peft≥0.11`, `deepspeed≥0.14`, `pydantic`, `pyyaml`, `opencv-python`, `Pillow`, `tqdm`, `accelerate`. Extras: `dev` (pytest, black, isort, flake8), `viz` (matplotlib, seaborn). **Torch/torchvision/torchaudio NOT in requirements** — server has them pre-installed. No console-script entry points; everything runs via `python <path>` or `torchrun`.

---

## Known rough edges

These are flagged for the refactor — see `REFACTORING.md` for priorities.

Resolved (P1):
- ~~`load_config()` reimplemented in ≥9 scripts~~ → now `ssd_vlm/utils/config.py:load_config`.
- ~~Prompt formatting reimplemented in 4 files~~ → all callers use `ssd_vlm.simplestream.format_ovo_prompt`. **Sampling prompts changed shape**: SSD training data generated before this refactor uses the old `Question:/Options:/Answer:` template and may need regeneration to match the new eval-time shape.
- ~~`OVOBenchEvaluator._extract_choice()` duplicates `simplestream.extract_choice()`~~ → removed; canonical version used everywhere (incl. `eval_dynamic_temperature.py` and smoke tests).
- ~~Task-set literals duplicated in `prepare_mini_data.py` / `prepare_ovo_subset.py`~~ → import from `simplestream`.

Outstanding:
- `train_lora.py` and `train_full_ft.py` share ~70% of the epoch loop.
- `configs/ablations/*.yaml` exist but `run_ablations.sh` doesn't consume them all.
- `tests/compare_1pct.py` and `configs/eval_ovo_base_1pct_t4.yaml` are untracked WIP from a T4/1%-subset experiment.

Resolved (P2):
- ~~`eval_ovo_base.yaml` ≈ `eval_ovo_ssd.yaml`~~ → `extends:` deep-merge in `load_config`; SSD config is now a 12-line leaf.
- ~~T4 quantization block repeated across configs~~ → `configs/_t4_nf4_sdpa.yaml` is the canonical profile, consumed via `extends:`.
- ~~`eval/eval_ovo_bench.py` 613 lines / mixed concerns~~ → metrics aggregation extracted to `ssd_vlm/eval_metrics.py:summarize_ovo_predictions`. File is now 428 lines.
- ~~Datasets decoded each video twice~~ → `load_video_frames_dual` returns both tensor and PIL frames in one pass; all 3 dataset classes use it.

---

## How to keep this file useful

- When a public function moves or is renamed, update the "Public APIs" table.
- When a new entry point is added, add a row to "Entry points".
- When configs are reorganized, refresh the "Configs" section.
- Don't paste code snippets here — paths and one-line descriptions only. Code drifts; paths are checkable.
- Don't duplicate `CLAUDE.md` (project rationale, paper method, frame budgets). This file is *where*, not *why*.
