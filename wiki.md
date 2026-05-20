# SSD-VLM Wiki

Compact codebase reference. Loaded alongside `CLAUDE.md` to skip re-exploration on planning tasks. `CLAUDE.md` covers *what* and *why*; this file covers *where*.

Last refreshed: 2026-05-15 (after precomputed-frames pipeline: extract_chunk_frames.py, load_precomputed_frames, dead cache removal). If file moves/renames invalidate paths below, refresh by re-running the `/plan` exploration.

---

## Module map (`ssd_vlm/`)

```
ssd_vlm/
├── __init__.py                      # lazy exports: PerceptionTestDataset, SSDSampleDataset
├── model_loading.py                 # load_vlm_processor_and_model(), quant+attn+pixel-cap logic
├── simplestream.py                  # OVO prompt formatting, scoring, task groups (LOCK/FORK)
├── eval_metrics.py                  # summarize_ovo_predictions() — aggregation logic split out of eval_ovo_bench.py
├── utils/
│   ├── __init__.py                  # re-exports load_config, set_global_seed, seed_worker
│   ├── config.py                    # load_config() with extends: deep-merge
│   └── seed.py                      # set_global_seed() + seed_worker for DataLoader determinism
├── sampling/
│   └── generate_samples.py          # Stage 1 entry — SSDSampleGenerator class
├── training/
│   ├── train_lora.py                # Stage 2 entry — LoRA (paper method)
│   ├── train_full_ft.py             # Stage 2 ablation — full FT (ZeRO-3)
│   └── utils.py                     # CosineWarmupScheduler, save_checkpoint, log_model_info
└── data/
    ├── __init__.py                  # exports OVOBenchDataset, SSDSampleDataset, SSD sample dataloader helpers
    ├── perception_test_dataset.py   # train-split loader, 4-frame uniform, 2× memory oversample
    ├── ssd_sample_dataset.py        # JSONL replay loader for LoRA training
    ├── ovo_bench_dataset.py         # OVO-Bench test loader; supports sample_ratio + use_precomputed_frames
    ├── qwen_exact_recent_decoder.py # vendored from EvolvingLMMs-Lab/SimpleStream: decode only the last N tail frames
    └── video_utils.py               # load_video_frames_dual() (+ use_simplestream_decode + precomputed_frame_dir), load_precomputed_frames(), resolve_frame_dir(), _fetch_simplestream_frames()
```

**Important path correction**: CLAUDE.md mentions `ssd_vlm/training/utils/model_loading.py` — actual location is `ssd_vlm/model_loading.py` (top-level), and `ssd_vlm/training/utils.py` is a single file (not a package).

### Public APIs

| Symbol | Defined in | Used by |
|---|---|---|
| `load_vlm_processor_and_model(...)` | `ssd_vlm/model_loading.py` | sampling, training, all eval scripts |
| `is_peft_adapter_path(...)` | `ssd_vlm/model_loading.py` | model loading (LoRA vs merged) |
| `format_ovo_prompt(task_type, question, options)` | `ssd_vlm/simplestream.py` | eval, sampling, entropy analysis |
| `extract_choice(text)` / `extract_mcq_answer(text)` / `extract_number(text)` / `extract_yes_no(text)` / `score_prediction(...)` | `ssd_vlm/simplestream.py` | eval_ovo_bench, sweeps; verbatim SimpleStream substring rules |
| `BR_PROMPT_TEMPLATE` / `REC_PROMPT_TEMPLATE` / `SSR_PROMPT_TEMPLATE` / `CRR_PROMPT_TEMPLATE` | `ssd_vlm/simplestream.py` | mirrored from EvolvingLMMs-Lab/SimpleStream/ovo_constants.py — required for SimpleStream Qwen3-VL 4f reproduction |
| `task_group(task_type)`, `BACKWARD_TASKS`/`REAL_TIME_TASKS`/`FORWARD_TASKS` | `ssd_vlm/simplestream.py` | eval, datasets, prep scripts |
| `LOCK_TASKS`/`FORK_TASKS` aliases | `ssd_vlm/data/ovo_bench_dataset.py` | OVO eval and entropy scripts |
| `_forward_question_and_gt(task, anno, test_info, fallback)` | `ssd_vlm/data/ovo_bench_dataset.py` | REC/SSR/CRR ground-truth and question construction (count / type→bool / annotation question) |
| `_stratified_sample(samples, ratio, seed, min_per_task)` | `ssd_vlm/data/ovo_bench_dataset.py` | sample fractions of the dataset at load time without per-ratio subset dirs |
| `CosineWarmupScheduler`, `save_checkpoint`, `log_model_info` | `ssd_vlm/training/utils.py` | both trainers |
| `load_checkpoint(path, model, optimizer=None, scheduler=None)` | `ssd_vlm/training/utils.py` | restores model + optimizer + LR scheduler state; returns `(epoch, step)` |
| `set_global_seed(seed, *, deterministic=False)` / `seed_worker(worker_id)` | `ssd_vlm/utils/seed.py` | called at top of `generate_samples.py` / `train_lora.py` / `train_full_ft.py`; DataLoaders wire `seed_worker` as `worker_init_fn` |
| `load_video_frames_dual(..., use_simplestream_decode=False, precomputed_frame_dir=None)` | `ssd_vlm/data/video_utils.py` | all three dataset classes; flags swap decode path |
| `load_precomputed_frames(frame_dir, num_frames, *, expected_fps, expected_chunk_duration)` | `ssd_vlm/data/video_utils.py` | reads last N PNGs + meta.json written by `extract_chunk_frames.py`; raises ValueError on fps/chunk_duration mismatch |
| `resolve_frame_dir(data_path, video_id, chunked_frames_dir=None)` | `ssd_vlm/data/video_utils.py` | returns `<chunked_frames>/<video_id>/` if meta.json present, else None |
| `_fetch_simplestream_frames(...)` | `ssd_vlm/data/video_utils.py` | SimpleStream-aligned decode (qwen_vl_utils.fetch_video + chunk-by-time + auto-exact-recent when chunk_duration*fps==1.0) |
| `fetch_recent_video_exact(ele, last_nframes, ...)` | `ssd_vlm/data/qwen_exact_recent_decoder.py` | tail-only decord/torchcodec decode; avoids CPU OOM on long OVO chunks |
| `create_ssd_sample_dataloader(s)` | `ssd_vlm/data/ssd_sample_dataset.py` | `train_lora.py`, `train_full_ft.py` |
| `apply_style()` | `figures/style.py` | every plot script |

Task sets (canonical in `simplestream.py`): `BACKWARD_TASKS=[EPM, ASI, HLD]` (Fork/memory), `REAL_TIME_TASKS=[OCR, ACR, ATR, STU, FPD, OJR]` (Lock/perception), `FORWARD_TASKS=[REC, SSR, CRR]`. Per-task prompt templates `BR_PROMPT_TEMPLATE` / `REC_PROMPT_TEMPLATE` / `SSR_PROMPT_TEMPLATE` / `CRR_PROMPT_TEMPLATE` are copied verbatim from `EvolvingLMMs-Lab/SimpleStream/ovo_constants.py` so eval numbers compare against the published SimpleStream leaderboard.

**Forward-task ground truth** (in `_forward_question_and_gt`): REC uses `test_info[i].count` (int), SSR/CRR use `test_info[i].type` (1→True, 0→False), CRR question text comes from the annotation-level `question`. Default-to-zero for these tasks (the pre-SimpleStream-fix path) silently scored ~0 on REC/SSR for every sample.

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

JSONL is written incrementally (every 100 batches) — survives interruption. The `.npy` frame cache that previously existed in `read_video_frames` was never populated (callers always passed `frame_indices`) and has been removed.

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
| `eval/eval_ovo_bench.py --config …` | 3 | `eval_ovo_{base,ssd,base_1pct_t4,…}.yaml`; SimpleStream Qwen3 parity uses `inference.simplestream_qwen3_per_frame_builder: true` |
| `eval/eval_frame_sweep.py` | 3 sweep | `eval_ovo_frame_sweep.yaml` |
| `eval/eval_temperature_sweep.py` | 3 sweep | `eval_temperature_sweep.yaml` |
| `eval/eval_dynamic_temperature.py` | 3 ablation | (config-driven) |
| `eval/eval_standard_ft_baseline.py` | 3 baseline | (config-driven) |
| `eval/eval_entropy_analysis.py`, `eval/compute_entropy.py`, `eval/compare_entropy.py` | analysis | — |
| `eval/score_results.py --base … --ssd …` | aggregation | — |
| `eval/statistical_tests.py` | significance | — |
| `figures/plot_all.py` | viz orchestrator | — |
| `figures/plot_*.py` (8 scripts) | viz | — |
| `scripts/diagnose_hld_repro.py` | HLD reproduction diagnostics: annotation/cache audit, stored-vs-regex-vs-substring scoring, cached PNG vs SimpleStream reference frame selection, and Qwen3 processor encoding comparison | — |
| `scripts/cache_ovo_recent_frames.py` | writes SimpleStream recent-window PNG caches from OVO chunk videos, optionally deleting mp4s after cache validation | — |
| `scripts/eval_hld_progress.py` | small HLD eval helper with per-sample progress logging and resumable JSONL output | — |
| `scripts/download_data.sh` / `download_mini_data.sh` / `download_ovo_sources.py` | data download | — |
| `scripts/download_extract_chunked.py` | **preferred** stream-download + extract HF `chunked_videos.tar.part*` directly to `chunked_videos/`, deleting each tar part as soon as it is consumed; `--include_list` restores only named mp4s from the streamed archive | — |
| `scripts/extract_chunk_frames.py --input_dir … --output_dir … [--delete_source]` | **D:\ space reclaim**: walks `chunked_videos/*.mp4`, extracts last-32 frames as PNG (short-edge 384) + meta.json per chunk to `chunked_frames/<id>/`, optionally deletes the mp4. Idempotent — skips dirs where meta.json + png count already match. | — |
| `scripts/prepare_mini_data.py` / `prepare_ovo_subset.py` / `chunk_ovo_subset.py` / `extract_ovo_src_subset.py` | data prep (local chunking path, fallback) | — |
| `scripts/smoke_qwen8b_load.py` | preflight | — |

PowerShell variants: `scripts/prepare_ovo_subset_pipeline.ps1`, `scripts/run_simplestream_baseline.ps1`.

---

## Configs (`configs/`)

**Main pipeline**: `sample_generation.yaml`, `train_lora.yaml`, `eval_ovo_base.yaml`, `eval_ovo_ssd.yaml`.

**Ablations** (`configs/ablations/`): `ablation_{sampling_temp, topk, no_oversample, oversample_ratio, dynamic_temperature, lora_only, full_ft_only, standard_ft, greedy_ssd, random_lora}.yaml`. **Not wired into `run_ablations.sh` yet** — orphaned until pipeline runner is extended.

**Sweeps**: `eval_ovo_frame_sweep.yaml`, `eval_temperature_sweep.yaml`.

**Low-resource subsets**: `eval_ovo_subset_{1,10}pct_{base,qwen2vl2b,qwen25vl3b}.yaml`, `eval_ovo_base_{1,10}pct_t4.yaml`.

**T4 SimpleStream-aligned (single-GPU 16GB)**:
- `eval_ovo_simplestream_10pct_t4.yaml` — int8 + sdpa + image-list encoding + `use_simplestream_decode: true` + `max_new_tokens: 256`. Matches SimpleStream Qwen3-VL 4f to within ~5pp per task on the 10% subset.
- `eval_ovo_simplestream_10pct_t4_nf4.yaml` — same setup but NF4 instead of int8 (faster, ~7pp HLD accuracy cost).
- `eval_ovo_hld_full_t4{,_int8}.yaml` — HLD-only full eval (186 samples) to quantify the quantization tax in isolation.

**Subset selection at load time** (no per-ratio directory anymore): `OVOBenchDataset(..., sample_ratio=0.25, sample_seed=42, sample_min_per_task=1)` does stratified-by-task source-grouped sampling. The `eval/eval_ovo_bench.py` CLI exposes `--sample_ratio/--sample_seed/--sample_min_per_task` and the eval config reads `data.sample_ratio` if set. **Forward tasks (REC/SSR/CRR) are sampled at the annotation level** — every chunk of a selected annotation is included so per-source-video accuracy stays well-defined.

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
| `tests/compare_1pct.py` | one-off — compares two result JSONs under `results/ovo_1pct/` (Qwen2VL-2B vs Qwen3VL-8B NF4) |
| `tests/compare_subsets.py` | 1% vs 10% vs paper comparison helper (aggregate + per-task) |
| `tests/compare_paper.py` | per-task delta vs OVO-Bench paper Table 1 leaderboard |
| `tests/compare_simplestream.py` | per-task delta vs SimpleStream Qwen3-VL 4f published numbers (`OCR…OJR` realtime + `EPM/ASI/HLD` backward) |
| `tests/analyze_variance.py` | Wilson 95% CI per task vs OVO-Bench paper — flags noise-equivalent vs statistically distinguishable |
| `tests/analyze_simplestream_ci.py` | same but vs SimpleStream paper |
| `tests/test_diagnose_hld_repro.py` | no-GPU checks for HLD annotation audit, scoring split, and PNG cache metadata adapters |
| `tests/test_qwen3_builder.py` | no-GPU checks for the SimpleStream Qwen3 explicit per-frame input-id builder |

No `pytest.ini`/`conftest.py`/`pyproject.toml`. Coverage gaps: **no test exercises `train_lora.py` or `train_full_ft.py`**; choice-extraction is tested via re-implementation in `smoke_test.py` (lines 649-780) rather than direct import.

| `tests/test_seed.py` | `set_global_seed` determinism across python/numpy/torch + `seed_worker` smoke |

| `tests/test_precomputed_frames.py` | `load_precomputed_frames`, `resolve_frame_dir`, `extract_chunk_frames._extract_one` round-trip; 10 pass / 3 skip (cv2-dependent extractor tests skip without OpenCV) |

---

## Packaging

Qwen3 SimpleStream reproduction uses `D:\conda_envs\env_ssd_simplestream_officialdeps` and `requirements-qwen3-officialdeps.txt`, which pins `transformers==4.57.6` and `accelerate==1.12.0`.

`setup.py`: package `ssd-vlm` v0.1.0. Install deps: `transformers≥4.51`, `peft≥0.11`, `deepspeed≥0.14`, `pydantic`, `pyyaml`, `numpy`, `opencv-python-headless`, `Pillow`, `tqdm`, `accelerate`. `requirements.txt` adds experiment/dev packages such as `qwen-vl-utils`, `datasets`, `bitsandbytes`, `safetensors`, `pandas`, `scipy`, `scikit-learn`, plotting, notebook, and lint/test tools. Extras: `dev` (pytest, pytest-cov, black, isort, flake8), `viz` (matplotlib, seaborn). **Torch/torchvision/torchaudio NOT in requirements** — server has them pre-installed. No console-script entry points; everything runs via `python <path>` or `torchrun`.

---

## OVO-Bench data layout (D:\ on the T4 box)

**D:\ is the Azure VM temporary disk** — wiped on VM stop/start. **Preferred bootstrap path: `scripts/download_extract_chunked.py`** stream-downloads HF's pre-chunked archive (152 GB total, 15 parts) and deletes each tar part as soon as the extractor reads it. Peak disk usage stays near `extracted_so_far + 1-2 parts`; the previously documented `download_ovo_sources.py + extract_ovo_src_subset.py + chunk_ovo_subset.py` chain works but produced ~100 GB of locally-chunked videos and requires `src_videos.tar` (43 GB) alongside, which overflows the 176 GB disk. Layout after the SimpleStream-reproduction refactor:

```
D:\ssd_video_data\
├── ovo_bench_new.json         # raw HF annotation, 1640 entries
├── ovo_bench_full.json        # same content, enriched with video_relpath fields
├── ovo_src_parts\             # 43 GB src_videos.tar.part{aa..ae}, downloaded once
├── src_videos\                # full extraction: ~644 source videos, ~43 GB
├── chunked_videos\            # ~100 GB mp4s; run extract_chunk_frames.py --delete_source to reclaim
├── chunked_frames\            # ~3–6 GB PNGs after extraction; layout: <video_id>/frame_NN.png + meta.json
│   └── <video_id>/
│       ├── frame_00.png … frame_31.png   # last 32 frames at fps=1, short-edge=384
│       └── meta.json                     # extraction_fps, chunk_duration, frame_indices, saved_count, …
├── hf_cache\                  # Qwen3-VL-8B weights + processor (~16 GB)
├── required_sources.txt       # 644 paths, written by prepare_ovo_subset.py --ratio 1.0
├── required_chunks.txt        # ~3035 chunk filenames
└── subset_report.json
```

**After running `extract_chunk_frames.py --delete_source`**: `chunked_videos/` → 0 bytes; `chunked_frames/` ≤ 6 GB. Flip `data.use_precomputed_frames: true` in the eval config to activate the PNG path — `OVOBenchDataset` will route each chunk through `load_precomputed_frames` instead of decoding the mp4.

**Per-ratio subset directories are gone.** Earlier the pipeline materialised `ovo_subset_1pct/`, `ovo_subset_10pct/`, etc. — each with its own `src_videos\` + `chunked_videos\`. After the dataset-level sampling refactor (`OVOBenchDataset.sample_ratio`) we keep one fullset on D:\ and sample at load time, which saves ~50 GB per ratio and avoids cross-subset drift.

HF model cache lives on D:\ too (`D:\hf_cache\`) via `HF_HOME` env var. **Set `HF_HOME=D:/hf_cache` + `FORCE_QWENVL_VIDEO_READER=decord` + `PYTHONIOENCODING=utf-8` before any eval run** — defaults land model weights on C:\ and torchvision's PyAV reader is broken in this env.

### SimpleStream Qwen3 Reproduction Notes

Use `conda run -p D:\conda_envs\env_ssd_simplestream_officialdeps ...` for Qwen3 SimpleStream reproduction. The retired C:\ `env_ssd_simplestream` env used `transformers 5.8.0` / `accelerate 1.13.0` and reproduced HLD at only 44.62%; the official-deps env reproduces HLD at 53.76%.

The official `EvolvingLMMs-Lab/SimpleStream` Qwen3 OVO path does not use the plain Qwen `apply_chat_template(... add_generation_prompt=True)` path. It encodes the selected frames first, emits one explicit `<|vision_start|>...<|vision_end|>` block per frame, computes Qwen3 rope position IDs manually, then calls `generate` from `inputs_embeds`. Use `inference.simplestream_qwen3_per_frame_builder: true` to enable that path. `configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder_officialdeps.yaml` replays the C:\ 4-frame HLD PNG cache with this builder and `max_new_tokens: 256`; model/HF cache still belongs on D:\.

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
- `tests/compare_1pct.py` and `configs/eval_ovo_base_{1,10}pct_t4.yaml` are untracked WIP from a T4 subset experiment.

Resolved (P2):
- ~~`eval_ovo_base.yaml` ≈ `eval_ovo_ssd.yaml`~~ → `extends:` deep-merge in `load_config`; SSD config is now a 12-line leaf.
- ~~T4 quantization block repeated across configs~~ → `configs/_t4_nf4_sdpa.yaml` is the canonical profile, consumed via `extends:`.
- ~~`eval/eval_ovo_bench.py` 613 lines / mixed concerns~~ → metrics aggregation extracted to `ssd_vlm/eval_metrics.py:summarize_ovo_predictions`. File is now 428 lines.
- ~~Datasets decoded each video twice~~ → `load_video_frames_dual` returns both tensor and PIL frames in one pass; all 3 dataset classes use it.
- ~~`read_video_frames` leaking cv2.VideoCapture handles on exception~~ → wrapped in `try/finally cap.release()`.
- ~~Dead `.npy`/`.npz` frame cache in `read_video_frames`~~ → removed; callers always passed `frame_indices`, so the cache branch was never reached. `cache_dir`/`enable_cache` params dropped from all three datasets.
- ~~Download-thread errors delayed 300 s before surfacing~~ → `error_box` checked before `thread.join`; join timeout reduced to 30 s.

---

## Full OVO Qwen3 Reproduction

- Read `docs/experiments/full_ovo_qwen_repro_2026-05-18.md` before full OVO/Qwen3 SimpleStream reproduction work.
- Canonical full OVO local config: `configs/eval_ovo_full_precomputed4_t4_int8_qwen3builder_officialdeps.yaml`.
- Rebuild missing `D:/ssd_video_data` with `scripts/bootstrap_ovo_full_precomputed.ps1`; it uses streaming precompute by default to avoid filling D: with all mp4 chunks.
- Qwen/SimpleStream-aligned PNG replay caches are produced by `scripts/stream_precompute_ovo_chunked.py` for full bootstrap and `scripts/precompute_ovo_simplestream_frames.py` for already-extracted mp4s. Both use `_fetch_simplestream_frames` instead of cv2 extraction.

---

## How to keep this file useful

- When a public function moves or is renamed, update the "Public APIs" table.
- When a new entry point is added, add a row to "Entry points".
- When configs are reorganized, refresh the "Configs" section.
- Don't paste code snippets here — paths and one-line descriptions only. Code drifts; paths are checkable.
- Don't duplicate `CLAUDE.md` (project rationale, paper method, frame budgets). This file is *where*, not *why*.
