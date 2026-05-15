# SSD-VLM Refactoring Plan

## Context

The codebase grew across four pipeline stages (sample → train → eval → viz) with several quick fixes layered on top (NF4/SDPA for T4 in `fa26e50`, SimpleStream subset pipeline in `3397bd5`). The result: ~600 lines of duplicated boilerplate, drift between near-identical files (`train_lora`/`train_full_ft`, `eval_ovo_base`/`eval_ovo_ssd`), and one large file (`eval/eval_ovo_bench.py`, 613 lines) mixing inference, scoring, and IO concerns.

**Intended outcome**: a smaller surface to maintain, with shared helpers used uniformly, so a future change to (e.g.) prompt formatting touches one file instead of four.

**Non-goals**: changing the paper method, the training hyperparameters, the metrics, or the file/directory layout users already know. Behavior must be identical pre/post each step (verified by `tests/smoke_test.py` and a smoke run on `eval_ovo_base_1pct_t4.yaml`).

Cross-reference: `wiki.md` (file-level inventory). Items below are ordered by ROI, not by file location.

---

## Status

| ID | Status | Notes |
|---|---|---|
| R1 | done | `ssd_vlm/utils/config.py:load_config`; 9 duplicates removed |
| R2 | done | All paths now call `simplestream.format_ovo_prompt`. **Behavior changed**: SSD sample-generation prompts switched from `Question:/Options:/Answer:` shape to the canonical SimpleStream MC shape (`A.`/`B.` + directive). Previously, sampling and eval used different prompts — now unified. |
| R3 | done | `OVOBenchEvaluator._extract_choice` removed; `simplestream.extract_choice` used everywhere |
| R4 | done | `simplestream.py` is the single source; `scripts/prepare_*` imports the canonical sets |
| R5 | done | `extends:` deep-merge in `load_config`; `eval_ovo_ssd.yaml` → `eval_ovo_base.yaml`. Chained extends supported. |
| R8 | done (partial) | Metrics aggregation pulled into `ssd_vlm/eval_metrics.py:summarize_ovo_predictions`; `eval/eval_ovo_bench.py` 545 → 428 lines. Kept the class in-place so sweep scripts' `from eval_ovo_bench import …` still works. |
| R9 | done | YAML profile `configs/_t4_nf4_sdpa.yaml`; `eval_ovo_base.yaml` and `sample_generation.yaml` extend it. Chose YAML inheritance over a Python `QuantizationProfile` module because the duplication was config-side, not Python. |
| R7 | done | `video_utils.load_video_frames_dual` decodes once and returns both tensor + PIL frames. The 3 dataset classes (`OVOBenchDataset`, `SSDSampleDataset`, `PerceptionTestDataset`) now decode each video once per item instead of twice. |
| R6 | deferred | Skipped for now — trainer code can't be validated without GPU; FullFTTrainer is ablation-only so risk/reward unfavorable until P3. |
| R10 | done | `run_ablations.sh` extended with ablations 10–12 (sampling temp / top-k / oversample ratio sweeps); `generate_samples.py` gained `--set section.key=value` CLI override |
| R11 | done | `tests/compare_1pct.py` parameterized with positional CLI args; defaults retained |
| R12 | pending | `test_train_lora_smoke.py` still needs GPU; precomputed-frames path covered by `test_precomputed_frames.py` |

## Priority 1 — High-ROI deduplication (do first)

### R1. Extract a shared `load_config` helper
**Problem**: identical 3–4 line `load_config(path) → dict` reimplemented in `ssd_vlm/sampling/generate_samples.py`, `ssd_vlm/training/train_lora.py`, `ssd_vlm/training/train_full_ft.py`, and every script under `eval/` (~9 files total).
**Fix**: add `ssd_vlm/utils/config.py` exposing `load_config(path: str | Path) -> dict`. Replace each local definition with `from ssd_vlm.utils.config import load_config`.
**Verify**: `pytest tests/` + run `python eval/eval_ovo_bench.py --config configs/eval_ovo_base_1pct_t4.yaml` and compare output to the pre-refactor JSON.

### R2. Stop reimplementing prompt formatting; always call `simplestream.format_ovo_prompt`
**Problem**: `eval/eval_ovo_bench.py` (~L131-141), `ssd_vlm/sampling/generate_samples.py` (~L68-83), `eval/eval_entropy_analysis.py` (~L84-95), `eval/compute_entropy.py` (~L32-37) each have their own prompt-builder. Two delegate to `simplestream.format_ovo_prompt`, two roll their own — meaning a prompt-shape change today silently affects only some paths.
**Fix**: delete local `_format_prompt`/`build_prompt` helpers; import `format_ovo_prompt` from `ssd_vlm.simplestream` everywhere.
**Verify**: prompt strings produced by sampling and eval should be byte-identical (add a diff assertion in `tests/test_simplestream.py`).

### R3. Use `simplestream.extract_choice` for letter parsing
**Problem**: `OVOBenchEvaluator._extract_choice()` in `eval/eval_ovo_bench.py:217-264` (~48 lines, 5 regex fallbacks) duplicates `simplestream.extract_choice()` (~L59-79). Smoke test `tests/smoke_test.py:649-780` then re-implements *both* to test them.
**Fix**: delete `_extract_choice` from the evaluator. Replace call sites with `from ssd_vlm.simplestream import extract_choice`. Rewrite the smoke test to import the canonical function rather than re-defining it.
**Verify**: per-task accuracy on `eval_ovo_base_1pct_t4.yaml` must match exactly.

### R4. Consolidate task-set constants
**Problem**: `ssd_vlm/data/ovo_bench_dataset.py:28-32` redefines `LOCK_TASKS`/`FORK_TASKS` constants that `ssd_vlm/simplestream.py:9-16` already defines. Some downstream files import from one, some from the other.
**Fix**: keep the definitions in `simplestream.py` only. In `ovo_bench_dataset.py`, replace the local literals with `from ssd_vlm.simplestream import REAL_TIME_TASKS as LOCK_TASKS, BACKWARD_TASKS as FORK_TASKS` (or re-export under new names if the existing import surface matters).
**Verify**: grep that no `[\"EPM\"`, `\"ASI\"`, …] literal lists exist outside `simplestream.py`.

---

## Priority 2 — Structural cleanups

### R5. Share base eval-config via YAML anchor or include
**Problem**: `configs/eval_ovo_base.yaml` (53 lines) and `configs/eval_ovo_ssd.yaml` (56 lines) differ in only `model_id` vs `model_path` + `is_merged: true`. Same drift risk for the four `eval_ovo_subset_*` variants and the `*_t4` configs.
**Fix**: factor a `configs/_eval_ovo_common.yaml` and load-merge in code (since PyYAML doesn't natively support includes, do the merge in `ssd_vlm/utils/config.py` — `load_config(path)` resolves a top-level `extends:` key by merging the parent first). Reduces every leaf config to ~10 lines.
**Verify**: `python eval/eval_ovo_bench.py` on each refactored config produces identical resolved config dict (log it once before/after).

### R6. Extract a `BaseTrainer` for the shared epoch loop
**Problem**: `ssd_vlm/training/train_lora.py:190-275` and `ssd_vlm/training/train_full_ft.py:120-191` repeat the tqdm wrapper, gradient-accumulation step, checkpoint-cadence check, and early stop. ~70% overlap.
**Fix**: introduce `ssd_vlm/training/base_trainer.py:BaseTrainer` exposing `_train_epoch`, `_evaluate`, `_save_checkpoint`. `LoRATrainer` and `FullFTTrainer` override only `_setup_model` (quant + LoRA vs. full).
**Verify**: a 1-step training smoke (mini config) on each trainer should produce the same first-step loss curve as before. `pytest tests/smoke_test.py::test_training_*`.

### R7. Unify video frame loading with `VideoFrameLoader`
**Problem**: `perception_test_dataset.py:164-198`, `ssd_sample_dataset.py:133-159`, `ovo_bench_dataset.py` each call both `load_video_frames` and `load_video_frame_images` with overlapping caching/parameter handling.
**Fix**: create `ssd_vlm/data/video_loader.py:VideoFrameLoader(num_frames, resolution, cache_dir)` exposing `load(video_id) -> (tensor, pil_images)`. Each dataset class holds one instance.
**Verify**: dataset `__getitem__` outputs (tensor shape + sample of PIL image array) should match byte-for-byte before/after. Add a `test_video_loader.py` covering at least the cache-hit path.

### R8. Split `eval/eval_ovo_bench.py` (613 lines)
**Problem**: single file mixes evaluator orchestration, per-task scoring, latency tracking, choice extraction, and CLI plumbing.
**Fix**: split into:
- `eval/ovo_bench/evaluator.py` (the `OVOBenchEvaluator` class)
- `eval/ovo_bench/metrics.py` (accuracy, latency, Lock/Fork aggregation)
- `eval/eval_ovo_bench.py` (thin CLI: arg parse → evaluator → write JSON)

Sweep scripts (`eval_frame_sweep.py`, `eval_temperature_sweep.py`, `eval_dynamic_temperature.py`) currently each subclass the evaluator + repeat the sweep loop — once `evaluator.py` is split out, add `eval/ovo_bench/sweep.py:run_sweep(param_name, param_values, …)` and rewrite the three sweep scripts as ~10-line entry points.

**Verify**: `python eval/eval_ovo_bench.py --config configs/eval_ovo_base_1pct_t4.yaml` produces the same JSON. Each sweep script's output JSON is unchanged on the mini config.

### R9. Centralize quantization profile (cleanup for `fa26e50`)
**Problem**: NF4 + SDPA + pixel-cap logic is split across `ssd_vlm/model_loading.py:41-82` and referenced separately in `sample_generation.yaml`, `eval_ovo_*_t4.yaml`, `eval_ovo_subset_*.yaml`. Comments about "T4 memory" appear in four files.
**Fix**: add `ssd_vlm/quantization.py` defining `QuantizationProfile` (name → dict of {load_in_4bit/8bit, bnb_4bit_quant_type, attn_impl, max_pixels, min_pixels, dtype}) with named presets: `a100_bf16`, `t4_nf4_sdpa`, `t4_int8`. Configs reference a single `quantization: t4_nf4_sdpa` key; `load_vlm_processor_and_model` resolves it.
**Verify**: `tests/smoke_oom.py` still passes; eval on `eval_ovo_base_1pct_t4.yaml` produces same JSON.

---

## Priority 3 — Hygiene

### R10. Wire ablation configs into `run_ablations.sh` or delete the orphans
**Problem**: `configs/ablations/ablation_*.yaml` (10 files) exist but the runner script doesn't iterate them all. Several reference flags that aren't read by any code (silent no-ops).
**Fix**: pick one — either extend `scripts/run_ablations.sh` to loop all 10 configs through the appropriate trainer/evaluator, OR delete the unused configs. Recommend: keep all 10, extend the runner with a `for cfg in configs/ablations/*.yaml; do …` loop dispatched by a `kind:` field in each YAML.

### R11. Land or remove the in-progress 1%/T4 experiment files
**Untracked**: `configs/eval_ovo_base_1pct_t4.yaml`, `tests/compare_1pct.py` (hardcoded paths). These are useful and should be promoted: move `compare_1pct.py` under `eval/`, parameterize the hardcoded paths, add to `setup.py`'s test discovery if appropriate.

### R12. Add direct tests for the untested paths
After R3/R6/R7 land, write thin tests:
- `tests/test_video_utils.py` — covers `load_video_frames` cache-hit and miss.
- `tests/test_train_lora_smoke.py` — instantiate `LoRATrainer` with mini config, run 1 step on a synthetic batch.

These should each be under 50 lines and not require GPU (mock the model where appropriate).

---

## Execution order

A safe sequence that keeps the smoke test green at every commit:

1. R1 (load_config) — pure additive, no behavior change.
2. R4 (task-set consolidation) — local imports only.
3. R3 (extract_choice) → R2 (format_ovo_prompt) — must come *with* a test diff to prove byte-identical output.
4. R5 (config extends:) — requires R1's `load_config` to host the merge.
5. R9 (quantization profile) — touches model loading + multiple configs; do separately.
6. R8 (split eval_ovo_bench) — biggest mechanical change; do after R2/R3 so the new modules already use the shared helpers.
7. R6 (BaseTrainer), R7 (VideoFrameLoader) — independent of each other, can interleave.
8. R10 / R11 / R12 — hygiene, any order, after the above stabilizes.

Each step should land as one commit, with `pytest tests/` + an OVO-Bench mini smoke (`eval_ovo_base_1pct_t4.yaml`) green in CI before pushing.

---

## Verification harness (run after every step)

```bash
# 1. unit + smoke (no GPU)
pytest tests/smoke_test.py tests/test_simplestream.py

# 2. 1%-subset eval on T4-style config — produces results/ovo_1pct.json
python eval/eval_ovo_bench.py --config configs/eval_ovo_base_1pct_t4.yaml

# 3. compare against committed baseline JSON
python tests/compare_1pct.py
```

Add the comparison step to CI once `tests/compare_1pct.py` is parameterized (R11).
