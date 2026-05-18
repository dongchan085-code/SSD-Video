# HLD Qwen3-VL-8B Reproduction Notes - 2026-05-18

Date/time basis: 2026-05-18 KST.

This note records the current SimpleStream-style Qwen3-VL-8B HLD reproduction state so future agents do not restart the same diagnosis from scratch. Update this file whenever a new HLD reproduction run changes the conclusion.

## Goal

Verify whether Qwen3-VL-8B gets the expected SimpleStream HLD score on OVO-Bench HLD with 4 recent frames.

Published comparison target used in local scripts:

- SimpleStream Qwen3-VL-8B 4f HLD: 52.1%.

## Current Assets

- HLD annotation subset: `data/ovo_hld_recent4/ovo_bench_hld.json`
- HLD precomputed frame cache: `data/ovo_hld_recent4/chunked_frames`
- Frame cache coverage: 186 HLD directories present.
- Runtime cache after restart: `D:/hf_cache`
- As of the initial 2026-05-18 check, `D:/ssd_video_data` was absent, but HLD precomputed PNG replay does not need the original mp4s.

## Validated Completed Runs

### Int8, precomputed4, standard image-list path

Result file:

- `results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_t4.json`

Configuration snapshot:

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Quantization: int8
- Device: single T4
- Frames: 4 recent frames
- Data path: `data/ovo_hld_recent4`
- Input frames: precomputed PNG cache
- Prompt/decode: SimpleStream-style prompt and decode, but not the official Qwen3 explicit per-frame builder

Metric:

- 186 / 186 HLD samples completed
- 77 / 186 correct
- Stored/release-regex accuracy: 41.3978%
- Official substring rescoring: 41.9355%
- Gap vs 52.1 target: about -10.7 pp
- Wilson 95% CI from diagnostic script: 34.56% to 48.58%; 52.1% is outside the interval.

Conclusion:

- This run does not reproduce the published 52.1% HLD target.
- The score gap is not explained by the local regex-vs-official substring scoring rule; rescoring changes only +0.54 pp.

### NF4 HLD full baseline

Result file:

- `results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.json`

Metric:

- 186 / 186 HLD samples completed
- 69 / 186 correct
- Accuracy: 37.10%

Conclusion:

- NF4 is lower than int8 on this HLD setup and is not the preferred path for checking the 52.1 target.

## Diagnostics Already Done

Annotation/cache audit:

- Command family: `scripts/diagnose_hld_repro.py audit-annotation`
- HLD annotations: 186
- Manifest rows: 186
- Duplicate ids: none
- Cache missing: 0
- Cache entries with fewer than 4 recent frames: 0
- Ground-truth outside options: 0
- Ground-truth outside A-D: 1
- Non-4-option HLD rows: 38
- Manifest GT mismatches: 0
- Manifest prompt mismatches: 0

Frame comparison against SimpleStream reference decode:

- Sample file: `results/diagnostics/hld_repro/hld_recent4_frame_compare_sample5.json`
- Compared reference videos: 5 / 5
- Timestamp mismatches: 0
- Chunk id mismatches: 0
- Frame-count mismatches: 0

Scoring comparison:

- File: `results/diagnostics/hld_repro/hld_int8_precomputed4_score_compare.json`
- Stored/release-regex: 41.3978%
- Official substring: 41.9355%
- Regex/substring differing rows: 1

Conclusion from diagnostics:

- The known data, frame-selection, and scoring checks do not explain the remaining HLD gap.
- The remaining likely variable is the Qwen3-specific input construction/model execution path.

## Completed Run: Official Qwen3 Explicit Per-frame Builder

Configuration:

- `configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder.yaml`

Expected output:

- `results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json`

Partial file:

- `results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.partial_predictions.jsonl`

Historical command context from 2026-05-18:

```powershell
$env:PYTHONPATH='C:/work/SSD-Video'
$env:HF_HOME='D:/hf_cache'
$env:HUGGINGFACE_HUB_CACHE='D:/hf_cache/hub'
$env:TRANSFORMERS_CACHE='D:/hf_cache/transformers'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
$env:FORCE_QWENVL_VIDEO_READER='decord'
$env:PYTHONIOENCODING='utf-8'
New-Item -ItemType Directory -Force -Path D:/hf_cache | Out-Null
# The original low-score run used the now-retired C:/Users/swsuser-j07/.conda/envs/env_ssd_simplestream env.
# Do not rerun that environment; use the official-dependency rerun command later in this document.
```

Final status:

- At 2026-05-18 09:39 KST, the run resumed from 59 partial predictions.
- At 2026-05-18 09:43 KST, partial progress was 71 / 186, with 29 correct and 40.85% cumulative accuracy.
- At 2026-05-18 09:57 KST, partial progress was 79 / 186, with 33 correct and 41.77% cumulative accuracy. The latest appended row was video id 339 at 09:53:20 KST; the run was still active as PID 12308.
- At 2026-05-18 10:38 KST, the run completed all 186 HLD samples.
- Final stored/release-regex accuracy: 83 / 186 = 44.6237%.
- Official substring rescoring: 44.6237% (0.0 pp difference from stored/release-regex scoring).
- Gap vs 52.1 target: -7.4763 pp.
- Wilson 95% CI from diagnostic script: 37.66% to 51.80%; 52.1% is outside the interval.
- Mean latency: 22195.6 ms; p50: 7131.4 ms; p95: 51866.7 ms.
- Reported peak allocated GPU memory in the result JSON: 13.30 GB. Live `nvidia-smi` memory during the resumed run was near 16.0 GB on T4.

Important resume note:

- The partial file had 59 unique rows before the 2026-05-18 resume.
- The evaluator uses `resume_partial: true` and skips completed `video_id` values, so it can safely continue from that JSONL.
- Do not delete the partial file unless intentionally restarting from zero.

## How To Check Status

```powershell
$rows = Get-Content results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.partial_predictions.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$n = @($rows).Count
$c = @($rows | Where-Object { $_.correct -eq $true }).Count
"partial=$n pending=$(186-$n) correct=$c acc=$([math]::Round(100*$c/[math]::Max($n,1),2)) last=$($rows[-1].video_id)"
nvidia-smi
```

The final output JSON now exists. Re-run the following diagnostics if the file is regenerated:

```powershell
conda run -p D:\conda_envs\env_ssd_simplestream_officialdeps python scripts/diagnose_hld_repro.py audit-results `
  --result-path results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json `
  --task HLD `
  --paper-accuracy 52.1

conda run -p D:\conda_envs\env_ssd_simplestream_officialdeps python scripts/diagnose_hld_repro.py score-compare `
  --result-path results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json `
  --task HLD
```

## Current Interpretation

Qwen3-VL-8B HLD performance now matches the 52.1% SimpleStream target when the T4/int8/sdpa run uses the official Qwen3 dependency pins (`transformers==4.57.6`, `accelerate==1.12.0`). The earlier low score was primarily a Transformers/Accelerate runtime mismatch, not annotation, prompt, frame selection, or scoring.

## 2026-05-18 Diagnosis Update

The most likely cause of the remaining gap is a runtime/reproduction-condition mismatch, not annotation, prompt, frame selection, or scoring.

Confirmed official SimpleStream Qwen3 conditions from the upstream release:

- `main_experiments/run_qwen3vl_ovo_4gpu.sh` runs 4 processes with `--mixed_precision bf16`, `recent_frames_only=4`, and no quantization.
- `lib/recent_window_eval_qwen3.py` loads `Qwen/Qwen3-VL-8B-Instruct` with `torch_dtype=torch.bfloat16` and `attn_implementation="flash_attention_2"`.
- `requirements-qwen3.txt` pins `transformers==4.57.6` and `accelerate==1.12.0`.

Current local T4 reproduction conditions:

- `configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder.yaml` uses `dtype=float16`, `load_in_8bit=true`, `attn_implementation=sdpa`, one T4, and precomputed PNG replay.
- The retired `env_ssd_simplestream` env reported `transformers 5.8.0`, `accelerate 1.13.0`, `torch 2.5.1`, CUDA 12.4 and produced the low 44.62% run.
- This means the local run differs from the release on precision, quantization, attention backend, GPU execution mode, and Transformers/Accelerate versions.

Additional observations:

- In the local HLD-only subset (`data/ovo_hld_recent4/ovo_bench_hld.json`), the HLD ground-truth option text is always `Unable to answer`. This is not a statement about every OVO-Bench task; it is specific to the HLD subset used for this reproduction.
- For this HLD subset, the score is effectively the rate at which the model selects the option letter containing `Unable to answer`.
- The qwen3builder run selected the `Unable to answer` option 83 / 186 times, exactly matching the 83 correct rows.
- Hitting 52.1% on this 186-row HLD subset would require about 97 correct rows, i.e. 14 more `Unable to answer` selections than the current int8/T4 run.
- The precomputed PNG cache is being used (`use_precomputed_frames=true`), has 186 HLD frame directories, and all cached `meta.json` files report `resize_shortest_edge=null`, `saved_count=4`. The cached image sizes are Qwen-style multiples/ranges such as 560-1008 wide by 336-672 high.

Ruled out or unlikely:

- Official substring rescoring does not change the qwen3builder result.
- HLD annotation/cache coverage and prompt manifest checks pass.
- Sampled SimpleStream reference frame comparison found no timestamp, chunk-id, or frame-count mismatches on the checked samples.
- The explicit per-frame Qwen3 builder is now aligned with the upstream input construction path.

Still worth checking only if exact official hardware is available:

- Run upstream SimpleStream unchanged in an environment pinned to `requirements-qwen3.txt`, with bf16 + flash-attention on supported GPUs.
- Compare pixel hashes or processor tensors for a sample of cached PNGs against upstream decode output; current diagnostics compare frame timing/counts, not full pixel identity.

## 2026-05-18 Official-Dependency T4 Rerun

Environment:

- Conda prefix: `D:/conda_envs/env_ssd_simplestream_officialdeps` (the active Qwen3 SimpleStream reproduction env)
- Created by cloning `env_ssd_simplestream`, then installing `transformers==4.57.6` and `accelerate==1.12.0`.
- Verified versions: `transformers 4.57.6`, `accelerate 1.12.0`, `torch 2.5.1`, CUDA 12.4, `bitsandbytes 0.49.2`.
- Runtime remained T4-constrained: `dtype=float16`, `load_in_8bit=true`, `attn_implementation=sdpa`, single T4, precomputed PNG replay.
- As of 2026-05-18, the old `C:/Users/swsuser-j07/.conda/envs/env_ssd_simplestream` env is retired and should not be used for Qwen3 SimpleStream reproduction.

Configuration:

- `configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder_officialdeps.yaml`

Result files:

- `results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4_officialdeps.json`
- `results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4_officialdeps.partial_predictions.jsonl`

Command:

```powershell
$env:PYTHONPATH='C:/work/SSD-Video'
$env:HF_HOME='D:/hf_cache'
$env:HUGGINGFACE_HUB_CACHE='D:/hf_cache/hub'
$env:TRANSFORMERS_CACHE='D:/hf_cache/transformers'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
$env:FORCE_QWENVL_VIDEO_READER='decord'
$env:PYTHONIOENCODING='utf-8'
conda run -p D:\conda_envs\env_ssd_simplestream_officialdeps python -u eval/eval_ovo_bench.py `
  --config configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder_officialdeps.yaml `
  --model_path Qwen/Qwen3-VL-8B-Instruct `
  --data_path C:/work/SSD-Video/data/ovo_hld_recent4 `
  --output_file results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4_officialdeps.json `
  --task_filter HLD `
  --sample_ratio 1.0
```

Final metric:

- 186 / 186 HLD samples completed.
- Stored/release-regex accuracy: 100 / 186 = 53.7634%.
- Official substring rescoring: 54.3011%.
- Gap vs 52.1 target: +1.6634 pp stored, +2.2011 pp official-substring.
- Wilson 95% CI: 46.59% to 60.78%; 52.1% is inside the interval.
- Mean latency: 9176.8 ms; p50: 8463.2 ms; p95: 18730.6 ms.
- Reported peak allocated GPU memory: 13.83 GB.

Comparison against the previous `transformers 5.8.0` / `accelerate 1.13.0` qwen3builder run:

- Previous: 83 / 186 = 44.6237%.
- Official-deps rerun: 100 / 186 = 53.7634%.
- Delta: +17 correct, +9.1398 pp.
- Row-level correctness flips: 25 total; 21 false-to-true, 4 true-to-false.

Conclusion:

- The HLD gap is reproduced as a dependency/runtime issue. Pinning the official Qwen3 Transformers/Accelerate stack is enough for the T4 int8/sdpa run to meet or exceed the 52.1 target.
- The run still is not bit-for-bit official hardware parity because it uses int8 + fp16 + sdpa + single T4 instead of unquantized bf16 + flash-attention on 4 GPUs, but the HLD score target is now matched.

## Random Seed Analysis

SimpleStream's OVO Qwen3 script uses Python randomness only for deterministic annotation ordering:

- `main_experiments/eval_qwen3vl_ovo.py` calls `random.seed(42)` and then shuffles the backward, realtime, and forward annotation lists.
- The shuffle affects evaluation order and multi-GPU shard assignment. It affects the evaluated subset only when `--max_samples_per_split` is used.
- Full HLD evaluation over all 186 rows should not change accuracy because of this shuffle.

Generation itself is greedy:

- `lib/recent_window_eval.py` calls `model.generate(..., do_sample=False)`.
- With `do_sample=False`, temperature/top-p/top-k sampling randomness is not active. Positive temperature scaling would not change argmax token choice, and Transformers warns that sampling flags such as `top_k` may be ignored.
- Remaining differences can still come from dependency versions, quantization, attention kernels, or low-level GPU non-determinism, but not from sampling seed in the normal sense.

## 2026-05-18 Full-Precision CPU-Offload Smoke

Configuration:

- `configs/eval_ovo_hld_precomputed4_t4_full_cpuoffload_smoke.yaml`

Result files:

- `results/ovo_simplestream_fullset/qwen3vl8b_full_cpuoffload_hld_precomputed4_qwen3builder_t4_smoke.json`
- `results/ovo_simplestream_fullset/qwen3vl8b_full_cpuoffload_hld_precomputed4_qwen3builder_t4_smoke.partial_predictions.jsonl`

Runtime:

- `dtype=float16`, `load_in_8bit=false`, `load_in_4bit=false`, `device_map=auto`, `max_memory: {0: 12GiB, cpu: 16GiB}`, `attn_implementation=sdpa`.
- Model load emitted the expected Accelerate warning that some parameters were offloaded to CPU.

Smoke result:

- 10 / 10 samples completed.
- Accuracy: 5 / 10 = 50.0%.
- For these first 10 HLD rows, predictions exactly matched the official-deps int8 run.
- Generation latency was roughly 7-10 seconds per sample after model load. A full 186-row CPU-offload run is possible in principle but is not a strong use of T4 time unless more evidence is needed.
