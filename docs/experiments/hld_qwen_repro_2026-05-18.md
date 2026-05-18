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

Command used on 2026-05-18:

```powershell
$env:PYTHONPATH='C:/work/SSD-Video'
$env:HF_HOME='D:/hf_cache'
$env:HUGGINGFACE_HUB_CACHE='D:/hf_cache/hub'
$env:TRANSFORMERS_CACHE='D:/hf_cache/transformers'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
$env:FORCE_QWENVL_VIDEO_READER='decord'
$env:PYTHONIOENCODING='utf-8'
New-Item -ItemType Directory -Force -Path D:/hf_cache | Out-Null
conda run -n env_ssd_simplestream python -u eval/eval_ovo_bench.py `
  --config configs/eval_ovo_hld_precomputed4_t4_int8_qwen3builder.yaml `
  --model_path Qwen/Qwen3-VL-8B-Instruct `
  --data_path C:/work/SSD-Video/data/ovo_hld_recent4 `
  --output_file results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json `
  --task_filter HLD `
  --sample_ratio 1.0
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
conda run -n env_ssd_simplestream python scripts/diagnose_hld_repro.py audit-results `
  --result-path results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json `
  --task HLD `
  --paper-accuracy 52.1

conda run -n env_ssd_simplestream python scripts/diagnose_hld_repro.py score-compare `
  --result-path results/ovo_simplestream_fullset/qwen3vl8b_int8_hld_precomputed4_qwen3builder_t4.json `
  --task HLD
```

## Current Interpretation

Qwen3-VL-8B HLD performance still does not match the 52.1% SimpleStream target on this T4/int8/precomputed4 setup. The explicit Qwen3 per-frame builder improves over the completed non-builder int8 run by about +3.23 pp (44.62% vs 41.40%), but it remains about -7.48 pp below the published target.
