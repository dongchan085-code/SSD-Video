# Full OVO Qwen3-VL-8B Reproduction Notes - 2026-05-18

Date/time basis: 2026-05-18 KST.

This note tracks the full OVO-Bench SimpleStream-style Qwen3-VL-8B reproduction path. Read this alongside `docs/experiments/hld_qwen_repro_2026-05-18.md` before changing full OVO/Qwen3 eval configs or rerunning the full benchmark.

## Goal

Stabilize full OVO-Bench evaluation under the same local constraints that reproduced HLD:

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Env: `D:/conda_envs/env_ssd_simplestream_officialdeps`
- Dependency pins: `transformers==4.57.6`, `accelerate==1.12.0`
- T4 runtime: int8, fp16, SDPA, single GPU
- Input construction: SimpleStream Qwen3 explicit per-frame builder
- Frame budget: 4 recent SimpleStream frames

## Current Local Status

As of 2026-05-18 13:48 KST, `D:/ssd_video_data` had been wiped or was absent except for unrelated data. The full OVO dataset therefore had to be bootstrapped again.

Recovered:

- Annotation: `D:/ssd_video_data/ovo_bench_new.json`
- Dataset load preflight: 3035 query units

Task counts from the annotation preflight:

```text
ACR 109
ASI 148
ATR 116
CRR 240
EPM 297
FPD 101
HLD 186
OCR 149
OJR 184
REC 698
SSR 629
STU 178
```

## Added Reproduction Assets

Config:

- `configs/eval_ovo_full_precomputed4_t4_int8_qwen3builder_officialdeps.yaml`

Data bootstrap wrapper:

- `scripts/bootstrap_ovo_full_precomputed.ps1`

Qwen/SimpleStream-aligned PNG precompute scripts:

- `scripts/precompute_ovo_simplestream_frames.py`
- `scripts/stream_precompute_ovo_chunked.py`

Why new precompute scripts exist:

- `scripts/extract_chunk_frames.py` uses cv2 frame reads and is useful for generic recent-frame extraction.
- The validated HLD reproduction used Qwen/SimpleStream exact recent decoding and Qwen-style image resizing before replay.
- `precompute_ovo_simplestream_frames.py` uses `_fetch_simplestream_frames(...)` and writes the `meta.json` layout consumed by `OVOBenchDataset`.
- `stream_precompute_ovo_chunked.py` is the preferred full OVO path on the T4 VM because it extracts one mp4 from the HF tar stream, writes its PNG replay cache, and deletes the temporary mp4 immediately.

## Bootstrap Command

The first long-running bootstrap was started at about 2026-05-18 13:51 KST:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts/bootstrap_ovo_full_precomputed.ps1 `
  -DataRoot D:\ssd_video_data `
  -CondaEnv D:\conda_envs\env_ssd_simplestream_officialdeps `
  -CacheDir D:\hf_cache `
  -RecentFrames 4 `
  -DeleteVideosAfterCache
```

Live logs:

- `logs/bootstrap_ovo_full_precomputed_20260518_135146.out.log`
- `logs/bootstrap_ovo_full_precomputed_20260518_135146.err.log`

At the first monitor check, the run had listed all 15 `chunked_videos.tar.part*` files, opened `partaa`, and started extracting into `D:/ssd_video_data/chunked_videos`.

Important correction from the 14:05 KST monitor check:

- The first wrapper version used a two-stage flow: extract all mp4 chunks, then precompute PNGs.
- That is too risky on the D: disk because the extracted mp4 size was tracking the tar stream too closely and could exceed available space before PNG conversion starts.
- The extraction was stopped at 917 mp4 files / about 53.2 GB extracted.
- `scripts/precompute_ovo_simplestream_frames.py` was started against the already-extracted mp4s with `--delete-videos-after-cache` to recover space.
- The wrapper was then changed so the default path is streaming precompute via `scripts/stream_precompute_ovo_chunked.py`; the old two-stage path is now behind `-UseTwoStageExtraction`.

## 2026-05-18 Bootstrap Completion

The streaming precompute run completed successfully:

- Log: `logs/stream_precompute_ovo_full_20260518_143427.out.log`
- Summary: `selected=3035`, `cached=2120`, `skipped=915`, `failed=0`, `missing=0`
- Final cache coverage: 3035 directories with `meta.json` under `D:/ssd_video_data/chunked_frames`
- Remaining `D:/ssd_video_data/chunked_videos` mp4 files: 0
- Remaining `D:/ssd_video_data/_chunked_parts` files: 0
- D: free space after bootstrap: about 136 GB

Dataset/cache preflight:

```text
OVOBenchDataset rows: 3035
First sample: video_id=0, frame_images=4, first frame size=(896, 672)
```

Smoke eval:

- Command used the full officialdeps config with `--max_samples 3`.
- Output: `results/ovo_simplestream_fullset/qwen3vl8b_int8_full_precomputed4_qwen3builder_t4_officialdeps_smoke3.json`
- Result: 3 / 3 samples completed, overall accuracy 33.33%.
- The smoke wrote 3 rows to the canonical partial file, and the full eval resumed from those rows.

## 2026-05-18 Full Eval Run

Started at about 2026-05-18 15:33 KST:

```powershell
conda run --no-capture-output -p D:\conda_envs\env_ssd_simplestream_officialdeps python -u eval\eval_ovo_bench.py `
  --config configs\eval_ovo_full_precomputed4_t4_int8_qwen3builder_officialdeps.yaml `
  --model_path Qwen/Qwen3-VL-8B-Instruct `
  --data_path D:/ssd_video_data `
  --output_file results/ovo_simplestream_fullset/qwen3vl8b_int8_full_precomputed4_qwen3builder_t4_officialdeps.json `
  --sample_ratio 1.0
```

Live logs:

- `logs/eval_ovo_full_qwen3_officialdeps_20260518_153338.err.log`
- `logs/eval_ovo_full_qwen3_officialdeps_20260518_153338.out.log`

Partial/result files:

- `results/ovo_simplestream_fullset/qwen3vl8b_int8_full_precomputed4_qwen3builder_t4_officialdeps.partial_predictions.jsonl`
- `results/ovo_simplestream_fullset/qwen3vl8b_int8_full_precomputed4_qwen3builder_t4_officialdeps.json`

Initial monitor check at 2026-05-18 15:36 KST:

- Partial rows: 19 / 3035
- Pending: 3016
- Correct so far: 8
- Cumulative accuracy so far: 42.11%
- Last completed `video_id`: 18
- GPU memory: about 15.2 GB on T4
- TQDM estimate: about 6-8 hours remaining after model load

## Expected Eval Command After Bootstrap

Run after `D:/ssd_video_data/chunked_frames` contains precomputed caches for all 3035 query units:

```powershell
$env:PYTHONPATH='C:/work/SSD-Video'
$env:HF_HOME='D:/hf_cache'
$env:HUGGINGFACE_HUB_CACHE='D:/hf_cache/hub'
$env:TRANSFORMERS_CACHE='D:/hf_cache/transformers'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
$env:FORCE_QWENVL_VIDEO_READER='decord'
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -p D:\conda_envs\env_ssd_simplestream_officialdeps python -u eval/eval_ovo_bench.py `
  --config configs/eval_ovo_full_precomputed4_t4_int8_qwen3builder_officialdeps.yaml `
  --model_path Qwen/Qwen3-VL-8B-Instruct `
  --data_path D:/ssd_video_data `
  --output_file results/ovo_simplestream_fullset/qwen3vl8b_int8_full_precomputed4_qwen3builder_t4_officialdeps.json `
  --sample_ratio 1.0
```

## Monitor Commands

Bootstrap process:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'bootstrap_ovo_full_precomputed|download_extract_chunked|precompute_ovo_simplestream' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

Log tail:

```powershell
Get-Content logs/bootstrap_ovo_full_precomputed_20260518_135146.out.log -Tail 40
Get-Content logs/bootstrap_ovo_full_precomputed_20260518_135146.err.log -Tail 40
```

Data size:

```powershell
Get-ChildItem D:/ssd_video_data/chunked_videos -File -ErrorAction SilentlyContinue |
  Measure-Object -Property Length -Sum
Get-ChildItem D:/ssd_video_data/chunked_frames -Directory -ErrorAction SilentlyContinue |
  Measure-Object
Get-Volume -DriveLetter D
```
