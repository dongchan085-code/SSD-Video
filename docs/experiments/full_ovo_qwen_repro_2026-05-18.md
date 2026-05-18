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

Qwen/SimpleStream-aligned PNG precompute script:

- `scripts/precompute_ovo_simplestream_frames.py`

Why a new precompute script exists:

- `scripts/extract_chunk_frames.py` uses cv2 frame reads and is useful for generic recent-frame extraction.
- The validated HLD reproduction used Qwen/SimpleStream exact recent decoding and Qwen-style image resizing before replay.
- `precompute_ovo_simplestream_frames.py` uses `_fetch_simplestream_frames(...)` and writes the `meta.json` layout consumed by `OVOBenchDataset`.

## Bootstrap Command

The long-running bootstrap was started at about 2026-05-18 13:51 KST:

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
