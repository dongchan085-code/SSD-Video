param(
    [string]$DataRoot = "D:\ssd_video_data",
    [string]$CondaEnv = "D:\conda_envs\env_ssd_simplestream_officialdeps",
    [string]$CacheDir = "D:\hf_cache",
    [int]$RecentFrames = 4,
    [switch]$SkipVideoDownload,
    [switch]$SkipPrecompute,
    [switch]$DeleteVideosAfterCache
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

function Invoke-CondaChecked {
    param([string[]]$Arguments)
    conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "conda $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-CondaRunArgs {
    param([string]$EnvOrPrefix)
    if ($EnvOrPrefix -match "^[A-Za-z]:\\" -or $EnvOrPrefix.StartsWith("/") -or $EnvOrPrefix.StartsWith("\")) {
        return @("run", "--no-capture-output", "-p", $EnvOrPrefix)
    }
    return @("run", "--no-capture-output", "-n", $EnvOrPrefix)
}

New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$env:PYTHONPATH = $ProjectDir
$env:HF_HOME = $CacheDir
$env:HUGGINGFACE_HUB_CACHE = Join-Path $CacheDir "hub"
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir "transformers"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:FORCE_QWENVL_VIDEO_READER = "decord"
$env:PYTHONIOENCODING = "utf-8"

$CondaRun = Get-CondaRunArgs $CondaEnv
$AnnoPath = Join-Path $DataRoot "ovo_bench_new.json"
$PartsDir = Join-Path $DataRoot "_chunked_parts"
$ChunkedDir = Join-Path $DataRoot "chunked_videos"
$FramesDir = Join-Path $DataRoot "chunked_frames"

Invoke-CondaChecked (@($CondaRun) + @(
    "python", "-u", "$ProjectDir\scripts\download_ovo_sources.py",
    "--data_root", $DataRoot,
    "--anno_path", $AnnoPath,
    "--skip_parts"
))

if (-not $SkipVideoDownload) {
    Invoke-CondaChecked (@($CondaRun) + @(
        "python", "-u", "$ProjectDir\scripts\download_extract_chunked.py",
        "--parts_dir", $PartsDir,
        "--output_dir", $ChunkedDir,
        "--max_parts_ahead", "1"
    ))
}

if (-not $SkipPrecompute) {
    $precomputeArgs = @($CondaRun) + @(
        "python", "-u", "$ProjectDir\scripts\precompute_ovo_simplestream_frames.py",
        "--data-path", $DataRoot,
        "--anno-path", $AnnoPath,
        "--chunked-dir", $ChunkedDir,
        "--output-dir", $FramesDir,
        "--recent-frames-only", "$RecentFrames",
        "--chunk-duration", "1.0",
        "--fps", "1.0"
    )
    if ($DeleteVideosAfterCache) {
        $precomputeArgs += "--delete-videos-after-cache"
    }
    Invoke-CondaChecked $precomputeArgs
}

Write-Host "OVO full precomputed data ready under $DataRoot"
