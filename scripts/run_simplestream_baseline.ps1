param(
    [string]$DataDir = "D:\ssd_video_data\ovo_subset_1pct",
    [string]$ResultsDir = ".\results",
    [string]$ModelId = "Qwen/Qwen3-VL-8B-Instruct",
    [string]$Config = ".\configs\eval_ovo_subset_1pct_base.yaml",
    [string]$CondaEnv = "D:\conda_envs\env_ssd_simplestream_officialdeps",
    [string]$CacheDir = "D:\ssd_video_data\hf_cache",
    [string]$OutputName = "ovo_base_simplestream_recent4_8bit.json",
    [int]$MaxSamples = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$env:HF_HOME = $CacheDir
$env:HUGGINGFACE_HUB_CACHE = Join-Path $CacheDir "hub"
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir "transformers"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:PYTHONPATH = $ProjectDir

function Get-CondaRunArgs {
    param([string]$EnvOrPrefix)
    if ($EnvOrPrefix -match "^[A-Za-z]:\\" -or $EnvOrPrefix.StartsWith("/") -or $EnvOrPrefix.StartsWith("\")) {
        return @("run", "-p", $EnvOrPrefix)
    }
    return @("run", "-n", $EnvOrPrefix)
}

$evalArgs = @(Get-CondaRunArgs $CondaEnv) + @(
    "python", "-u", "$ProjectDir\eval\eval_ovo_bench.py",
    "--config", $Config,
    "--model_path", $ModelId,
    "--data_path", $DataDir,
    "--output_file", "$ResultsDir\$OutputName"
)
if ($MaxSamples -gt 0) {
    $evalArgs += @("--max_samples", "$MaxSamples")
}

conda @evalArgs
if ($LASTEXITCODE -ne 0) {
    throw "SimpleStream baseline evaluation failed with exit code $LASTEXITCODE"
}

Write-Host "Baseline results: $ResultsDir\$OutputName"
