param(
    [double]$Ratio = 0.01,
    [string]$DataRoot = "D:\ssd_video_data",
    [string]$CondaEnv = "env_ssd_simplestream",
    [switch]$DownloadParts,
    [switch]$OverwriteChunks
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

if ($Ratio -eq 0.01) {
    $SubsetName = "ovo_subset_1pct"
} elseif ($Ratio -eq 0.10) {
    $SubsetName = "ovo_subset_10pct"
} else {
    $SubsetName = "ovo_subset_$([int]($Ratio * 10000))bp"
}

$AnnoPath = Join-Path $DataRoot "ovo_bench_new.json"
$PartsDir = Join-Path $DataRoot "ovo_src_parts"
$SubsetDir = Join-Path $DataRoot $SubsetName
$SrcDir = Join-Path $SubsetDir "src_videos"
$ChunkDir = Join-Path $SubsetDir "chunked_videos"

$downloadArgs = @(
    "run", "-n", $CondaEnv, "python", "-u", "$ProjectDir\scripts\download_ovo_sources.py",
    "--data_root", $DataRoot,
    "--parts_dir", $PartsDir,
    "--anno_path", $AnnoPath,
    "--max_gb", "100"
)
if (-not $DownloadParts) {
    $downloadArgs += "--skip_parts"
}
Invoke-CondaChecked $downloadArgs

Invoke-CondaChecked @(
    "run", "-n", $CondaEnv, "python", "-u", "$ProjectDir\scripts\prepare_ovo_subset.py",
    "--anno_path", $AnnoPath,
    "--output_dir", $SubsetDir,
    "--ratio", "$Ratio",
    "--seed", "42",
    "--min_per_task", "1"
)

if (-not (Get-ChildItem -Path $PartsDir -Filter "src_videos.tar.part*" -ErrorAction SilentlyContinue)) {
    throw "No source-video tar parts found in $PartsDir. Re-run with -DownloadParts once, then retry."
}

Invoke-CondaChecked @(
    "run", "-n", $CondaEnv, "python", "-u", "$ProjectDir\scripts\extract_ovo_src_subset.py",
    "--parts_dir", $PartsDir,
    "--required_sources", "$SubsetDir\required_sources.txt",
    "--output_dir", $SrcDir
)

$chunkArgs = @(
    "run", "-n", $CondaEnv, "python", "-u", "$ProjectDir\scripts\chunk_ovo_subset.py",
    "--anno_path", "$SubsetDir\ovo_bench_subset.json",
    "--src_dir", $SrcDir,
    "--output_dir", $ChunkDir
)
if ($OverwriteChunks) {
    $chunkArgs += "--overwrite"
}
Invoke-CondaChecked $chunkArgs

Write-Host "Prepared subset: $SubsetDir"
