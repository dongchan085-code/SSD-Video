#!/bin/bash
# Download minimal data for local end-to-end pipeline validation.
#
# Perception Test: sample split (~215MB videos + ~3MB annotations)
#   Source: https://github.com/google-deepmind/perception_test
#
# OVO-Bench: annotation JSON only (~150KB) + synthetic placeholder videos
#   Source: https://github.com/joeleelyf/ovo-bench
#
# Total download: ~220MB

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

PT_DIR="${PROJECT_DIR}/data/perception_test_mini"
OVO_DIR="${PROJECT_DIR}/data/ovo_bench_mini"

echo "==========================================="
echo "Downloading Mini Validation Data"
echo "==========================================="

# ── Perception Test (sample split) ───────────────────────────────────
echo ""
echo "Step 1: Perception Test sample split"
echo "====================================="
mkdir -p "${PT_DIR}"

# Download sample videos (~215MB)
if [ ! -d "${PT_DIR}/videos" ] || [ -z "$(ls -A ${PT_DIR}/videos 2>/dev/null)" ]; then
    echo "Downloading sample videos..."
    curl -L -o "${PT_DIR}/sample_videos.zip" \
        "https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip"
    mkdir -p "${PT_DIR}/videos"
    unzip -o "${PT_DIR}/sample_videos.zip" -d "${PT_DIR}/videos/"
    rm -f "${PT_DIR}/sample_videos.zip"
    echo "Sample videos downloaded."
else
    echo "Videos already exist, skipping download."
fi

# Download MC question annotations
if [ ! -f "${PT_DIR}/mc_question_annotations.json" ]; then
    echo "Downloading MC question annotations..."
    curl -L -o "${PT_DIR}/mc_annotations.zip" \
        "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip"
    unzip -o "${PT_DIR}/mc_annotations.zip" -d "${PT_DIR}/"
    rm -f "${PT_DIR}/mc_annotations.zip"
    echo "Annotations downloaded."
else
    echo "Annotations already exist, skipping download."
fi

# ── OVO-Bench annotations ───────────────────────────────────────────
echo ""
echo "Step 2: OVO-Bench annotations"
echo "=============================="
mkdir -p "${OVO_DIR}"

if [ ! -f "${OVO_DIR}/ovo_bench_raw.json" ]; then
    echo "Downloading OVO-Bench annotations from GitHub..."
    curl -L -o "${OVO_DIR}/ovo_bench_raw.json" \
        "https://raw.githubusercontent.com/joeleelyf/ovo-bench/main/data/ovo_bench_new.json"
    echo "OVO-Bench annotations downloaded."
else
    echo "OVO-Bench annotations already exist, skipping download."
fi

# ── Convert data to our format ───────────────────────────────────────
echo ""
echo "Step 3: Converting data to pipeline format"
echo "==========================================="
python3 "${SCRIPT_DIR}/prepare_mini_data.py" \
    --pt_dir "${PT_DIR}" \
    --ovo_dir "${OVO_DIR}" \
    --max_pt_samples 20 \
    --max_ovo_samples 10

echo ""
echo "==========================================="
echo "Mini data download complete!"
echo "==========================================="
echo "  Perception Test: ${PT_DIR}"
echo "  OVO-Bench:       ${OVO_DIR}"
echo ""
