#!/bin/bash
# Download Perception Test and OVO-Bench datasets

set -e

# Script configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="${PROJECT_DIR}/data"

echo "Downloading SSD-VLM datasets..."
echo "Data directory: ${DATA_DIR}"

# Create data directory
mkdir -p "${DATA_DIR}"

# Download Perception Test
echo ""
echo "Downloading Perception Test dataset..."
mkdir -p "${DATA_DIR}/perception_test"

# Note: These are placeholder URLs - replace with actual dataset URLs
# The actual Perception Test is available at: https://github.com/google-research/perception-test

echo "Perception Test download (placeholder)"
echo "To download the actual dataset, visit: https://github.com/google-research/perception-test"

# Create placeholder annotation files
cat > "${DATA_DIR}/perception_test/train_split.json" << 'EOF'
{
  "video_ids": ["video_0", "video_1", "video_2"],
  "split": "train",
  "num_videos": 3
}
EOF

cat > "${DATA_DIR}/perception_test/train_annotations.json" << 'EOF'
{
  "video_0": {
    "question": "What is the person doing?",
    "options": ["Walking", "Running", "Jumping", "Sitting"],
    "answer_idx": 0,
    "skill": "perception",
    "task_type": "action_recognition"
  },
  "video_1": {
    "question": "How many objects are in the scene?",
    "options": ["1", "2", "3", "4"],
    "answer_idx": 2,
    "skill": "memory",
    "task_type": "counting"
  },
  "video_2": {
    "question": "What color is the main object?",
    "options": ["Red", "Blue", "Green", "Yellow"],
    "answer_idx": 1,
    "skill": "perception",
    "task_type": "color_recognition"
  }
}
EOF

# Download OVO-Bench
echo ""
echo "Downloading OVO-Bench dataset..."
mkdir -p "${DATA_DIR}/ovo_bench"

# OVO-Bench is available at: https://github.com/zihangxu98/OVO-Bench

echo "OVO-Bench download (placeholder)"
echo "To download the actual dataset, visit: https://github.com/zihangxu98/OVO-Bench"

# Create placeholder annotation files
cat > "${DATA_DIR}/ovo_bench/test_split.json" << 'EOF'
{
  "video_ids": ["ovo_0", "ovo_1", "ovo_2"],
  "split": "test",
  "num_videos": 3
}
EOF

cat > "${DATA_DIR}/ovo_bench/test_annotations.json" << 'EOF'
{
  "ovo_0": {
    "question": "What is the text visible in the image?",
    "options": ["Hello", "World", "Test", "Sample"],
    "answer_idx": 0,
    "task_type": "OCR"
  },
  "ovo_1": {
    "question": "What action is being performed?",
    "options": ["Lift", "Push", "Pull", "Carry"],
    "answer_idx": 1,
    "task_type": "ATR"
  },
  "ovo_2": {
    "question": "Where is the object?",
    "options": ["Left", "Right", "Top", "Bottom"],
    "answer_idx": 2,
    "task_type": "OJR"
  }
}
EOF

# Create video directories
mkdir -p "${DATA_DIR}/perception_test/videos"
mkdir -p "${DATA_DIR}/ovo_bench/videos"

echo ""
echo "Dataset structure created at ${DATA_DIR}"
echo "Note: Placeholder files created. Please download actual datasets from:"
echo "  - Perception Test: https://github.com/google-research/perception-test"
echo "  - OVO-Bench: https://github.com/zihangxu98/OVO-Bench"
echo ""
echo "Directory structure:"
tree -L 2 "${DATA_DIR}" 2>/dev/null || find "${DATA_DIR}" -type d | head -20

echo ""
echo "Download script completed!"
