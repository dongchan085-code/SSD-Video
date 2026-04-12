"""
Convert downloaded Perception Test and OVO-Bench data into the format
expected by the SSD-VLM pipeline code.

Usage:
    python3 scripts/prepare_mini_data.py \
        --pt_dir data/perception_test_mini \
        --ovo_dir data/ovo_bench_mini \
        --max_pt_samples 20 \
        --max_ovo_samples 10
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Perception Test adapter ──────────────────────────────────────────

def convert_perception_test(pt_dir: str, max_samples: int = 20):
    """
    Convert Perception Test MC annotations to our pipeline format.

    Expected input files (from download_mini_data.sh):
      - {pt_dir}/videos/*.mp4
      - {pt_dir}/mc_question_valid_annotations.json  (or similar)

    Output files:
      - {pt_dir}/train_annotations.json
      - {pt_dir}/train_split.json
    """
    pt_path = Path(pt_dir)
    videos_dir = pt_path / "videos"

    # Find annotation file (name may vary across Perception Test releases)
    ann_file = None
    for candidate in [
        "mc_question_valid_annotations.json",
        "mc_question_train_annotations.json",
        "valid_annotations.json",
        "sample_annotations.json",
    ]:
        if (pt_path / candidate).exists():
            ann_file = pt_path / candidate
            break

    # Also search inside any sub-directories created by unzip
    if ann_file is None:
        for f in pt_path.rglob("*.json"):
            if "mc" in f.name.lower() or "annotation" in f.name.lower():
                ann_file = f
                break

    if ann_file is None:
        logger.warning("No Perception Test annotations found. Creating synthetic data.")
        _create_synthetic_pt_data(pt_path, max_samples)
        return

    logger.info(f"Using annotations: {ann_file}")

    with open(ann_file) as f:
        raw_annotations = json.load(f)

    # Find available video files
    if videos_dir.exists():
        available_videos = {
            p.stem for p in videos_dir.iterdir() if p.suffix == ".mp4"
        }
    else:
        available_videos = set()
        # Check if videos are in a sub-folder
        for d in pt_path.iterdir():
            if d.is_dir() and d.name != ".frame_cache":
                for p in d.iterdir():
                    if p.suffix == ".mp4":
                        available_videos.add(p.stem)
                        # Move to videos_dir for consistency
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        p.rename(videos_dir / p.name)

    logger.info(f"Found {len(available_videos)} videos")

    # The Perception Test MC format varies. Handle both dict-of-dicts
    # and list-of-dicts.
    annotations = {}
    video_ids = []

    if isinstance(raw_annotations, dict):
        items = list(raw_annotations.items())[:max_samples]
        for video_id, ann in items:
            if available_videos and video_id not in available_videos:
                continue
            annotations[video_id] = _normalise_pt_annotation(ann)
            video_ids.append(video_id)
    elif isinstance(raw_annotations, list):
        for ann in raw_annotations[:max_samples * 2]:
            video_id = str(ann.get("video_id", ann.get("id", "")))
            if available_videos and video_id not in available_videos:
                continue
            annotations[video_id] = _normalise_pt_annotation(ann)
            video_ids.append(video_id)
            if len(video_ids) >= max_samples:
                break

    # If we couldn't match any videos, use synthetic data
    if not video_ids:
        logger.warning("No matching videos found. Creating synthetic data.")
        _create_synthetic_pt_data(pt_path, max_samples)
        return

    # Write our format
    split_data = {"video_ids": video_ids}
    with open(pt_path / "train_split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    with open(pt_path / "train_annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    logger.info(f"Wrote {len(video_ids)} Perception Test samples")


def _normalise_pt_annotation(ann: dict) -> dict:
    """Normalise a single Perception Test annotation to our schema."""
    # Handle different key names across PT releases
    question = ann.get("question", ann.get("text", ""))
    options = ann.get("options", ann.get("choices", []))

    # Ensure options is a list of strings
    if isinstance(options, dict):
        options = list(options.values())

    answer_id = ann.get("answer_id", ann.get("answer_idx", ann.get("correct", 0)))

    # Skill/area field
    area = ann.get("area", ann.get("skill", ann.get("category", "")))

    return {
        "question": question,
        "options": options,
        "answer_id": answer_id,
        "area": area,
        "reasoning": ann.get("reasoning", ann.get("task_type", "")),
    }


def _create_synthetic_pt_data(pt_path: Path, num_samples: int = 10):
    """Create synthetic Perception Test data when real data isn't available."""
    logger.info("Creating synthetic Perception Test data...")

    videos_dir = pt_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}
    video_ids = []
    skills = ["memory", "memory", "Semantics", "Physics", "Abstraction"]
    questions = [
        "What happened before the person sat down?",
        "What did the person do earlier in the video?",
        "What color is the main object?",
        "What will happen if the object falls?",
        "How many objects are on the table?",
    ]

    for i in range(min(num_samples, len(questions))):
        video_id = f"synth_video_{i:04d}"
        video_ids.append(video_id)

        # Create synthetic video (10 frames of colored noise)
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            _write_synthetic_video(str(video_path), num_frames=10)

        annotations[video_id] = {
            "question": questions[i % len(questions)],
            "options": ["Option A", "Option B", "Option C"],
            "answer_id": i % 3,
            "area": skills[i % len(skills)],
            "reasoning": "Descriptive",
        }

    split_data = {"video_ids": video_ids}
    with open(pt_path / "train_split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    with open(pt_path / "train_annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    logger.info(f"Created {len(video_ids)} synthetic PT samples")


# ── OVO-Bench adapter ────────────────────────────────────────────────

# Task taxonomy
LOCK_TASKS = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
FORK_TASKS = {"EPM", "ASI", "HLD"}
# Only Backward and Real-Time tasks are MC format
MC_TASKS = LOCK_TASKS | FORK_TASKS


def convert_ovo_bench(ovo_dir: str, max_samples: int = 10):
    """
    Convert OVO-Bench annotations to our evaluation format.

    Expected input:
      - {ovo_dir}/ovo_bench_raw.json  (from GitHub download)

    Output:
      - {ovo_dir}/test_split.json
      - {ovo_dir}/test_annotations.json
      - {ovo_dir}/videos/*.mp4  (synthetic placeholders)
    """
    ovo_path = Path(ovo_dir)
    raw_file = ovo_path / "ovo_bench_raw.json"

    if not raw_file.exists():
        logger.warning("OVO-Bench raw annotations not found. Creating synthetic data.")
        _create_synthetic_ovo_data(ovo_path, max_samples)
        return

    with open(raw_file) as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} raw OVO-Bench entries")

    videos_dir = ovo_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}
    video_ids = []

    # Select a balanced subset: some Lock, some Fork
    lock_count = 0
    fork_count = 0
    max_per_category = max_samples // 2

    for entry in raw_data:
        task = entry.get("task", "")
        if task not in MC_TASKS:
            continue

        # Balance Lock and Fork
        is_lock = task in LOCK_TASKS
        is_fork = task in FORK_TASKS
        if is_lock and lock_count >= max_per_category:
            continue
        if is_fork and fork_count >= max_per_category:
            continue

        entry_id = str(entry.get("id", len(video_ids)))
        video_id = f"ovo_{entry_id}"

        # Create synthetic video placeholder (real OVO videos are too large)
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            _write_synthetic_video(str(video_path), num_frames=10)

        # Convert options: OVO-Bench uses a list of 4 options
        options = entry.get("options", [])
        gt = entry.get("gt", 0)

        annotations[video_id] = {
            "question": entry.get("question", ""),
            "options": options,
            "answer_idx": gt,
            "task_type": task,
            "video_relpath": f"videos/{video_id}.mp4",
        }
        video_ids.append(video_id)

        if is_lock:
            lock_count += 1
        if is_fork:
            fork_count += 1

        if len(video_ids) >= max_samples:
            break

    split_data = {"video_ids": video_ids}
    with open(ovo_path / "test_split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    with open(ovo_path / "test_annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    logger.info(f"Wrote {len(video_ids)} OVO-Bench samples "
                f"(Lock: {lock_count}, Fork: {fork_count})")


def _create_synthetic_ovo_data(ovo_path: Path, max_samples: int = 10):
    """Create synthetic OVO-Bench data as fallback."""
    logger.info("Creating synthetic OVO-Bench data...")

    videos_dir = ovo_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["OCR", "ATR", "OJR", "EPM", "ASI", "STU", "ACR", "FPD", "HLD", "OJR"]
    questions = [
        "What text is visible on the sign?",
        "What is the person wearing?",
        "Where is the object located?",
        "What happened before this moment?",
        "Who was interacting with whom?",
        "What does the text say?",
        "What action is being performed?",
        "What is the fine detail visible?",
        "What was discussed earlier?",
        "How many items are there?",
    ]

    annotations = {}
    video_ids = []

    for i in range(min(max_samples, len(tasks))):
        video_id = f"ovo_synth_{i:04d}"
        video_ids.append(video_id)

        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            _write_synthetic_video(str(video_path), num_frames=10)

        annotations[video_id] = {
            "question": questions[i],
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer_idx": i % 4,
            "task_type": tasks[i],
            "video_relpath": f"videos/{video_id}.mp4",
        }

    split_data = {"video_ids": video_ids}
    with open(ovo_path / "test_split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    with open(ovo_path / "test_annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    logger.info(f"Created {len(video_ids)} synthetic OVO-Bench samples")


# ── Synthetic video writer ───────────────────────────────────────────

def _write_synthetic_video(
    path: str,
    num_frames: int = 10,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
):
    """Write a short synthetic video with colored noise frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    rng = np.random.RandomState(hash(path) % (2**31))
    for _ in range(num_frames):
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare mini validation data for SSD-VLM pipeline")
    parser.add_argument("--pt_dir", default="./data/perception_test_mini",
                        help="Perception Test data directory")
    parser.add_argument("--ovo_dir", default="./data/ovo_bench_mini",
                        help="OVO-Bench data directory")
    parser.add_argument("--max_pt_samples", type=int, default=20,
                        help="Max Perception Test samples")
    parser.add_argument("--max_ovo_samples", type=int, default=10,
                        help="Max OVO-Bench samples")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    convert_perception_test(args.pt_dir, args.max_pt_samples)
    convert_ovo_bench(args.ovo_dir, args.max_ovo_samples)


if __name__ == "__main__":
    main()
