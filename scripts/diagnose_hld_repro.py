"""Diagnose SimpleStream HLD reproduction gaps.

The commands here are deliberately read-only. They compare annotations,
cached PNG frames, prediction scoring, frame selection, and Qwen3-VL prompt
encoding so HLD gaps can be attributed before running another expensive eval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssd_vlm.simplestream import extract_mcq_answer, format_ovo_prompt  # noqa: E402


DEFAULT_SS_REF_DIR = Path(r"C:/Users/swsuser-j07/AppData/Local/Temp/ss_ref")
MULTICHOICE_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR", "EPM", "ASI", "HLD"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Optional[Path], payload: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _jsonish(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Optional[Path], rows: Sequence[Dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonish(row.get(key, "")) for key in keys})


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _gt_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        return ord(upper) - ord("A")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else None


def _gt_to_letter(value: Any) -> Optional[str]:
    if isinstance(value, str):
        upper = value.strip().upper()
        if len(upper) == 1 and "A" <= upper <= "Z":
            return upper
    gt_int = _gt_to_int(value)
    if gt_int is None or gt_int < 0 or gt_int >= 26:
        return None
    return chr(65 + gt_int)


def _safe_prompt_hash(task: str, question: str, options: Sequence[Any]) -> str:
    prompt = format_ovo_prompt(task, question, list(options))
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _annotation_id(anno: Dict[str, Any]) -> str:
    return str(anno.get("id", anno.get("video_id", "")))


def _annotation_task(anno: Dict[str, Any]) -> str:
    return str(anno.get("task", anno.get("task_type", "")))


def _annotation_gt(anno: Dict[str, Any]) -> Any:
    return anno.get("gt", anno.get("answer_idx"))


def _annotation_options(anno: Dict[str, Any]) -> List[Any]:
    options = anno.get("options", [])
    return list(options) if isinstance(options, list) else []


def _canonical_annotation_row(anno: Dict[str, Any]) -> Dict[str, Any]:
    task = _annotation_task(anno)
    question = str(anno.get("question", ""))
    options = _annotation_options(anno)
    gt = _annotation_gt(anno)
    gt_int = _gt_to_int(gt)
    return {
        "id": _annotation_id(anno),
        "task": task,
        "question": question,
        "options": options,
        "gt": gt,
        "gt_int": gt_int,
        "gt_letter": _gt_to_letter(gt),
        "n_options": len(options),
        "gt_in_options": bool(gt_int is not None and 0 <= gt_int < len(options)),
        "gt_in_abcd": bool(gt_int is not None and 0 <= gt_int <= 3),
        "prompt_hash": _safe_prompt_hash(task, question, options),
    }


def load_annotations(path: Path, task: Optional[str] = None) -> List[Dict[str, Any]]:
    annotations = _read_json(path)
    if not isinstance(annotations, list):
        raise ValueError(f"Annotation JSON must be a list: {path}")
    rows = [_canonical_annotation_row(item) for item in annotations]
    if task:
        rows = [row for row in rows if row["task"] == task]
    return rows


def _load_cache_record(cache_dir: Optional[Path], video_id: str, recent: int = 4) -> Dict[str, Any]:
    if cache_dir is None:
        return {
            "cache_present": None,
            "cache_format": None,
            "cache_frame_count": None,
            "cache_selected_count": None,
            "cache_indices": None,
            "cache_timestamps": None,
            "cache_chunk_ids": None,
        }

    item_dir = cache_dir / str(video_id)
    if not item_dir.exists():
        return {
            "cache_present": False,
            "cache_format": None,
            "cache_frame_count": 0,
            "cache_selected_count": 0,
            "cache_indices": [],
            "cache_timestamps": [],
            "cache_chunk_ids": [],
        }

    source_meta = item_dir / "metadata.json"
    precomputed_meta = item_dir / "meta.json"

    if source_meta.exists():
        meta = _read_json(source_meta)
        frames = list(meta.get("frames") or [])
        selected = frames[-recent:]
        return {
            "cache_present": True,
            "cache_format": "simplestream_png_cache",
            "cache_frame_count": len(frames),
            "cache_selected_count": len(selected),
            "cache_indices": [int(item.get("frame_index")) for item in selected],
            "cache_timestamps": [float(item.get("timestamp")) for item in selected],
            "cache_chunk_ids": [int(item.get("chunk_id")) for item in selected],
        }

    if precomputed_meta.exists():
        meta = _read_json(precomputed_meta)
        indices = [int(x) for x in meta.get("frame_indices", [])]
        timestamps = [float(x) for x in meta.get("frame_timestamps", [])]
        chunk_duration = max(float(meta.get("chunk_duration", 1.0)), 1e-6)
        selected_indices = indices[-recent:]
        selected_timestamps = timestamps[-recent:]
        return {
            "cache_present": True,
            "cache_format": "precomputed_frames",
            "cache_frame_count": int(meta.get("saved_count", len(indices))),
            "cache_selected_count": len(selected_indices),
            "cache_indices": selected_indices,
            "cache_timestamps": selected_timestamps,
            "cache_chunk_ids": [int(ts // chunk_duration) for ts in selected_timestamps],
        }

    return {
        "cache_present": False,
        "cache_format": "unknown",
        "cache_frame_count": len(list(item_dir.glob("*.png"))),
        "cache_selected_count": 0,
        "cache_indices": [],
        "cache_timestamps": [],
        "cache_chunk_ids": [],
    }


def audit_annotations(
    anno_path: Path,
    *,
    task: str = "HLD",
    manifest_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    recent_frames: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    official_rows = load_annotations(anno_path, task=task)
    manifest_rows = load_annotations(manifest_path, task=task) if manifest_path else []
    manifest_by_id = {row["id"]: row for row in manifest_rows}

    duplicate_ids = [item for item, count in Counter(row["id"] for row in official_rows).items() if count > 1]
    rows: List[Dict[str, Any]] = []
    for row in official_rows:
        cache = _load_cache_record(cache_dir, row["id"], recent=recent_frames)
        manifest = manifest_by_id.get(row["id"])
        out = {
            **row,
            "manifest_present": manifest is not None if manifest_path else None,
            "manifest_gt_match": (manifest.get("gt_int") == row["gt_int"]) if manifest else None,
            "manifest_prompt_match": (manifest.get("prompt_hash") == row["prompt_hash"]) if manifest else None,
            "manifest_options_match": (manifest.get("options") == row["options"]) if manifest else None,
            **cache,
        }
        rows.append(out)

    missing_from_manifest = sorted(set(row["id"] for row in official_rows) - set(manifest_by_id)) if manifest_path else []
    extra_in_manifest = sorted(set(manifest_by_id) - set(row["id"] for row in official_rows)) if manifest_path else []
    summary = {
        "command": "audit-annotation",
        "task": task,
        "anno_path": str(anno_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "n_annotations": len(official_rows),
        "n_manifest": len(manifest_rows) if manifest_path else None,
        "duplicate_ids": duplicate_ids,
        "non4_options": sum(1 for row in official_rows if row["n_options"] != 4),
        "gt_outside_options": sum(1 for row in official_rows if not row["gt_in_options"]),
        "gt_outside_abcd": sum(1 for row in official_rows if not row["gt_in_abcd"]),
        "cache_missing": sum(1 for row in rows if row["cache_present"] is False),
        "cache_lt_recent": sum(
            1
            for row in rows
            if row["cache_selected_count"] is not None and int(row["cache_selected_count"]) < recent_frames
        ),
        "missing_from_manifest": missing_from_manifest,
        "extra_in_manifest": extra_in_manifest,
        "manifest_gt_mismatches": sum(1 for row in rows if row["manifest_gt_match"] is False),
        "manifest_prompt_mismatches": sum(1 for row in rows if row["manifest_prompt_match"] is False),
    }
    return summary, rows


def _normalise_prediction(raw: Dict[str, Any]) -> Dict[str, Any]:
    task = raw.get("task_type", raw.get("task", ""))
    video_id = raw.get("video_id", raw.get("id", ""))
    response = raw.get("answer_text", raw.get("response"))
    gt = raw.get("ground_truth", raw.get("gt"))
    if isinstance(gt, str) and len(gt.strip()) == 1 and gt.strip().isalpha():
        gt_int = _gt_to_int(gt)
    else:
        gt_int = _gt_to_int(gt)
    pred = raw.get("predicted", raw.get("pred"))
    return {
        "video_id": str(video_id),
        "task_type": str(task),
        "question": raw.get("question"),
        "options": raw.get("options") if isinstance(raw.get("options"), list) else [],
        "ground_truth": gt,
        "ground_truth_int": gt_int,
        "ground_truth_letter": _gt_to_letter(gt),
        "predicted": pred,
        "answer_text": response,
        "stored_correct": raw.get("correct"),
        "frame_indices": raw.get("frame_indices"),
        "frame_timestamps": raw.get("frame_timestamps"),
        "chunk_ids": raw.get("chunk_ids"),
    }


def load_predictions(path: Path, task: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    raw_rows: List[Dict[str, Any]] = []

    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        raw_rows = list(payload["predictions"])
    elif isinstance(payload, dict):
        for section in ("backward", "realtime", "forward"):
            if isinstance(payload.get(section), list):
                raw_rows.extend(payload[section])
        simplestream = payload.get("simplestream")
        if not raw_rows and isinstance(simplestream, dict):
            for section_rows in simplestream.values():
                if isinstance(section_rows, list):
                    raw_rows.extend(section_rows)
    elif isinstance(payload, list):
        raw_rows = payload

    rows = [_normalise_prediction(row) for row in raw_rows]
    if task:
        rows = [row for row in rows if row["task_type"] == task]
    return rows


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    phat = correct / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return center, half


def _stored_correct_as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def audit_results(
    result_path: Path,
    *,
    task: str = "HLD",
    paper_accuracy: Optional[float] = 52.1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = load_predictions(result_path, task=task)
    for row in rows:
        row["stored_correct_bool"] = _stored_correct_as_bool(row["stored_correct"])
        row["answer_len"] = len(str(row.get("answer_text") or ""))
        row["n_options"] = len(row.get("options") or [])
        row["gt_in_abcd"] = bool(
            row["ground_truth_int"] is not None and 0 <= int(row["ground_truth_int"]) <= 3
        )
        row["gt_in_options"] = bool(
            row["ground_truth_int"] is not None
            and row["n_options"] > 0
            and 0 <= int(row["ground_truth_int"]) < row["n_options"]
        )

    total = len(rows)
    correct = sum(1 for row in rows if row["stored_correct_bool"] is True)
    center, half = _wilson_interval(correct, total)
    gt_dist = Counter(str(row["ground_truth_int"]) for row in rows)
    pred_dist = Counter(str(row["predicted"]) for row in rows)
    confusion = Counter((str(row["ground_truth_int"]), str(row["predicted"])) for row in rows)
    acc_pct = 100.0 * correct / total if total else None
    summary: Dict[str, Any] = {
        "command": "audit-results",
        "task": task,
        "result_path": str(result_path),
        "num_total": total,
        "num_correct": correct,
        "accuracy_percent": acc_pct,
        "paper_accuracy_percent": paper_accuracy,
        "delta_vs_paper_pp": (acc_pct - paper_accuracy) if acc_pct is not None and paper_accuracy is not None else None,
        "wilson95_percent": (
            [100.0 * (center - half), 100.0 * (center + half)] if center is not None and half is not None else None
        ),
        "paper_in_wilson95": (
            bool(100.0 * (center - half) <= paper_accuracy <= 100.0 * (center + half))
            if center is not None and half is not None and paper_accuracy is not None
            else None
        ),
        "gt_distribution": dict(gt_dist),
        "pred_distribution": dict(pred_dist),
        "top_confusions": [
            {"ground_truth": gt, "predicted": pred, "count": count}
            for (gt, pred), count in confusion.most_common(12)
        ],
        "invalid_gt_outside_options": sum(1 for row in rows if not row["gt_in_options"]),
        "gt_outside_abcd": sum(1 for row in rows if not row["gt_in_abcd"]),
        "empty_answers": sum(1 for row in rows if not str(row.get("answer_text") or "").strip()),
    }
    return summary, rows


def _official_substring_correct(response: Any, gt_letter: Optional[str]) -> bool:
    if gt_letter is None or response is None:
        return False
    return gt_letter in str(response)


def _release_regex_correct(response: Any, gt_letter: Optional[str]) -> bool:
    if gt_letter is None or response is None:
        return False
    pred = extract_mcq_answer(str(response))
    return bool(pred is not None and pred.upper() == gt_letter.upper())


def compare_scoring(result_path: Path, *, task: str = "HLD") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = load_predictions(result_path, task=task)
    by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        response = row.get("answer_text")
        gt_letter = row.get("ground_truth_letter")
        stored = _stored_correct_as_bool(row.get("stored_correct"))
        release_regex = _release_regex_correct(response, gt_letter)
        official_substring = _official_substring_correct(response, gt_letter)
        pred_letter = extract_mcq_answer(str(response)) if response is not None else None
        item = {
            **row,
            "ground_truth_letter": gt_letter,
            "regex_pred_letter": pred_letter,
            "stored_correct_bool": stored,
            "release_regex_correct": release_regex,
            "official_substring_correct": official_substring,
            "regex_vs_substring_diff": release_regex != official_substring,
            "stored_vs_regex_diff": stored is not None and stored != release_regex,
            "stored_vs_substring_diff": stored is not None and stored != official_substring,
        }
        out_rows.append(item)

        task_name = item["task_type"]
        by_task[task_name]["n"] += 1
        by_task[task_name]["stored"] += int(stored is True)
        by_task[task_name]["release_regex"] += int(release_regex)
        by_task[task_name]["official_substring"] += int(official_substring)
        by_task[task_name]["regex_substring_diff"] += int(release_regex != official_substring)

    task_summaries = {}
    for task_name, counts in by_task.items():
        n = counts["n"]
        task_summaries[task_name] = {
            "n": n,
            "stored_percent": 100.0 * counts["stored"] / n if n else None,
            "release_regex_percent": 100.0 * counts["release_regex"] / n if n else None,
            "official_substring_percent": 100.0 * counts["official_substring"] / n if n else None,
            "official_minus_regex_pp": 100.0 * (counts["official_substring"] - counts["release_regex"]) / n if n else None,
            "regex_substring_diff_n": counts["regex_substring_diff"],
        }

    summary = {
        "command": "score-compare",
        "task": task,
        "result_path": str(result_path),
        "tasks": task_summaries,
        "num_rows": len(out_rows),
        "num_regex_substring_diffs": sum(1 for row in out_rows if row["regex_vs_substring_diff"]),
        "num_stored_regex_diffs": sum(1 for row in out_rows if row["stored_vs_regex_diff"]),
        "num_stored_substring_diffs": sum(1 for row in out_rows if row["stored_vs_substring_diff"]),
    }
    return summary, out_rows


def _ids_from_args(ids: Optional[str], result_path: Optional[Path], task: str, limit: Optional[int]) -> List[str]:
    values: List[str] = []
    if ids:
        values.extend([part.strip() for part in ids.split(",") if part.strip()])
    if result_path:
        values.extend([row["video_id"] for row in load_predictions(result_path, task=task)])
    deduped = list(dict.fromkeys(values))
    return deduped[:limit] if limit is not None else deduped


def _load_simplestream_decode(ss_ref_dir: Path):
    if not ss_ref_dir.exists():
        raise FileNotFoundError(f"SimpleStream reference dir not found: {ss_ref_dir}")
    if str(ss_ref_dir) not in sys.path:
        sys.path.insert(0, str(ss_ref_dir))
    os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    from lib.recent_window_eval import decode_video_to_chunks_qwen

    return decode_video_to_chunks_qwen


def compare_frames(
    *,
    video_ids: Sequence[str],
    chunked_dir: Path,
    cache_dir: Path,
    ss_ref_dir: Path = DEFAULT_SS_REF_DIR,
    recent_frames: int = 4,
    fps: float = 1.0,
    chunk_duration: float = 1.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    present_video_ids = [video_id for video_id in video_ids if (chunked_dir / f"{video_id}.mp4").exists()]
    decode_video_to_chunks_qwen = _load_simplestream_decode(ss_ref_dir) if present_video_ids else None
    rows: List[Dict[str, Any]] = []
    for video_id in video_ids:
        video_path = chunked_dir / f"{video_id}.mp4"
        cache = _load_cache_record(cache_dir, video_id, recent=recent_frames)
        if not video_path.exists():
            rows.append({
                "video_id": video_id,
                "video_present": False,
                **cache,
                "reference_present": False,
                "same_timestamps": None,
                "same_chunk_ids": None,
                "same_frame_count": None,
            })
            continue

        if decode_video_to_chunks_qwen is None:
            raise RuntimeError("internal error: reference decoder was not loaded for present videos")
        chunks, backend = decode_video_to_chunks_qwen(
            video_path=str(video_path),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames,
        )
        recent_chunks = chunks[-recent_frames:]
        ref_timestamps: List[float] = []
        ref_chunk_ids: List[int] = []
        for chunk in recent_chunks:
            ref_timestamps.extend([float(x) for x in chunk.frame_timestamps])
            ref_chunk_ids.extend([int(chunk.chunk_index)] * len(chunk.frames))

        cache_ts = [round(float(x), 3) for x in cache.get("cache_timestamps") or []]
        ref_ts = [round(float(x), 3) for x in ref_timestamps]
        row = {
            "video_id": video_id,
            "video_present": True,
            "reference_present": True,
            "reference_backend": backend,
            "reference_frame_count": len(ref_timestamps),
            "reference_timestamps": ref_timestamps,
            "reference_chunk_ids": ref_chunk_ids,
            **cache,
            "same_timestamps": cache_ts == ref_ts,
            "same_chunk_ids": (cache.get("cache_chunk_ids") or []) == ref_chunk_ids,
            "same_frame_count": int(cache.get("cache_selected_count") or 0) == len(ref_timestamps),
        }
        rows.append(row)

    summary = {
        "command": "frame-compare",
        "chunked_dir": str(chunked_dir),
        "cache_dir": str(cache_dir),
        "ss_ref_dir": str(ss_ref_dir),
        "num_requested": len(video_ids),
        "num_videos_present": sum(1 for row in rows if row["video_present"]),
        "num_reference_compared": sum(1 for row in rows if row["reference_present"]),
        "timestamp_mismatches": sum(1 for row in rows if row["same_timestamps"] is False),
        "chunk_id_mismatches": sum(1 for row in rows if row["same_chunk_ids"] is False),
        "frame_count_mismatches": sum(1 for row in rows if row["same_frame_count"] is False),
    }
    return summary, rows


def _load_cache_images(frame_dir: Path, recent_frames: int = 4) -> List[Any]:
    from PIL import Image

    source_meta = frame_dir / "metadata.json"
    precomputed_meta = frame_dir / "meta.json"
    if source_meta.exists():
        meta = _read_json(source_meta)
        frame_records = list(meta.get("frames") or [])[-recent_frames:]
        paths = [frame_dir / str(item["file"]) for item in frame_records]
    elif precomputed_meta.exists():
        paths = sorted(frame_dir.glob("frame_*.png"))[-recent_frames:]
    else:
        paths = sorted(frame_dir.glob("*.png"))[-recent_frames:]
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    frames: List[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.load()
        frames.append(image)
    return frames


def compare_encoding(
    *,
    model_path: str,
    frame_dir: Path,
    task: str = "HLD",
    question: str = "",
    options: Optional[Sequence[Any]] = None,
    recent_frames: int = 4,
    merge_size: int = 2,
) -> Dict[str, Any]:
    from transformers import AutoProcessor

    frames = _load_cache_images(frame_dir, recent_frames=recent_frames)
    prompt = format_ovo_prompt(task, question, list(options or []))
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    current_messages = [{
        "role": "user",
        "content": [{"type": "image", "image": frame} for frame in frames] + [{"type": "text", "text": prompt}],
    }]
    current_inputs = processor.apply_chat_template(
        current_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    enc_messages = [{
        "role": "user",
        "content": [{"type": "image", "image": frame} for frame in frames] + [{"type": "text", "text": "."}],
    }]
    enc_inputs = processor.apply_chat_template(
        enc_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    grid = enc_inputs["image_grid_thw"][0:] if hasattr(enc_inputs["image_grid_thw"], "shape") else enc_inputs["image_grid_thw"]
    grid_rows = [[int(x) for x in row.tolist()] for row in grid]
    tokens_per_frame = [
        max(1, int(row[0] * row[1] * row[2]) // max(1, int(merge_size) ** 2))
        for row in grid_rows
    ]

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or image_token_id < 0:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    if image_token_id is None or image_token_id < 0:
        image_token_id = 0

    question_ids = tokenizer.encode(prompt, add_special_tokens=False)
    reference_ids: List[int] = []
    reference_ids.append(im_start_id)
    reference_ids.extend(tokenizer.encode("user\n", add_special_tokens=False))
    for token_count in tokens_per_frame:
        reference_ids.append(vision_start_id)
        reference_ids.extend([image_token_id] * int(token_count))
        reference_ids.append(vision_end_id)
    reference_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    reference_ids.extend(question_ids)
    reference_ids.append(im_end_id)
    reference_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    reference_ids.append(im_start_id)
    reference_ids.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

    current_input_ids = current_inputs["input_ids"][0].tolist()
    return {
        "command": "encode-compare",
        "model_path": model_path,
        "frame_dir": str(frame_dir),
        "task": task,
        "num_frames": len(frames),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "current_input_len": len(current_input_ids),
        "reference_manual_input_len": len(reference_ids),
        "input_len_delta_reference_minus_current": len(reference_ids) - len(current_input_ids),
        "current_vision_start_count": current_input_ids.count(vision_start_id),
        "current_vision_end_count": current_input_ids.count(vision_end_id),
        "reference_vision_start_count": len(tokens_per_frame),
        "reference_vision_end_count": len(tokens_per_frame),
        "image_grid_thw": grid_rows,
        "tokens_per_frame": tokens_per_frame,
        "total_vision_tokens": sum(tokens_per_frame),
        "merge_size": int(merge_size),
    }


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_ann = sub.add_parser("audit-annotation", help="Check HLD annotation/cache consistency.")
    audit_ann.add_argument("--anno-path", type=Path, required=True)
    audit_ann.add_argument("--manifest-path", type=Path, default=None)
    audit_ann.add_argument("--cache-dir", type=Path, default=None)
    audit_ann.add_argument("--task", default="HLD")
    audit_ann.add_argument("--recent-frames", type=int, default=4)
    add_common_output_args(audit_ann)

    audit_res = sub.add_parser("audit-results", help="Summarize a prediction JSON.")
    audit_res.add_argument("--result-path", type=Path, required=True)
    audit_res.add_argument("--task", default="HLD")
    audit_res.add_argument("--paper-accuracy", type=float, default=52.1)
    add_common_output_args(audit_res)

    score = sub.add_parser("score-compare", help="Compare stored, regex, and substring scoring.")
    score.add_argument("--result-path", type=Path, required=True)
    score.add_argument("--task", default="HLD")
    add_common_output_args(score)

    frame = sub.add_parser("frame-compare", help="Compare cached frames with SimpleStream reference decode.")
    frame.add_argument("--ids", default=None, help="Comma-separated video ids.")
    frame.add_argument("--result-path", type=Path, default=None, help="Read ids from a result JSON.")
    frame.add_argument("--task", default="HLD")
    frame.add_argument("--chunked-dir", type=Path, required=True)
    frame.add_argument("--cache-dir", type=Path, required=True)
    frame.add_argument("--ss-ref-dir", type=Path, default=DEFAULT_SS_REF_DIR)
    frame.add_argument("--recent-frames", type=int, default=4)
    frame.add_argument("--fps", type=float, default=1.0)
    frame.add_argument("--chunk-duration", type=float, default=1.0)
    frame.add_argument("--limit", type=int, default=None)
    add_common_output_args(frame)

    enc = sub.add_parser("encode-compare", help="Compare current processor encoding with SimpleStream Qwen3 explicit blocks.")
    enc.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    enc.add_argument("--frame-dir", type=Path, required=True)
    enc.add_argument("--task", default="HLD")
    enc.add_argument("--question", default="")
    enc.add_argument("--options-json", default="[]", help="JSON list of options.")
    enc.add_argument("--recent-frames", type=int, default=4)
    enc.add_argument("--merge-size", type=int, default=2)
    enc.add_argument("--out-json", type=Path, default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit-annotation":
        summary, rows = audit_annotations(
            args.anno_path,
            task=args.task,
            manifest_path=args.manifest_path,
            cache_dir=args.cache_dir,
            recent_frames=args.recent_frames,
        )
        _write_json(args.out_json, {"summary": summary, "rows": rows})
        _write_csv(args.out_csv, rows)
        _print_json(summary)
        return 0

    if args.command == "audit-results":
        summary, rows = audit_results(args.result_path, task=args.task, paper_accuracy=args.paper_accuracy)
        _write_json(args.out_json, {"summary": summary, "rows": rows})
        _write_csv(args.out_csv, rows)
        _print_json(summary)
        return 0

    if args.command == "score-compare":
        summary, rows = compare_scoring(args.result_path, task=args.task)
        _write_json(args.out_json, {"summary": summary, "rows": rows})
        _write_csv(args.out_csv, rows)
        _print_json(summary)
        return 0

    if args.command == "frame-compare":
        ids = _ids_from_args(args.ids, args.result_path, args.task, args.limit)
        if not ids:
            raise SystemExit("No video ids provided. Use --ids or --result-path.")
        summary, rows = compare_frames(
            video_ids=ids,
            chunked_dir=args.chunked_dir,
            cache_dir=args.cache_dir,
            ss_ref_dir=args.ss_ref_dir,
            recent_frames=args.recent_frames,
            fps=args.fps,
            chunk_duration=args.chunk_duration,
        )
        _write_json(args.out_json, {"summary": summary, "rows": rows})
        _write_csv(args.out_csv, rows)
        _print_json(summary)
        return 0

    if args.command == "encode-compare":
        try:
            options = json.loads(args.options_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--options-json must be a JSON list: {exc}") from exc
        if not isinstance(options, list):
            raise SystemExit("--options-json must be a JSON list")
        summary = compare_encoding(
            model_path=args.model_path,
            frame_dir=args.frame_dir,
            task=args.task,
            question=args.question,
            options=options,
            recent_frames=args.recent_frames,
            merge_size=args.merge_size,
        )
        _write_json(args.out_json, summary)
        _print_json(summary)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
