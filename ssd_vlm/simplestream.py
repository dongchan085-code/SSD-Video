"""SimpleStream-compatible OVO-Bench prompts, task splits, and scoring."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

BACKWARD_TASK_SET = set(BACKWARD_TASKS)
REAL_TIME_TASK_SET = set(REAL_TIME_TASKS)
FORWARD_TASK_SET = set(FORWARD_TASKS)
MULTIPLE_CHOICE_TASKS = BACKWARD_TASK_SET | REAL_TIME_TASK_SET

MC_DIRECTIVE = "Only give the best option's letter directly."
REC_DIRECTIVE = "Only answer with a number."
YES_NO_DIRECTIVE = "Only answer Yes or No."

# Official OVO-Bench prompt templates (mirrored from JoeLeelyf/ovo-bench/constant.py).
# These are used so per-task numbers can be compared against the published
# leaderboard without prompt-format confounds.
REC_PROMPT_TEMPLATE = (
    "You're watching a video in which people may perform a certain type of action repetively. "
    "The person performing this kind of action are referred to as 'they' in the following statement.\n"
    "You're task is to count how many times have different people in the video perform this kind of action in total.\n"
    "One complete motion counts as one.\n"
    "Now, answer the following question: {question}\n"
    "Provide your answer as a single number (e.g., 0, 1, 2, 3) indicating the total count.\n"
    "Do not include any additional text or explanation in your response."
)
SSR_PROMPT_TEMPLATE = (
    "You're watching a tutorial video which contain a sequential of steps.\n"
    "The following is one step from the whole procedures:\n{step}\n"
    "Your task is to determine if the man or woman in the video is currently performing this step.\n"
    "Answer only with \"Yes\" or \"No\".\n"
    "Do not include any additional text or explanation in your response."
)
CRR_PROMPT_TEMPLATE = (
    "You're responsible of answering questions based on the video content.\n"
    "The following question are relevant to the latest frames, i.e. the end of the video.\n{question}\n"
    "Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, "
    "provide enough information for answering the question.\n"
    "Answer only with \"Yes\" or \"No\".\n"
    "Do not include any additional text or explanation in your response."
)


def task_group(task_type: str) -> str:
    if task_type in BACKWARD_TASK_SET:
        return "backward"
    if task_type in REAL_TIME_TASK_SET:
        return "realtime"
    if task_type in FORWARD_TASK_SET:
        return "forward"
    return "unknown"


def format_options(options: Iterable[Any]) -> str:
    return "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))


def format_ovo_prompt(task_type: str, question: str, options: Optional[List[Any]] = None) -> str:
    """Return the prompt shape used by the SimpleStream OVO evaluator."""
    options = options or []
    if task_type in MULTIPLE_CHOICE_TASKS:
        return (
            f"{question}\n"
            f"{format_options(options)}\n"
            f"{MC_DIRECTIVE}"
        )
    if task_type == "REC":
        return REC_PROMPT_TEMPLATE.format(question=question)
    if task_type == "SSR":
        return SSR_PROMPT_TEMPLATE.format(step=question)
    if task_type == "CRR":
        return CRR_PROMPT_TEMPLATE.format(question=question)
    if options:
        return (
            f"{question}\n"
            f"{format_options(options)}\n"
            f"{MC_DIRECTIVE}"
        )
    return question


def extract_choice(text: str) -> Optional[int]:
    """Extract a 0-based multiple-choice answer from generated text."""
    text_stripped = text.strip()
    text_upper = text_stripped.upper()

    patterns = [
        r"ANSWER\s*(?:IS|:|=)?\s*([A-D])\b",
        r"OPTION\s*([A-D])\b",
        r"^([A-D])[\.\)\s:]",
        r"[\(\[]\s*([A-D])\s*[\)\]]",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return ord(match.group(1)) - ord("A")

    match = re.search(r"\b([0-3])\b", text_stripped)
    if match:
        return int(match.group(1))
    return None


def extract_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def extract_yes_no(text: str) -> Optional[bool]:
    lowered = text.strip().lower()
    if re.search(r"\byes\b", lowered):
        return True
    if re.search(r"\bno\b", lowered):
        return False
    return None


def normalize_ground_truth(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        upper = stripped.upper()
        if len(upper) == 1 and "A" <= upper <= "D":
            return ord(upper) - ord("A")
        if upper in {"YES", "TRUE"}:
            return True
        if upper in {"NO", "FALSE"}:
            return False
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return value


def score_prediction(task_type: str, generated_text: str, ground_truth: Any) -> Dict[str, Any]:
    """Score one SimpleStream OVO prediction."""
    gt = normalize_ground_truth(ground_truth)
    if task_type in MULTIPLE_CHOICE_TASKS:
        pred = extract_choice(generated_text)
        return {"predicted": pred, "ground_truth": gt, "correct": pred == gt}

    if task_type == "REC":
        pred = extract_number(generated_text)
        if pred is None:
            return {"predicted": None, "ground_truth": gt, "correct": False}
        try:
            correct = float(pred) == float(gt)
        except (TypeError, ValueError):
            correct = False
        return {"predicted": pred, "ground_truth": gt, "correct": correct}

    if task_type in {"SSR", "CRR"}:
        pred = extract_yes_no(generated_text)
        return {"predicted": pred, "ground_truth": gt, "correct": pred == gt}

    pred = extract_choice(generated_text)
    return {"predicted": pred, "ground_truth": gt, "correct": pred == gt}


def aggregate_group_accuracy(predictions: List[Dict[str, Any]], group: str) -> Optional[float]:
    selected = [p for p in predictions if p.get("ovo_split") == group]
    if not selected:
        return None
    return sum(1 for p in selected if p.get("correct")) / len(selected)


def prediction_to_simplestream_record(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Keep field names compact and compatible with SimpleStream-style arrays."""
    return {
        "id": prediction.get("video_id"),
        "source_id": prediction.get("source_id", prediction.get("video_id")),
        "task": prediction.get("task_type"),
        "question": prediction.get("question"),
        "options": prediction.get("options"),
        "gt": prediction.get("ground_truth"),
        "pred": prediction.get("predicted"),
        "response": prediction.get("answer_text"),
        "correct": prediction.get("correct"),
        "frame_indices": prediction.get("frame_indices"),
        "frame_timestamps": prediction.get("frame_timestamps"),
        "chunk_ids": prediction.get("chunk_ids"),
    }
