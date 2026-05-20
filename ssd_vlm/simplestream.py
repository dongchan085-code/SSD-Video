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

# Prompt templates verbatim from EvolvingLMMs-Lab/SimpleStream/ovo_constants.py.
# Reproducing the SimpleStream Qwen3-VL + 4f result requires bit-for-bit prompt
# parity — substituting OVO-Bench's longer prompts changes wording the model
# was tested against and shifts per-task accuracy by several points.
BR_PROMPT_TEMPLATE = (
    "{question}\n"
    "Options: {options}\n"
    "Only give the best option's letter directly."
)
REC_PROMPT_TEMPLATE = (
    "{question}\n"
    "Only give a number as answer."
)
SSR_PROMPT_TEMPLATE = (
    "Is this person performing the tutorial step: {step}\n"
    "Answer Yes or No only."
)
CRR_PROMPT_TEMPLATE = (
    "{question}\n"
    "Is there enough information in the provided video to answer the question? "
    "Answer Yes or No only."
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
    """Format BR options as SimpleStream does: `A. opt; B. opt; ...;` (single line)."""
    return "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"


def format_ovo_prompt(task_type: str, question: str, options: Optional[List[Any]] = None) -> str:
    """Return the prompt shape used by the SimpleStream OVO evaluator."""
    options = options or []
    if task_type in MULTIPLE_CHOICE_TASKS:
        return BR_PROMPT_TEMPLATE.format(question=question, options=format_options(options))
    if task_type == "REC":
        return REC_PROMPT_TEMPLATE.format(question=question)
    if task_type == "SSR":
        return SSR_PROMPT_TEMPLATE.format(step=question)
    if task_type == "CRR":
        return CRR_PROMPT_TEMPLATE.format(question=question)
    if options:
        return BR_PROMPT_TEMPLATE.format(question=question, options=format_options(options))
    return question


def extract_mcq_answer(response: Optional[str]) -> Optional[str]:
    """Mirror SimpleStream/ovo_constants.extract_br_answer.

    Returns a letter 'A'..'D' or None. Falls back to 1..4 -> A..D mapping.
    """
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return chr(64 + int(m.group(1)))
    return None


def extract_choice(text: str) -> Optional[int]:
    """0-based letter index (A->0..D->3). Wraps extract_mcq_answer for compat."""
    letter = extract_mcq_answer(text)
    return None if letter is None else ord(letter) - ord("A")


def extract_number(text: str) -> Optional[int]:
    """Mirror SimpleStream/ovo_constants.score_rec digit concatenation.

    SimpleStream joins all digit runs into one int (so "1, 2 times" -> 12).
    We return that joined int (or None when no digits exist) — score_prediction
    handles the equality check against the integer ground truth.
    """
    if text is None or not str(text).strip():
        return None
    nums = re.findall(r"\d+", str(text))
    if not nums:
        return None
    try:
        return int("".join(nums))
    except ValueError:
        return None


def extract_yes_no(text: str) -> Optional[bool]:
    """Mirror SimpleStream's score_yesno substring rule.

    The substring rule is intentional and matches SimpleStream verbatim — so
    "NONE" / "KNOW" / "ANNOTATION" return False because they contain "NO".
    Keep parity rather than tightening, otherwise SSR/CRR numbers drift from
    the published leaderboard. The check order is: "NO" first (and "N"
    standalone), then "YES" (and "Y" standalone).
    """
    if text is None or not str(text).strip():
        return None
    s = str(text).strip().upper()
    if (s == "N") or ("NO" in s):
        return False
    if (s == "Y") or ("YES" in s):
        return True
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
            correct = int(pred) == int(gt)
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
