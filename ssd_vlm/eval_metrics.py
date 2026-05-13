"""OVO-Bench prediction aggregation.

Pure functions that turn a list of per-sample prediction dicts into the
metrics blob that ``eval/eval_ovo_bench.py`` writes to disk. Kept out of
the evaluator class so it stays testable without instantiating a model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ssd_vlm.simplestream import (
    BACKWARD_TASK_SET,
    FORWARD_TASK_SET,
    REAL_TIME_TASK_SET,
    aggregate_group_accuracy,
    prediction_to_simplestream_record,
)


def _subset_accuracy(
    task_results: Mapping[str, Mapping[str, int]],
    task_filter: Iterable[str],
) -> float:
    task_filter = set(task_filter)
    correct = sum(r["correct"] for task, r in task_results.items() if task in task_filter)
    total = sum(r["total"] for task, r in task_results.items() if task in task_filter)
    return correct / total if total > 0 else 0.0


def summarize_ovo_predictions(
    predictions: Sequence[Dict[str, Any]],
    *,
    lock_tasks: Iterable[str],
    fork_tasks: Iterable[str],
    decoding_meta: Mapping[str, Any],
    streaming_meta: Mapping[str, Any],
    save_predictions: bool = True,
    peak_gpu_memory_gb: float | None = None,
) -> Dict[str, Any]:
    """Aggregate per-sample predictions into the OVO-Bench results dict."""
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])

    task_results: Dict[str, Dict[str, int]] = {}
    for p in predictions:
        task = p.get("task_type", "unknown")
        bucket = task_results.setdefault(task, {"correct": 0, "total": 0})
        bucket["correct"] += int(bool(p["correct"]))
        bucket["total"] += 1

    per_task_accuracy = {
        task: (r["correct"] / r["total"] if r["total"] else 0.0)
        for task, r in task_results.items()
    }

    accuracy = correct / total if total > 0 else 0.0
    lock_accuracy = _subset_accuracy(task_results, lock_tasks)
    fork_accuracy = _subset_accuracy(task_results, fork_tasks)
    realtime_accuracy = _subset_accuracy(task_results, REAL_TIME_TASK_SET)
    backward_accuracy = _subset_accuracy(task_results, BACKWARD_TASK_SET)

    forward_correct = sum(
        r["correct"] for task, r in task_results.items() if task in FORWARD_TASK_SET
    )
    forward_total = sum(
        r["total"] for task, r in task_results.items() if task in FORWARD_TASK_SET
    )
    forward_accuracy = (forward_correct / forward_total) if forward_total > 0 else None

    latencies: List[float] = [p["latency_ms"] for p in predictions]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    pure_memory_correct = sum(
        1 for p in predictions if p.get("pure_memory") and p["correct"]
    )
    pure_memory_total = sum(1 for p in predictions if p.get("pure_memory"))

    simple_predictions = {
        split: [
            prediction_to_simplestream_record(p)
            for p in predictions
            if p.get("ovo_split") == split
        ]
        for split in ("backward", "realtime", "forward")
    }

    rt_bwd_values = [
        v for v in (
            aggregate_group_accuracy(predictions, "realtime"),
            aggregate_group_accuracy(predictions, "backward"),
        )
        if v is not None
    ]
    three_way_values = [
        v for v in (
            aggregate_group_accuracy(predictions, "realtime"),
            aggregate_group_accuracy(predictions, "backward"),
            aggregate_group_accuracy(predictions, "forward"),
        )
        if v is not None
    ]

    return {
        "overall_accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "per_task_accuracy": per_task_accuracy,
        "lock_accuracy": lock_accuracy,
        "fork_accuracy": fork_accuracy,
        "realtime_accuracy": realtime_accuracy,
        "backward_accuracy": backward_accuracy,
        "forward_accuracy": forward_accuracy,
        "rt_bwd_avg": float(np.mean(rt_bwd_values)) if rt_bwd_values else accuracy,
        "ovo_total_avg_3way": float(np.mean(three_way_values)) if three_way_values else accuracy,
        "ovo_avg": float(np.mean(rt_bwd_values)) if rt_bwd_values else accuracy,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "throughput_samples_per_sec": float(1000.0 / mean_lat) if mean_lat > 0 else 0.0,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "pure_memory_accuracy": (
            pure_memory_correct / pure_memory_total
            if pure_memory_total > 0 else None
        ),
        "pure_memory_n": pure_memory_total,
        "decoding": dict(decoding_meta),
        "streaming": dict(streaming_meta),
        "simplestream": simple_predictions,
        "predictions": list(predictions) if save_predictions else None,
    }
