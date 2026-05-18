"""Run a small HLD eval with per-sample progress logging."""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from eval.eval_ovo_bench import OVOBenchEvaluator
from ssd_vlm.eval_metrics import summarize_ovo_predictions
from ssd_vlm.simplestream import score_prediction


def _print(line: str) -> None:
    print(line, flush=True)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _load_partial(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    predictions: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default="D:/ssd_video_data")
    parser.add_argument("--anno-path", default="D:/ssd_video_data/ovo_bench_full.json")
    parser.add_argument("--chunked-dir", default="D:/ssd_video_data/chunked_videos")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--partial-file", required=True)
    parser.add_argument("--task-type", default="HLD")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    partial_path = Path(args.partial_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    _print(
        "LOAD_START "
        f"model={args.model_path} dtype={args.dtype} max_samples={args.max_samples} "
        f"max_new_tokens={args.max_new_tokens} load_in_8bit={args.load_in_8bit}"
    )
    evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map="cuda",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=False,
        attn_implementation=args.attn_implementation,
        num_frames=4,
        frame_sampling_strategy="simplestream",
        resize_shortest_edge=224,
        max_new_tokens=args.max_new_tokens,
        batch_size=1,
        recent_frames_only=4,
        chunk_duration=1.0,
        fps=1.0,
        use_cache=True,
        use_simplestream_decode=True,
    )
    _print("LOAD_DONE")

    dataset = evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split="test",
        anno_path=args.anno_path,
        chunked_dir=args.chunked_dir,
        sample_ratio=1.0,
    )
    dataset.samples = [
        sample for sample in dataset.samples if sample.get("task_type") == args.task_type
    ][: args.max_samples]
    total = len(dataset.samples)

    predictions: List[Dict[str, Any]] = _load_partial(partial_path) if args.resume else []
    completed_ids = {prediction["video_id"] for prediction in predictions}
    pending_indices = [
        idx for idx, sample in enumerate(dataset.samples)
        if sample.get("video_id") not in completed_ids
    ]

    _print(
        "EVAL_START "
        f"task={args.task_type} total={total} completed={len(predictions)} "
        f"pending={len(pending_indices)} ids="
        + ",".join(str(sample["video_id"]) for sample in dataset.samples)
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.perf_counter()
    correct_so_far = sum(bool(prediction["correct"]) for prediction in predictions)

    mode = "a" if args.resume else "w"
    with partial_path.open(mode, encoding="utf-8") as partial_handle:
        for idx in pending_indices:
            meta = dataset.samples[idx]
            video_id = meta.get("video_id")
            done_before = len(predictions)
            _print(
                "SAMPLE_START "
                f"{done_before + 1}/{total} id={video_id} "
                f"source={meta.get('source_id', video_id)}"
            )

            sample = dataset[idx]
            frames = sample.get("frame_images", sample["frames"])
            sample_start = time.perf_counter()
            answer_text = evaluator._generate_answer(
                question=sample["question"],
                options=sample["options"],
                frames=frames,
                task_type=sample.get("task_type", ""),
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                do_sample=False,
            )
            latency_ms = (time.perf_counter() - sample_start) * 1000.0

            task_type = sample.get("task_type", "unknown")
            scored = score_prediction(task_type, answer_text, sample["answer_idx"])
            prediction = {
                "video_id": sample["video_id"],
                "source_id": sample.get("source_id", sample["video_id"]),
                "question": sample["question"],
                "options": sample["options"],
                "ground_truth": scored["ground_truth"],
                "predicted": scored["predicted"],
                "answer_text": answer_text,
                "correct": bool(scored["correct"]),
                "task_type": task_type,
                "ovo_split": sample.get("ovo_split"),
                "latency_ms": latency_ms,
                "pure_memory": sample.get("pure_memory", False),
                "frame_indices": sample.get("frame_indices"),
                "frame_timestamps": sample.get("frame_timestamps"),
                "chunk_ids": sample.get("chunk_ids"),
            }
            predictions.append(prediction)
            partial_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            partial_handle.flush()

            correct_so_far += int(prediction["correct"])
            done = len(predictions)
            elapsed = time.perf_counter() - start_time
            avg = elapsed / max(1, done - len(completed_ids))
            eta = avg * (total - done)
            acc = correct_so_far / done if done else 0.0
            pred_counts = collections.Counter(p["predicted"] for p in predictions)
            mem = ""
            if torch.cuda.is_available():
                mem = (
                    f" gpu_alloc_gb={torch.cuda.memory_allocated() / 1e9:.2f}"
                    f" gpu_peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
                )

            _print(
                "PROGRESS "
                f"{done}/{total} id={video_id} ok={int(prediction['correct'])} "
                f"pred={prediction['predicted']} gt={prediction['ground_truth']} "
                f"acc={acc:.4f} correct={correct_so_far} "
                f"latency={latency_ms / 1000.0:.1f}s "
                f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)} "
                f"pred_counts={dict(sorted(pred_counts.items()))}"
                + mem
            )

    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    )
    results = summarize_ovo_predictions(
        predictions,
        lock_tasks=evaluator.lock_tasks,
        fork_tasks=evaluator.fork_tasks,
        decoding_meta={
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "do_sample": False,
            "use_cache": evaluator.use_cache,
            "max_new_tokens": args.max_new_tokens,
            "load_in_8bit": args.load_in_8bit,
        },
        streaming_meta={
            "recent_frames_only": evaluator.recent_frames_only,
            "chunk_duration": evaluator.chunk_duration,
            "fps": evaluator.fps,
            "use_simplestream_decode": evaluator.use_simplestream_decode,
        },
        save_predictions=True,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
    )
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print(
        "FINAL "
        f"task={args.task_type} accuracy={results['per_task_accuracy'].get(args.task_type)} "
        f"correct={results['num_correct']} total={results['num_total']} "
        f"mean_latency_ms={results['mean_latency_ms']:.1f} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
