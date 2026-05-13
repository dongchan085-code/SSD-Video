"""
OVO-Bench Evaluation for SSD-VLM.
Evaluates vision language models with 4-frame streaming budget.
Adapted from SimpleStream evaluation protocol.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ssd_vlm.data.ovo_bench_dataset import FORK_TASKS, LOCK_TASKS, OVOBenchDataset
from ssd_vlm.model_loading import load_vlm_processor_and_model
from ssd_vlm.simplestream import (
    BACKWARD_TASK_SET,
    FORWARD_TASK_SET,
    REAL_TIME_TASK_SET,
    aggregate_group_accuracy,
    format_ovo_prompt,
    prediction_to_simplestream_record,
    score_prediction,
)
from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class OVOBenchEvaluator:
    """Evaluator for OVO-Bench benchmark."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_memory: Optional[Dict[Any, str]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        max_new_tokens: int = 512,
        batch_size: int = 16,
        recent_frames_only: Optional[int] = None,
        chunk_duration: float = 1.0,
        fps: float = 1.0,
        use_cache: bool = True,
    ):
        """
        Initialize OVO-Bench evaluator.
        
        Args:
            model_path: Path to model (can be model ID or local path)
            dtype: Data type
            device_map: Device mapping
            num_frames: Number of frames (typically 4)
            max_new_tokens: Max generation tokens
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.recent_frames_only = recent_frames_only or num_frames
        self.chunk_duration = chunk_duration
        self.fps = fps
        self.use_cache = use_cache

        logger.info(f"Loading model from: {model_path}")
        self.processor, self.model = load_vlm_processor_and_model(
            model_path=model_path,
            dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Task definitions
        # Temporal Lock tasks: real-time perception (sharp distributions needed)
        self.lock_tasks = LOCK_TASKS
        # Temporal Fork tasks: backward tracing / memory (flatter distributions needed)
        self.fork_tasks = FORK_TASKS
    
    def load_ovo_dataset(
        self,
        data_path: str,
        split: str = "test",
        anno_path: Optional[str] = None,
        chunked_dir: Optional[str] = None,
    ) -> OVOBenchDataset:
        """
        Load OVO-Bench dataset.
        
        Args:
            data_path: Path to OVO-Bench data directory
            split: Dataset split (test, val, etc.)
        
        Returns:
            List of test samples
        """
        return OVOBenchDataset(
            data_path=data_path,
            split=split,
            num_frames=self.num_frames,
            frame_sampling_strategy=self.frame_sampling_strategy,
            resize_shortest_edge=self.resize_shortest_edge,
            anno_path=anno_path,
            chunked_dir=chunked_dir,
            recent_frames_only=self.recent_frames_only,
            chunk_duration=self.chunk_duration,
            fps=self.fps,
        )
    
    @torch.no_grad()
    def _generate_answer(
        self,
        question: str,
        options: List[str],
        frames: Any,
        task_type: str = "",
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate answer for a single sample.
        
        Args:
            question: Question text
            options: List of options
            frames: [num_frames, 3, H, W] tensor
            temperature: Generation temperature
            top_k: Top-k sampling
        
        Returns:
            Generated text
        """
        prompt = format_ovo_prompt(task_type, question, options)
        
        # Prepare input — pass as a video item so Qwen3-VL temporal-packs the frames.
        if isinstance(frames, list):
            video_content = [{"type": "video", "video": frames}]
        else:
            video_content = [{"type": "video", "video": [frames]}]

        messages = [
            {
                "role": "user",
                "content": video_content + [{"type": "text", "text": prompt}],
            }
        ]
        
        # Unified apply_chat_template (Qwen3-VL API)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        # Move to device
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                 for k, v in inputs.items()}
        
        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=self.use_cache,
        )
        
        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return answer.strip()
    
    def evaluate(
        self,
        samples: Sequence[Dict[str, Any]],
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        do_sample: bool = False,
        save_predictions: bool = True,
        output_file: Optional[str] = None,
        partial_predictions_file: Optional[str] = None,
        resume_partial: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on OVO-Bench.
        
        Args:
            samples: List of samples
            temperature: Generation temperature
            top_k: Top-k sampling
            save_predictions: Whether to save predictions
            output_file: Output file for predictions
        
        Returns:
            Results dictionary with metrics
        """
        predictions = []
        completed_ids = set()
        partial_path = Path(partial_predictions_file) if partial_predictions_file else None
        if partial_path and resume_partial and partial_path.exists():
            with open(partial_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    prediction = json.loads(line)
                    predictions.append(prediction)
                    completed_ids.add(prediction["video_id"])
            logger.info("Loaded %d partial predictions from %s", len(predictions), partial_path)

        pending_samples = [
            sample for sample in samples
            if sample.get("video_id") not in completed_ids
        ]

        logger.info(f"Evaluating {len(pending_samples)} pending samples out of {len(samples)}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        partial_handle = None
        if partial_path:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_handle = open(partial_path, "a", encoding="utf-8")

        try:
            with tqdm(total=len(pending_samples), desc="Evaluating") as pbar:
                for sample in pending_samples:
                    # Generate answer via model inference
                    frames = sample.get("frame_images", sample["frames"])
                    t_start = time.perf_counter()
                    answer_text = self._generate_answer(
                        question=sample["question"],
                        options=sample["options"],
                        frames=frames,
                        task_type=sample.get("task_type", ""),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    latency_ms = (time.perf_counter() - t_start) * 1000.0

                    task_type = sample.get("task_type", "unknown")
                    scored = score_prediction(task_type, answer_text, sample["answer_idx"])
                    answer_idx = scored["predicted"]
                    is_correct = bool(scored["correct"])

                    prediction = {
                        "video_id": sample["video_id"],
                        "source_id": sample.get("source_id", sample["video_id"]),
                        "question": sample["question"],
                        "options": sample["options"],
                        "ground_truth": scored["ground_truth"],
                        "predicted": answer_idx,
                        "answer_text": answer_text,
                        "correct": is_correct,
                        "task_type": task_type,
                        "ovo_split": sample.get("ovo_split"),
                        "latency_ms": latency_ms,
                        "pure_memory": sample.get("pure_memory", False),
                        "frame_indices": sample.get("frame_indices"),
                        "frame_timestamps": sample.get("frame_timestamps"),
                        "chunk_ids": sample.get("chunk_ids"),
                    }
                    predictions.append(prediction)
                    if partial_handle:
                        partial_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        partial_handle.flush()
                    
                    pbar.update(1)
        finally:
            if partial_handle:
                partial_handle.close()

        correct = sum(1 for prediction in predictions if prediction["correct"])
        total = len(predictions)
        task_results = {}
        for prediction in predictions:
            task_type = prediction.get("task_type", "unknown")
            if task_type not in task_results:
                task_results[task_type] = {"correct": 0, "total": 0}
            task_results[task_type]["correct"] += int(bool(prediction["correct"]))
            task_results[task_type]["total"] += 1
        
        # Compute metrics
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-task accuracy
        per_task_accuracy = {}
        for task_type, results in task_results.items():
            acc = results["correct"] / results["total"] if results["total"] > 0 else 0.0
            per_task_accuracy[task_type] = acc
        
        # Lock vs Fork split
        lock_correct = sum(results["correct"] for task, results in task_results.items() 
                          if task in self.lock_tasks)
        lock_total = sum(results["total"] for task, results in task_results.items() 
                        if task in self.lock_tasks)
        lock_accuracy = lock_correct / lock_total if lock_total > 0 else 0.0
        
        fork_correct = sum(results["correct"] for task, results in task_results.items() 
                          if task in self.fork_tasks)
        fork_total = sum(results["total"] for task, results in task_results.items() 
                        if task in self.fork_tasks)
        fork_accuracy = fork_correct / fork_total if fork_total > 0 else 0.0

        realtime_correct = sum(results["correct"] for task, results in task_results.items()
                               if task in REAL_TIME_TASK_SET)
        realtime_total = sum(results["total"] for task, results in task_results.items()
                             if task in REAL_TIME_TASK_SET)
        realtime_accuracy = realtime_correct / realtime_total if realtime_total > 0 else 0.0

        backward_correct = sum(results["correct"] for task, results in task_results.items()
                               if task in BACKWARD_TASK_SET)
        backward_total = sum(results["total"] for task, results in task_results.items()
                             if task in BACKWARD_TASK_SET)
        backward_accuracy = backward_correct / backward_total if backward_total > 0 else 0.0

        forward_correct = sum(results["correct"] for task, results in task_results.items()
                              if task in FORWARD_TASK_SET)
        forward_total = sum(results["total"] for task, results in task_results.items()
                            if task in FORWARD_TASK_SET)
        forward_accuracy = forward_correct / forward_total if forward_total > 0 else None
        
        latencies = [p["latency_ms"] for p in predictions]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        pure_memory_correct = sum(
            1 for p in predictions if p.get("pure_memory") and p["correct"]
        )
        pure_memory_total = sum(
            1 for p in predictions if p.get("pure_memory")
        )

        simple_predictions = {
            "backward": [
                prediction_to_simplestream_record(p)
                for p in predictions if p.get("ovo_split") == "backward"
            ],
            "realtime": [
                prediction_to_simplestream_record(p)
                for p in predictions if p.get("ovo_split") == "realtime"
            ],
            "forward": [
                prediction_to_simplestream_record(p)
                for p in predictions if p.get("ovo_split") == "forward"
            ],
        }
        rt_bwd_values = [
            value for value in [
                aggregate_group_accuracy(predictions, "realtime"),
                aggregate_group_accuracy(predictions, "backward"),
            ]
            if value is not None
        ]
        three_way_values = [
            value for value in [
                aggregate_group_accuracy(predictions, "realtime"),
                aggregate_group_accuracy(predictions, "backward"),
                aggregate_group_accuracy(predictions, "forward"),
            ]
            if value is not None
        ]

        results = {
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
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else None
            ),
            "pure_memory_accuracy": (
                pure_memory_correct / pure_memory_total
                if pure_memory_total > 0 else None
            ),
            "pure_memory_n": pure_memory_total,
            "decoding": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "use_cache": self.use_cache,
            },
            "streaming": {
                "recent_frames_only": self.recent_frames_only,
                "chunk_duration": self.chunk_duration,
                "fps": self.fps,
            },
            "simplestream": simple_predictions,
            "predictions": predictions if save_predictions else None,
        }
        
        # Save results
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate SSD-VLM on OVO-Bench")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_file", type=str, default="./results/ovo_results.json",
                       help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Optional smoke-test limit after dataset loading")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create evaluator
    evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        max_memory=config["model"].get("max_memory"),
        load_in_8bit=config["model"].get("load_in_8bit", False),
        load_in_4bit=config["model"].get("load_in_4bit", False),
        attn_implementation=config["model"].get("attn_implementation"),
        max_pixels=config["model"].get("max_pixels"),
        min_pixels=config["model"].get("min_pixels"),
        num_frames=config["inference"].get("num_frames", 4),
        frame_sampling_strategy=config["inference"].get(
            "frame_sampling_strategy",
            config["evaluation"].get("frame_sampling_strategy", "uniform"),
        ),
        resize_shortest_edge=config["inference"].get(
            "resize_shortest_edge",
            config["evaluation"].get("resize_shortest_edge", 224),
        ),
        max_new_tokens=config["inference"].get("max_new_tokens", 512),
        batch_size=config["data"].get("batch_size", 16),
        recent_frames_only=config["inference"].get(
            "recent_frames_only",
            config["inference"].get("num_frames", 4),
        ),
        chunk_duration=config["inference"].get("chunk_duration", 1.0),
        fps=config["inference"].get("fps", 1.0),
        use_cache=config["inference"].get("use_cache", True),
    )
    
    # Load dataset
    samples = evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
        anno_path=config["data"].get("anno_path"),
        chunked_dir=config["data"].get("chunked_dir"),
    )
    max_samples = args.max_samples or config["evaluation"].get("max_samples")
    if max_samples:
        samples = [samples[i] for i in range(min(int(max_samples), len(samples)))]
        logger.info("Using max_samples=%d", len(samples))
    
    # Evaluate
    results = evaluator.evaluate(
        samples=samples,
        temperature=config["inference"].get("temperature", 1.0),
        top_k=config["inference"].get("top_k", 1),
        top_p=config["inference"].get("top_p", 1.0),
        do_sample=config["inference"].get("do_sample", False),
        save_predictions=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
        partial_predictions_file=config["evaluation"].get(
            "partial_predictions_file",
            str(Path(args.output_file).with_suffix(".partial_predictions.jsonl")),
        ),
        resume_partial=config["evaluation"].get("resume_partial", True),
    )
    
    # Print summary
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Lock Task Accuracy: {results['lock_accuracy']:.4f}")
    logger.info(f"Fork Task Accuracy: {results['fork_accuracy']:.4f}")
    
    for task_type, accuracy in results["per_task_accuracy"].items():
        logger.info(f"{task_type} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
