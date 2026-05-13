"""
Dynamic temperature baseline evaluation for SSD-VLM.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from eval_ovo_bench import OVOBenchEvaluator

from ssd_vlm.simplestream import extract_choice
from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class DynamicTemperatureEvaluator(OVOBenchEvaluator):
    """Evaluator that varies temperature by Lock/Fork task type."""

    def __init__(
        self,
        model_path: str,
        temperature_lock: float = 0.3,
        temperature_fork: float = 1.2,
        **kwargs,
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.temperature_lock = temperature_lock
        self.temperature_fork = temperature_fork

    def _get_temperature(self, task_type: str) -> float:
        if task_type in self.lock_tasks:
            return self.temperature_lock
        if task_type in self.fork_tasks:
            return self.temperature_fork
        return self.temperature_fork

    def evaluate(
        self,
        samples: Sequence[Dict[str, Any]],
        save_predictions: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        predictions = []
        task_results = {}
        temperature_usage = {}

        logger.info(f"Evaluating {len(samples)} samples with dynamic temperature")

        for sample in samples:
            task_type = sample.get("task_type", "unknown")
            temperature = self._get_temperature(task_type)
            top_k = 10 if temperature > 0 else 1

            temperature_usage[temperature] = temperature_usage.get(temperature, 0) + 1

            answer_text = self._generate_answer(
                question=sample["question"],
                options=sample["options"],
                frames=sample["frames"],
                temperature=temperature,
                top_k=top_k,
            )
            answer_idx = extract_choice(answer_text)
            if answer_idx is None:
                answer_idx = 0

            is_correct = answer_idx == sample["answer_idx"]
            correct += int(is_correct)
            total += 1

            if task_type not in task_results:
                task_results[task_type] = {"correct": 0, "total": 0}
            task_results[task_type]["correct"] += int(is_correct)
            task_results[task_type]["total"] += 1

            predictions.append({
                "video_id": sample["video_id"],
                "question": sample["question"],
                "task_type": task_type,
                "temperature_used": temperature,
                "ground_truth": sample["answer_idx"],
                "predicted": answer_idx,
                "answer_text": answer_text,
                "correct": is_correct,
            })

        per_task_accuracy = {
            task_type: results["correct"] / results["total"]
            for task_type, results in task_results.items()
            if results["total"] > 0
        }
        lock_correct = sum(
            results["correct"] for task, results in task_results.items()
            if task in self.lock_tasks
        )
        lock_total = sum(
            results["total"] for task, results in task_results.items()
            if task in self.lock_tasks
        )
        fork_correct = sum(
            results["correct"] for task, results in task_results.items()
            if task in self.fork_tasks
        )
        fork_total = sum(
            results["total"] for task, results in task_results.items()
            if task in self.fork_tasks
        )

        results = {
            "overall_accuracy": correct / total if total > 0 else 0.0,
            "num_correct": correct,
            "num_total": total,
            "per_task_accuracy": per_task_accuracy,
            "lock_accuracy": lock_correct / lock_total if lock_total > 0 else 0.0,
            "fork_accuracy": fork_correct / fork_total if fork_total > 0 else 0.0,
            "temperature_lock": self.temperature_lock,
            "temperature_fork": self.temperature_fork,
            "temperature_usage": temperature_usage,
            "predictions": predictions if save_predictions else None,
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dynamic-temperature baseline on OVO-Bench"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench", help="Path to OVO-Bench data")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/dynamic_temperature_baseline.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    evaluator = DynamicTemperatureEvaluator(
        model_path=args.model_path,
        temperature_lock=config["inference"].get("temperature_lock", 0.3),
        temperature_fork=config["inference"].get("temperature_fork", 1.2),
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        num_frames=config["inference"].get("num_frames", 4),
        frame_sampling_strategy=config["inference"].get("frame_sampling_strategy", "uniform"),
        resize_shortest_edge=config["inference"].get("resize_shortest_edge", 224),
        max_new_tokens=config["inference"].get("max_new_tokens", 512),
        batch_size=config["data"].get("batch_size", 16),
    )
    samples = evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
    )
    results = evaluator.evaluate(
        samples=samples,
        save_predictions=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
    )

    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Lock Task Accuracy: {results['lock_accuracy']:.4f}")
    logger.info(f"Fork Task Accuracy: {results['fork_accuracy']:.4f}")


if __name__ == "__main__":
    main()
