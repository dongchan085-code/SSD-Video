"""
Standard supervised FT baseline evaluation for SSD-VLM.
"""

import argparse
import logging
from typing import Dict

import yaml

from eval_ovo_bench import OVOBenchEvaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a standard supervised FT checkpoint on OVO-Bench"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench", help="Path to OVO-Bench data")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/standard_ft_baseline.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
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
        temperature=config["inference"].get("temperature", 1.0),
        top_k=config["inference"].get("top_k", 1),
        save_predictions=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
    )

    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Lock Task Accuracy: {results['lock_accuracy']:.4f}")
    logger.info(f"Fork Task Accuracy: {results['fork_accuracy']:.4f}")


if __name__ == "__main__":
    main()
