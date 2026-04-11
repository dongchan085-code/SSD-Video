"""
Score and aggregate OVO-Bench evaluation results.
Produces summary statistics and comparative analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResultsScorer:
    """Score and aggregate evaluation results."""
    
    @staticmethod
    def score_single(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single evaluation result.
        
        Args:
            results: Result dictionary from evaluator
        
        Returns:
            Scored result with additional metrics
        """
        scored = {
            "overall_accuracy": results.get("overall_accuracy", 0.0),
            "lock_accuracy": results.get("lock_accuracy", 0.0),
            "fork_accuracy": results.get("fork_accuracy", 0.0),
            "per_task_accuracy": results.get("per_task_accuracy", {}),
            "num_correct": results.get("num_correct", 0),
            "num_total": results.get("num_total", 0),
        }
        
        # Compute improvement over baseline (if available)
        if "baseline_accuracy" in results:
            scored["improvement_over_baseline"] = (
                scored["overall_accuracy"] - results["baseline_accuracy"]
            )
        
        return scored
    
    @staticmethod
    def compare_results(
        base_results: Dict[str, Any],
        ssd_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare base model and SSD-VLM results.
        
        Args:
            base_results: Base model evaluation results
            ssd_results: SSD-VLM evaluation results
        
        Returns:
            Comparison metrics
        """
        comparison = {
            "base": {
                "overall_accuracy": base_results.get("overall_accuracy", 0.0),
                "lock_accuracy": base_results.get("lock_accuracy", 0.0),
                "fork_accuracy": base_results.get("fork_accuracy", 0.0),
            },
            "ssd": {
                "overall_accuracy": ssd_results.get("overall_accuracy", 0.0),
                "lock_accuracy": ssd_results.get("lock_accuracy", 0.0),
                "fork_accuracy": ssd_results.get("fork_accuracy", 0.0),
            },
            "improvement": {
                "overall_accuracy": (
                    ssd_results.get("overall_accuracy", 0.0) - 
                    base_results.get("overall_accuracy", 0.0)
                ),
                "lock_accuracy": (
                    ssd_results.get("lock_accuracy", 0.0) - 
                    base_results.get("lock_accuracy", 0.0)
                ),
                "fork_accuracy": (
                    ssd_results.get("fork_accuracy", 0.0) - 
                    base_results.get("fork_accuracy", 0.0)
                ),
            },
        }
        
        # Per-task comparison
        base_per_task = base_results.get("per_task_accuracy", {})
        ssd_per_task = ssd_results.get("per_task_accuracy", {})
        
        comparison["per_task_improvement"] = {}
        for task_type in set(list(base_per_task.keys()) + list(ssd_per_task.keys())):
            base_acc = base_per_task.get(task_type, 0.0)
            ssd_acc = ssd_per_task.get(task_type, 0.0)
            comparison["per_task_improvement"][task_type] = ssd_acc - base_acc
        
        return comparison
    
    @staticmethod
    def aggregate_frame_sweep(
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate frame sweep results.
        
        Args:
            results: Results for each frame budget
        
        Returns:
            Aggregated statistics
        """
        frame_budgets = []
        accuracies = []
        lock_accuracies = []
        fork_accuracies = []
        
        for frame_key in sorted(results.keys()):
            if "frames_" in frame_key:
                num_frames = int(frame_key.split("_")[1])
                result = results[frame_key]
                
                frame_budgets.append(num_frames)
                accuracies.append(result.get("overall_accuracy", 0.0))
                lock_accuracies.append(result.get("lock_accuracy", 0.0))
                fork_accuracies.append(result.get("fork_accuracy", 0.0))
        
        aggregated = {
            "frame_budgets": frame_budgets,
            "overall_accuracies": accuracies,
            "lock_accuracies": lock_accuracies,
            "fork_accuracies": fork_accuracies,
            "best_frame_budget": frame_budgets[np.argmax(accuracies)] if accuracies else None,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "improvement_4_to_32": (accuracies[-1] - accuracies[0]) if len(accuracies) >= 2 else 0.0,
        }
        
        return aggregated
    
    @staticmethod
    def aggregate_temperature_sweep(
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate temperature sweep results.
        
        Args:
            results: Results for each temperature
        
        Returns:
            Aggregated statistics
        """
        temperatures = []
        accuracies = []
        lock_accuracies = []
        fork_accuracies = []
        
        for temp_key in sorted(results.keys()):
            if "temp_" in temp_key:
                temp_str = temp_key.split("_")[1]
                temperature = float(temp_str)
                result = results[temp_key]
                
                temperatures.append(temperature)
                accuracies.append(result.get("overall_accuracy", 0.0))
                lock_accuracies.append(result.get("lock_accuracy", 0.0))
                fork_accuracies.append(result.get("fork_accuracy", 0.0))
        
        aggregated = {
            "temperatures": temperatures,
            "overall_accuracies": accuracies,
            "lock_accuracies": lock_accuracies,
            "fork_accuracies": fork_accuracies,
            "best_temperature": temperatures[np.argmax(accuracies)] if accuracies else None,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
        }
        
        return aggregated


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str):
    """Save JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Main scoring script."""
    parser = argparse.ArgumentParser(description="Score OVO-Bench results")
    parser.add_argument("--base_results", type=str, help="Path to base model results")
    parser.add_argument("--ssd_results", type=str, help="Path to SSD-VLM results")
    parser.add_argument("--frame_sweep_dir", type=str, help="Path to frame sweep results directory")
    parser.add_argument("--temperature_sweep_dir", type=str, help="Path to temperature sweep results directory")
    parser.add_argument("--output_file", type=str, default="./results/scored_results.json",
                       help="Output file for scored results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    scorer = ResultsScorer()
    scored_results = {}
    
    # Score base and SSD results
    if args.base_results and args.ssd_results:
        base_results = load_json(args.base_results)
        ssd_results = load_json(args.ssd_results)
        
        scored_results["comparison"] = scorer.compare_results(base_results, ssd_results)
        
        logger.info("Comparison Results:")
        logger.info(f"  Base Overall: {scored_results['comparison']['base']['overall_accuracy']:.4f}")
        logger.info(f"  SSD Overall: {scored_results['comparison']['ssd']['overall_accuracy']:.4f}")
        logger.info(f"  Improvement: {scored_results['comparison']['improvement']['overall_accuracy']:.4f}")
    
    # Aggregate frame sweep
    if args.frame_sweep_dir:
        frame_sweep_file = Path(args.frame_sweep_dir) / "frame_sweep_results.json"
        if frame_sweep_file.exists():
            frame_results = load_json(str(frame_sweep_file))
            scored_results["frame_sweep"] = scorer.aggregate_frame_sweep(frame_results)
            
            logger.info("Frame Sweep Results:")
            logger.info(f"  Best Frame Budget: {scored_results['frame_sweep']['best_frame_budget']}")
            logger.info(f"  Best Accuracy: {scored_results['frame_sweep']['best_accuracy']:.4f}")
    
    # Aggregate temperature sweep
    if args.temperature_sweep_dir:
        temp_sweep_file = Path(args.temperature_sweep_dir) / "temperature_sweep_results.json"
        if temp_sweep_file.exists():
            temp_results = load_json(str(temp_sweep_file))
            scored_results["temperature_sweep"] = scorer.aggregate_temperature_sweep(temp_results)
            
            logger.info("Temperature Sweep Results:")
            logger.info(f"  Best Temperature: {scored_results['temperature_sweep']['best_temperature']}")
            logger.info(f"  Best Accuracy: {scored_results['temperature_sweep']['best_accuracy']:.4f}")
    
    # Save scored results
    save_json(scored_results, args.output_file)
    logger.info(f"Scored results saved to {args.output_file}")


if __name__ == "__main__":
    main()
