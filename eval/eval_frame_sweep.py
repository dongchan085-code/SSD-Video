"""
Frame Budget Sweep Evaluation for SSD-VLM.
Tests performance across different frame budgets: 4, 8, 16, 32.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from eval_ovo_bench import OVOBenchEvaluator

from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class FrameSweepEvaluator:
    """Evaluator for frame budget sweep."""
    
    def __init__(
        self,
        model_path: str,
        frame_budgets: List[int] = [4, 8, 16, 32],
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        """
        Initialize frame sweep evaluator.
        
        Args:
            model_path: Path to model
            frame_budgets: List of frame budgets to test
            dtype: Data type
            device_map: Device mapping
        """
        self.model_path = model_path
        self.frame_budgets = frame_budgets
        self.dtype = dtype
        self.device_map = device_map
    
    def sweep(
        self,
        data_path: str,
        split: str,
        output_dir: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform frame budget sweep.
        
        Args:
            samples: List of OVO-Bench samples
            output_dir: Output directory for results
        
        Returns:
            Results for each frame budget
        """
        results = {}
        
        for num_frames in self.frame_budgets:
            logger.info(f"\nEvaluating with {num_frames} frames")
            
            # Create evaluator for this frame budget
            evaluator = OVOBenchEvaluator(
                model_path=self.model_path,
                dtype=self.dtype,
                device_map=self.device_map,
                num_frames=num_frames,
            )
            
            # Evaluate
            samples = evaluator.load_ovo_dataset(
                data_path=data_path,
                split=split,
            )
            result = evaluator.evaluate(
                samples=samples,
                temperature=1.0,
                top_k=1,
                save_predictions=False,
                output_file=None,
            )
            
            results[f"frames_{num_frames}"] = result
            
            logger.info(f"  Accuracy: {result['overall_accuracy']:.4f}")
            logger.info(f"  Lock: {result['lock_accuracy']:.4f}, Fork: {result['fork_accuracy']:.4f}")
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / "frame_sweep_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nFrame sweep results saved to {output_file}")
        
        return results


def main():
    """Main script for frame sweep."""
    parser = argparse.ArgumentParser(description="Frame budget sweep for SSD-VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_dir", type=str, default="./results/frame_sweep",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create sweep evaluator
    sweep_evaluator = FrameSweepEvaluator(
        model_path=args.model_path,
        frame_budgets=config["sweep"].get("frame_budgets", [4, 8, 16, 32]),
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
    )
    
    # Run sweep
    results = sweep_evaluator.sweep(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
