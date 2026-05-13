"""
Temperature Sweep Evaluation for base model.
Tests sampling temperature impact: 0.5 to 1.5.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from eval_ovo_bench import OVOBenchEvaluator

from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class TemperatureSweepEvaluator:
    """Evaluator for temperature sweep."""
    
    def __init__(
        self,
        model_path: str,
        temperatures: List[float] = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5],
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        """
        Initialize temperature sweep evaluator.
        
        Args:
            model_path: Path to model
            temperatures: List of temperatures to test
            dtype: Data type
            device_map: Device mapping
        """
        self.model_path = model_path
        self.temperatures = temperatures
        self.dtype = dtype
        self.device_map = device_map
    
    def sweep(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform temperature sweep.
        
        Args:
            samples: List of OVO-Bench samples
            output_dir: Output directory for results
        
        Returns:
            Results for each temperature
        """
        results = {}
        
        for temperature in self.temperatures:
            logger.info(f"\nEvaluating with temperature={temperature}")
            
            # Create evaluator for this temperature
            evaluator = OVOBenchEvaluator(
                model_path=self.model_path,
                dtype=self.dtype,
                device_map=self.device_map,
            )
            
            # Evaluate
            top_k = 10 if temperature > 0 else 1
            result = evaluator.evaluate(
                samples=samples,
                temperature=temperature,
                top_k=top_k,
                save_predictions=False,
                output_file=None,
            )
            
            results[f"temp_{temperature:.1f}"] = result
            
            logger.info(f"  Accuracy: {result['overall_accuracy']:.4f}")
            logger.info(f"  Lock: {result['lock_accuracy']:.4f}, Fork: {result['fork_accuracy']:.4f}")
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / "temperature_sweep_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nTemperature sweep results saved to {output_file}")
        
        return results


def main():
    """Main script for temperature sweep."""
    parser = argparse.ArgumentParser(description="Temperature sweep for base model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_dir", type=str, default="./results/temperature_sweep",
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
    
    # Create base evaluator to load dataset
    base_evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
        dtype=config["model"].get("dtype", "bfloat16"),
    )
    
    samples = base_evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
    )
    
    # Create sweep evaluator
    sweep_evaluator = TemperatureSweepEvaluator(
        model_path=args.model_path,
        temperatures=config["sweep"].get("temperatures", [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]),
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
    )
    
    # Run sweep
    results = sweep_evaluator.sweep(
        samples=samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
