"""
Entropy analysis for Lock-Fork hypothesis verification.
Compares output distribution entropy between base model and SSD-VLM.
Tests the Lock-Fork hypothesis mechanistically.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from ssd_vlm.data.ovo_bench_dataset import FORK_TASKS, LOCK_TASKS, OVOBenchDataset
from ssd_vlm.model_loading import load_vlm_processor_and_model
from ssd_vlm.simplestream import format_ovo_prompt
from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class EntropyAnalyzer:
    """Analyzer for output distribution entropy across task types."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        num_frames: int = 4,
        max_new_tokens: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize entropy analyzer.
        
        Args:
            model_path: Path to model
            dtype: Data type
            device_map: Device mapping
            num_frames: Number of frames
            max_new_tokens: Max generation tokens
            batch_size: Batch size
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        logger.info(f"Loading model from: {model_path}")
        self.processor, self.model = load_vlm_processor_and_model(
            model_path=model_path,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Task definitions
        self.lock_tasks = LOCK_TASKS
        self.fork_tasks = FORK_TASKS
    
    def load_ovo_dataset(self, data_path: str, split: str = "test") -> OVOBenchDataset:
        """
        Load OVO-Bench dataset.
        
        Args:
            data_path: Path to OVO-Bench data directory
            split: Dataset split
        
        Returns:
            List of test samples
        """
        return OVOBenchDataset(
            data_path=data_path,
            split=split,
            num_frames=self.num_frames,
        )
    
    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy from logits.
        
        Args:
            logits: [vocab_size] logits
        
        Returns:
            Shannon entropy value
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        probs = probs.clamp(min=1e-10)
        entropy = -torch.sum(probs * torch.log(probs)).item()
        
        return entropy
    
    @torch.no_grad()
    def analyze_entropy(
        self,
        samples: Sequence[Dict[str, Any]],
        save_per_sample: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze output distribution entropy.
        
        Args:
            samples: List of samples
            save_per_sample: Whether to save per-sample entropy
            output_file: Output file for results
        
        Returns:
            Results dictionary with entropy statistics
        """
        entropy_data = {
            "lock": [],
            "fork": [],
            "unknown": [],
        }
        
        per_sample_entropy = []
        
        logger.info(f"Analyzing entropy for {len(samples)} samples")
        
        with tqdm(total=len(samples), desc="Analyzing") as pbar:
            for sample in samples:
                task_type = sample.get("task_type", "unknown")

                # Forward pass to get logits at answer position
                prompt = format_ovo_prompt(task_type, sample["question"], sample["options"])
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": sample["frames"]},
                    {"type": "text", "text": prompt},
                ]}]
                # Unified apply_chat_template (Qwen3-VL API)
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                          for k, v in inputs.items()}

                outputs = self.model(**inputs)
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                entropy = self._compute_entropy(last_logits)

                # Categorize
                if task_type in self.lock_tasks:
                    entropy_data["lock"].append(entropy)
                elif task_type in self.fork_tasks:
                    entropy_data["fork"].append(entropy)
                else:
                    entropy_data["unknown"].append(entropy)

                # Store per-sample
                per_sample_entropy.append({
                    "video_id": sample["video_id"],
                    "task_type": task_type,
                    "entropy": entropy,
                })

                pbar.update(1)
        
        # Compute statistics
        lock_entropy = np.array(entropy_data["lock"])
        fork_entropy = np.array(entropy_data["fork"])
        
        # Compute means and standard deviations
        lock_mean = lock_entropy.mean() if len(lock_entropy) > 0 else 0.0
        lock_std = lock_entropy.std() if len(lock_entropy) > 0 else 0.0
        fork_mean = fork_entropy.mean() if len(fork_entropy) > 0 else 0.0
        fork_std = fork_entropy.std() if len(fork_entropy) > 0 else 0.0
        
        # Perform statistical tests
        # T-test: are Lock and Fork entropies significantly different?
        if len(lock_entropy) > 1 and len(fork_entropy) > 1:
            t_stat, p_value = stats.ttest_ind(lock_entropy, fork_entropy)
            cohen_d = (lock_mean - fork_mean) / np.sqrt(
                ((len(lock_entropy) - 1) * lock_std**2 + 
                 (len(fork_entropy) - 1) * fork_std**2) / 
                (len(lock_entropy) + len(fork_entropy) - 2)
            ) if (lock_std**2 + fork_std**2) > 0 else 0.0
        else:
            t_stat = 0.0
            p_value = 1.0
            cohen_d = 0.0
        
        # Bootstrap confidence intervals for entropy difference
        entropy_diff = []
        n_bootstrap = 1000
        for _ in range(n_bootstrap):
            lock_sample = np.random.choice(lock_entropy, size=len(lock_entropy), replace=True)
            fork_sample = np.random.choice(fork_entropy, size=len(fork_entropy), replace=True)
            entropy_diff.append(lock_sample.mean() - fork_sample.mean())
        
        entropy_diff = np.array(entropy_diff)
        ci_lower = np.percentile(entropy_diff, 2.5)
        ci_upper = np.percentile(entropy_diff, 97.5)
        
        results = {
            "lock_entropy": {
                "mean": float(lock_mean),
                "std": float(lock_std),
                "count": len(lock_entropy),
                "values": lock_entropy.tolist(),
            },
            "fork_entropy": {
                "mean": float(fork_mean),
                "std": float(fork_std),
                "count": len(fork_entropy),
                "values": fork_entropy.tolist(),
            },
            "statistical_tests": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohen_d": float(cohen_d),
                "entropy_diff_ci_lower": float(ci_lower),
                "entropy_diff_ci_upper": float(ci_upper),
            },
            "per_sample_entropy": per_sample_entropy if save_per_sample else None,
        }
        
        # Save results
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze output distribution entropy for Lock-Fork hypothesis"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_file", type=str,
                       default="./results/entropy_analysis.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create analyzer
    analyzer = EntropyAnalyzer(
        model_path=args.model_path,
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        num_frames=config["inference"].get("num_frames", 4),
        max_new_tokens=config["inference"].get("max_new_tokens", 512),
        batch_size=config["data"].get("batch_size", 16),
    )
    
    # Load dataset
    samples = analyzer.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
    )
    
    # Analyze
    results = analyzer.analyze_entropy(
        samples=samples,
        save_per_sample=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
    )
    
    # Print summary
    logger.info("\n=== Lock vs Fork Entropy Analysis ===")
    logger.info(f"Lock tasks (OCR, ATR, OJR, STU):")
    logger.info(f"  Mean entropy: {results['lock_entropy']['mean']:.4f} ± {results['lock_entropy']['std']:.4f}")
    logger.info(f"  Count: {results['lock_entropy']['count']}")
    logger.info(f"\nFork tasks (EPM, ASI):")
    logger.info(f"  Mean entropy: {results['fork_entropy']['mean']:.4f} ± {results['fork_entropy']['std']:.4f}")
    logger.info(f"  Count: {results['fork_entropy']['count']}")
    logger.info(f"\nStatistical Significance:")
    logger.info(f"  t-statistic: {results['statistical_tests']['t_statistic']:.4f}")
    logger.info(f"  p-value: {results['statistical_tests']['p_value']:.4e}")
    logger.info(f"  Cohen's d: {results['statistical_tests']['cohen_d']:.4f}")
    logger.info(f"  Entropy diff 95% CI: [{results['statistical_tests']['entropy_diff_ci_lower']:.4f}, {results['statistical_tests']['entropy_diff_ci_upper']:.4f}]")


if __name__ == "__main__":
    main()
