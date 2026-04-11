"""
Statistical significance testing for SSD-VLM ablation studies.
Provides rigorous statistical analysis for reviewer concerns.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import binom_test

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Analyzer for statistical significance testing."""
    
    def __init__(self, task_categories: Optional[Dict[str, str]] = None):
        """
        Initialize analyzer.
        
        Args:
            task_categories: Mapping from task name to Lock/Fork category
        """
        self.task_categories = task_categories or {
            "OCR": "lock",
            "ATR": "lock",
            "OJR": "lock",
            "STU": "lock",
            "ACR": "lock",
            "FPD": "lock",
            "EPM": "fork",
            "ASI": "fork",
            "HLD": "fork",
        }
    
    def mcnemar_test(
        self,
        base_correct: np.ndarray,
        ssd_correct: np.ndarray,
    ) -> Dict[str, float]:
        """
        McNemar's test for paired samples.
        Tests if SSD-VLM significantly differs from base model.
        
        Args:
            base_correct: Binary array of base model correctness
            ssd_correct: Binary array of SSD-VLM correctness
        
        Returns:
            Dictionary with test statistics
        """
        # Build contingency table
        # n01 = base wrong, SSD right (improvement)
        # n10 = base right, SSD wrong (regression)
        n01 = np.sum((base_correct == 0) & (ssd_correct == 1))
        n10 = np.sum((base_correct == 1) & (ssd_correct == 0))
        
        # McNemar's test: chi-square approximation
        if (n01 + n10) > 0:
            chi2 = (n01 - n10)**2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2 = 0.0
            p_value = 1.0
        
        return {
            "n_improvement": int(n01),
            "n_regression": int(n10),
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
        }
    
    def bootstrap_ci(
        self,
        values: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals.
        
        Args:
            values: Array of values
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval level (0.95 = 95%)
        
        Returns:
            Dictionary with CI bounds and estimate
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(sample.mean())
        
        bootstrap_means = np.array(bootstrap_means)
        
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            "estimate": float(values.mean()),
            "ci_lower": float(np.percentile(bootstrap_means, lower_percentile)),
            "ci_upper": float(np.percentile(bootstrap_means, upper_percentile)),
            "std_error": float(bootstrap_means.std()),
        }
    
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            group1: First group values
            group2: Second group values
        
        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((group1.mean() - group2.mean()) / pooled_std)
    
    def analyze_lock_fork_improvement(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze improvement asymmetry between Lock and Fork tasks.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            Analysis results
        """
        lock_improvements = []
        fork_improvements = []
        
        for pred in predictions:
            task_type = pred.get("task_type", "unknown")
            is_improvement = pred.get("is_improvement", False)
            
            if task_type in self.task_categories:
                category = self.task_categories[task_type]
                if category == "lock":
                    lock_improvements.append(float(is_improvement))
                else:
                    fork_improvements.append(float(is_improvement))
        
        lock_improvements = np.array(lock_improvements)
        fork_improvements = np.array(fork_improvements)
        
        # Improvement rates
        lock_rate = lock_improvements.mean() if len(lock_improvements) > 0 else 0.0
        fork_rate = fork_improvements.mean() if len(fork_improvements) > 0 else 0.0
        
        # Statistical test: chi-square test for independence
        lock_improvement_count = int(lock_improvements.sum())
        lock_total = len(lock_improvements)
        fork_improvement_count = int(fork_improvements.sum())
        fork_total = len(fork_improvements)
        
        # Build contingency table
        contingency = np.array([
            [lock_improvement_count, lock_total - lock_improvement_count],
            [fork_improvement_count, fork_total - fork_improvement_count],
        ])
        
        if (contingency.sum() > 0) and (contingency.min() >= 0):
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        else:
            chi2 = 0.0
            p_value = 1.0
            dof = 1
            expected = np.array([[0, 0], [0, 0]])
        
        # Cohen's h for effect size (proportions)
        if lock_rate > 0 and fork_rate > 0:
            h = 2 * (np.arcsin(np.sqrt(lock_rate)) - np.arcsin(np.sqrt(fork_rate)))
        else:
            h = 0.0
        
        return {
            "lock_improvement_rate": float(lock_rate),
            "fork_improvement_rate": float(fork_rate),
            "lock_total": int(lock_total),
            "fork_total": int(fork_total),
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "cohens_h": float(h),
            "significant_at_0.05": p_value < 0.05,
        }
    
    def compare_models(
        self,
        base_results: Dict[str, Any],
        ssd_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between base and SSD-VLM.
        
        Args:
            base_results: Results from base model
            ssd_results: Results from SSD-VLM
        
        Returns:
            Comprehensive comparison dictionary
        """
        # Extract predictions
        base_preds = base_results.get("predictions", [])
        ssd_preds = ssd_results.get("predictions", [])
        
        if len(base_preds) != len(ssd_preds):
            logger.warning("Prediction lists have different lengths")
        
        # Build correctness arrays
        min_len = min(len(base_preds), len(ssd_preds))
        base_correct = np.array([p.get("correct", False) for p in base_preds[:min_len]])
        ssd_correct = np.array([p.get("correct", False) for p in ssd_preds[:min_len]])
        
        # Overall accuracies
        base_acc = base_correct.mean()
        ssd_acc = ssd_correct.mean()
        improvement = ssd_acc - base_acc
        
        # McNemar's test
        mcnemar = self.mcnemar_test(base_correct, ssd_correct)
        
        # Bootstrap CI for improvement
        improvements = np.array([
            float(ssd_correct[i]) - float(base_correct[i])
            for i in range(len(base_correct))
        ])
        improvement_ci = self.bootstrap_ci(improvements)
        
        # Cohen's d for effect size
        cohens_d_value = self.cohens_d(ssd_correct.astype(float), base_correct.astype(float))
        
        # Per-task analysis
        task_analysis = {}
        for task_type in set(p.get("task_type", "unknown") for p in base_preds[:min_len]):
            task_preds_base = [p for p in base_preds if p.get("task_type") == task_type]
            task_preds_ssd = [p for p in ssd_preds if p.get("task_type") == task_type]
            
            if len(task_preds_base) > 0 and len(task_preds_ssd) > 0:
                base_task_correct = np.array([p.get("correct", False) for p in task_preds_base])
                ssd_task_correct = np.array([p.get("correct", False) for p in task_preds_ssd])
                
                task_analysis[task_type] = {
                    "base_accuracy": float(base_task_correct.mean()),
                    "ssd_accuracy": float(ssd_task_correct.mean()),
                    "improvement": float(ssd_task_correct.mean() - base_task_correct.mean()),
                }
        
        return {
            "overall": {
                "base_accuracy": float(base_acc),
                "ssd_accuracy": float(ssd_acc),
                "improvement": float(improvement),
                "improvement_ci_lower": float(improvement_ci["ci_lower"]),
                "improvement_ci_upper": float(improvement_ci["ci_upper"]),
                "cohens_d": float(cohens_d_value),
            },
            "mcnemar_test": mcnemar,
            "task_analysis": task_analysis,
        }


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Perform statistical significance testing for SSD-VLM"
    )
    parser.add_argument("--base_results", type=str, required=True,
                       help="Path to base model results JSON")
    parser.add_argument("--ssd_results", type=str, required=True,
                       help="Path to SSD-VLM results JSON")
    parser.add_argument("--output_file", type=str,
                       default="./results/statistical_analysis.json",
                       help="Output file for statistical results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load results
    logger.info(f"Loading base results from {args.base_results}")
    base_results = load_json(args.base_results)
    
    logger.info(f"Loading SSD results from {args.ssd_results}")
    ssd_results = load_json(args.ssd_results)
    
    # Perform analysis
    analyzer = StatisticalAnalyzer()
    comparison = analyzer.compare_models(base_results, ssd_results)
    
    # Save results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")
    
    # Print summary
    logger.info("\n=== Statistical Significance Analysis ===")
    logger.info(f"Base Model Accuracy: {comparison['overall']['base_accuracy']:.4f}")
    logger.info(f"SSD-VLM Accuracy: {comparison['overall']['ssd_accuracy']:.4f}")
    logger.info(f"Improvement: {comparison['overall']['improvement']:.4f}")
    logger.info(f"95% CI: [{comparison['overall']['improvement_ci_lower']:.4f}, {comparison['overall']['improvement_ci_upper']:.4f}]")
    logger.info(f"Cohen's d: {comparison['overall']['cohens_d']:.4f}")
    logger.info(f"\nMcNemar's Test:")
    logger.info(f"  Improvements: {comparison['mcnemar_test']['n_improvement']}")
    logger.info(f"  Regressions: {comparison['mcnemar_test']['n_regression']}")
    logger.info(f"  Chi-square: {comparison['mcnemar_test']['chi_square']:.4f}")
    logger.info(f"  p-value: {comparison['mcnemar_test']['p_value']:.4e}")
    logger.info(f"  Significant at 0.05: {comparison['mcnemar_test']['significant_at_0.05']}")
    
    if comparison['task_analysis']:
        logger.info(f"\nPer-Task Analysis:")
        for task_type, metrics in comparison['task_analysis'].items():
            logger.info(f"  {task_type}:")
            logger.info(f"    Base: {metrics['base_accuracy']:.4f}")
            logger.info(f"    SSD: {metrics['ssd_accuracy']:.4f}")
            logger.info(f"    Δ: {metrics['improvement']:.4f}")


if __name__ == "__main__":
    main()
