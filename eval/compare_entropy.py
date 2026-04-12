"""
Compare entropy results between base model and SSD-VLM.

Usage:
    python eval/compare_entropy.py \
        --base results/entropy_base.json \
        --ssd  results/entropy_ssd.json \
        --output results/entropy_comparison.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def welch_t_test(vals_a, vals_b):
    """Welch's t-test (unequal variance)."""
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) < 2 or len(b) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t_statistic": float(t), "p_value": float(p)}


def cohens_d(vals_a, vals_b):
    """Cohen's d effect size with pooled std."""
    a, b = np.array(vals_a), np.array(vals_b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def bootstrap_ci(vals_a, vals_b, n_boot=5000, ci=0.95):
    """Bootstrap CI for mean(A) - mean(B)."""
    a, b = np.array(vals_a), np.array(vals_b)
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return {"ci_lower": float(np.percentile(diffs, lo)),
            "ci_upper": float(np.percentile(diffs, hi)),
            "mean_diff": float(np.mean(diffs))}


def compare(base: Dict, ssd: Dict) -> Dict[str, Any]:
    """Build full comparison dict."""
    out: Dict[str, Any] = {}

    for category in ("lock", "fork"):
        ek = f"{category}_entropy"
        rk = f"{category}_rank"

        base_e = base.get(ek, {}).get("values", [])
        ssd_e = ssd.get(ek, {}).get("values", [])
        base_r = base.get(rk, {}).get("values", [])
        ssd_r = ssd.get(rk, {}).get("values", [])

        out[f"{category}_entropy_base"] = float(np.mean(base_e)) if base_e else None
        out[f"{category}_entropy_ssd"] = float(np.mean(ssd_e)) if ssd_e else None
        out[f"{category}_rank_base"] = float(np.mean(base_r)) if base_r else None
        out[f"{category}_rank_ssd"] = float(np.mean(ssd_r)) if ssd_r else None

        # Welch's t-test: base vs SSD entropy
        t_result = welch_t_test(base_e, ssd_e)
        out[f"p_value_{category}"] = t_result["p_value"]
        out[f"t_stat_{category}"] = t_result["t_statistic"]
        out[f"cohens_d_{category}"] = cohens_d(base_e, ssd_e)

        # Bootstrap CI for entropy difference
        if base_e and ssd_e:
            ci = bootstrap_ci(base_e, ssd_e)
            out[f"{category}_entropy_diff_ci"] = ci

        # Rank comparison
        if base_r and ssd_r:
            rank_t = welch_t_test(base_r, ssd_r)
            out[f"p_value_{category}_rank"] = rank_t["p_value"]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs SSD-VLM entropy")
    parser.add_argument("--base", required=True, help="Base entropy JSON")
    parser.add_argument("--ssd", required=True, help="SSD entropy JSON")
    parser.add_argument("--output", default="./results/entropy_comparison.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    with open(args.base) as f:
        base = json.load(f)
    with open(args.ssd) as f:
        ssd = json.load(f)

    result = compare(base, ssd)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=== Entropy Comparison ===")
    for cat in ("lock", "fork"):
        be = result.get(f"{cat}_entropy_base")
        se = result.get(f"{cat}_entropy_ssd")
        p = result.get(f"p_value_{cat}")
        d = result.get(f"cohens_d_{cat}")
        logger.info(f"  {cat.upper()}  base={be:.4f}  ssd={se:.4f}  "
                     f"p={p:.2e}  d={d:.3f}")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
