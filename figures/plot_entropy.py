"""
Entropy analysis visualisation: Lock vs Fork entropy distributions
for base model and SSD-VLM.

Usage:
    python figures/plot_entropy.py \
        --entropy results/entropy_comparison.json \
        --base_entropy results/entropy_base.json \
        --ssd_entropy results/entropy_ssd.json \
        --output figures/outputs/entropy_analysis.pdf
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, COLORS, save_figure

logger = logging.getLogger(__name__)


# ── Mock data generators (used when real results are unavailable) ───

def _mock_entropy_data() -> Dict[str, Any]:
    """Generate synthetic entropy distributions for testing."""
    rng = np.random.default_rng(42)
    return {
        "base": {
            "lock": rng.normal(1.8, 0.4, 120).clip(0.2).tolist(),
            "fork": rng.normal(4.2, 0.8, 60).clip(0.5).tolist(),
            "lock_rank": rng.exponential(3, 120).clip(1).tolist(),
            "fork_rank": rng.exponential(8, 60).clip(1).tolist(),
        },
        "ssd": {
            "lock": rng.normal(1.7, 0.35, 120).clip(0.2).tolist(),
            "fork": rng.normal(3.0, 0.6, 60).clip(0.5).tolist(),
            "lock_rank": rng.exponential(2.5, 120).clip(1).tolist(),
            "fork_rank": rng.exponential(5, 60).clip(1).tolist(),
        },
        "comparison": {
            "p_value_lock": 0.23,
            "p_value_fork": 3.2e-8,
        },
    }


def _load_json(path: Optional[str]) -> Optional[Dict]:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def _collect_data(args) -> Dict[str, Any]:
    """Load real data if available, otherwise return mock."""
    base_data = _load_json(args.base_entropy)
    ssd_data = _load_json(args.ssd_entropy)
    comp_data = _load_json(args.entropy)

    if base_data and ssd_data:
        data = {
            "base": {
                "lock": base_data.get("lock_entropy", {}).get("values", []),
                "fork": base_data.get("fork_entropy", {}).get("values", []),
                "lock_rank": base_data.get("lock_rank", {}).get("values", []),
                "fork_rank": base_data.get("fork_rank", {}).get("values", []),
            },
            "ssd": {
                "lock": ssd_data.get("lock_entropy", {}).get("values", []),
                "fork": ssd_data.get("fork_entropy", {}).get("values", []),
                "lock_rank": ssd_data.get("lock_rank", {}).get("values", []),
                "fork_rank": ssd_data.get("fork_rank", {}).get("values", []),
            },
            "comparison": comp_data or {},
        }
        return data

    logger.warning("Using mock entropy data (real results not found)")
    return _mock_entropy_data()


# ── Plotting ────────────────────────────────────────────────────────

def plot_entropy(data: Dict[str, Any], output_path: str):
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), dpi=300,
                             gridspec_kw={"wspace": 0.35})

    comp = data.get("comparison", {})

    # ── Panel A: Violin / box plots ────────────────────────────────
    ax = axes[0]

    groups = [
        data["base"]["lock"],
        data["ssd"]["lock"],
        data["base"]["fork"],
        data["ssd"]["fork"],
    ]
    positions = [1, 2, 4, 5]
    colors_list = [COLORS["base"], COLORS["ssd"],
                   COLORS["base"], COLORS["ssd"]]

    parts = ax.violinplot(groups, positions=positions, showmeans=True,
                          showmedians=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.5)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    # Overlay box plot
    bp = ax.boxplot(groups, positions=positions, widths=0.25,
                    patch_artist=True, zorder=3,
                    showfliers=False, medianprops=dict(color="black", lw=1))
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["Lock Tasks", "Fork Tasks"])
    ax.set_ylabel("Output Entropy (nats)")
    ax.set_title("(a) Entropy Distribution", fontsize=11, fontweight="bold")

    # p-value annotations
    p_lock = comp.get("p_value_lock", None)
    p_fork = comp.get("p_value_fork", None)
    if p_lock is not None:
        sig = "n.s." if p_lock > 0.05 else f"p={p_lock:.1e}"
        ymax = max(max(groups[0]), max(groups[1])) * 1.05
        ax.annotate(sig, xy=(1.5, ymax), ha="center", fontsize=8,
                    color="gray")
    if p_fork is not None:
        sig = f"p={p_fork:.1e}" if p_fork < 0.05 else "n.s."
        ymax = max(max(groups[2]), max(groups[3])) * 1.05
        ax.annotate(sig, xy=(4.5, ymax), ha="center", fontsize=8,
                    color=COLORS["fork"])

    # Legend
    base_patch = mpatches.Patch(color=COLORS["base"], alpha=0.7, label="Base")
    ssd_patch = mpatches.Patch(color=COLORS["ssd"], alpha=0.7, label="SSD-VLM")
    ax.legend(handles=[base_patch, ssd_patch], loc="upper left",
              fontsize=8, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Answer-token rank bar chart ───────────────────────
    ax2 = axes[1]

    categories = ["Lock", "Fork"]
    base_ranks = [
        np.mean(data["base"]["lock_rank"]) if data["base"]["lock_rank"] else 0,
        np.mean(data["base"]["fork_rank"]) if data["base"]["fork_rank"] else 0,
    ]
    ssd_ranks = [
        np.mean(data["ssd"]["lock_rank"]) if data["ssd"]["lock_rank"] else 0,
        np.mean(data["ssd"]["fork_rank"]) if data["ssd"]["fork_rank"] else 0,
    ]
    base_rank_err = [
        np.std(data["base"]["lock_rank"]) / max(np.sqrt(len(data["base"]["lock_rank"])), 1),
        np.std(data["base"]["fork_rank"]) / max(np.sqrt(len(data["base"]["fork_rank"])), 1),
    ]
    ssd_rank_err = [
        np.std(data["ssd"]["lock_rank"]) / max(np.sqrt(len(data["ssd"]["lock_rank"])), 1),
        np.std(data["ssd"]["fork_rank"]) / max(np.sqrt(len(data["ssd"]["fork_rank"])), 1),
    ]

    x = np.arange(len(categories))
    w = 0.3
    ax2.bar(x - w / 2, base_ranks, w, yerr=base_rank_err, capsize=3,
            color=COLORS["base"], alpha=0.8, label="Base", zorder=3)
    ax2.bar(x + w / 2, ssd_ranks, w, yerr=ssd_rank_err, capsize=3,
            color=COLORS["ssd"], alpha=0.8, label="SSD-VLM", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Mean GT-Token Rank")
    ax2.set_title("(b) Answer Token Rank", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# need mpatches for legend
import matplotlib.patches as mpatches


def main():
    parser = argparse.ArgumentParser(
        description="Plot Lock-Fork entropy analysis")
    parser.add_argument("--entropy", default=None,
                        help="entropy comparison JSON")
    parser.add_argument("--base_entropy", default=None,
                        help="base model entropy JSON")
    parser.add_argument("--ssd_entropy", default=None,
                        help="SSD-VLM entropy JSON")
    parser.add_argument("--output",
                        default="./figures/outputs/entropy_analysis.pdf")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data = _collect_data(args)
    plot_entropy(data, args.output)
    logger.info(f"Done: {args.output}")


if __name__ == "__main__":
    main()
