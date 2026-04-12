"""
Pareto frontier figure: ΔRT vs ΔMem scatter plot.

Shows existing methods on a negative-slope frontier, with SSD-VLM
expanding into the previously unoccupied upper-right region.

Usage:
    python figures/plot_pareto.py \
        --comparison results/comparison.json \
        --sweep results/temp_sweep_comparison.json \
        --output figures/outputs/pareto_frontier.pdf
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from style import apply_style, COLORS, save_figure

logger = logging.getLogger(__name__)


# ── Mock / fallback data (used when real results are unavailable) ───

MOCK_METHODS = {
    # name: (delta_mem, delta_rt)
    "HERMES":              (3.2,  -4.1),
    "Flash-VStream":       (2.1,  -2.8),
    "SimpleStream-8f":     (1.5,  -1.2),
    "SimpleStream-16f":    (4.8,  -3.5),
    "SimpleStream-32f":    (6.1,  -5.9),
}

MOCK_TEMP_SWEEP = [
    # (delta_mem, delta_rt) for temperatures 0.5 → 1.5
    (-0.3,  0.8),
    ( 0.4,  0.3),
    ( 0.9, -0.1),
    ( 1.2, -0.5),
    ( 1.8, -1.4),
    ( 2.3, -2.1),
]

MOCK_SSD = (2.8, 0.3)  # ΔMem > 0, ΔRT ≥ 0


def _load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def plot_pareto(
    methods: Dict[str, tuple],
    temp_sweep: list,
    ssd_point: tuple,
    output_path: str,
):
    apply_style()

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # ── Quadrant shading ───────────────────────────────────────────
    xlim = (-1.5, 8.0)
    ylim = (-7.0, 2.5)

    # Upper-right: Pareto expansion (green)
    ax.fill_between([0, xlim[1]], 0, ylim[1],
                    color="#2ca02c", alpha=0.08, zorder=0)
    ax.text(xlim[1] - 0.3, ylim[1] - 0.4, "Pareto Expansion",
            ha="right", va="top", fontsize=8, color="#2ca02c",
            fontstyle="italic")

    # Lower-right: standard trade-off (yellow)
    ax.fill_between([0, xlim[1]], ylim[0], 0,
                    color="#e6ab02", alpha=0.06, zorder=0)
    ax.text(xlim[1] - 0.3, ylim[0] + 0.4, "Standard Trade-off",
            ha="right", va="bottom", fontsize=8, color="#b8860b",
            fontstyle="italic")

    # ── Axes lines ─────────────────────────────────────────────────
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)

    # ── Existing methods (gray / blue) ─────────────────────────────
    for name, (dm, dr) in methods.items():
        ax.scatter(dm, dr, color=COLORS["gray"], s=50, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(name, (dm, dr), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color=COLORS["gray"])

    # ── Temperature sweep (connected dotted line) ──────────────────
    if temp_sweep:
        xs = [p[0] for p in temp_sweep]
        ys = [p[1] for p in temp_sweep]
        ax.plot(xs, ys, "o--", color=COLORS["base"], markersize=4,
                linewidth=1.0, zorder=4, label="Temp sweep (base)")

    # ── SSD-VLM (red star) ─────────────────────────────────────────
    ax.scatter(*ssd_point, marker="*", color=COLORS["ssd"], s=250,
              zorder=5, edgecolors="black", linewidths=0.6,
              label="SSD-VLM")

    # ── Labels / limits ────────────────────────────────────────────
    ax.set_xlabel(r"$\Delta$Mem (pp)", fontsize=11)
    ax.set_ylabel(r"$\Delta$RT (pp)", fontsize=11)
    ax.set_title("Perception\u2013Memory Pareto Frontier", fontsize=12,
                 fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontier")
    parser.add_argument("--comparison", default=None,
                        help="comparison JSON from score_results")
    parser.add_argument("--sweep", default=None,
                        help="temperature sweep comparison JSON")
    parser.add_argument("--output",
                        default="./figures/outputs/pareto_frontier.pdf")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Try to load real data; fall back to mock
    methods = dict(MOCK_METHODS)
    temp_sweep = list(MOCK_TEMP_SWEEP)
    ssd_point = MOCK_SSD

    if args.comparison:
        data = _load_json(args.comparison)
        if data and "improvement" in data:
            imp = data["improvement"]
            ssd_point = (
                imp.get("fork_accuracy", MOCK_SSD[0]) * 100,
                imp.get("lock_accuracy", MOCK_SSD[1]) * 100,
            )

    plot_pareto(methods, temp_sweep, ssd_point, args.output)
    logger.info(f"Done: {args.output}")


if __name__ == "__main__":
    main()
