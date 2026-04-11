"""
Entropy distribution figure for Lock-Fork hypothesis verification.
Violin/box plot showing entropy differences between base and SSD-VLM.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from style import apply_style, setup_figure, format_axes, save_figure, COLORS


def generate_mock_entropy_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock entropy data for demonstration."""
    np.random.seed(42)
    
    # Base model entropy
    # Lock: low entropy (sharp), Fork: high entropy (flat)
    base_lock_entropy = np.random.normal(1.8, 0.4, size=100)
    base_fork_entropy = np.random.normal(4.2, 0.8, size=100)
    
    # SSD-VLM entropy
    # Lock: maintained low, Fork: reduced (compressed)
    ssd_lock_entropy = np.random.normal(1.7, 0.35, size=100)
    ssd_fork_entropy = np.random.normal(3.0, 0.6, size=100)
    
    return base_lock_entropy, base_fork_entropy, ssd_lock_entropy, ssd_fork_entropy


def plot_entropy_comparison(
    base_lock: np.ndarray,
    base_fork: np.ndarray,
    ssd_lock: np.ndarray,
    ssd_fork: np.ndarray,
    output_path: str = "entropy_comparison.pdf",
):
    """
    Plot entropy distribution comparison.
    
    Args:
        base_lock: Base model entropy for Lock tasks
        base_fork: Base model entropy for Fork tasks
        ssd_lock: SSD-VLM entropy for Lock tasks
        ssd_fork: SSD-VLM entropy for Fork tasks
        output_path: Output file path
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    
    # Prepare data for violin plot
    data_to_plot = [base_lock, base_fork, ssd_lock, ssd_fork]
    positions = [1, 2, 3.5, 4.5]
    colors_list = [COLORS["base"], COLORS["base"], COLORS["ssd"], COLORS["ssd"]]
    
    # Create violin plot
    parts = ax.violinplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )
    
    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    
    # Color other components
    for partname in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.5)
    
    # Add box plots for clarity
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
    )
    
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors_list[i])
        box.set_alpha(0.3)
        box.set_edgecolor("black")
        box.set_linewidth(0.5)
    
    # Format axes
    format_axes(
        ax,
        ylabel="Output Token Entropy",
        title="Lock-Fork Entropy Hypothesis Verification",
        ylim=[0, 6],
    )
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Base\nLock", "Base\nFork", "SSD\nLock", "SSD\nFork"],
        fontsize=10,
    )
    
    # Add vertical line to separate base and SSD
    ax.axvline(x=2.75, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.5, 5.7, "Base Model", fontsize=10, ha="center", fontweight="bold")
    ax.text(4.0, 5.7, "SSD-VLM", fontsize=10, ha="center", fontweight="bold")
    
    # Add grid
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_entropy_statistics(
    base_lock: np.ndarray,
    base_fork: np.ndarray,
    ssd_lock: np.ndarray,
    ssd_fork: np.ndarray,
    output_path: str = "entropy_statistics.pdf",
):
    """
    Plot entropy statistics with error bars.
    
    Args:
        base_lock: Base model entropy for Lock tasks
        base_fork: Base model entropy for Fork tasks
        ssd_lock: SSD-VLM entropy for Lock tasks
        ssd_fork: SSD-VLM entropy for Fork tasks
        output_path: Output file path
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    
    # Compute statistics
    means = [base_lock.mean(), base_fork.mean(), ssd_lock.mean(), ssd_fork.mean()]
    stds = [base_lock.std(), base_fork.std(), ssd_lock.std(), ssd_fork.std()]
    
    # Standard error
    n = [len(base_lock), len(base_fork), len(ssd_lock), len(ssd_fork)]
    sems = [stds[i] / np.sqrt(n[i]) for i in range(4)]
    
    # Bar plot with error bars
    x_pos = np.array([1, 2, 3.5, 4.5])
    colors_list = [COLORS["base"], COLORS["base"], COLORS["ssd"], COLORS["ssd"]]
    
    bars = ax.bar(
        x_pos,
        means,
        yerr=sems,
        capsize=5,
        color=colors_list,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )
    
    # Add value labels
    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + sem + 0.15,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Format axes
    format_axes(
        ax,
        ylabel="Mean Output Token Entropy",
        title="Mean Entropy with Standard Error",
        ylim=[0, 5.5],
    )
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        ["Base\nLock", "Base\nFork", "SSD\nLock", "SSD\nFork"],
        fontsize=10,
    )
    
    # Add vertical line
    ax.axvline(x=2.75, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_entropy_combined(
    base_lock: np.ndarray,
    base_fork: np.ndarray,
    ssd_lock: np.ndarray,
    ssd_fork: np.ndarray,
    output_path: str = "entropy_combined.pdf",
):
    """
    Plot combined entropy figure.
    
    Args:
        base_lock: Base model entropy for Lock tasks
        base_fork: Base model entropy for Fork tasks
        ssd_lock: SSD-VLM entropy for Lock tasks
        ssd_fork: SSD-VLM entropy for Fork tasks
        output_path: Output file path
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        dpi=300,
    )
    
    # Panel A: Violin plot
    data_to_plot = [base_lock, base_fork, ssd_lock, ssd_fork]
    positions = [1, 2, 3.5, 4.5]
    colors_list = [COLORS["base"], COLORS["base"], COLORS["ssd"], COLORS["ssd"]]
    
    parts = ax1.violinplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )
    
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
    
    for partname in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.5)
    
    bp = ax1.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
    )
    
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors_list[i])
        box.set_alpha(0.3)
        box.set_edgecolor("black")
        box.set_linewidth(0.5)
    
    format_axes(
        ax1,
        ylabel="Output Token Entropy",
        title="A. Entropy Distribution Comparison",
        ylim=[0, 6],
    )
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(
        ["Base\nLock", "Base\nFork", "SSD\nLock", "SSD\nFork"],
        fontsize=10,
    )
    
    ax1.axvline(x=2.75, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.text(1.5, 5.7, "Base Model", fontsize=10, ha="center", fontweight="bold")
    ax1.text(4.0, 5.7, "SSD-VLM", fontsize=10, ha="center", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    # Panel B: Mean with error bars
    means = [base_lock.mean(), base_fork.mean(), ssd_lock.mean(), ssd_fork.mean()]
    stds = [base_lock.std(), base_fork.std(), ssd_lock.std(), ssd_fork.std()]
    n = [len(base_lock), len(base_fork), len(ssd_lock), len(ssd_fork)]
    sems = [stds[i] / np.sqrt(n[i]) for i in range(4)]
    
    x_pos = np.array([1, 2, 3.5, 4.5])
    
    bars = ax2.bar(
        x_pos,
        means,
        yerr=sems,
        capsize=5,
        color=colors_list,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )
    
    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + sem + 0.15,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    format_axes(
        ax2,
        ylabel="Mean Output Token Entropy",
        title="B. Mean Entropy with Standard Error",
        ylim=[0, 5.5],
    )
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        ["Base\nLock", "Base\nFork", "SSD\nLock", "SSD\nFork"],
        fontsize=10,
    )
    
    ax2.axvline(x=2.75, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def main():
    """Generate entropy analysis figures."""
    parser = argparse.ArgumentParser(description="Plot entropy analysis results")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--use_mock_data", action="store_true", default=True,
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply style
    apply_style()
    
    print("Generating entropy analysis figures...")
    
    # Generate mock data
    base_lock, base_fork, ssd_lock, ssd_fork = generate_mock_entropy_data()
    
    # Generate figures
    print("\n1. Plotting entropy distribution comparison...")
    plot_entropy_comparison(
        base_lock,
        base_fork,
        ssd_lock,
        ssd_fork,
        output_path=str(Path(args.output_dir) / "entropy_distribution.pdf"),
    )
    
    print("2. Plotting entropy statistics...")
    plot_entropy_statistics(
        base_lock,
        base_fork,
        ssd_lock,
        ssd_fork,
        output_path=str(Path(args.output_dir) / "entropy_statistics.pdf"),
    )
    
    print("3. Plotting combined entropy figure...")
    plot_entropy_combined(
        base_lock,
        base_fork,
        ssd_lock,
        ssd_fork,
        output_path=str(Path(args.output_dir) / "entropy_combined.pdf"),
    )
    
    print(f"\nFigures saved to {args.output_dir}")
    print("Output files:")
    print("  - entropy_distribution.pdf")
    print("  - entropy_statistics.pdf")
    print("  - entropy_combined.pdf")


if __name__ == "__main__":
    main()
