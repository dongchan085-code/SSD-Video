"""
Comprehensive ablation study figure for SSD-VLM.
Multi-panel figure showing ablation trade-offs and overall accuracy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, setup_figure, format_axes, save_figure, COLORS


def generate_mock_ablation_data() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock ablation results for demonstration."""
    ablation_names = [
        "Baseline\n(Base Model)",
        "SSD-VLM\n(Full)",
        "No\nOversample",
        "LoRA\nOnly",
        "Full FT\nOnly",
        "Standard\nFT",
        "Dynamic\nTemp",
    ]
    
    # Simulated metrics
    # Δ Response Time (lower is better - negative means faster)
    delta_rt = np.array([-0.02, -0.05, -0.04, -0.06, -0.08, -0.02, -0.01])
    
    # Δ Memory (higher is better - positive means improvement)
    delta_mem = np.array([0.0, 0.035, 0.020, 0.015, 0.022, 0.008, 0.005])
    
    # Overall Accuracy improvement
    accuracy_improvement = np.array([0.0, 0.035, 0.020, 0.018, 0.025, 0.010, 0.012])
    
    return ablation_names, delta_rt, delta_mem, accuracy_improvement


def plot_ablation_tradeoff(
    ablation_names: List[str],
    delta_rt: np.ndarray,
    delta_mem: np.ndarray,
    output_path: str = "ablation_tradeoff.pdf",
):
    """
    Plot Panel A: ΔRT vs ΔMem scatter plot.
    
    Args:
        ablation_names: List of ablation names
        delta_rt: Response time deltas
        delta_mem: Memory deltas
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    # Create scatter plot
    colors = [COLORS["ssd"] if i == 1 else COLORS["base"] for i in range(len(ablation_names))]
    sizes = [200 if i == 1 else 100 for i in range(len(ablation_names))]
    
    scatter = ax.scatter(
        delta_rt,
        delta_mem,
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )
    
    # Annotate points
    for i, name in enumerate(ablation_names):
        offset_x = 0.005 if i == 1 else 0.002
        offset_y = 0.002
        ax.annotate(
            name.replace("\n", " "),
            (delta_rt[i], delta_mem[i]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            ha="left",
        )
    
    # Draw "green zone" region
    # Positive ΔMem and negative ΔRT (faster and more accurate)
    ax.fill_between(
        [-0.1, 0.0],
        0.0,
        0.05,
        alpha=0.1,
        color=COLORS["memory"],
        label="Favorable zone",
    )
    
    # Format axes
    format_axes(
        ax,
        xlabel="ΔResponse Time (faster ←)",
        ylabel="ΔMemory Accuracy (+)",
        title="Panel A: Ablation Trade-off Space",
        xlim=[-0.1, 0.01],
        ylim=[-0.005, 0.045],
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Add baseline reference lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_ablation_accuracy(
    ablation_names: List[str],
    accuracy_improvement: np.ndarray,
    output_path: str = "ablation_accuracy.pdf",
):
    """
    Plot Panel B: Overall accuracy improvement comparison.
    
    Args:
        ablation_names: List of ablation names
        accuracy_improvement: Accuracy improvements
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    # Create bar plot
    colors = [COLORS["ssd"] if i == 1 else COLORS["base"] for i in range(len(ablation_names))]
    
    x_pos = np.arange(len(ablation_names))
    bars = ax.bar(
        x_pos,
        accuracy_improvement,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, accuracy_improvement)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    
    # Format axes
    format_axes(
        ax,
        ylabel="ΔAccuracy",
        title="Panel B: Overall Accuracy Improvement",
        ylim=[0, 0.045],
    )
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ablation_names, rotation=45, ha="right", fontsize=9)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_ablation_combined(
    ablation_names: List[str],
    delta_rt: np.ndarray,
    delta_mem: np.ndarray,
    accuracy_improvement: np.ndarray,
    output_path: str = "ablation_combined.pdf",
):
    """
    Plot combined figure with both panels.
    
    Args:
        ablation_names: List of ablation names
        delta_rt: Response time deltas
        delta_mem: Memory deltas
        accuracy_improvement: Accuracy improvements
        output_path: Output file path
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 4),
        dpi=300,
    )
    
    # Panel A: Trade-off
    colors = [COLORS["ssd"] if i == 1 else COLORS["base"] for i in range(len(ablation_names))]
    sizes = [200 if i == 1 else 100 for i in range(len(ablation_names))]
    
    ax1.scatter(
        delta_rt,
        delta_mem,
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )
    
    # Annotate points
    for i, name in enumerate(ablation_names):
        offset_x = 0.005 if i == 1 else 0.002
        offset_y = 0.002
        ax1.annotate(
            name.replace("\n", " "),
            (delta_rt[i], delta_mem[i]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            ha="left",
        )
    
    ax1.fill_between(
        [-0.1, 0.0],
        0.0,
        0.05,
        alpha=0.1,
        color=COLORS["memory"],
    )
    
    format_axes(
        ax1,
        xlabel="ΔResponse Time (faster ←)",
        ylabel="ΔMemory Accuracy (+)",
        title="A. Ablation Trade-off Space",
        xlim=[-0.1, 0.01],
        ylim=[-0.005, 0.045],
    )
    
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    
    # Panel B: Accuracy
    x_pos = np.arange(len(ablation_names))
    bars = ax2.bar(
        x_pos,
        accuracy_improvement,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )
    
    for i, (bar, val) in enumerate(zip(bars, accuracy_improvement)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    
    format_axes(
        ax2,
        ylabel="ΔAccuracy",
        title="B. Overall Accuracy Improvement",
        ylim=[0, 0.045],
    )
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ablation_names, rotation=45, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def main():
    """Generate ablation figures."""
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory with ablation result JSONs")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply style
    apply_style()
    
    print("Generating ablation study figures...")
    
    # Generate mock data
    ablation_names, delta_rt, delta_mem, accuracy_improvement = generate_mock_ablation_data()
    
    # Generate figures
    print("\n1. Plotting trade-off space...")
    plot_ablation_tradeoff(
        ablation_names,
        delta_rt,
        delta_mem,
        output_path=str(Path(args.output_dir) / "ablation_tradeoff.pdf"),
    )
    
    print("2. Plotting accuracy comparison...")
    plot_ablation_accuracy(
        ablation_names,
        accuracy_improvement,
        output_path=str(Path(args.output_dir) / "ablation_accuracy.pdf"),
    )
    
    print("3. Plotting combined figure...")
    plot_ablation_combined(
        ablation_names,
        delta_rt,
        delta_mem,
        accuracy_improvement,
        output_path=str(Path(args.output_dir) / "ablation_combined.pdf"),
    )
    
    print(f"\nFigures saved to {args.output_dir}")
    print("Output files:")
    print("  - ablation_tradeoff.pdf")
    print("  - ablation_accuracy.pdf")
    print("  - ablation_combined.pdf")


if __name__ == "__main__":
    main()
