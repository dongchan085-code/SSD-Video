"""
Plot Lock vs Fork task asymmetry.
Bar chart showing differential improvements across task types.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, format_axes, save_figure, setup_figure


def generate_mock_data() -> tuple:
    """Generate mock data for visualization."""
    # Mock per-task improvements
    tasks = ["OCR", "ATR", "OJR", "STU", "EPM", "ASI"]
    task_categories = ["Lock", "Lock", "Lock", "Lock", "Fork", "Fork"]
    
    base_accuracies = np.array([0.62, 0.58, 0.55, 0.61, 0.45, 0.42])
    ssd_accuracies = np.array([0.67, 0.61, 0.58, 0.65, 0.46, 0.43])
    
    improvements = ssd_accuracies - base_accuracies
    
    return tasks, task_categories, base_accuracies, ssd_accuracies, improvements


def plot_lock_fork_asymmetry(
    tasks: list,
    task_categories: list,
    base_accuracies: np.ndarray,
    ssd_accuracies: np.ndarray,
    improvements: np.ndarray,
    output_path: str = "./figures/outputs/lock_fork_asymmetry.pdf",
):
    """
    Plot Lock vs Fork task improvements.
    
    Args:
        tasks: Task names
        task_categories: Category for each task
        base_accuracies: Base model accuracies
        ssd_accuracies: SSD-VLM accuracies
        improvements: Accuracy improvements
        output_path: Output file path
    """
    fig, (ax1, ax2) = setup_figure(figsize=(7, 3), ncols=2, nrows=1)
    
    # Colors for Lock vs Fork
    colors = [COLORS["lock"] if cat == "Lock" else COLORS["fork"] for cat in task_categories]
    
    # Plot 1: Accuracy comparison
    x = np.arange(len(tasks))
    width = 0.35
    
    ax1.bar(x - width/2, base_accuracies, width, label="Base", color=COLORS["base"], alpha=0.8)
    ax1.bar(x + width/2, ssd_accuracies, width, label="SSD-VLM", color=COLORS["ssd"], alpha=0.8)
    
    ax1.set_xlabel("Task Type", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Per-Task Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=10)
    ax1.set_ylim([0, 0.8])
    ax1.legend(fontsize=9, frameon=False)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Plot 2: Improvements by category
    lock_improvements = improvements[[i for i, cat in enumerate(task_categories) if cat == "Lock"]]
    fork_improvements = improvements[[i for i, cat in enumerate(task_categories) if cat == "Fork"]]
    
    categories = ["Lock Tasks\n(Perception)", "Fork Tasks\n(Reasoning)"]
    means = [np.mean(lock_improvements), np.mean(fork_improvements)]
    stds = [np.std(lock_improvements), np.std(fork_improvements)]
    
    bars = ax2.bar(
        categories,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS["lock"], COLORS["fork"]],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    
    ax2.set_ylabel("Accuracy Improvement (%)", fontsize=11)
    ax2.set_title("Task Category Asymmetry", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, max(means) * 1.3])
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{mean*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()
    
    print(f"Lock-Fork asymmetry plot saved to {output_path}")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(description="Plot Lock vs Fork task asymmetry")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    if args.use_mock_data:
        tasks, categories, base_acc, ssd_acc, improvements = generate_mock_data()
    else:
        tasks = []
        categories = []
        base_acc = np.array([])
        ssd_acc = np.array([])
        improvements = np.array([])
    
    # Plot
    output_path = Path(args.output_dir) / "lock_fork_asymmetry.pdf"
    plot_lock_fork_asymmetry(tasks, categories, base_acc, ssd_acc, improvements, str(output_path))


if __name__ == "__main__":
    main()
