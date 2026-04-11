"""
Plot perception vs memory tradeoff.
Scatter plot showing the "green zone" of improved performance.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, format_axes, save_figure, setup_figure


def generate_mock_data() -> tuple:
    """Generate mock data for visualization."""
    # Mock data: (memory_budget, accuracy_delta)
    np.random.seed(42)
    
    # Base model points
    base_memory = np.linspace(100, 1000, 15)
    base_accuracy = np.random.uniform(-2, 2, 15)
    
    # SSD-VLM points (in green zone)
    ssd_memory = np.linspace(150, 800, 20)
    ssd_accuracy = np.clip(np.random.normal(3, 1.5, 20), 0.5, 6)
    
    return (
        base_memory,
        base_accuracy,
        ssd_memory,
        ssd_accuracy,
    )


def plot_tradeoff(
    base_memory: np.ndarray,
    base_accuracy: np.ndarray,
    ssd_memory: np.ndarray,
    ssd_accuracy: np.ndarray,
    output_path: str = "./figures/outputs/perception_memory_tradeoff.pdf",
):
    """
    Plot perception vs memory tradeoff.
    
    Args:
        base_memory: Memory usage for base model
        base_accuracy: Accuracy improvement for base
        ssd_memory: Memory usage for SSD-VLM
        ssd_accuracy: Accuracy improvement for SSD
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    # Plot base model scatter
    ax.scatter(
        base_memory,
        base_accuracy,
        s=80,
        alpha=0.6,
        color=COLORS["base"],
        label="Base Model",
        marker="o",
        edgecolors="black",
        linewidth=0.5,
    )
    
    # Plot SSD-VLM scatter
    ax.scatter(
        ssd_memory,
        ssd_accuracy,
        s=100,
        alpha=0.8,
        color=COLORS["ssd"],
        label="SSD-VLM (Green Zone)",
        marker="s",
        edgecolors="black",
        linewidth=0.5,
    )
    
    # Add green zone shading
    ax.axhspan(1.5, 5.5, alpha=0.1, color=COLORS["ssd"], label="Target Region")
    ax.axvspan(100, 800, alpha=0.05, color="green")
    
    # Formatting
    format_axes(
        ax,
        xlabel="Memory Budget (MB)",
        ylabel="Accuracy Improvement (%)",
        title="Perception-Memory Tradeoff\nSSD-VLM Green Zone",
        xlim=[50, 1050],
        ylim=[-3, 7],
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.axvline(x=400, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)
    
    # Legend
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()
    
    print(f"Tradeoff plot saved to {output_path}")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(description="Plot perception-memory tradeoff")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    if args.use_mock_data:
        base_memory, base_accuracy, ssd_memory, ssd_accuracy = generate_mock_data()
    else:
        # Load from actual results
        base_memory = np.array([])
        base_accuracy = np.array([])
        ssd_memory = np.array([])
        ssd_accuracy = np.array([])
    
    # Plot
    output_path = Path(args.output_dir) / "perception_memory_tradeoff.pdf"
    plot_tradeoff(base_memory, base_accuracy, ssd_memory, ssd_accuracy, str(output_path))


if __name__ == "__main__":
    main()
