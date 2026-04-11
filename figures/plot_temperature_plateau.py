"""
Plot temperature sweep results.
Line plot showing base model plateau vs SSD-VLM stability.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, apply_style, format_axes, save_figure, setup_figure


def generate_mock_data() -> tuple:
    """Generate mock data for visualization."""
    temperatures = np.array([0.5, 0.7, 0.9, 1.0, 1.2, 1.5])
    
    # Base model accuracy (relatively flat)
    base_accuracy = np.array([0.52, 0.54, 0.55, 0.55, 0.54, 0.53]) * 100
    
    # SSD-VLM (stable improvement)
    ssd_accuracy = np.array([0.54, 0.57, 0.59, 0.60, 0.59, 0.58]) * 100
    
    return temperatures, base_accuracy, ssd_accuracy


def plot_temperature_sweep(
    temperatures: np.ndarray,
    base_accuracy: np.ndarray,
    ssd_accuracy: np.ndarray,
    output_path: str = "./figures/outputs/temperature_sweep.pdf",
):
    """
    Plot temperature sweep results.
    
    Args:
        temperatures: Temperature values
        base_accuracy: Base model accuracies
        ssd_accuracy: SSD-VLM accuracies
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4, 3))
    
    # Plot lines
    ax.plot(
        temperatures,
        base_accuracy,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=8,
        label="Base Model",
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    
    ax.plot(
        temperatures,
        ssd_accuracy,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=8,
        label="SSD-VLM",
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    
    # Fill between for improvement region
    ax.fill_between(
        temperatures,
        base_accuracy,
        ssd_accuracy,
        alpha=0.1,
        color=COLORS["ssd"],
        label="SSD Improvement",
    )
    
    # Formatting
    format_axes(
        ax,
        xlabel="Sampling Temperature",
        ylabel="Accuracy (%)",
        title="Temperature Sweep:\nSSD-VLM Stability",
        xlim=[0.4, 1.6],
        ylim=[50, 62],
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Legend
    ax.legend(loc="best", fontsize=9, frameon=False)
    
    # Add annotation
    ax.annotate(
        "Sampling Temperature\nfor SSD Training (1.5)",
        xy=(1.5, ssd_accuracy[-1]),
        xytext=(1.3, 58),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8,
        ha="center",
    )
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()
    
    print(f"Temperature sweep plot saved to {output_path}")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(description="Plot temperature sweep results")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    if args.use_mock_data:
        temperatures, base_acc, ssd_acc = generate_mock_data()
    else:
        temperatures = np.array([])
        base_acc = np.array([])
        ssd_acc = np.array([])
    
    # Plot
    output_path = Path(args.output_dir) / "temperature_sweep.pdf"
    plot_temperature_sweep(temperatures, base_acc, ssd_acc, str(output_path))


if __name__ == "__main__":
    main()
