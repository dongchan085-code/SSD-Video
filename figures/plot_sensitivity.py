"""
Hyperparameter sensitivity figure for SSD-VLM.
Line plots showing accuracy vs sampling temperature, top-k, and oversample ratio.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, setup_figure, format_axes, save_figure, COLORS


def generate_mock_sensitivity_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock sensitivity analysis data."""
    np.random.seed(42)
    
    # Temperature sensitivity (0.5 to 2.0)
    temps = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    acc_temp = np.array([0.65, 0.68, 0.70, 0.72, 0.73, 0.71])
    acc_temp_ssd = np.array([0.68, 0.72, 0.735, 0.748, 0.755, 0.745])
    
    # Top-k sensitivity (5, 10, 20, 50)
    topks = np.array([5, 10, 20, 50])
    acc_topk = np.array([0.68, 0.70, 0.69, 0.67])
    acc_topk_ssd = np.array([0.74, 0.755, 0.74, 0.72])
    
    # Oversample ratio sensitivity (1.0, 1.5, 2.0, 3.0, 4.0)
    ratios = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    acc_ratio = np.array([0.70, 0.70, 0.70, 0.70, 0.70])
    acc_ratio_ssd = np.array([0.73, 0.745, 0.755, 0.752, 0.75])
    
    return temps, acc_temp, acc_temp_ssd, topks, acc_topk, acc_topk_ssd


def plot_temperature_sensitivity(
    temperatures: np.ndarray,
    acc_base: np.ndarray,
    acc_ssd: np.ndarray,
    output_path: str = "temperature_sensitivity.pdf",
):
    """
    Plot temperature sensitivity.
    
    Args:
        temperatures: Temperature values
        acc_base: Base model accuracy
        acc_ssd: SSD-VLM accuracy
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    ax.plot(
        temperatures,
        acc_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model",
    )
    
    ax.plot(
        temperatures,
        acc_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    # Highlight default temperature (1.5)
    ax.axvline(x=1.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.5, 0.665, "Default (1.5)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        ax,
        xlabel="Sampling Temperature",
        ylabel="Accuracy",
        title="Sampling Temperature Sensitivity",
        xlim=[0.4, 2.1],
        ylim=[0.64, 0.76],
    )
    
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_topk_sensitivity(
    topks: np.ndarray,
    acc_base: np.ndarray,
    acc_ssd: np.ndarray,
    output_path: str = "topk_sensitivity.pdf",
):
    """
    Plot top-k sensitivity.
    
    Args:
        topks: Top-k values
        acc_base: Base model accuracy
        acc_ssd: SSD-VLM accuracy
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    ax.plot(
        topks,
        acc_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model",
    )
    
    ax.plot(
        topks,
        acc_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    # Highlight default top-k (10)
    ax.axvline(x=10, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(10, 0.665, "Default (10)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        ax,
        xlabel="Top-k",
        ylabel="Accuracy",
        title="Top-k Sampling Sensitivity",
        xlim=[2, 55],
        ylim=[0.64, 0.76],
    )
    
    ax.set_xscale("log")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_oversample_sensitivity(
    ratios: np.ndarray,
    acc_base: np.ndarray,
    acc_ssd: np.ndarray,
    output_path: str = "oversample_sensitivity.pdf",
):
    """
    Plot oversample ratio sensitivity.
    
    Args:
        ratios: Oversample ratios
        acc_base: Base model accuracy
        acc_ssd: SSD-VLM accuracy
        output_path: Output file path
    """
    fig, ax = setup_figure(figsize=(4.5, 3.5))
    
    ax.plot(
        ratios,
        acc_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model (no change)",
    )
    
    ax.plot(
        ratios,
        acc_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    # Highlight default oversample (2.0)
    ax.axvline(x=2.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.0, 0.72, "Default (2.0)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        ax,
        xlabel="Memory Skill Oversample Ratio",
        ylabel="Accuracy",
        title="Oversample Ratio Sensitivity",
        xlim=[0.8, 4.2],
        ylim=[0.70, 0.76],
    )
    
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def plot_sensitivity_combined(
    temperatures: np.ndarray,
    acc_temp_base: np.ndarray,
    acc_temp_ssd: np.ndarray,
    topks: np.ndarray,
    acc_topk_base: np.ndarray,
    acc_topk_ssd: np.ndarray,
    ratios: np.ndarray,
    acc_ratio_base: np.ndarray,
    acc_ratio_ssd: np.ndarray,
    output_path: str = "sensitivity_combined.pdf",
):
    """
    Plot combined sensitivity figure with 3 panels.
    
    Args:
        temperatures: Temperature values
        acc_temp_base: Base model accuracy for temperature
        acc_temp_ssd: SSD-VLM accuracy for temperature
        topks: Top-k values
        acc_topk_base: Base model accuracy for top-k
        acc_topk_ssd: SSD-VLM accuracy for top-k
        ratios: Oversample ratios
        acc_ratio_base: Base model accuracy for oversample
        acc_ratio_ssd: SSD-VLM accuracy for oversample
        output_path: Output file path
    """
    apply_style()
    
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 3.5),
        dpi=300,
    )
    
    # Panel A: Temperature
    axes[0].plot(
        temperatures,
        acc_temp_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model",
    )
    
    axes[0].plot(
        temperatures,
        acc_temp_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    axes[0].axvline(x=1.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].text(1.5, 0.665, "Default\n(1.5)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        axes[0],
        xlabel="Sampling Temperature",
        ylabel="Accuracy",
        title="A. Temperature Sensitivity",
        xlim=[0.4, 2.1],
        ylim=[0.64, 0.76],
    )
    
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    
    # Panel B: Top-k
    axes[1].plot(
        topks,
        acc_topk_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model",
    )
    
    axes[1].plot(
        topks,
        acc_topk_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    axes[1].axvline(x=10, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].text(10, 0.665, "Default\n(10)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        axes[1],
        xlabel="Top-k",
        ylabel="Accuracy",
        title="B. Top-k Sensitivity",
        xlim=[2, 55],
        ylim=[0.64, 0.76],
    )
    
    axes[1].set_xscale("log")
    axes[1].legend(loc="lower left", fontsize=9, frameon=False)
    axes[1].grid(True, alpha=0.3, linestyle="--", which="both")
    
    # Panel C: Oversample ratio
    axes[2].plot(
        ratios,
        acc_ratio_base,
        "o-",
        color=COLORS["base"],
        linewidth=2,
        markersize=6,
        label="Base Model (no change)",
    )
    
    axes[2].plot(
        ratios,
        acc_ratio_ssd,
        "s-",
        color=COLORS["ssd"],
        linewidth=2,
        markersize=6,
        label="SSD-VLM",
    )
    
    axes[2].axvline(x=2.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[2].text(2.0, 0.72, "Default\n(2.0)", fontsize=8, ha="center", alpha=0.7)
    
    format_axes(
        axes[2],
        xlabel="Memory Skill Oversample Ratio",
        ylabel="Accuracy",
        title="C. Oversample Ratio Sensitivity",
        xlim=[0.8, 4.2],
        ylim=[0.70, 0.76],
    )
    
    axes[2].legend(loc="lower right", fontsize=9, frameon=False)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def main():
    """Generate sensitivity analysis figures."""
    parser = argparse.ArgumentParser(description="Plot hyperparameter sensitivity results")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory with sensitivity result JSONs")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply style
    apply_style()
    
    print("Generating hyperparameter sensitivity figures...")
    
    # Generate mock data
    temps, acc_temp_base, acc_temp_ssd, topks, acc_topk_base, acc_topk_ssd = \
        generate_mock_sensitivity_data()
    
    # Generate oversample ratio data separately
    ratios = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    acc_ratio_base = np.array([0.70, 0.70, 0.70, 0.70, 0.70])
    acc_ratio_ssd = np.array([0.73, 0.745, 0.755, 0.752, 0.75])
    
    # Generate individual figures
    print("\n1. Plotting temperature sensitivity...")
    plot_temperature_sensitivity(
        temps,
        acc_temp_base,
        acc_temp_ssd,
        output_path=str(Path(args.output_dir) / "temperature_sensitivity.pdf"),
    )
    
    print("2. Plotting top-k sensitivity...")
    plot_topk_sensitivity(
        topks,
        acc_topk_base,
        acc_topk_ssd,
        output_path=str(Path(args.output_dir) / "topk_sensitivity.pdf"),
    )
    
    print("3. Plotting oversample ratio sensitivity...")
    plot_oversample_sensitivity(
        ratios,
        acc_ratio_base,
        acc_ratio_ssd,
        output_path=str(Path(args.output_dir) / "oversample_sensitivity.pdf"),
    )
    
    print("4. Plotting combined sensitivity figure...")
    plot_sensitivity_combined(
        temps,
        acc_temp_base,
        acc_temp_ssd,
        topks,
        acc_topk_base,
        acc_topk_ssd,
        ratios,
        acc_ratio_base,
        acc_ratio_ssd,
        output_path=str(Path(args.output_dir) / "sensitivity_combined.pdf"),
    )
    
    print(f"\nFigures saved to {args.output_dir}")
    print("Output files:")
    print("  - temperature_sensitivity.pdf")
    print("  - topk_sensitivity.pdf")
    print("  - oversample_sensitivity.pdf")
    print("  - sensitivity_combined.pdf")


if __name__ == "__main__":
    main()
