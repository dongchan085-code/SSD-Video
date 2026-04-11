"""
Generate all publication-quality figures for SSD-VLM paper.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from plot_perception_memory_tradeoff import plot_tradeoff
from plot_lock_fork_asymmetry import plot_lock_fork_asymmetry
from plot_temperature_plateau import plot_temperature_sweep
from style import apply_style


def main():
    """Generate all figures."""
    parser = argparse.ArgumentParser(description="Generate all SSD-VLM figures")
    parser.add_argument("--base_results", type=str, help="Path to base results JSON")
    parser.add_argument("--ssd_results", type=str, help="Path to SSD results JSON")
    parser.add_argument("--frame_sweep_dir", type=str, help="Path to frame sweep directory")
    parser.add_argument("--temperature_sweep_dir", type=str, help="Path to temperature sweep directory")
    parser.add_argument("--output_dir", type=str, default="./figures/outputs",
                       help="Output directory for figures")
    parser.add_argument("--use_mock_data", action="store_true", default=True,
                       help="Use mock data for demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply style globally
    apply_style()
    
    print("Generating SSD-VLM figures...")
    
    # Figure 1: Perception-Memory Tradeoff
    print("\n1. Generating perception-memory tradeoff figure...")
    if args.use_mock_data:
        from plot_perception_memory_tradeoff import generate_mock_data
        base_mem, base_acc, ssd_mem, ssd_acc = generate_mock_data()
    else:
        base_mem = np.array([])
        base_acc = np.array([])
        ssd_mem = np.array([])
        ssd_acc = np.array([])
    
    plot_tradeoff(
        base_mem,
        base_acc,
        ssd_mem,
        ssd_acc,
        output_path=str(Path(args.output_dir) / "perception_memory_tradeoff.pdf"),
    )
    
    # Figure 2: Lock vs Fork Asymmetry
    print("2. Generating Lock vs Fork asymmetry figure...")
    if args.use_mock_data:
        from plot_lock_fork_asymmetry import generate_mock_data as generate_lock_fork_data
        tasks, categories, base_acc, ssd_acc, improvements = generate_lock_fork_data()
    else:
        tasks = []
        categories = []
        base_acc = np.array([])
        ssd_acc = np.array([])
        improvements = np.array([])
    
    plot_lock_fork_asymmetry(
        tasks,
        categories,
        base_acc,
        ssd_acc,
        improvements,
        output_path=str(Path(args.output_dir) / "lock_fork_asymmetry.pdf"),
    )
    
    # Figure 3: Temperature Sweep
    print("3. Generating temperature sweep figure...")
    if args.use_mock_data:
        from plot_temperature_plateau import generate_mock_data as generate_temp_data
        temperatures, base_acc, ssd_acc = generate_temp_data()
    else:
        temperatures = np.array([])
        base_acc = np.array([])
        ssd_acc = np.array([])
    
    plot_temperature_sweep(
        temperatures,
        base_acc,
        ssd_acc,
        output_path=str(Path(args.output_dir) / "temperature_sweep.pdf"),
    )
    
    print(f"\nAll figures generated successfully in {args.output_dir}")
    print("Output files:")
    print(f"  - perception_memory_tradeoff.pdf")
    print(f"  - lock_fork_asymmetry.pdf")
    print(f"  - temperature_sweep.pdf")


if __name__ == "__main__":
    main()
