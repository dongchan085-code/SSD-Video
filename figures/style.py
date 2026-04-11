"""
Matplotlib style settings for publication-quality SSD-VLM figures.
Based on Nature journal style guidelines.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Paper style settings
PAPER_STYLE = {
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    
    # Figure
    "figure.figsize": (3.5, 2.8),
    "figure.dpi": 300,
    
    # Lines and patches
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "patch.linewidth": 0.5,
    "patch.edgecolor": "black",
    
    # Axes
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    
    # Grid
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "gray",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    
    # Legend
    "legend.frameon": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",
    
    # Colors
    "axes.prop_cycle": plt.cycler(
        "color",
        [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
        ]
    ),
}

# Color scheme
COLORS = {
    "base": "#1f77b4",      # Blue
    "ssd": "#ff7f0e",       # Orange
    "lock": "#2ca02c",      # Green
    "fork": "#d62728",      # Red
    "memory": "#9467bd",    # Purple
    "gray": "#7f7f7f",      # Gray
}


def apply_style():
    """Apply paper style to matplotlib."""
    rcParams.update(PAPER_STYLE)


def setup_figure(figsize=(3.5, 2.8), ncols=1, nrows=1):
    """
    Setup a figure with paper style.
    
    Args:
        figsize: Figure size
        ncols: Number of columns
        nrows: Number of rows
    
    Returns:
        fig, ax or fig, axes
    """
    apply_style()
    
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        dpi=300,
    )
    
    return fig, ax


def add_legend(ax, labels, colors, loc="best", title=None):
    """Add a legend with custom colors."""
    patches = [
        mpatches.Patch(color=COLORS[color], label=label)
        for label, color in zip(labels, colors)
    ]
    
    ax.legend(
        handles=patches,
        loc=loc,
        frameon=False,
        fontsize=9,
        title=title,
    )


def format_axes(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    """Format axes with standard style."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return ax


def save_figure(fig, output_path, dpi=300, bbox_inches="tight", pad_inches=0.1):
    """Save figure with consistent settings."""
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Figure saved to {output_path}")


# Example usage
if __name__ == "__main__":
    apply_style()
    
    # Test plot
    fig, ax = setup_figure()
    
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], "o-", color=COLORS["base"], label="Base")
    ax.plot([1, 2, 3, 4], [2, 5, 3, 4], "s-", color=COLORS["ssd"], label="SSD-VLM")
    
    format_axes(
        ax,
        xlabel="Frame Budget",
        ylabel="Accuracy",
        title="Example Plot",
        ylim=[0, 6],
    )
    
    add_legend(ax, ["Base", "SSD-VLM"], ["base", "ssd"])
    
    plt.tight_layout()
    save_figure(fig, "/tmp/test_figure.png")
    plt.close()
