"""
Visualization Utilities 🎨
==========================
Theme configurations and helper functions for generating slick, publication-ready plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """Sets a modern, clean plotting theme."""
    # Reset to defaults to avoid stacking
    plt.rcdefaults()
    
    # Base Style
    sns.set_style("whitegrid")
    sns.set_context("talk") # Larger font for readability
    
    # Custom RC params for the "Slick" look
    rc_params = {
        'figure.figsize': (12, 6),
        'figure.dpi': 300,
        
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        
        # Axes
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Colors
        'axes.edgecolor': '#2c3e50',
        'text.color': '#2c3e50',
        'xtick.color': '#2c3e50',
        'ytick.color': '#2c3e50',
    }
    plt.rcParams.update(rc_params)

def save_plot(name: str, folder: str = "plots/descriptive_stats", formats=['pdf']):
    """
    Saves the current figure to the specified folder in PDF format.
    
    Args:
        name (str): Filename without extension (e.g., 'f1_score_comparison').
        folder (str): Target directory inside the project root or absolute path.
        formats (list): List of formats to save (default: ['pdf']).
    """
    # Find project root (assuming we are in notebooks/ or ml/ or root)
    # Simple heuristic: look for 'data' or 'plots' in CWD or parent
    
    # Setup output path
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    for fmt in formats:
        path = os.path.join(folder, f"{name}.{fmt}")
        plt.savefig(path, bbox_inches='tight', dpi=300, format=fmt)
        print(f"Saved plot: {path}")

def get_palette():
    """Returns a custom professional palette."""
    # Navy, Teal, Grey scheme
    return ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
