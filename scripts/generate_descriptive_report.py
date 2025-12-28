"""
Descriptive Stats Report Generator 📊
====================================
Generates 50+ slick plots visualizing the Golden Dataset.
Focuses on Distributions, Correlations, and Advanced Comparisons.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from itertools import combinations

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "golden_dataset.parquet"
PLOTS_DIR = PROJECT_DIR / "plots" / "descriptive_stats"

# Add ML folder to path for visuals
sys.path.append(str(PROJECT_DIR))
from ml.visuals import set_style, save_plot, get_palette

def generate_report():
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter numeric
    df_num = df.select_dtypes(include=[np.number])
    
    # Apply Slick Style
    set_style()
    palette = get_palette()
    
    print(f"Generating Plots in {PLOTS_DIR}...")
    
    # --- 1. Top 10 Feature Distributions (Histograms) ---
    print("1. Generating Distributions...")
    # Select key metrics (mix of Matches, PvP, Objectives)
    key_metrics = [
        'wins_matches', 'wl_ratio_matches', 'matches_matches', 
        'kills_pvp', 'kd_ratio_pvp', 'damage_dealt_pvp', 'accuracy_pvp',
        'wools_objectives', 'cores_objectives', 'monuments_objectives'
    ]
    
    # Ensure they exist
    valid_metrics = [m for m in key_metrics if m in df.columns]
    
    for metric in valid_metrics:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[metric], kde=True, color=palette[0], edgecolor=None, alpha=0.6)
        plt.title(f"Distribution of {metric.replace('_', ' ').title()}")
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Count")
        
        # Log scale if skewed
        if df[metric].skew() > 4:
            plt.yscale('log')
            plt.title(f"Distribution of {metric.replace('_', ' ').title()} (Log Scale)")
            
        save_plot(f"dist_{metric}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 2. Top 10 Correlations (Scatter with Regression) ---
    print("2. Generating Scatters...")
    # Interesting pairs
    pairs = [
        ('matches_matches', 'wins_matches'),
        ('kills_pvp', 'wins_matches'),
        ('damage_dealt_pvp', 'kills_pvp'),
        ('accuracy_pvp', 'kd_ratio_pvp'),
        ('wools_objectives', 'wins_matches'),
        ('cores_objectives', 'wins_matches'),
        ('matches_matches', 'kills_pvp'),
        ('killstreak_pvp', 'kd_ratio_pvp'),
        ('arrows_hit_pvp', 'damage_dealt_pvp'),
        ('monuments_objectives', 'wools_objectives')
    ]
    
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(12, 6))
            sns.regplot(x=df[x], y=df[y], scatter_kws={'alpha':0.3, 'color': palette[1]}, line_kws={'color': palette[3]})
            plt.title(f"{y.title()} vs {x.title()}")
            save_plot(f"scatter_{y}_vs_{x}", folder=str(PLOTS_DIR))
            plt.close()

    # --- 3. Box Plots (Outlier Detection) ---
    print("3. Generating Box Plots...")
    for metric in valid_metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[metric], color=palette[2])
        plt.title(f"Box Plot: {metric.replace('_', ' ').title()}")
        save_plot(f"box_{metric}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 4. Violin Plots (Density) ---
    print("4. Generating Violin Plots...")
    for metric in valid_metrics:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=df[metric], color=palette[1], inner="quartile")
        plt.title(f"Violin Plot: {metric.replace('_', ' ').title()}")
        save_plot(f"violin_{metric}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 5. Cumulative Distribution Functions (CDF) ---
    print("5. Generating CDFs...")
    for metric in valid_metrics:
        plt.figure(figsize=(12, 6))
        sns.ecdfplot(data=df, x=metric, color=palette[0], linewidth=3)
        plt.title(f"Cumulative Distribution: {metric.replace('_', ' ').title()}")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        save_plot(f"cdf_{metric}", folder=str(PLOTS_DIR))
        plt.close()
        
    # --- 6. Correlation Heatmap (Master) ---
    print("6. Generating Master Heatmap...")
    plt.figure(figsize=(20, 16))
    # Correlation of valid metrics
    corr = df[valid_metrics].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5)
    plt.title("Correlation Matrix of Top Metrics")
    save_plot("master_correlation_heatmap", folder=str(PLOTS_DIR))
    plt.close()

    print(f"\nDone! Generated ~{len(valid_metrics)*5 + len(pairs)} plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
