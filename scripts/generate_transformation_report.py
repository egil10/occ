"""
Data Transformation Report 🔄
=============================
Generates 25 plots using Log, Sqrt, Box-Cox, Z-Score, and Rank transformations.
Uncovers hidden patterns in skewed data.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "golden_dataset.parquet"
PLOTS_DIR = PROJECT_DIR / "plots" / "descriptive_stats"

# Add ML folder
sys.path.append(str(PROJECT_DIR))
from ml.visuals import set_style, save_plot, get_palette

def generate_report():
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    set_style()
    palette = get_palette()
    
    metrics = ['matches_matches', 'wins_matches', 'kills_pvp', 'damage_dealt_pvp', 'wools_objectives']
    
    print(f"Generating 25 Transformation Plots in {PLOTS_DIR}...")
    
    # --- 1. Log Transformation Histograms ---
    print("1. Log Transforms...")
    for m in metrics:
        plt.figure(figsize=(12, 6))
        
        # Original
        plt.subplot(1, 2, 1)
        sns.histplot(df[m], kde=True, color=palette[0])
        plt.title(f"Original: {m}")
        
        # Log Transformed
        plt.subplot(1, 2, 2)
        log_data = np.log1p(df[m])
        sns.histplot(log_data, kde=True, color=palette[1])
        plt.title(f"Log(x+1): {m}")
        
        save_plot(f"trans_01_log_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 2. Q-Q Plots (Box-Cox / Normality Check) ---
    print("2. Q-Q Plots (Box-Cox)...")
    pt = PowerTransformer(method='yeo-johnson')
    
    for m in metrics:
        data = df[[m]].dropna()
        # Transform
        transformed = pt.fit_transform(data).flatten()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        stats.probplot(transformed, dist="norm", plot=ax)
        plt.title(f"Q-Q Plot (Box-Cox Transformed): {m}")
        # Customize colors manually since stats.probplot is rigid
        ax.get_lines()[0].set_markerfacecolor(palette[0])
        ax.get_lines()[0].set_markeredgecolor(palette[0])
        ax.get_lines()[1].set_color('red')
        
        save_plot(f"trans_02_qq_boxcox_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 3. Log-Log Scatter Plots (Elasticity) ---
    print("3. Log-Log Scatters...")
    pairs = [
        ('matches_matches', 'wins_matches'),
        ('kills_pvp', 'wins_matches'),
        ('damage_dealt_pvp', 'kills_pvp'),
        ('wools_objectives', 'wins_matches'),
        ('kills_pvp', 'deaths_pvp')
    ]
    
    for x, y in pairs:
        plt.figure(figsize=(12, 6))
        
        # Log-Log data
        lx = np.log1p(df[x])
        ly = np.log1p(df[y])
        
        sns.regplot(x=lx, y=ly, scatter_kws={'alpha':0.2, 'color': palette[2]}, line_kws={'color': 'black'})
        plt.title(f"Log-Log Scatter: {y} vs {x}")
        plt.xlabel(f"Log({x})")
        plt.ylabel(f"Log({y})")
        
        save_plot(f"trans_03_loglog_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 4. Z-Score Standardized Comparison ---
    print("4. Standardized Trends (Z-Scores)...")
    scaler = StandardScaler()
    df_std = df.copy()
    df_std[metrics] = scaler.fit_transform(df[metrics])
    
    # Sort by 'wins_matches' to see trends in other variables relative to winning
    df_sorted = df_std.sort_values('wins_matches').reset_index(drop=True)
    
    # We plot rolling means of Z-scores to see trends cleanly
    window = 100
    
    for m in metrics:
        if m == 'wins_matches': continue
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(df_sorted['wins_matches'].rolling(window).mean(), label='Wins (Z-Score)', color='black', alpha=0.3,  linestyle='--')
        plt.plot(df_sorted[m].rolling(window).mean(), label=f'{m} (Z-Score)', color=palette[3], linewidth=2)
        
        plt.title(f"Trend Comparison: {m} vs Wins (Standardized, Rolling Mean {window})")
        plt.xlabel("Player Rank (Sorted by Wins)")
        plt.ylabel("Z-Score (Standard Deviations)")
        plt.legend()
        plt.axhline(0, color='grey', linewidth=0.5)
        
        save_plot(f"trans_04_zscore_trend_{m}", folder=str(PLOTS_DIR))
        plt.close()
        
    # --- 5. Rank-Order / Lorenz Curves ---
    print("5. Lorenz Curves (Inequality)...")
    # Inequality visualization
    for m in metrics:
        data = np.sort(df[m].dropna())
        # Cumulative sum normalized
        lorenz = np.cumsum(data) / np.sum(data)
        # Population share
        share = np.arange(1, len(lorenz)+1) / len(lorenz)
        
        plt.figure(figsize=(12, 6))
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Equality')
        plt.plot(share, lorenz, color=palette[4], linewidth=3, label='Lorenz Curve')
        
        # Gini Coeff
        gini = 1 - 2 * np.trapz(lorenz, share)
        plt.title(f"Lorenz Curve: {m} (Gini = {gini:.2f})")
        plt.xlabel("Cumulative Share of Players")
        plt.ylabel(f"Cumulative Share of {m}")
        plt.legend()
        plt.fill_between(share, share, lorenz, alpha=0.1, color=palette[4])
        
        save_plot(f"trans_05_lorenz_{m}", folder=str(PLOTS_DIR))
        plt.close()

    print(f"\nDone! Generated 25 Transformation Plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
