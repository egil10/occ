"""
Distribution Fit Report 📐
==========================
Generates 25 plots analyzing the statistical distribution of key metrics.
Overlays theoretical models (Normal, Log-Norm, Exp, Gamma, Pareto) to find fits.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
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
    
    # Feature Engineering
    df['kd_ratio_pvp'] = df['kills_pvp'] / df['deaths_pvp'].replace(0, 1)

    set_style()
    palette = get_palette()
    
    # Metrics to analyze (Continuous & Count data)
    metrics = [
        'matches_matches', 
        'wins_matches', 
        'kills_pvp', 
        'kd_ratio_pvp', 
        'wools_objectives'
    ]
    
    print(f"Generating 25 Distribution Fit Plots in {PLOTS_DIR}...")
    
    # --- 1. Normal Distribution Fit ---
    print("1. Fitting Normal Distribution...")
    for m in metrics:
        data = df[m].dropna()
        data = data[data > 0] # Avoid zeros for some fits
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data, stat="density", color=palette[0], alpha=0.4, label="Empirical")
        
        # Fit Normal
        mu, std = stats.norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        
        plt.plot(x, p, 'k', linewidth=2, color='red', label=f'Normal Fit\n($\mu$={mu:.1f}, $\sigma$={std:.1f})')
        plt.title(f"Normal Fit: {m}")
        plt.legend()
        save_plot(f"dist_01_normal_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 2. Log-Normal Distribution Fit ---
    print("2. Fitting Log-Normal Distribution...")
    for m in metrics:
        data = df[m].dropna()
        data = data[data > 0]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data, stat="density", color=palette[1], alpha=0.4, label="Empirical")
        
        # Fit Log-Normal
        shape, loc, scale = stats.lognorm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.lognorm.pdf(x, shape, loc, scale)
        
        plt.plot(x, p, linewidth=2, color='purple', label=f'Log-Norm Fit\n(s={shape:.2f})')
        plt.title(f"Log-Normal Fit: {m}")
        plt.legend()
        save_plot(f"dist_02_lognormal_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 3. Exponential Distribution Fit ---
    print("3. Fitting Exponential Distribution...")
    for m in metrics:
        data = df[m].dropna()
        # Exp is good for non-negative
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data, stat="density", color=palette[2], alpha=0.4, label="Empirical")
        
        # Fit Exponential
        loc, scale = stats.expon.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.expon.pdf(x, loc, scale)
        
        plt.plot(x, p, linewidth=2, color='green', label=f'Exponential Fit')
        plt.title(f"Exponential Fit: {m}")
        plt.legend()
        save_plot(f"dist_03_expon_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 4. Gamma Distribution Fit ---
    print("4. Fitting Gamma Distribution...")
    for m in metrics:
        data = df[m].dropna()
        data = data[data > 0]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data, stat="density", color=palette[3], alpha=0.4, label="Empirical")
        
        # Fit Gamma
        a, loc, scale = stats.gamma.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.gamma.pdf(x, a, loc, scale)
        
        plt.plot(x, p, linewidth=2, color='blue', label=f'Gamma Fit\n(a={a:.2f})')
        plt.title(f"Gamma Fit: {m}")
        plt.legend()
        save_plot(f"dist_04_gamma_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 5. Power Law / Pareto Check (Log-Log) ---
    print("5. Power Law (Log-Log Survival)...")
    for m in metrics:
        data = df[m].dropna()
        data = data[data > 0]
        
        plt.figure(figsize=(12, 6))
        
        # Calculate CCDF (Survival Function)
        sorted_data = np.sort(data)
        y = 1. - np.arange(len(sorted_data)) / float(len(sorted_data))
        
        plt.plot(sorted_data, y, marker='.', linestyle='none', color=palette[4], alpha=0.5, label="Empirical CCDF")
        plt.xscale('log')
        plt.yscale('log')
        
        plt.title(f"Power Law Check (Log-Log): {m}")
        plt.ylabel("P(X > x) [CCDF]")
        plt.xlabel(f"{m} (Log Scale)")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Reference Line (Slope -1)
        # Just visual anchor
        x_ref = np.linspace(sorted_data.min(), sorted_data.max(), 100)
        # Simple arbitrary 1/x line for visual check
        y_ref = x_ref**(-1.0) * (y.max()/y_ref.max()) # Scaled
        # plt.plot(x_ref, y_ref, 'k--', label="Slope -1 (Zipf)")

        save_plot(f"dist_05_powerlaw_{m}", folder=str(PLOTS_DIR))
        plt.close()

    print(f"\nDone! Generated 25 Distribution Fit Plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
