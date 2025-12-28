"""
Regression Analysis Report 📉
=============================
Generates 25 regression plots to find relationships in the data.
Includes Linear, Polynomial, Lowess, Robust, and Logistic regressions.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "golden_dataset.parquet"
PLOTS_DIR = PROJECT_DIR / "plots" / "descriptive_stats"

# Add ML folder to path
sys.path.append(str(PROJECT_DIR))
from ml.visuals import set_style, save_plot, get_palette

def generate_report():
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Feature Engineering needed for Logistic
    # Create binary target for logistic regression plots
    df['is_winner'] = (df['wins_matches'] > df['wins_matches'].median()).astype(int)
    
    # Avoid division by zero for ratios
    df['kd_ratio_pvp'] = df['kills_pvp'] / df['deaths_pvp'].replace(0, 1)

    set_style()
    palette = get_palette()
    
    print(f"Generating 25 Regression Plots in {PLOTS_DIR}...")
    
    # --- The Pairs ---
    # We define 5 key relationship pairs
    pairs = [
        ('kills_pvp', 'wins_matches'),            # Slaying -> Winning?
        ('wools_objectives', 'wins_matches'),     # Objectives -> Winning?
        ('matches_matches', 'wins_matches'),      # Experience -> Winning?
        ('damage_dealt_pvp', 'kills_pvp'),        # Damage -> Kills (Likely Linear)
        ('kd_ratio_pvp', 'wins_matches')          # Efficiency -> Winning?
    ]
    
    # --- 1. Linear Regression (Standard) ---
    print("1. Standard Linear Regressions...")
    for i, (x, y) in enumerate(pairs):
        plt.figure(figsize=(12, 6))
        sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha':0.3, 'color': palette[0]}, line_kws={'color': 'red'})
        plt.title(f"Linear: {y} vs {x}")
        # Correlation in title
        corr = df[[x, y]].corr().iloc[0,1]
        plt.title(f"Linear: {y} vs {x} (R={corr:.2f})")
        
        save_plot(f"reg_01_linear_{i}_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 2. Polynomial Regression (Order 2) ---
    print("2. Polynomial Regressions...")
    for i, (x, y) in enumerate(pairs):
        plt.figure(figsize=(12, 6))
        sns.regplot(x=x, y=y, data=df, order=2, scatter_kws={'alpha':0.3, 'color': palette[1]}, line_kws={'color': 'black'})
        plt.title(f"Polynomial (Order 2): {y} vs {x}")
        save_plot(f"reg_02_poly_{i}_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()
        
    # --- 3. Robust Regression (Outlier Resistant) ---
    print("3. Robust Regressions...")
    for i, (x, y) in enumerate(pairs):
        plt.figure(figsize=(12, 6))
        # Robust uses statsmodels usually, sns supports it
        sns.regplot(x=x, y=y, data=df, robust=True, scatter_kws={'alpha':0.3, 'color': palette[2]}, line_kws={'color': 'green'})
        plt.title(f"Robust Regression: {y} vs {x}")
        save_plot(f"reg_03_robust_{i}_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 4. Lowess Smoothing (Non-Parametric) ---
    print("4. Lowess Smoothing...")
    for i, (x, y) in enumerate(pairs):
        plt.figure(figsize=(12, 6))
        # Sampled because Lowess is slow on 3k points
        sample = df.sample(min(len(df), 500)) 
        sns.regplot(x=x, y=y, data=sample, lowess=True, scatter_kws={'alpha':0.3, 'color': palette[3]}, line_kws={'color': 'blue'})
        plt.title(f"Lowess Trend: {y} vs {x}")
        save_plot(f"reg_04_lowess_{i}_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 5. Logistic Regression (Binary Outcome) ---
    print("5. Logistic Regressions (Probability of Winning > Median)...")
    # For logistic, Y must be binary. We use 'is_winner' vs X
    for i, (x, y) in enumerate(pairs):
        # We replace original Y with 'is_winner' and check if X predicts it
        target = 'is_winner'
        
        plt.figure(figsize=(12, 6))
        sns.regplot(x=x, y=target, data=df, logistic=True, y_jitter=0.03, 
                    scatter_kws={'alpha':0.1, 'color': palette[4]}, line_kws={'color': 'purple'})
        plt.title(f"Logistic: Prob(Winner) vs {x}")
        plt.ylabel("Probability of being a 'Winner' (Top 50%)")
        save_plot(f"reg_05_logistic_{i}_winner_{x}", folder=str(PLOTS_DIR))
        plt.close()

    print(f"\nDone! Generated 25 Regression Plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
