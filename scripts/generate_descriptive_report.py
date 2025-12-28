"""
The Ultimate Descriptive Report 📊
==================================
Generates 50 plots using 25 DIFFERENT visualization types.
Focuses on variety, depth, and aesthetics.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from math import pi
from pandas.plotting import parallel_coordinates, andrews_curves
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "golden_dataset.parquet"
PLOTS_DIR = PROJECT_DIR / "plots" / "descriptive_stats"

# Add ML folder to path
sys.path.append(str(PROJECT_DIR))
from ml.visuals import set_style, save_plot, get_palette

# --- Helper: Synthetic Feature Creation ---
def create_synthetic_features(df):
    df = df.copy()
    # Feature Engineering
    df['kd_ratio_pvp'] = df['kills_pvp'] / df['deaths_pvp'].replace(0, 1)
    if 'arrows_hit_pvp' in df.columns:
        df['accuracy_pvp'] = df['arrows_hit_pvp'] / df['arrows_shot_pvp'].replace(0, 1)
    else:
        df['accuracy_pvp'] = 0.0

    # Winner Tier
    df['Winner Tier'] = pd.qcut(df['wins_matches'], q=[0, 0.5, 0.75, 0.9, 1.0], 
                                labels=['Casual', 'Competitor', 'Pro', 'Elite'])
    
    # PvP Class
    kd_med = df['kd_ratio_pvp'].median()
    k_med = df['kills_pvp'].median()
    conds = [
        (df['kills_pvp'] > k_med) & (df['kd_ratio_pvp'] > kd_med),
        (df['kills_pvp'] > k_med),
        (df['deaths_pvp'] > df['deaths_pvp'].median()),
    ]
    choices = ['Slayer', 'Berserker', 'Feeder']
    df['PvP Class'] = np.select(conds, choices, default='Novice')

    # Playstyle
    obj_q = df['wools_objectives'].quantile(0.8)
    pvp_q = df['kills_pvp'].quantile(0.8)
    df['Playstyle'] = np.where((df['wools_objectives'] > obj_q) & (df['kills_pvp'] > pvp_q), 'Hybrid God',
                      np.where(df['wools_objectives'] > obj_q, 'Objective Main',
                      np.where(df['kills_pvp'] > pvp_q, 'PvP Main', 'Generalist')))
    return df

# --- Helper: Radar Chart ---
def plot_radar(df, categories, group_col, title, filename):
    # Aggregate
    data = df.groupby(group_col)[categories].mean()
    # Normalize min-max
    data = (data - data.min()) / (data.max() - data.min())
    
    # Setup
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    palette = get_palette()
    for i, (idx, row) in enumerate(data.iterrows()):
        values = row.tolist()
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=idx, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.xticks(angles[:-1], [c.replace('_', '\n') for c in categories])
    plt.title(title, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    save_plot(filename, folder=str(PLOTS_DIR))
    plt.close()

# --- MAIN GENERATOR ---
def generate_report():
    print("Loading Data...")
    raw_df = pd.read_parquet(DATA_PATH)
    df = create_synthetic_features(raw_df)
    
    # Filter numeric for some plots, but keep cats for hue
    set_style()
    palette = get_palette()
    
    print(f"Generating 50 Plots (25 Types x 2) in {PLOTS_DIR}...")
    
    # Define Metrics
    met_dist = ['wins_matches', 'kills_pvp']
    met_cat = ['Winner Tier', 'Playstyle']
    met_cat_num = [('Winner Tier', 'kills_pvp'), ('Playstyle', 'wools_objectives')]
    met_bi = [('matches_matches', 'wins_matches'), ('kills_pvp', 'deaths_pvp')]
    
    # 1. Histogram
    for m in met_dist:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[m], kde=False, color=palette[0])
        plt.title(f"1. Histogram: {m}")
        save_plot(f"01_hist_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # 2. KDE Plot
    for m in met_dist:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df[m], fill=True, color=palette[1])
        plt.title(f"2. KDE: {m}")
        save_plot(f"02_kde_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # 3. ECDF Plot
    for m in met_dist:
        plt.figure(figsize=(12, 6))
        sns.ecdfplot(data=df, x=m, hue='Winner Tier', palette=palette)
        plt.title(f"3. ECDF: {m} by Tier")
        save_plot(f"03_ecdf_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # 4. Rug Plot
    for m in met_dist:
        plt.figure(figsize=(12, 2)) # Wide and short
        sns.rugplot(data=df.sample(500), x=m, height=1, color=palette[2]) # Sampled for clarity
        plt.title(f"4. Rug Plot: {m} (Sample 500)")
        plt.yticks([])
        save_plot(f"04_rug_{m}", folder=str(PLOTS_DIR))
        plt.close()

    # 5. Box Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=cat, y=num, data=df, palette=palette)
        plt.title(f"5. Box Plot: {num} by {cat}")
        save_plot(f"05_box_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 6. Violin Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=cat, y=num, data=df, palette=palette, inner="stick")
        plt.title(f"6. Violin Plot: {num} by {cat}")
        save_plot(f"06_violin_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 7. Strip Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.stripplot(x=cat, y=num, data=df.sample(1000), alpha=0.5, jitter=True, palette=palette)
        plt.title(f"7. Strip Plot: {num} by {cat}")
        save_plot(f"07_strip_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 8. Swarm Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        # Heavy downsample for Swarm to prevent crash
        sns.swarmplot(x=cat, y=num, data=df.sample(300), palette=palette, size=4) 
        plt.title(f"8. Swarm: {num} by {cat} (Sample 300)")
        save_plot(f"08_swarm_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 9. Boxen Plot (Letter Value)
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.boxenplot(x=cat, y=num, data=df, palette=palette)
        plt.title(f"9. Boxen Plot: {num} by {cat}")
        save_plot(f"09_boxen_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 10. Point Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.pointplot(x=cat, y=num, data=df, capsize=.2, color=palette[0])
        plt.title(f"10. Point Mean: {num} by {cat}")
        save_plot(f"10_point_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 11. Bar Plot
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=cat, y=num, data=df, estimator=np.median, palette=palette)
        plt.title(f"11. Bar (Median): {num} by {cat}")
        save_plot(f"11_bar_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 12. Count Plot
    for cat in met_cat:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=cat, data=df, palette=palette)
        plt.title(f"12. Count Plot: {cat}")
        save_plot(f"12_count_{cat}", folder=str(PLOTS_DIR))
        plt.close()
    
    # 13. Lollipop Chart
    for cat, num in met_cat_num:
        plt.figure(figsize=(12, 6))
        agg = df.groupby(cat)[num].mean()
        plt.stem(agg.index, agg.values)
        plt.title(f"13. Lollipop: Avg {num} by {cat}")
        save_plot(f"13_lollipop_{num}", folder=str(PLOTS_DIR))
        plt.close()

    # 14. Scatter Plot
    for x, y in met_bi:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=x, y=y, data=df, alpha=0.5, color=palette[3])
        plt.title(f"14. Scatter: {y} vs {x}")
        save_plot(f"14_scatter_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 15. Reg Plot
    for x, y in met_bi:
        plt.figure(figsize=(12, 6))
        sns.regplot(x=x, y=y, data=df.sample(1000), scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f"15. RegPlot: {y} vs {x}")
        save_plot(f"15_reg_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 16. Residual Plot
    for x, y in met_bi:
        plt.figure(figsize=(12, 6))
        sns.residplot(x=x, y=y, data=df.sample(1000), lowess=True, color=palette[2])
        plt.title(f"16. Residuals: {y} vs {x}")
        save_plot(f"16_resid_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 17. Hexbin Plot
    for x, y in met_bi:
        plt.figure(figsize=(10, 8))
        plt.hexbin(df[x], df[y], gridsize=20, cmap='Blues')
        plt.colorbar(label='Count')
        plt.title(f"17. Hexbin: {y} vs {x}")
        save_plot(f"17_hexbin_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 18. Joint Plot
    for x, y in met_bi:
        # JointGrid manages its own figure
        g = sns.jointplot(x=x, y=y, data=df.sample(2000), kind="kde", color=palette[0])
        g.fig.suptitle(f"18. Joint KDE: {y} vs {x}", y=1.02)
        save_plot(f"18_joint_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 19. 2D KDE Contour
    for x, y in met_bi:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(x=x, y=y, data=df.sample(2000), fill=True, cmap="mako")
        plt.title(f"19. 2D Contour: {y} vs {x}")
        save_plot(f"19_contour_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 20. Bubble Plot (Scatter + Size)
    for x, y in met_bi:
        plt.figure(figsize=(12, 6))
        # Size by 'matches_matches' or 'kd_ratio'
        size_var = 'matches_matches' if x != 'matches_matches' else 'kd_ratio_pvp'
        sns.scatterplot(x=x, y=y, size=size_var, data=df.sample(500), sizes=(20, 200), alpha=0.5, hue='Winner Tier')
        plt.title(f"20. Bubble: {y} vs {x} (Size={size_var})")
        save_plot(f"20_bubble_{y}_{x}", folder=str(PLOTS_DIR))
        plt.close()

    # 21. Heatmap (Correlation)
    plt.figure(figsize=(12, 10))
    cols = ['matches_matches', 'wins_matches', 'kills_pvp', 'deaths_pvp', 'kd_ratio_pvp', 'wools_objectives']
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("21. Correlation Heatmap")
    save_plot("21_heatmap_1", folder=str(PLOTS_DIR))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    cols2 = ['accuracy_pvp', 'damage_dealt_pvp', 'arrows_hit_pvp', 'cores_objectives', 'monuments_objectives']
    # Filter only useful ones
    cols2_valid = [c for c in cols2 if c in df.columns]
    if len(cols2_valid) > 1:
        sns.heatmap(df[cols2_valid].corr(), annot=True, cmap='viridis')
        plt.title("21. Correlation Heatmap (Secondary)")
        save_plot("21_heatmap_2", folder=str(PLOTS_DIR))
        plt.close()

    # 22. Parallel Coordinates
    plt.figure(figsize=(12, 6))
    subset = df[cols + ['Winner Tier']].sample(100)
    # Normalize for displaying together
    for c in cols:
        subset[c] = (subset[c] - subset[c].min()) / (subset[c].max() - subset[c].min())
    parallel_coordinates(subset, 'Winner Tier', color=palette, alpha=0.5)
    plt.title("22. Parallel Coordinates (Normalized, Sample 100)")
    save_plot("22_parallel_coords_1", folder=str(PLOTS_DIR))
    plt.close()
    
    # 2nd Example
    plt.figure(figsize=(12, 6))
    subset2 = df[['kills_pvp', 'wools_objectives', 'wins_matches', 'Playstyle']].sample(100)
    for c in subset2.columns[:-1]:
        subset2[c] = (subset2[c] - subset2[c].min()) / (subset2[c].max() - subset2[c].min())
    parallel_coordinates(subset2, 'Playstyle', color=palette, alpha=0.5)
    plt.title("22. Parallel Coordinates (Playstyle)")
    save_plot("22_parallel_coords_2", folder=str(PLOTS_DIR))
    plt.close()

    # 23. Andrews Curves
    plt.figure(figsize=(12, 6))
    andrews_curves(subset, 'Winner Tier', color=palette, alpha=0.5)
    plt.title("23. Andrews Curves (Sample 100)")
    save_plot("23_andrews_1", folder=str(PLOTS_DIR))
    plt.close()
    
    # 2nd Example
    plt.figure(figsize=(12, 6))
    andrews_curves(subset2, 'Playstyle', color=palette, alpha=0.5)
    plt.title("23. Andrews Curves (Playstyle)")
    save_plot("23_andrews_2", folder=str(PLOTS_DIR))
    plt.close()
    
    # 24. Radar Chart
    radar_cols = ['wins_matches', 'kills_pvp', 'kd_ratio_pvp', 'wools_objectives', 'matches_matches']
    plot_radar(df, radar_cols, 'Winner Tier', "24. Radar: Winner Archetypes", "24_radar_winner")
    plot_radar(df, radar_cols, 'Playstyle', "24. Radar: Playstyle Archetypes", "24_radar_playstyle")

    # 25. Diverging Bars (Z-Score Deviation for 'Elite' vs Mean)
    # Compare Elite means to Global means
    global_mean = df[cols].mean()
    elite_mean = df[df['Winner Tier'] == 'Elite'][cols].mean()
    z_scores = (elite_mean - global_mean) / df[cols].std()
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if x < 0 else 'green' for x in z_scores]
    plt.hlines(y=z_scores.index, xmin=0, xmax=z_scores, color=colors)
    plt.plot(z_scores, z_scores.index, "o")
    plt.axvline(0, color='black', linestyle='--')
    plt.title("25. Diverging Bars: Elite Tier Deviation (Z-Score)")
    save_plot("25_diverging_elite", folder=str(PLOTS_DIR))
    plt.close()
    
    # 2nd Example (Playstyle 'Hybrid God')
    hybrid_mean = df[df['Playstyle'] == 'Hybrid God'][cols].mean()
    z_scores_h = (hybrid_mean - global_mean) / df[cols].std()
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if x < 0 else 'green' for x in z_scores_h]
    plt.hlines(y=z_scores_h.index, xmin=0, xmax=z_scores_h, color=colors)
    plt.plot(z_scores_h, z_scores_h.index, "o")
    plt.axvline(0, color='black', linestyle='--')
    plt.title("25. Diverging Bars: Hybrid God Deviation (Z-Score)")
    save_plot("25_diverging_hybrid", folder=str(PLOTS_DIR))
    plt.close()

    print(f"\nDone! Generated 50+ Plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
