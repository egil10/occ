"""
Tree Analysis Report 🌳
=======================
Generates 25 plots using Decision Trees and Random Forests.
Focuses on Feature Importance, Decision Boundaries, and Leakage Detection.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
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
    
    # 5 Binary Targets
    targets = {
        'is_elite_winner': df['wins_matches'] > df['wins_matches'].quantile(0.90),
        'is_slayer': df['kills_pvp'] > df['kills_pvp'].quantile(0.90),
        'is_obj_god': df['wools_objectives'] > df['wools_objectives'].quantile(0.90),
        'is_veteran': df['matches_matches'] > df['matches_matches'].quantile(0.90),
        'is_efficient': df['kd_ratio_pvp'] > df['kd_ratio_pvp'].quantile(0.90)
    }

    set_style()
    palette = get_palette()
    
    print(f"Generating 25 Tree Analysis Plots in {PLOTS_DIR}...")
    
    # --- 1. Single Decision Tree Structure (Depth 3) ---
    print("1. Decision Tree Structures (Leakage Checks)...")
    for name, y in targets.items():
        # X: Use numeric columns, fill NaN
        X = df.select_dtypes(include=[np.number]).fillna(0)
        # Drop ID
        if 'player_id' in X.columns: X = X.drop(columns=['player_id'])
        
        # Intentionally KEEP potental leaks to find them!
        # But drop the target source itself implies triviality, but let's see correlations
        # Actually, for "is_elite_winner" (wins), we must drop 'wins_matches' or it's trivial.
        # But we KEEP related things like 'matches' to see if that leaks.
        
        clf = DecisionTreeClassifier(max_depth=3, random_state=808)
        clf.fit(X, y)
        
        plt.figure(figsize=(16, 8))
        plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], 
                  filled=True, rounded=True, fontsize=10)
        plt.title(f"Decision Tree Path: Predicting {name}")
        save_plot(f"tree_01_structure_{name}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 2. Feature Importance (Random Forest) ---
    print("2. RF Feature Importance...")
    for name, y in targets.items():
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if 'player_id' in X.columns: X = X.drop(columns=['player_id'])
        
        # Drop likely target source for rigorous importance
        # e.g. for is_slayer, drop kills
        if 'kills' in name and 'kills_pvp' in X.columns: X = X.drop(columns=['kills_pvp'])
        if 'winner' in name and 'wins_matches' in X.columns: X = X.drop(columns=['wins_matches'])
        if 'obj' in name and 'wools_objectives' in X.columns: X = X.drop(columns=['wools_objectives'])
        if 'veteran' in name and 'matches_matches' in X.columns: X = X.drop(columns=['matches_matches'])
        
        rf = RandomForestClassifier(n_estimators=50, random_state=808)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances.values, y=importances.index, palette=palette)
        plt.title(f"Top 10 Predictors for {name} (RF)")
        plt.xlabel("Gini Importance")
        save_plot(f"tree_02_importance_{name}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 3. Partial Dependence Plots (Top Feature) ---
    print("3. Partial Dependence Plots...")
    for i, (name, y) in enumerate(targets.items()):
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if 'player_id' in X.columns: X = X.drop(columns=['player_id'])
        
        # Train simple model
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=808).fit(X, y)
        
        # Get top feature
        top_feat = pd.Series(rf.feature_importances_, index=X.columns).idxmax()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        PartialDependenceDisplay.from_estimator(rf, X, [top_feat], ax=ax, line_kw={"color": palette[0], "linewidth": 3})
        plt.title(f"PDP: Impact of {top_feat} on {name}")
        save_plot(f"tree_03_pdp_{name}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 4. 2D Decision Boundaries (Top 2 Features) ---
    print("4. 2D Decision Boundaries...")
    for name, y in targets.items():
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if 'player_id' in X.columns: X = X.drop(columns=['player_id'])
        
        # Find top 2 features to plot
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=808).fit(X, y)
        top_2 = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(2).index.tolist()
        
        # Retrain on just 2 features for 2D plotting
        X_2d = X[top_2]
        clf_2d = DecisionTreeClassifier(max_depth=5).fit(X_2d, y)
        
        plt.figure(figsize=(12, 6))
        
        # Create mesh
        x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
        y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                             np.arange(y_min, y_max, (y_max-y_min)/100))
        
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        sns.scatterplot(x=X_2d.iloc[:, 0], y=X_2d.iloc[:, 1], hue=y, palette={False:'grey', True:'red'}, alpha=0.6)
        
        plt.title(f"Decision Boundary: {name} ({top_2[0]} vs {top_2[1]})")
        save_plot(f"tree_04_boundary_{name}", folder=str(PLOTS_DIR))
        plt.close()

    # --- 5. Tree Depth vs Accuracy (Overfitting Check) ---
    print("5. Complexity Curves (Depth vs Accuracy)...")
    for name, y in targets.items():
        X = df.select_dtypes(include=[np.number]).fillna(0)
        if 'player_id' in X.columns: X = X.drop(columns=['player_id'])
        
        train_ws = []
        depths = range(1, 15)
        
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, random_state=808)
            clf.fit(X, y)
            train_ws.append(clf.score(X, y))
            
        plt.figure(figsize=(12, 6))
        plt.plot(depths, train_ws, marker='o', color=palette[1], linewidth=2)
        plt.title(f"Model Complexity: {name}")
        plt.xlabel("Tree Max Depth")
        plt.ylabel("Training Accuracy")
        plt.grid(True, linestyle='--')
        save_plot(f"tree_05_complexity_{name}", folder=str(PLOTS_DIR))
        plt.close()

    print(f"\nDone! Generated 25 Tree Analysis Plots in {PLOTS_DIR}")

if __name__ == "__main__":
    generate_report()
