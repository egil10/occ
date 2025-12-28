"""
Comprehensive Data Analysis Script
===================================
Performs descriptive statistics, correlation analysis, ML, and visualization
on Overcast leaderboard data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PLOTS_DIR = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class LeaderboardAnalyzer:
    """Comprehensive analyzer for Overcast leaderboard data."""
    
    def __init__(self, data_file: str = "data/processed/leaderboard_full.parquet"):
        """Load data."""
        self.data_path = Path(data_file)
        print("=" * 80)
        print("OVERCAST LEADERBOARD DATA ANALYSIS")
        print("=" * 80)
        print(f"Loading data from: {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        print(f"✓ Loaded {len(self.df):,} entries")
        print(f"✓ Categories: {', '.join(self.df['category'].unique())}")
        print("=" * 80)
        
        # Split by category
        self.matches_df = self.df[self.df['category'] == 'matches'].copy()
        self.pvp_df = self.df[self.df['category'] == 'pvp'].copy()
        self.objectives_df = self.df[self.df['category'] == 'objectives'].copy()
    
    def descriptive_stats(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)
        
        # Overall stats
        print(f"\nTotal Players: {len(self.df['username'].unique()):,}")
        print(f"Total Entries: {len(self.df):,}")
        print(f"\nEntries per category:")
        print(self.df['category'].value_counts())
        
        # Matches category
        if len(self.matches_df) > 0:
            print("\n" + "-" * 80)
            print("MATCHES CATEGORY")
            print("-" * 80)
            numeric_cols = self.matches_df.select_dtypes(include=[np.number]).columns
            stats_df = self.matches_df[numeric_cols].describe()
            print(stats_df)
            
            print("\nTop 5 Players by Total Matches:")
            top_matches = self.matches_df.nlargest(5, 'matches')[['username', 'matches', 'wins', 'losses', 'wl']]
            print(top_matches.to_string(index=False))
        
        # PVP category
        if len(self.pvp_df) > 0:
            print("\n" + "-" * 80)
            print("PVP CATEGORY")
            print("-" * 80)
            numeric_cols = self.pvp_df.select_dtypes(include=[np.number]).columns
            stats_df = self.pvp_df[numeric_cols].describe()
            print(stats_df)
            
            print("\nTop 5 Players by Kills:")
            top_pvp = self.pvp_df.nlargest(5, 'kills')[['username', 'kills', 'deaths', 'killstreak', 'damageDealt']]
            print(top_pvp.to_string(index=False))
        
        # Objectives category
        if len(self.objectives_df) > 0:
            print("\n" + "-" * 80)
            print("OBJECTIVES CATEGORY")
            print("-" * 80)
            numeric_cols = self.objectives_df.select_dtypes(include=[np.number]).columns
            stats_df = self.objectives_df[numeric_cols].describe()
            print(stats_df)
        
        return self.df.describe()
    
    def correlation_analysis(self):
        """Analyze correlations between variables."""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Matches correlations
        if len(self.matches_df) > 0:
            print("\nMATCHES - Top Correlations:")
            matches_numeric = self.matches_df.select_dtypes(include=[np.number])
            corr_matrix = matches_numeric.corr()
            
            # Find strong correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False, key=abs)
            print(corr_df.head(10).to_string(index=False))
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, linewidths=1)
            plt.title('Matches Category - Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'correlation_matches.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {PLOTS_DIR / 'correlation_matches.png'}")
            plt.close()
        
        # PVP correlations
        if len(self.pvp_df) > 0:
            print("\nPVP - Top Correlations:")
            pvp_numeric = self.pvp_df.select_dtypes(include=[np.number])
            corr_matrix = pvp_numeric.corr()
            
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False, key=abs)
            print(corr_df.head(10).to_string(index=False))
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, linewidths=1)
            plt.title('PVP Category - Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'correlation_pvp.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {PLOTS_DIR / 'correlation_pvp.png'}")
            plt.close()
    
    def scatter_plots(self):
        """Create comprehensive scatterplot analysis."""
        print("\n" + "=" * 80)
        print("SCATTERPLOT ANALYSIS")
        print("=" * 80)
        
        # Matches: Wins vs Losses
        if len(self.matches_df) > 0 and 'wins' in self.matches_df.columns and 'losses' in self.matches_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Matches Category - Relationship Analysis', fontsize=18, fontweight='bold')
            
            # 1. Wins vs Losses
            ax = axes[0, 0]
            ax.scatter(self.matches_df['losses'], self.matches_df['wins'], alpha=0.6, s=50)
            ax.set_xlabel('Losses', fontsize=12)
            ax.set_ylabel('Wins', fontsize=12)
            ax.set_title('Wins vs Losses', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trendline
            z = np.polyfit(self.matches_df['losses'].dropna(), self.matches_df['wins'].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(self.matches_df['losses'], p(self.matches_df['losses']), "r--", alpha=0.8, linewidth=2)
            
            # 2. Total Matches vs W/L Ratio
            if 'wl' in self.matches_df.columns:
                ax = axes[0, 1]
                ax.scatter(self.matches_df['matches'], self.matches_df['wl'], alpha=0.6, s=50, c='green')
                ax.set_xlabel('Total Matches', fontsize=12)
                ax.set_ylabel('W/L Ratio', fontsize=12)
                ax.set_title('Total Matches vs W/L Ratio', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # 3. Wins Distribution
            ax = axes[1, 0]
            ax.hist(self.matches_df['wins'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Wins', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Wins', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. W/L Ratio Distribution
            if 'wl' in self.matches_df.columns:
                ax = axes[1, 1]
                ax.hist(self.matches_df['wl'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax.set_xlabel('W/L Ratio', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Distribution of W/L Ratio', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'scatterplots_matches.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {PLOTS_DIR / 'scatterplots_matches.png'}")
            plt.close()
        
        # PVP: Kills vs Deaths, Arrows analysis
        if len(self.pvp_df) > 0 and 'kills' in self.pvp_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('PVP Category - Relationship Analysis', fontsize=18, fontweight='bold')
            
            # 1. Kills vs Deaths
            if 'deaths' in self.pvp_df.columns:
                ax = axes[0, 0]
                ax.scatter(self.pvp_df['deaths'], self.pvp_df['kills'], alpha=0.6, s=50, c='red')
                ax.set_xlabel('Deaths', fontsize=12)
                ax.set_ylabel('Kills', fontsize=12)
                ax.set_title('Kills vs Deaths', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Trendline
                valid_data = self.pvp_df[['deaths', 'kills']].dropna()
                if len(valid_data) > 0:
                    z = np.polyfit(valid_data['deaths'], valid_data['kills'], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data['deaths'], p(valid_data['deaths']), "r--", alpha=0.8, linewidth=2)
            
            # 2. Arrows Shot vs Arrows Hit
            if 'arrowsShot' in self.pvp_df.columns and 'arrowsHit' in self.pvp_df.columns:
                ax = axes[0, 1]
                ax.scatter(self.pvp_df['arrowsShot'], self.pvp_df['arrowsHit'], alpha=0.6, s=50, c='purple')
                ax.set_xlabel('Arrows Shot', fontsize=12)
                ax.set_ylabel('Arrows Hit', fontsize=12)
                ax.set_title('Arrows Shot vs Arrows Hit', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Calculate accuracy
                self.pvp_df['arrow_accuracy'] = (self.pvp_df['arrowsHit'] / self.pvp_df['arrowsShot'] * 100).fillna(0)
            
            # 3. Kills Distribution
            ax = axes[1, 0]
            ax.hist(self.pvp_df['kills'].dropna(), bins=50, alpha=0.7, color='crimson', edgecolor='black')
            ax.set_xlabel('Kills', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Kills', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. K/D Ratio
            if 'deaths' in self.pvp_df.columns:
                self.pvp_df['kd_ratio'] = (self.pvp_df['kills'] / self.pvp_df['deaths']).replace([np.inf, -np.inf], np.nan)
                ax = axes[1, 1]
                kd_data = self.pvp_df['kd_ratio'].dropna()
                kd_data = kd_data[kd_data < 10]  # Remove outliers
                ax.hist(kd_data, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
                ax.set_xlabel('K/D Ratio', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Distribution of K/D Ratio', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'scatterplots_pvp.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {PLOTS_DIR / 'scatterplots_pvp.png'}")
            plt.close()
    
    def ml_analysis(self):
        """Machine learning analysis."""
        print("\n" + "=" * 80)
        print("MACHINE LEARNING ANALYSIS")
        print("=" * 80)
        
        # Predict W/L ratio based on other features (Matches)
        if len(self.matches_df) > 0 and 'wl' in self.matches_df.columns:
            print("\n--- Predicting W/L Ratio (Matches Category) ---")
            
            feature_cols = ['matches', 'wins', 'losses']
            data = self.matches_df[feature_cols + ['wl']].dropna()
            
            if len(data) > 10:
                X = data[feature_cols]
                y = data['wl']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                lr_r2 = r2_score(y_test, lr_pred)
                lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
                
                print(f"\nLinear Regression:")
                print(f"  R² Score: {lr_r2:.4f}")
                print(f"  RMSE: {lr_rmse:.4f}")
                print(f"  Coefficients: {dict(zip(feature_cols, lr.coef_))}")
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_r2 = r2_score(y_test, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
                
                print(f"\nRandom Forest:")
                print(f"  R² Score: {rf_r2:.4f}")
                print(f"  RMSE: {rf_rmse:.4f}")
                print(f"  Feature Importances: {dict(zip(feature_cols, rf.feature_importances_))}")
                
                # Plot predictions
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                axes[0].scatter(y_test, lr_pred, alpha=0.6)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0].set_xlabel('Actual W/L Ratio', fontsize=12)
                axes[0].set_ylabel('Predicted W/L Ratio', fontsize=12)
                axes[0].set_title(f'Linear Regression (R²={lr_r2:.3f})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].scatter(y_test, rf_pred, alpha=0.6, c='green')
                axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[1].set_xlabel('Actual W/L Ratio', fontsize=12)
                axes[1].set_ylabel('Predicted W/L Ratio', fontsize=12)
                axes[1].set_title(f'Random Forest (R²={rf_r2:.3f})', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / 'ml_wl_prediction.png', dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {PLOTS_DIR / 'ml_wl_prediction.png'}")
                plt.close()
        
        # Predict Kills based on other features (PVP)
        if len(self.pvp_df) > 0 and 'kills' in self.pvp_df.columns:
            print("\n--- Predicting Kills (PVP Category) ---")
            
            feature_cols = ['deaths', 'arrowsShot', 'arrowsHit', 'damageDealt']
            available_cols = [col for col in feature_cols if col in self.pvp_df.columns]
            
            if len(available_cols) > 0:
                data = self.pvp_df[available_cols + ['kills']].dropna()
                
                if len(data) > 10:
                    X = data[available_cols]
                    y = data['kills']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Random Forest
                    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf.fit(X_train, y_train)
                    rf_pred = rf.predict(X_test)
                    rf_r2 = r2_score(y_test, rf_pred)
                    
                    print(f"\nRandom Forest:")
                    print(f"  R² Score: {rf_r2:.4f}")
                    print(f"  Feature Importances: {dict(zip(available_cols, rf.feature_importances_))}")
    
    def clustering_analysis(self):
        """Perform clustering analysis."""
        print("\n" + "=" * 80)
        print("CLUSTERING ANALYSIS")
        print("=" * 80)
        
        # K-Means on PVP data
        if len(self.pvp_df) > 0 and 'kills' in self.pvp_df.columns and 'deaths' in self.pvp_df.columns:
            print("\nPerforming K-Means clustering on PVP data...")
            
            feature_cols = ['kills', 'deaths']
            if 'arrowsHit' in self.pvp_df.columns:
                feature_cols.append('arrowsHit')
            
            data = self.pvp_df[feature_cols].dropna()
            
            if len(data) > 10:
                # Standardize
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                # K-Means
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(data_scaled)
                
                # Plot
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(data['kills'], data['deaths'], c=clusters, cmap='viridis', 
                                    alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
                plt.xlabel('Kills', fontsize=12)
                plt.ylabel('Deaths', fontsize=12)
                plt.title('Player Clustering (K-Means, k=3)', fontsize=16, fontweight='bold')
                plt.colorbar(scatter, label='Cluster')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / 'clustering_pvp.png', dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {PLOTS_DIR / 'clustering_pvp.png'}")
                plt.close()
                
                # Print cluster stats
                self.pvp_df['cluster'] = np.nan
                self.pvp_df.loc[data.index, 'cluster'] = clusters
                
                print("\nCluster Statistics:")
                for i in range(3):
                    cluster_data = self.pvp_df[self.pvp_df['cluster'] == i]
                    print(f"\nCluster {i} (n={len(cluster_data)}):")
                    print(f"  Avg Kills: {cluster_data['kills'].mean():.0f}")
                    print(f"  Avg Deaths: {cluster_data['deaths'].mean():.0f}")
                    if 'kd_ratio' in cluster_data.columns:
                        kd_clean = cluster_data['kd_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(kd_clean) > 0:
                            print(f"  Avg K/D: {kd_clean.mean():.2f}")
    
    def run_full_analysis(self):
        """Run complete analysis suite."""
        self.descriptive_stats()
        self.correlation_analysis()
        self.scatter_plots()
        self.ml_analysis()
        self.clustering_analysis()
        
        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Plots saved to: {PLOTS_DIR}")
        print("=" * 80)


def main():
    """Main execution."""
    analyzer = LeaderboardAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
