"""
Golden Dataset Creator 🌟
=======================
Creates a merged dataset containing ONLY players who appear in 
ALL three leaderboards (Matches, PvP, Objectives).

This is the high-quality dataset for ML modeling.
"""

import pandas as pd
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_FILE = PROCESSED_DIR / "golden_dataset.parquet"
OUTPUT_CSV = PROCESSED_DIR / "golden_dataset.csv"

def create_golden_dataset():
    print("\n" + "="*80)
    print("CREATING GOLDEN DATASET (INTERSECTION OF ALL DATA)")
    print("="*80 + "\n")
    
    # Load full data
    df = pd.read_parquet(PROCESSED_DIR / 'leaderboard_full.parquet')
    print(f"Total records loaded: {len(df):,}")
    
    # Split by category
    matches = df[df['category'] == 'matches'].add_suffix('_matches')
    pvp = df[df['category'] == 'pvp'].add_suffix('_pvp')
    objectives = df[df['category'] == 'objectives'].add_suffix('_objectives')
    
    # Rename player_id columns back to just 'player_id' for merging
    matches = matches.rename(columns={'player_id_matches': 'player_id'}).drop('category_matches', axis=1)
    pvp = pvp.rename(columns={'player_id_pvp': 'player_id'}).drop('category_pvp', axis=1)
    objectives = objectives.rename(columns={'player_id_objectives': 'player_id'}).drop('category_objectives', axis=1)
    
    # Merge Inner Join (Intersection)
    # 1. Matches + PvP
    merged_1 = pd.merge(matches, pvp, on='player_id', how='inner')
    print(f"  Players in Matches + PvP: {len(merged_1):,}")
    
    # 2. + Objectives
    golden_df = pd.merge(merged_1, objectives, on='player_id', how='inner')
    
    print("\n" + "="*80)
    print("✓ GOLDEN DATASET CREATED!")
    print("="*80)
    print(f"Total Unique Players: {len(golden_df):,}")
    print(f"Total Features: {len(golden_df.columns)}")
    print("\nFeature Columns:")
    for col in golden_df.columns:
        if col != 'player_id':
            print(f"  - {col}")
            
    # Save
    golden_df.to_parquet(OUTPUT_FILE, index=False)
    golden_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n💾 SAVED TO:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {OUTPUT_CSV}")
    print("="*80 + "\n")

if __name__ == "__main__":
    create_golden_dataset()
