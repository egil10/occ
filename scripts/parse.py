"""
Anonymous Data Parser
=====================
Converts leaderboard data to JSON with anonymized player IDs
"""

import json
import pandas as pd
from pathlib import Path
import hashlib

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


def anonymize_username(username: str) -> str:
    """Create anonymous player ID from username."""
    # Create SHA256 hash of username
    hash_obj = hashlib.sha256(username.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    # Use first 12 characters as player ID
    return f"player_{hash_hex[:12]}"


def parse_matches(filepath):
    """Parse matches leaderboard with anonymization."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 6 lines)
    data_lines = lines[6:]
    players = []
    
    i = 0
    while i < len(data_lines):
        try:
            if i + 6 >= len(data_lines):
                break
            
            username = data_lines[i]
            player_id = anonymize_username(username)
            
            player = {
                'category': 'matches',
                'player_id': player_id,  # Anonymized!
                'position': data_lines[i + 1],
                'matches': int(data_lines[i + 2]),
                'wins': int(data_lines[i + 3]),
                'losses': int(data_lines[i + 4]),
                'wl_ratio': float(data_lines[i + 5]),
                'ties': int(data_lines[i + 6])
            }
            players.append(player)
            i += 7
        except (ValueError, IndexError):
            i += 1
    
    return players


def parse_pvp(filepath):
    """Parse PVP leaderboard with anonymization."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 8 lines for PVP - Username, Position, Kills, Deaths, Killstreak, Damage, Arrows Shot, Arrows Hit)
    data_lines = lines[8:]
    players = []
    
    i = 0
    while i < len(data_lines):
        try:
            if i + 7 >= len(data_lines):
                break
            
            username = data_lines[i]
            player_id = anonymize_username(username)
            
            # Clean damage dealt (remove heart symbol)
            damage_str = data_lines[i + 5].replace('♥', '').replace('â™¥', '').strip()
            
            player = {
                'category': 'pvp',
                'player_id': player_id,  # Anonymized!
                'position': data_lines[i + 1],
                'kills': int(data_lines[i + 2]),
                'deaths': int(data_lines[i + 3]),
                'killstreak': int(data_lines[i + 4]),
                'damage_dealt': float(damage_str),
                'arrows_shot': int(data_lines[i + 6]),
                'arrows_hit': int(data_lines[i + 7])
            }
            players.append(player)
            i += 8
        except (ValueError, IndexError) as e:
            i += 1
    
    return players


def parse_objectives(filepath):
    """Parse objectives leaderboard with anonymization."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 10 lines)
    data_lines = lines[10:]
    players = []
    
    i = 0
    while i < len(data_lines):
        try:
            if i + 7 >= len(data_lines):
                break
            
            username = data_lines[i]
            player_id = anonymize_username(username)
            
            player = {
                'category': 'objectives',
                'player_id': player_id,  # Anonymized!
                'position': data_lines[i + 1],
                'wools': int(data_lines[i + 2]),
                'wools_touched': int(data_lines[i + 3]),
                'flags': int(data_lines[i + 4]),
                'flags_picked': int(data_lines[i + 5]),
                'cores': int(data_lines[i + 6]),
                'monuments': int(data_lines[i + 7])
            }
            players.append(player)
            i += 8
        except (ValueError, IndexError):
            i += 1
    
    return players


def main():
    """Parse all data files with anonymization."""
    print("\n" + "="*80)
    print("PARSING LEADERBOARD DATA (WITH ANONYMIZATION)")
    print("="*80 + "\n")
    
    all_data = []
    stats = {}
    
    # Parse each category
    for name, parser in [('matches', parse_matches), ('pvp', parse_pvp), ('objectives', parse_objectives)]:
        filepath = RAW_DIR / f"{name}.txt"
        
        if filepath.exists():
            print(f"[{name.upper()}] Parsing...")
            try:
                data = parser(filepath)
                all_data.extend(data)
                stats[name] = len(data)
                print(f"  ✓ {len(data):,} players (anonymized)\n")
                
                # Save individual JSON
                with open(OUTPUT_DIR / f"{name}.json", 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                stats[name] = 0
        else:
            print(f"[{name.upper()}] File not found: {filepath}\n")
            stats[name] = 0
    
    # Save combined files
    if all_data:
        # JSON
        with open(OUTPUT_DIR / 'leaderboard_full.json', 'w') as f:
            json.dump(all_data, f, indent=2)
        
        # Parquet
        df = pd.DataFrame(all_data)
        df.to_parquet(OUTPUT_DIR / 'leaderboard_full.parquet', index=False)
        
        print("="*80)
        print("✓ PARSING COMPLETE!")
        print("="*80)
        print(f"Total entries: {len(all_data):,}")
        for cat, count in stats.items():
            if count > 0:
                print(f"  {cat}: {count:,}")
        
        print(f"\n🔒 ANONYMIZATION:")
        print(f"  All usernames replaced with player_XXXX IDs")
        print(f"  Based on SHA256 hash (irreversible)")
        
        print(f"\n💾 SAVED TO: {OUTPUT_DIR}/")
        print("  - leaderboard_full.json (SAFE TO PUBLISH)")
        print("  - leaderboard_full.parquet (SAFE TO PUBLISH)")
        print("  - matches.json (SAFE TO PUBLISH)")
        print("  - pvp.json (SAFE TO PUBLISH)")
        print("  - objectives.json (SAFE TO PUBLISH)")
        print("\n🔐 RAW DATA (PRIVATE):")
        print(f"  - {RAW_DIR}/*.txt (GIT-IGNORED)")
        print("="*80 + "\n")
    else:
        print("="*80)
        print("✗ NO DATA FOUND")
        print("="*80)
        print("Please paste data into:")
        print(f"  - {RAW_DIR}/matches.txt")
        print(f"  - {RAW_DIR}/pvp.txt")
        print(f"  - {RAW_DIR}/objectives.txt")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
