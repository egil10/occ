"""
Username Extractor
==================
Extract unique usernames from leaderboard data for text analysis
"""

from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def extract_usernames_from_matches(filepath):
    """Extract usernames from matches leaderboard."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 6 lines)
    data_lines = lines[6:]
    usernames = []
    
    i = 0
    while i < len(data_lines):
        if i + 6 >= len(data_lines):
            break
        usernames.append(data_lines[i])
        i += 7
    
    return usernames


def extract_usernames_from_pvp(filepath):
    """Extract usernames from PVP leaderboard."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 8 lines)
    data_lines = lines[8:]
    usernames = []
    
    i = 0
    while i < len(data_lines):
        if i + 7 >= len(data_lines):
            break
        usernames.append(data_lines[i])
        i += 8
    
    return usernames


def extract_usernames_from_objectives(filepath):
    """Extract usernames from objectives leaderboard."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip headers (first 10 lines)
    data_lines = lines[10:]
    usernames = []
    
    i = 0
    while i < len(data_lines):
        if i + 7 >= len(data_lines):
            break
        usernames.append(data_lines[i])
        i += 8
    
    return usernames


def main():
    """Extract all unique usernames."""
    print("\n" + "="*80)
    print("EXTRACTING USERNAMES FOR TEXT ANALYSIS")
    print("="*80 + "\n")
    
    all_usernames = set()
    
    # Extract from each file
    for name, extractor in [
        ('matches', extract_usernames_from_matches),
        ('pvp', extract_usernames_from_pvp),
        ('objectives', extract_usernames_from_objectives)
    ]:
        filepath = RAW_DIR / f"{name}.txt"
        
        if filepath.exists():
            print(f"[{name.upper()}] Extracting usernames...")
            try:
                usernames = extractor(filepath)
                all_usernames.update(usernames)
                print(f"  ✓ Found {len(usernames):,} usernames\n")
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
        else:
            print(f"[{name.upper()}] File not found: {filepath}\n")
    
    # Save unique usernames
    if all_usernames:
        output_file = RAW_DIR / "usernames.txt"
        
        # Sort for consistency
        sorted_usernames = sorted(all_usernames)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted_usernames))
        
        print("="*80)
        print("✓ EXTRACTION COMPLETE!")
        print("="*80)
        print(f"Total unique usernames: {len(all_usernames):,}")
        print(f"\n💾 SAVED TO: {output_file}")
        print("🔐 STATUS: GIT-IGNORED (PRIVATE)")
        print("\n📊 Ready for text analysis!")
        print("="*80 + "\n")
    else:
        print("="*80)
        print("✗ NO USERNAMES FOUND")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
