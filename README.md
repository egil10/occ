# Overcast Leaderboard Analysis

Simple pipeline: **Paste data → Parse → Analyze**

## Structure

```
occ/
├── scripts/           # Utility scripts
│   ├── parse.py      # Convert txt → json (with anonymization)
│   └── analyze.py    # Automated analysis
│
├── notebooks/         # Jupyter notebooks for exploration
│   └── 01_matches_descriptive_analysis.ipynb
│
├── ml/                # Machine learning experiments
│
├── data/
│   ├── raw/          # 👉 Paste data here (PRIVATE, git-ignored)
│   └── processed/    # Anonymized JSON/Parquet (PUBLIC)
│
└── plots/            # Generated visualizations
```

## Quick Start

### 1. Add Data
Paste leaderboard data into `data/raw/`:
- `matches.txt`
- `pvp.txt`
- `objectives.txt`

### 2. Parse (Anonymizes!)
```bash
python scripts/parse.py
```

### 3. Explore
```bash
jupyter notebook notebooks/01_matches_descriptive_analysis.ipynb
```

### 4. Run Full Analysis
```bash
python scripts/analyze.py
```

## Privacy

✅ **Anonymized**: Usernames → SHA256 hashed player IDs  
✅ **Git-Safe**: Raw data is private, processed is public  
✅ **Ethical**: Safe to share JSON files

## Install

```bash
pip install -r requirements.txt
```