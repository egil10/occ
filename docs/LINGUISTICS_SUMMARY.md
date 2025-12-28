# Username Linguistics Analysis - Results Summary 🎮

**Date:** 2025-12-28  
**Dataset:** 5,026 unique player usernames from OCC leaderboards

---

## 📊 Key Statistics

- **Total unique usernames:** 5,026
- **Average length:** 8.3 characters
- **Length range:** Variable (see plots)
- **Digit usage:** 5.5% of all characters
- **Names with numbers:** 1,009 (20.1%)

---

## 🔍 Analysis Components

### 1. **Character Frequency Analysis**
- Compared letter frequency to standard English
- Identified overused/underused letters in gaming culture
- Analyzed special character usage patterns

### 2. **N-Gram Analysis**
- **Bigrams (2-char):** Most common 2-character combinations
- **Trigrams (3-char):** Most common 3-character patterns
- Reveals popular naming conventions and patterns

### 3. **Case Pattern Analysis**
- lowercase vs UPPERCASE vs CamelCase vs Mixed
- Shows stylistic preferences in username creation

### 4. **Leet Speak Detection**
- Identified 1337 speak patterns (0→o, 1→i, 3→e, etc.)
- Measured prevalence of leet substitutions

### 5. **Prefix/Suffix Analysis**
- Most common starting and ending patterns
- Cultural markers (xX, YT, TTV, etc.)

### 6. **Number Usage Patterns**
- Most popular numbers in usernames
- Cultural significance (69, 420, birth years, etc.)

---

## 📈 Generated Visualizations

All plots saved to `plots/`:

1. **`username_length_distribution.png`** - Distribution of username lengths
2. **`letter_frequency_analysis.png`** - Letter usage vs English comparison
3. **`ngram_analysis.png`** - Top bigrams and trigrams
4. **`case_pattern_distribution.png`** - Case style preferences
5. **`number_usage_analysis.png`** - Popular numbers in usernames

---

## 🔐 Privacy Note

- Raw usernames remain in `data/raw/usernames.txt` (git-ignored)
- All published analysis contains only aggregate statistics
- No individual usernames are exposed in outputs

---

## 💡 Insights for Gaming Culture

This analysis reveals:
- **Naming conventions** unique to gaming communities
- **Linguistic patterns** that differ from standard language
- **Cultural markers** embedded in username choices
- **Evolution of internet language** (leet speak, meme numbers)

---

## 🚀 Future Analysis Ideas

- Temporal analysis (if username creation dates available)
- Cross-platform comparison (compare to other gaming communities)
- Sentiment/tone analysis of username semantics
- Predictive modeling of username popularity trends

---

**Notebook:** `notebooks/02_username_linguistics.ipynb`  
**Data Source:** OCC Minecraft Server Leaderboards
