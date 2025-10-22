# Smart Multi-Dimensional Clustering

## 🎯 Overview

Stage 4 now uses **multi-dimensional similarity** to cluster theories more intelligently by considering:

1. **Content Similarity (80% weight)**
   - Mechanisms (50%)
   - Key players (30%)
   - Pathways (20%)

2. **Categorical Similarity (20% weight)**
   - Level of explanation (Molecular, Cellular, Organismal, etc.)
   - Type of cause (Intrinsic, Extrinsic, Both)
   - Temporal focus (Developmental, Lifelong, Late-life, etc.)
   - Adaptiveness (Adaptive, Non-adaptive, Both)

## 📊 How It Works

### Step 1: Extract Comprehensive Signature

For each theory, extract:
```python
{
    'mechanisms': ['ROS accumulation', 'oxidative damage', ...],
    'key_players': ['mitochondria', 'ROS', 'DNA', ...],
    'pathways': ['oxidative phosphorylation', 'NF-κB', ...],
    'level_of_explanation': 'Molecular',
    'type_of_cause': 'Intrinsic',
    'temporal_focus': 'Lifelong',
    'adaptiveness': 'Non-adaptive'
}
```

### Step 2: Calculate Multi-Dimensional Similarity

#### A. Content Similarity (Jaccard)
```python
mechanisms_sim = |mechanisms1 ∩ mechanisms2| / |mechanisms1 ∪ mechanisms2|
key_players_sim = |players1 ∩ players2| / |players1 ∪ players2|
pathways_sim = |pathways1 ∩ pathways2| / |pathways1 ∪ pathways2|

content_sim = mechanisms_sim * 0.5 + key_players_sim * 0.3 + pathways_sim * 0.2
```

#### B. Categorical Similarity (Exact Match)
```python
categorical_matches = 0
categorical_total = 0

if both have level_of_explanation:
    categorical_total += 1
    if level1 == level2:
        categorical_matches += 1

# Same for type_of_cause, temporal_focus, adaptiveness

categorical_sim = categorical_matches / categorical_total
```

#### C. Final Similarity
```python
final_sim = content_sim * 0.8 + categorical_sim * 0.2
```

### Step 3: Cluster by Combined Similarity

Theories cluster together if:
- **Name similarity** (70%) + **Mechanism similarity** (30%) >= 0.6 (for novel theories)
- **Mechanism similarity** >= 0.6 (for unmapped theories)

## 🔍 Example Scenarios

### Scenario 1: Same Mechanisms, Same Level

**Theory A:**
```json
{
  "mechanisms": ["ROS accumulation", "oxidative damage", "mitochondrial dysfunction"],
  "key_players": ["mitochondria", "ROS", "DNA"],
  "pathways": ["oxidative phosphorylation"],
  "level_of_explanation": "Molecular",
  "type_of_cause": "Intrinsic"
}
```

**Theory B:**
```json
{
  "mechanisms": ["ROS accumulation", "oxidative damage", "DNA damage"],
  "key_players": ["mitochondria", "ROS", "proteins"],
  "pathways": ["oxidative phosphorylation", "NF-κB"],
  "level_of_explanation": "Molecular",
  "type_of_cause": "Intrinsic"
}
```

**Similarity Calculation:**
```
Mechanisms: 2/4 = 0.50
Key players: 2/4 = 0.50
Pathways: 1/2 = 0.50

Content similarity: 0.50 * 0.5 + 0.50 * 0.3 + 0.50 * 0.2 = 0.50

Categorical matches: 2/2 = 1.00
  - Level: Molecular == Molecular ✓
  - Type: Intrinsic == Intrinsic ✓

Categorical similarity: 1.00

Final similarity: 0.50 * 0.8 + 1.00 * 0.2 = 0.60

Result: ✅ WILL CLUSTER (>= 0.6 threshold)
```

### Scenario 2: Similar Mechanisms, Different Level

**Theory A:**
```json
{
  "mechanisms": ["cellular senescence", "SASP secretion"],
  "level_of_explanation": "Cellular",
  "type_of_cause": "Intrinsic"
}
```

**Theory B:**
```json
{
  "mechanisms": ["cellular senescence", "tissue dysfunction"],
  "level_of_explanation": "Tissue/Organ",
  "type_of_cause": "Intrinsic"
}
```

**Similarity Calculation:**
```
Mechanisms: 1/3 = 0.33
Content similarity: 0.33

Categorical matches: 1/2 = 0.50
  - Level: Cellular != Tissue/Organ ✗
  - Type: Intrinsic == Intrinsic ✓

Categorical similarity: 0.50

Final similarity: 0.33 * 0.8 + 0.50 * 0.2 = 0.36

Result: ❌ WON'T CLUSTER (< 0.6 threshold)
```

### Scenario 3: Different Mechanisms, Same Categories

**Theory A:**
```json
{
  "mechanisms": ["telomere shortening", "replicative senescence"],
  "level_of_explanation": "Cellular",
  "type_of_cause": "Intrinsic",
  "temporal_focus": "Lifelong"
}
```

**Theory B:**
```json
{
  "mechanisms": ["DNA methylation changes", "epigenetic drift"],
  "level_of_explanation": "Cellular",
  "type_of_cause": "Intrinsic",
  "temporal_focus": "Lifelong"
}
```

**Similarity Calculation:**
```
Mechanisms: 0/4 = 0.00
Content similarity: 0.00

Categorical matches: 3/3 = 1.00
  - Level: Cellular == Cellular ✓
  - Type: Intrinsic == Intrinsic ✓
  - Temporal: Lifelong == Lifelong ✓

Categorical similarity: 1.00

Final similarity: 0.00 * 0.8 + 1.00 * 0.2 = 0.20

Result: ❌ WON'T CLUSTER (< 0.6 threshold)
```

**Conclusion:** Categories alone aren't enough - theories need similar mechanisms to cluster.

## 🎛️ Tuning the Weights

### Current Weights

```python
# Final similarity
content_sim * 0.8 + categorical_sim * 0.2

# Content breakdown
mechanisms * 0.5 + key_players * 0.3 + pathways * 0.2

# Novel theory clustering
name_sim * 0.7 + mechanism_sim * 0.3
```

### Adjustment Scenarios

#### 1. Emphasize Categorical Matching More

If you want theories at the same level/type to cluster more easily:

```python
# Line 233
final_sim = content_sim * 0.7 + categorical_sim * 0.3  # Increase categorical to 30%
```

**Effect:** Theories with same level/type but moderate mechanism overlap will cluster.

#### 2. Emphasize Key Players Over Mechanisms

If key players are more reliable than mechanism descriptions:

```python
# Line 221
content_sim = mech_sim * 0.4 + player_sim * 0.4 + pathway_sim * 0.2  # Equal weight
```

**Effect:** Theories sharing key players will cluster more easily.

#### 3. Strict Mechanism Matching

If you want only very similar mechanisms to cluster:

```python
# Line 221
content_sim = mech_sim * 0.7 + player_sim * 0.2 + pathway_sim * 0.1  # Heavy mechanism weight
```

**Effect:** Only theories with high mechanism overlap will cluster.

## 📋 Benefits of Multi-Dimensional Clustering

### 1. **More Intelligent Grouping**

**Before (mechanism-only):**
```
Group 1: All theories mentioning "ROS" (200 theories)
  - Mix of molecular, cellular, and organismal theories
  - Mix of intrinsic and extrinsic causes
```

**After (multi-dimensional):**
```
Group 1: Molecular ROS theories, intrinsic (45 theories)
Group 2: Cellular ROS theories, intrinsic (38 theories)
Group 3: Organismal ROS theories, extrinsic (25 theories)
```

### 2. **Better Separation**

Theories that share some mechanisms but are fundamentally different (different levels, different causes) won't cluster together.

### 3. **Richer Group Metadata**

Each group now has:
- Shared mechanisms
- Shared key players
- Shared pathways
- Dominant level of explanation
- Dominant type of cause
- Dominant temporal focus

### 4. **Validation**

Categorical matching acts as a validation:
- If mechanisms match BUT categories don't → Lower similarity
- If mechanisms AND categories match → Higher similarity

## 🧪 Testing

### Test Similarity Calculation

```python
from src.normalization.stage4_theory_grouping_improved import ImprovedTheoryGrouper

grouper = ImprovedTheoryGrouper()

# Create test theories
theory1 = {
    'stage3_metadata': {
        'mechanisms': ['ROS accumulation', 'oxidative damage'],
        'key_players': ['mitochondria', 'ROS'],
        'pathways': ['oxidative phosphorylation'],
        'level_of_explanation': 'Molecular',
        'type_of_cause': 'Intrinsic'
    }
}

theory2 = {
    'stage3_metadata': {
        'mechanisms': ['ROS accumulation', 'DNA damage'],
        'key_players': ['mitochondria', 'DNA'],
        'pathways': ['oxidative phosphorylation'],
        'level_of_explanation': 'Molecular',
        'type_of_cause': 'Intrinsic'
    }
}

# Compute signatures
sig1 = grouper._compute_mechanism_signature(theory1)
sig2 = grouper._compute_mechanism_signature(theory2)

# Get detailed breakdown
breakdown = grouper._get_similarity_breakdown(sig1, sig2)

print(f"Total similarity: {breakdown['total']:.2f}")
print(f"Content similarity: {breakdown['content_similarity']:.2f}")
print(f"Categorical similarity: {breakdown['categorical_similarity']:.2f}")
print(f"Breakdown: {breakdown['breakdown']}")
```

### Run Full Pipeline

```bash
python src/normalization/stage4_theory_grouping_improved.py \
  --input output/stage3_extracted_improved_TEST.json \
  --output output/stage4_groups_SMART.json \
  --overlap-threshold 0.6
```

### Analyze Results

```bash
python -c "
import json

with open('output/stage4_groups_SMART.json') as f:
    data = json.load(f)
    groups = data['groups']
    
    print('Group Analysis:')
    print(f'Total groups: {len(groups)}')
    
    # Analyze by level of explanation
    levels = {}
    for g in groups:
        level = g.get('level_of_explanation', 'Unknown')
        levels[level] = levels.get(level, 0) + 1
    
    print('\nGroups by Level of Explanation:')
    for level, count in sorted(levels.items(), key=lambda x: x[1], reverse=True):
        print(f'  {level}: {count} groups')
    
    # Analyze by type of cause
    types = {}
    for g in groups:
        type_cause = g.get('type_of_cause', 'Unknown')
        types[type_cause] = types.get(type_cause, 0) + 1
    
    print('\nGroups by Type of Cause:')
    for type_cause, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f'  {type_cause}: {count} groups')
"
```

## 📊 Expected Improvements

### Before (Mechanism-Only)

```
39 groups:
- Free Radical Theory: 290 theories (all ROS-related, mixed levels)
- Disposable Soma Theory: 217 theories (all trade-off related)
```

### After (Multi-Dimensional)

```
60-80 groups:
- Free Radical Theory (Molecular, Intrinsic): 120 theories
- Free Radical Theory (Cellular, Intrinsic): 85 theories
- Free Radical Theory (Organismal, Both): 45 theories
- Disposable Soma Theory (Organismal, Adaptive): 150 theories
- Disposable Soma Theory (Population, Adaptive): 67 theories
```

**Benefits:**
- More granular groups
- Better separation by level and type
- Easier to analyze and compare

## 🎯 Summary

The enhanced clustering now considers:

| Dimension | Weight | Purpose |
|-----------|--------|---------|
| Mechanisms | 40% (0.5 × 0.8) | Core similarity |
| Key Players | 24% (0.3 × 0.8) | Supporting evidence |
| Pathways | 16% (0.2 × 0.8) | Biological context |
| Categories | 20% | Validation & refinement |

This creates **smarter, more coherent groups** that share not just mechanisms, but also the same level of explanation and type of cause.
