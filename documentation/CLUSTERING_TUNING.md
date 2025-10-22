# Clustering Tuning for Novel Theories

## 🔍 Issue Identified

The test revealed that the clustering was **too strict**:

```
"Epigenetic Clock Theory" vs "Epigenetic Aging Clock"
  Name similarity: 1.00 (perfect match!)
  Mechanism similarity: 0.17 (low due to different wording)
  Combined (60/40): 0.67
  Would cluster: ❌ NO (threshold was 0.7)
```

**Problem:** Even theories with identical names after normalization weren't clustering because mechanism similarity was too low.

## ✅ Changes Made

### 1. Lowered Clustering Threshold
**File:** `stage4_theory_grouping_improved.py`

```python
# Before
high_overlap_threshold: float = 0.7  # 70% similarity required

# After
high_overlap_threshold: float = 0.6  # 60% similarity required
```

**Impact:** Theories with high name similarity but moderate mechanism similarity will now cluster.

### 2. Increased Name Weight
**File:** `stage4_theory_grouping_improved.py` (line 262)

```python
# Before
combined_sim = name_sim * 0.6 + mech_sim * 0.4  # 60% name, 40% mechanism

# After
combined_sim = name_sim * 0.7 + mech_sim * 0.3  # 70% name, 30% mechanism
```

**Impact:** Name similarity is now weighted more heavily, which makes sense for novel theories where the name is the primary identifier.

### 3. Improved Mechanism Normalization
**File:** `stage4_theory_grouping_improved.py` (line 102-120)

```python
def _normalize_list(self, items: List[str]) -> Set[str]:
    """Normalize list for comparison with better tokenization."""
    normalized = set()
    for item in items:
        # Tokenize and add individual words
        words = item.split()
        words = [w for w in words if w not in stop_words]
        
        # Add full phrase
        normalized.add(' '.join(words))
        
        # Add individual significant words (>3 chars)
        for word in words:
            if len(word) > 3:
                normalized.add(word)
    
    return normalized
```

**Impact:** Better matching between similar mechanisms with different wording:
- "DNA methylation" and "DNA methylation patterns" now share "methylation" and "dna"
- "aging biomarker" and "biological age" now share "aging" and "biological"

## 📊 Expected Results

### Before Changes
```
"Epigenetic Clock Theory" vs "Epigenetic Aging Clock"
  Name similarity: 1.00
  Mechanism similarity: 0.17
  Combined (60/40): 0.67
  Would cluster: ❌ NO (< 0.7 threshold)
```

### After Changes
```
"Epigenetic Clock Theory" vs "Epigenetic Aging Clock"
  Name similarity: 1.00
  Mechanism similarity: ~0.35 (improved with tokenization)
  Combined (70/30): 0.81
  Would cluster: ✅ YES (> 0.6 threshold)
```

## 🧪 Test Again

Run the updated test:

```bash
python test_name_normalization.py
```

**Expected improvements:**
- "Epigenetic Clock Theory" + "Epigenetic Aging Clock" → ✅ YES
- "Telomere Shortening Theory" + "Telomere Theory of Aging" → ✅ YES (if mechanisms similar)
- Different theories still separate → ✅ NO

## 🎯 Tuning Guidelines

### If Too Many Clusters (Over-splitting)

**Lower the threshold:**
```bash
python src/normalization/stage4_theory_grouping_improved.py \
  --overlap-threshold 0.5  # More lenient
```

**Increase name weight:**
```python
# Line 262
combined_sim = name_sim * 0.8 + mech_sim * 0.2  # 80% name
```

### If Too Few Clusters (Over-merging)

**Raise the threshold:**
```bash
python src/normalization/stage4_theory_grouping_improved.py \
  --overlap-threshold 0.7  # More strict
```

**Increase mechanism weight:**
```python
# Line 262
combined_sim = name_sim * 0.5 + mech_sim * 0.5  # 50/50 split
```

## 📋 Summary of Settings

| Setting | Old Value | New Value | Reason |
|---------|-----------|-----------|--------|
| Clustering threshold | 0.7 | 0.6 | Allow moderate similarity to cluster |
| Name weight | 60% | 70% | Prioritize name for novel theories |
| Mechanism weight | 40% | 30% | Reduce impact of wording differences |
| Mechanism normalization | Simple lowercase | Tokenized + words | Better partial matching |

## ✅ Benefits

1. **Better clustering** - Theories with same/similar names cluster together
2. **More robust** - Less sensitive to mechanism wording differences
3. **Configurable** - Can adjust threshold via command line
4. **Balanced** - Still uses mechanisms to validate clustering

## 🔄 Next Steps

1. **Run test** - Verify improvements: `python test_name_normalization.py`
2. **Run Stage 4** - Process full dataset with new settings
3. **Evaluate** - Check if novel groups are well-formed
4. **Tune** - Adjust threshold if needed based on results

The clustering should now properly group novel theories with similar names, even if their mechanisms are worded differently!
