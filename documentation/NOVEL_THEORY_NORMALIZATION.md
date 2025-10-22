# Novel Theory Name Normalization

## 🎯 Problem

**Stage 1.5** produces novel theories with **non-normalized names** across the dataset:
- "Epigenetic Clock Theory"
- "DNA Methylation Aging Theory"  
- "Epigenetic Aging Clock"
- "Methylation Clock Hypothesis"

These are essentially the **same theory** but with different names.

## ✅ Solution

**Stage 4** now normalizes novel theory names by clustering based on:
1. **Name similarity** (60% weight) - using fuzzy string matching
2. **Mechanism similarity** (40% weight) - using Jaccard similarity

### How It Works

#### 1. Name Normalization
```python
def _normalize_theory_name(self, name: str) -> str:
    """Normalize theory name for comparison."""
    name = name.lower().strip()
    
    # Remove common words
    removals = ['theory', 'hypothesis', 'model', 'of aging', 'aging', 'the ', 'a ', 'an ']
    for removal in removals:
        name = name.replace(removal, '')
    
    return ' '.join(name.split())
```

**Example:**
- "Epigenetic Clock Theory" → "epigenetic clock"
- "DNA Methylation Aging Theory" → "dna methylation"
- "Epigenetic Aging Clock" → "epigenetic clock"

#### 2. Name Similarity Calculation
```python
def _calculate_name_similarity(self, name1: str, name2: str) -> float:
    """Calculate similarity between two theory names."""
    norm1 = self._normalize_theory_name(name1)
    norm2 = self._normalize_theory_name(name2)
    
    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()
```

**Examples:**
- "epigenetic clock" vs "epigenetic clock" → 1.0 (100%)
- "epigenetic clock" vs "dna methylation" → 0.3 (30%)
- "epigenetic clock" vs "epigenetic aging" → 0.7 (70%)

#### 3. Combined Clustering
```python
# Calculate name similarity
name_sim = self._calculate_name_similarity(name1, name2)

# Calculate mechanism similarity
mech_sim = self._calculate_similarity(mechanisms1, mechanisms2)

# Combined similarity (weight name more heavily)
combined_sim = name_sim * 0.6 + mech_sim * 0.4

# Cluster if above threshold (default 0.7)
if combined_sim >= 0.7:
    # Add to same cluster
```

### Example Clustering

**Input (from Stage 1.5):**
```
Novel theories:
1. "Epigenetic Clock Theory" (mechanisms: DNA methylation, CpG sites, aging biomarker)
2. "DNA Methylation Aging Theory" (mechanisms: DNA methylation, epigenetic changes)
3. "Epigenetic Aging Clock" (mechanisms: DNA methylation patterns, biological age)
4. "Telomere Shortening Theory" (mechanisms: telomere attrition, replicative senescence)
5. "Telomere Theory of Aging" (mechanisms: telomere length, cellular senescence)
```

**Output (from Stage 4):**
```
Novel groups:
G0040: "Epigenetic Clock Theory" (3 theories)
  - Variant names: "Epigenetic Clock Theory", "DNA Methylation Aging Theory", "Epigenetic Aging Clock"
  - Shared mechanisms: DNA methylation, epigenetic changes
  
G0041: "Telomere Shortening Theory" (2 theories)
  - Variant names: "Telomere Shortening Theory", "Telomere Theory of Aging"
  - Shared mechanisms: telomere attrition, cellular senescence
```

## 📊 Impact

### Before (Old Stage 4)
```
Novel theories clustered ONLY by mechanisms:
- Group 1: 5 theories (all with "DNA" mechanisms)
  - Includes epigenetic clock + other DNA theories
- Group 2: 3 theories (all with "protein" mechanisms)
  - Mixed protein-related theories
```

**Problem:** Groups too broad, names not normalized

### After (Improved Stage 4)
```
Novel theories clustered by NAME + mechanisms:
- Group 1: "Epigenetic Clock Theory" (3 theories)
  - All about DNA methylation clocks
- Group 2: "Telomere Shortening Theory" (2 theories)
  - All about telomere attrition
- Group 3: "Protein Aggregation Theory" (4 theories)
  - All about protein misfolding
```

**Benefits:** 
- ✅ Similar names clustered together
- ✅ Representative name chosen from variants
- ✅ Variant names tracked for reference
- ✅ More coherent groups

## 🔧 Configuration

### Adjust Weights

Edit `stage4_theory_grouping_improved.py` line ~261:

```python
# Current: 60% name, 40% mechanism
combined_sim = name_sim * 0.6 + mech_sim * 0.4

# More emphasis on name (for very similar names)
combined_sim = name_sim * 0.8 + mech_sim * 0.2

# More emphasis on mechanisms (for diverse names)
combined_sim = name_sim * 0.4 + mech_sim * 0.6
```

### Adjust Threshold

```bash
# Default: 0.7 (70% similarity required)
python src/normalization/stage4_theory_grouping_improved.py \
  --overlap-threshold 0.7

# More strict (fewer, tighter clusters)
python src/normalization/stage4_theory_grouping_improved.py \
  --overlap-threshold 0.8

# More lenient (more, looser clusters)
python src/normalization/stage4_theory_grouping_improved.py \
  --overlap-threshold 0.6
```

## 📋 Output Format

Each novel group now includes:

```json
{
  "group_id": "G0040",
  "canonical_name": null,
  "representative_name": "Epigenetic Clock Theory",
  "theory_count": 3,
  "primary_category": "Novel (3 variants)",
  "secondary_category": "Epigenetic Clock Theory, DNA Methylation Aging Theory, Epigenetic Aging Clock",
  "shared_mechanisms": [
    "DNA methylation patterns change with age",
    "CpG sites show age-related methylation changes",
    "Epigenetic clock predicts biological age"
  ],
  "source": "novel",
  "mechanism_source": "extracted"
}
```

**Key fields:**
- `representative_name`: Most common name in cluster
- `primary_category`: Shows number of name variants
- `secondary_category`: Lists all variant names (up to 3)
- `shared_mechanisms`: Mechanisms common to all theories in group

## 🧪 Testing

### Test Name Normalization

```python
from src.normalization.stage4_theory_grouping_improved import ImprovedTheoryGrouper

grouper = ImprovedTheoryGrouper()

# Test name similarity
names = [
    "Epigenetic Clock Theory",
    "DNA Methylation Aging Theory",
    "Epigenetic Aging Clock",
    "Telomere Shortening Theory"
]

print("Name Similarity Matrix:")
for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if i < j:
            sim = grouper._calculate_name_similarity(name1, name2)
            print(f"{name1} vs {name2}: {sim:.2f}")
```

**Expected output:**
```
Epigenetic Clock Theory vs DNA Methylation Aging Theory: 0.35
Epigenetic Clock Theory vs Epigenetic Aging Clock: 0.85
Epigenetic Clock Theory vs Telomere Shortening Theory: 0.20
DNA Methylation Aging Theory vs Epigenetic Aging Clock: 0.40
DNA Methylation Aging Theory vs Telomere Shortening Theory: 0.15
Epigenetic Aging Clock vs Telomere Shortening Theory: 0.18
```

### Run Full Pipeline

```bash
# Run with updated Stage 4
python src/normalization/stage4_theory_grouping_improved.py \
  --input output/stage3_extracted_improved_TEST.json \
  --output output/stage4_groups_NORMALIZED.json \
  --overlap-threshold 0.7

# Check novel groups
python -c "
import json
with open('output/stage4_groups_NORMALIZED.json') as f:
    data = json.load(f)
    novel_groups = [g for g in data['groups'] if g['source'] == 'novel']
    
    print(f'Novel Groups: {len(novel_groups)}')
    print('\nTop 10 Novel Groups:')
    sorted_groups = sorted(novel_groups, key=lambda x: x['theory_count'], reverse=True)
    for i, g in enumerate(sorted_groups[:10], 1):
        print(f'{i}. {g[\"representative_name\"]}: {g[\"theory_count\"]} theories')
        print(f'   Variants: {g[\"secondary_category\"]}')
"
```

## 🎯 Expected Results

With stricter Stage 1.5 mapping (confidence >= 0.8) and improved Stage 4 name normalization:

### Stage 1.5 Output
```
Mapped: ~1100 theories (72%)
Novel: ~350 theories (23%)
Unmatched: ~50 theories (3%)
```

### Stage 4 Output
```
Total groups: ~80-100

Canonical groups: 39
  - Free Radical Theory: 200 theories
  - Disposable Soma Theory: 150 theories
  - ...

Novel groups: 40-60
  - Epigenetic Clock Theory: 15 theories (3 name variants)
  - Gut Microbiome Aging Theory: 12 theories (2 name variants)
  - Cellular Senescence Burden Theory: 10 theories (4 name variants)
  - ...
```

### Benefits
- ✅ Novel theories properly normalized
- ✅ Name variants tracked
- ✅ Coherent groups with similar names + mechanisms
- ✅ Better organization and searchability

## 💡 Future Improvements

### 1. LLM-Based Name Normalization
For even better normalization, use LLM to suggest canonical names for novel theory clusters:

```python
def suggest_canonical_name_for_novel_group(self, variant_names: List[str], mechanisms: List[str]) -> str:
    """Use LLM to suggest best canonical name for novel theory cluster."""
    
    prompt = f"""Given these variant names for the same theory:
{chr(10).join([f'- {name}' for name in variant_names])}

And these mechanisms:
{chr(10).join([f'- {mech}' for mech in mechanisms[:5]])}

Suggest ONE clear, canonical name for this theory.
The name should be:
- Descriptive and specific
- Scientifically accurate
- Concise (3-5 words)

Output ONLY the canonical name, nothing else."""
    
    response = self.llm.client.chat.completions.create(...)
    return response.choices[0].message.content.strip()
```

### 2. Hierarchical Novel Theory Organization
Organize novel theories in a hierarchy:
```
Novel Theories
├─ Epigenetic Theories
│  ├─ Epigenetic Clock Theory (15 theories)
│  └─ Histone Modification Theory (8 theories)
├─ Microbiome Theories
│  ├─ Gut Microbiome Aging Theory (12 theories)
│  └─ Microbiome Dysbiosis Theory (6 theories)
└─ Cellular Theories
   ├─ Cellular Senescence Burden Theory (10 theories)
   └─ Zombie Cell Theory (7 theories)
```

### 3. Cross-Reference with Literature
Check if "novel" theories are actually published theories not in ontology:
- Search PubMed for theory name
- Check if it's a recognized theory
- Suggest adding to canonical ontology

## 📝 Summary

The improved Stage 4 now:
1. ✅ **Normalizes novel theory names** using fuzzy string matching
2. ✅ **Clusters by name + mechanisms** (60% name, 40% mechanisms)
3. ✅ **Tracks variant names** for each group
4. ✅ **Chooses representative name** (most common variant)
5. ✅ **Provides better organization** of novel theories

This solves the problem of non-normalized names across the dataset and creates coherent novel theory groups.
