# Stage 3: Theory Grouping by Shared Mechanisms

## Overview

Stage 3 identifies and groups theories that share the same or highly similar mechanisms. It combines:
1. **Stage 1 theories** - Matched to canonical names via fuzzy matching
2. **Stage 2 theories** - Validated and extracted via LLM

## Approach

### Grouping Strategy

#### 1. Stage 1 Theories (Fuzzy Matched)
- Automatically grouped by **canonical name**
- Example: All theories matched to "Free Radical Theory" form one group

#### 2. Stage 2 Theories (LLM Extracted)
- Grouped by **mechanism similarity** using Jaccard similarity
- Compares:
  - **Mechanisms** (60% weight)
  - **Key players** (20% weight)
  - **Pathways** (20% weight)

### Similarity Thresholds

- **Exact match**: 100% similarity (identical mechanisms)
- **High overlap**: 80% similarity (highly similar mechanisms)

### Mechanism Signature

For each theory, we compute a signature containing:
```python
{
  'mechanisms': set(['mechanism1', 'mechanism2', ...]),
  'key_players': set(['mTOR', 'AMPK', ...]),
  'pathways': set(['mTOR pathway', 'AMPK pathway', ...])
}
```

Two theories are grouped if their signatures have ≥80% Jaccard similarity.

## Algorithm

```
1. Group Stage 1 theories by canonical name
   - All theories with canonical_name="Free Radical Theory" → Group G0001

2. Group Stage 2 theories by mechanism similarity
   a. Compute mechanism signature for each theory
   b. Greedy clustering:
      - Take first ungrouped theory as seed
      - Find all theories with ≥80% similarity
      - Create group
      - Repeat until all theories grouped

3. Compute shared characteristics for each group
   - Shared mechanisms = intersection of all mechanisms
   - Shared key players = intersection of all key players
   - Shared pathways = intersection of all pathways
```

## Output Structure

### Theory Group

```json
{
  "group_id": "G0042",
  "canonical_name": null,
  "representative_name": "DNA Damage Accumulation Theory",
  "theory_ids": ["T000123", "T000456", "T000789"],
  "theory_count": 3,
  "primary_category": "Molecular and Cellular Damage Theories",
  "secondary_category": "Protein and DNA Damage Theories",
  "shared_mechanisms": [
    "Accumulation of DNA damage in cells",
    "Impaired DNA repair capacity with age",
    "Activation of DNA damage response pathways"
  ],
  "shared_key_players": [
    "DNA",
    "p53",
    "ATM",
    "DNA repair enzymes"
  ],
  "shared_pathways": [
    "p53 pathway",
    "ATM/ATR pathway",
    "DNA repair pathways"
  ],
  "level_of_explanation": "Molecular",
  "type_of_cause": "Intrinsic",
  "temporal_focus": "Lifelong",
  "adaptiveness": "Non-adaptive",
  "source": "stage2"
}
```

### Complete Output File

```json
{
  "metadata": {
    "stage": "stage3_theory_grouping",
    "approach": "mechanism-based grouping",
    "statistics": {
      "total_theories": 7675,
      "stage1_matched": 1469,
      "stage2_valid": 4500,
      "total_groups": 2500,
      "singleton_groups": 800,
      "avg_group_size": 3.1
    },
    "thresholds": {
      "exact_match": 1.0,
      "high_overlap": 0.8
    }
  },
  "groups": [...],
  "theories": [...]
}
```

## Usage

### Test Run (with Stage 2 test data):
```bash
python test_stage3_grouping.py
```

### Full Run:
```bash
python -m src.normalization.stage3_theory_grouping \
    --stage1 output/stage1_fuzzy_matched.json \
    --stage2 output/stage2_llm_extracted.json \
    --output output/stage3_theory_groups.json
```

### With Custom Thresholds:
```bash
python -m src.normalization.stage3_theory_grouping \
    --exact-threshold 1.0 \
    --overlap-threshold 0.75
```

## Input Requirements

### From Stage 1 (`stage1_fuzzy_matched.json`):
```json
{
  "matched_theories": [
    {
      "theory_id": "T000001",
      "original_name": "Free Radical Theory of Aging",
      "match_result": {
        "matched": true,
        "canonical_name": "Free Radical Theory"
      }
    }
  ]
}
```

### From Stage 2 (`stage2_llm_extracted.json`):
```json
{
  "valid_theories": [
    {
      "theory_id": "T000002",
      "original_name": "DNA Damage Theory",
      "stage2_metadata": {
        "mechanisms": ["DNA damage accumulation", "Impaired repair"],
        "key_players": ["DNA", "p53", "ATM"],
        "pathways": ["p53", "ATM/ATR"],
        "primary_category": "Molecular and Cellular Damage Theories",
        "secondary_category": "Protein and DNA Damage Theories"
      }
    }
  ]
}
```

## Expected Results

### On Full Dataset (~7,675 theories):

**Input**:
- Stage 1 matched: 1,469 theories
- Stage 2 valid: ~4,500 theories
- Total: ~5,969 theories

**Expected Output**:
- Total groups: ~2,500-3,000
- Singleton groups: ~800-1,000 (unique theories)
- Multi-theory groups: ~1,500-2,000
- Avg group size: ~2-3 theories per group
- Compression: ~50-60% reduction

### Benefits:

1. **Identifies duplicates**: Same theory mentioned in multiple papers
2. **Finds variants**: Theories with slightly different names but same mechanisms
3. **Enables analysis**: Can analyze theory frequency and popularity
4. **Reduces redundancy**: From 7,675 → ~2,500 unique theories

## Example Groups

### Group 1: Free Radical Theory (from Stage 1)
- **Canonical**: Free Radical Theory
- **Theory count**: 262
- **Source**: stage1
- All theories matched to "Free Radical Theory" in fuzzy matching

### Group 2: DNA Damage Accumulation (from Stage 2)
- **Representative**: DNA Damage Accumulation Theory
- **Theory count**: 15
- **Source**: stage2
- **Shared mechanisms**:
  - Accumulation of DNA damage
  - Impaired DNA repair
  - Activation of damage response
- **Shared key players**: DNA, p53, ATM, DNA repair enzymes
- **Shared pathways**: p53, ATM/ATR, DNA repair

### Group 3: Mitochondrial Dysfunction (from Stage 2)
- **Representative**: Mitochondrial Dysfunction Theory
- **Theory count**: 8
- **Source**: stage2
- **Shared mechanisms**:
  - Mitochondrial dysfunction
  - Increased ROS production
  - Impaired energy metabolism
- **Shared key players**: mitochondria, ROS, ATP, electron transport chain
- **Shared pathways**: oxidative phosphorylation, ROS signaling

## Statistics Tracked

- Total theories processed
- Theories from Stage 1 vs Stage 2
- Total groups created
- Singleton groups (1 theory)
- Multi-theory groups (2+ theories)
- Average group size
- Compression ratio

## Files

- `src/normalization/stage3_theory_grouping.py` - Main implementation
- `test_stage3_grouping.py` - Test script
- `STAGE3_README.md` - This file

## Next Steps

After Stage 3:
1. Analyze group distribution
2. Identify most popular theories
3. Find novel theories (singletons with unique mechanisms)
4. Export final normalized dataset
5. Create visualization of theory landscape
