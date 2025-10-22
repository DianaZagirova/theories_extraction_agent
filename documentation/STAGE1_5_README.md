# Stage 1.5: LLM-Based Mapping to Canonical Theories

## Overview

Stage 1.5 sits between Stage 1 (fuzzy matching) and Stage 2 (full extraction). It uses LLM to intelligently map unmatched theories to canonical theories from the ontology.

## Purpose

**Problem**: Stage 1 fuzzy matching only catches 19.1% of theories, leaving 80.9% for expensive Stage 2 processing.

**Solution**: Use LLM to map theories to canonical names before full extraction.

**Benefits**:
- Higher match rate (19.1% → 35-45%)
- Lower Stage 2 costs ($35 → $15-20)
- Validates theories before extraction
- Identifies novel theories

## How It Works

### Input
- **Unmatched theories from Stage 1** (6,206 theories)
- **Canonical theories from ontology** (46 theories with mechanisms)

### Process

```
For each batch of 30 theories:
  1. Create prompt with:
     - Theory name + concept description
     - Confidence flag (if medium)
     - All 46 canonical theories + mechanisms
  
  2. Ask LLM to:
     - Validate: Is this a real aging theory?
     - Map: Does it match a canonical theory?
     - Classify: Mapped, Novel, or Invalid
  
  3. Parse response:
     - Mapped → Add to matched_theories
     - Novel → Add to novel_theories  
     - Invalid → Add to invalid_theories
     - Unmatched → Add to still_unmatched
```

### Output
- **Mapped theories**: Matched to canonical names (go to Stage 3)
- **Novel theories**: Valid but don't match ontology (go to Stage 2)
- **Still unmatched**: Valid but uncertain (go to Stage 2)
- **Invalid theories**: Not real aging theories (filtered out)

## Prompt Design

### Key Features

1. **Validation First**
   - Checks if theory is genuine
   - Extra validation for medium-confidence theories
   - Filters out pseudoscience

2. **Semantic Matching**
   - Compares concepts, not just names
   - Uses canonical mechanisms for matching
   - Considers similarity, not exact match

3. **Batch Processing**
   - Processes 30 theories at once
   - Saves tokens and time
   - Includes theory IDs to prevent truncation issues

4. **Conservative Mapping**
   - Only maps if confidence ≥ 0.7
   - Proposes names for novel theories
   - Clear reasoning for each decision

### Example Prompt Structure

```
You are an expert in aging biology...

# CANONICAL THEORIES IN ONTOLOGY

- **Free Radical Theory**
  Category: Molecular and Cellular Damage Theories
  Key mechanisms:
    - Excessive ROS production causing oxidative damage
    - Mitochondrial dysfunction leading to increased ROS
    - Accumulation of oxidative damage

- **Telomere Theory**
  ...

# THEORIES TO VALIDATE AND MAP

1. **ROS-Induced Cellular Aging Theory**
   ID: T000123
   Concept: Reactive oxygen species cause damage to cellular components...

2. **Chromosomal End Shortening Hypothesis** ⚠️ MEDIUM CONFIDENCE
   ID: T000456
   Concept: Progressive shortening of chromosome ends...

# INSTRUCTIONS
...

# OUTPUT FORMAT
{
  "mappings": [
    {
      "theory_id": "T000123",
      "is_valid_theory": true,
      "is_mapped": true,
      "canonical_name": "Free Radical Theory",
      "mapping_confidence": 0.92
    },
    ...
  ]
}
```

## Usage

### Test Run (50 theories)
```bash
python test_stage1_5_mapping.py
```

### Full Run
```bash
python -m src.normalization.stage1_5_llm_mapping \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage1_5_llm_mapped.json \
    --batch-size 30
```

### Custom Parameters
```bash
python -m src.normalization.stage1_5_llm_mapping \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage1_5_llm_mapped.json \
    --batch-size 25 \
    --max-theories 100
```

## Output Structure

```json
{
  "metadata": {
    "stage": "stage1_5_llm_mapping",
    "batch_size": 30,
    "statistics": {
      "total_processed": 6206,
      "valid_theories": 5200,
      "invalid_theories": 1006,
      "mapped_to_canonical": 2100,
      "novel_theories": 2800,
      "batch_count": 207
    }
  },
  "mapped_theories": [
    {
      "theory_id": "T000123",
      "theory_name": "ROS-Induced Cellular Aging",
      "concept_text": "...",
      "stage1_5_result": {
        "is_valid_theory": true,
        "validation_reasoning": "Valid theory explaining ROS damage",
        "is_mapped": true,
        "canonical_name": "Free Radical Theory",
        "mapping_confidence": 0.92,
        "is_novel": false
      },
      "match_result": {
        "matched": true,
        "canonical_name": "Free Radical Theory",
        "match_type": "llm_mapping",
        "confidence": 0.92
      }
    }
  ],
  "novel_theories": [
    {
      "theory_id": "T000456",
      "theory_name": "Epigenetic Noise Theory",
      "stage1_5_result": {
        "is_valid_theory": true,
        "is_mapped": false,
        "is_novel": true,
        "proposed_name": "Epigenetic Noise Accumulation Theory"
      }
    }
  ],
  "still_unmatched": [...],
  "invalid_theories": [...]
}
```

## Expected Results

### On Full Dataset (6,206 unmatched from Stage 1)

| Category | Count | Percentage |
|----------|-------|------------|
| **Mapped to canonical** | ~2,100 | 34% |
| **Novel theories** | ~2,800 | 45% |
| **Still unmatched** | ~300 | 5% |
| **Invalid theories** | ~1,000 | 16% |

### Impact on Pipeline

**Before Stage 1.5**:
- Stage 1 matched: 1,469 (19.1%)
- To Stage 2: 6,206 (80.9%)
- Stage 2 cost: $35

**After Stage 1.5**:
- Stage 1 matched: 1,469 (19.1%)
- Stage 1.5 mapped: ~2,100 (27.4%)
- **Total matched: ~3,569 (46.5%)** ✅
- To Stage 2: ~3,100 (40.4%)
- Stage 2 cost: **$17-20** ✅
- Invalid filtered: ~1,000 (13%)

### Cost Analysis

**Stage 1.5 Cost**:
- 6,206 theories / 30 per batch = 207 batches
- ~2,000 tokens per batch (input + output)
- Total: ~414,000 tokens
- Cost: **~$2-3** (GPT-4o-mini)

**Stage 2 Savings**:
- Theories reduced: 6,206 → 3,100 (50% reduction)
- Cost savings: **~$15-18**

**Net Savings**: $15-18 - $2-3 = **$12-15 per run** ✅

## Integration with Pipeline

### Updated Pipeline Flow

```
7,675 theories
    ↓
┌─────────────────────┐
│ STAGE 1             │
│ Fuzzy Matching      │
└─────────────────────┘
    ↓
1,469 matched (19.1%)
6,206 unmatched (80.9%)
    ↓
┌─────────────────────┐
│ STAGE 1.5 (NEW!)    │  ← $2-3 cost
│ LLM Mapping         │
└─────────────────────┘
    ↓
├─ 2,100 mapped (27.4%) ────────┐
├─ 2,800 novel (36.5%) ─────┐   │
├─ 300 unmatched (3.9%) ────┤   │
└─ 1,000 invalid (13%) [X]  │   │
                             ↓   │
                    ┌─────────────────────┐
                    │ STAGE 2             │  ← $17-20 cost (vs $35)
                    │ LLM Extraction      │
                    └─────────────────────┘
                             ↓
                    ~2,300 valid
                             ↓
                             ├───────────────┘
                             ↓
                    Total: ~5,900 theories
                             ↓
                    ┌─────────────────────┐
                    │ STAGE 3             │
                    │ Theory Grouping     │
                    └─────────────────────┘
```

## Advantages

### 1. Higher Match Rate ✅
- Stage 1 alone: 19.1%
- Stage 1 + 1.5: **46.5%** (+140%)

### 2. Lower Costs ✅
- Stage 2 cost: $35 → $17-20 (-50%)
- Stage 1.5 cost: $2-3
- Net savings: **$12-15 per run**

### 3. Better Quality ✅
- Validates theories before extraction
- Filters out invalid theories (13%)
- Identifies novel theories explicitly

### 4. Semantic Understanding ✅
- Matches concepts, not just names
- Uses canonical mechanisms
- LLM reasoning for each decision

### 5. Batch Efficiency ✅
- 30 theories per batch
- Saves tokens and time
- Prevents truncation with IDs

## Limitations

### 1. LLM Dependency
- Requires API access
- Subject to rate limits
- Non-deterministic (can vary slightly)

### 2. Cost
- Adds $2-3 per run
- But saves $12-15 overall

### 3. Processing Time
- Adds ~15-20 minutes
- But reduces Stage 2 time by ~20-25 minutes
- Net time savings: ~5-10 minutes

### 4. Batch Size Trade-off
- Larger batches: Cheaper, faster, but risk truncation
- Smaller batches: More reliable, but more expensive
- Recommended: 25-30 theories per batch

## Best Practices

### 1. Batch Size
- **Recommended**: 30 theories
- **Conservative**: 20-25 theories
- **Aggressive**: 40-50 theories (risk truncation)

### 2. Validation
- Always validate medium-confidence theories
- Check for pseudoscience
- Require clear reasoning

### 3. Mapping Confidence
- **High (≥0.8)**: Very confident match
- **Medium (0.7-0.8)**: Probable match
- **Low (<0.7)**: Don't map, send to Stage 2

### 4. Novel Theories
- Propose clear, descriptive names
- Document why they don't match ontology
- Send to Stage 2 for full extraction

## Testing

### Test with 50 theories:
```bash
python test_stage1_5_mapping.py
```

Expected output:
- Mapped: ~15-20 theories
- Novel: ~20-25 theories
- Invalid: ~5-10 theories

### Validate results:
```bash
python3 << 'EOF'
import json

with open('output/stage1_5_llm_mapped_TEST.json', 'r') as f:
    data = json.load(f)

print(f"Mapped: {len(data['mapped_theories'])}")
print(f"Novel: {len(data['novel_theories'])}")
print(f"Invalid: {len(data['invalid_theories'])}")

# Check sample mappings
for theory in data['mapped_theories'][:3]:
    result = theory['stage1_5_result']
    print(f"\n{theory['theory_name']}")
    print(f"  → {result['canonical_name']} ({result['mapping_confidence']:.2f})")
EOF
```

## Next Steps

After Stage 1.5:
1. **Mapped theories** → Go directly to Stage 3 (grouped with Stage 1 matches)
2. **Novel + unmatched** → Go to Stage 2 for full extraction
3. **Invalid theories** → Filtered out, not processed further

## Files

- `src/normalization/stage1_5_llm_mapping.py` - Main implementation
- `test_stage1_5_mapping.py` - Test script
- `STAGE1_5_README.md` - This documentation
