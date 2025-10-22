# Stage 2: Group Normalization and Standardization

## Overview

Stage 2 processes the output from Stage 1.5 to further normalize and group theory names using LLM-assisted grouping.

## What It Does

1. **Extracts names needing normalization:**
   - Mapped theories from stage 1.5 that are NOT in the initial ontology (6,926 names)
   - Unmapped theories using their original names (14,959 names)
   - Total unique names: **14,403**

2. **Groups alphabetically:**
   - Groups by first letter/symbol (38 groups: A-Z, symbols, Greek letters)
   - Creates batches of max 100 names per batch
   - Combines small consecutive groups when possible
   - Total batches: **157**

3. **LLM grouping and standardization:**
   - For each batch, LLM groups similar theories together
   - Creates standardized group names (not too generic, preserves meaning)
   - Tracks existing groups across iterations for consistency
   - Validates all input names appear in output

## Input/Output

**Input:** `output/stage1_5_llm_mapped.json`
- Contains mapped_theories and unmapped_theories from stage 1.5

**Output:** `output/stage2_grouped_theories.json`
```json
{
  "metadata": {
    "stage": "stage2_group_normalization",
    "total_input_names": 14403,
    "total_groups": <number>,
    "total_batches": 157,
    "total_input_tokens": <number>,
    "total_output_tokens": <number>,
    "total_cost": <cost>,
    "timestamp": "..."
  },
  "groups": {
    "Standardized Group Name 1": ["Theory 1", "Theory 2", ...],
    "Standardized Group Name 2": ["Theory 3", ...],
    ...
  },
  "initial_ontology_theories": ["Theory A", "Theory B", ...]
}
```

## Usage

### Run Full Stage 2

```bash
cd /home/diana.z/hack/theories_extraction_agent
python src/normalization/stage2_group_normalization.py
```

### Test with Small Subset

```bash
python test_stage2_small.py
```

### Programmatic Usage

```python
from src.normalization.stage2_group_normalization import Stage2GroupNormalizer

normalizer = Stage2GroupNormalizer(
    stage1_5_path='output/stage1_5_llm_mapped.json',
    ontology_path='ontology/groups_ontology_alliases.json'
)

normalizer.run(
    output_path='output/stage2_grouped_theories.json',
    max_batch_size=100
)
```

## Key Features

### 1. Alphabetical Batching
- Groups names by first character for semantic similarity
- Automatically splits large groups (>100 names)
- Combines small consecutive groups to optimize batch size

### 2. Iterative Group Tracking
- Maintains `existing_groups` dictionary across batches
- Passes existing group names to LLM for consistency
- Example: If batch 1 creates "Insulin/Igf-1 Signaling Pathway Theory", batch 2 will reuse this name for similar theories

### 3. Validation
- Checks all input names appear in output
- Missing names are automatically added as individual groups
- Warns about extra names in output

### 4. Cost Tracking
- Tracks token usage and cost per batch
- Model: gpt-4o-mini ($0.150/1M input, $0.600/1M output)
- Estimated cost for full run: ~$2-3

## Example Output

From test run with 20 names:
```
Groups created: 16

Group: Sexual Selection and Genetic Models
  - "Good Genes" Model of Sexual Selection

Group: Cognitive Aging Theories
  - "Speed"-Hypothesis of Cognitive Aging
  - "Use It or Lose It" Neuronal Activation Theory

Group: Protein Misreading and Regulation Theories
  - +1 Protein Molecular Misreading Theory of Aging
  - 14-3-3 Protein Regulatory Theory of Aging
  - 14-3-3 proteins as regulators of lifespan via interaction with SIR-2.1 and DAF-16/FOXO
  - 14-3-3zeta-mediated aging regulatory network
```

## Statistics (Expected)

- **Input:** 14,403 unique theory names
- **Batches:** 157 batches (max 100 names each)
- **Processing time:** ~2-3 hours (with 1s delay between batches)
- **Cost:** ~$2-3 USD
- **Output:** ~1,000-3,000 standardized groups (estimated)

## Next Steps After Stage 2

After running stage 2, you will have:
1. Grouped and standardized theory names
2. Reduced 14,403 unique names to ~1,000-3,000 groups
3. Consistent naming across similar theories

You can then:
- Review the groups for quality
- Merge groups if needed
- Map back to original theory entries
- Update the ontology with new standardized names
