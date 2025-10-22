# Stage 2: LLM-Based Theory Validation and Data Extraction

## Overview

Stage 2 processes theories that were **not matched** in Stage 1 fuzzy matching. It uses Claude (Anthropic) to:
1. Validate if each theory is genuinely a theory of aging
2. Extract detailed metadata and categorization

## Features

### 1. Theory Validation
- Checks if the theory explains WHY/HOW aging occurs
- Ensures it proposes causal mechanisms (not just descriptions)
- Verifies generalizability beyond narrow contexts
- Filters out specific findings, biomarkers, or interventions

### 2. Categorization
- **Primary Category**: Top-level classification from ontology
  - Evolutionary Theories of Aging
  - Molecular and Cellular Damage Theories
  - Systemic/Integrative Theories
  - Sociological and Psychological Theories
- **Secondary Categories**: Subcategories within primary
- **Novel Detection**: Identifies theories that don't fit existing categories

### 3. Metadata Extraction

#### Biological/Molecular Details:
- **Key Players**: Main biological/molecular/cellular actors (e.g., mTOR, AMPK, SIRT1, telomeres)
- **Pathways**: Specific molecular pathways (e.g., mTOR, AMPK, sirtuins, p53)
- **Mechanism of Action**: Primary process explaining aging (1-2 sentences)

#### Classification:
- **Level of Explanation**: Molecular, Cellular, Tissue/Organ, Organismal, Population, Societal
- **Type of Cause**: Intrinsic, Extrinsic, Both
- **Temporal Focus**: Developmental, Reproductive, Post-reproductive, Lifelong, Late-life, Not-stated
- **Adaptiveness**: Adaptive, Non-adaptive, Both/Context-dependent, Not-stated

#### Quality:
- **Extraction Confidence**: 0.0-1.0 score

## Usage

### Test Run (5 theories):
```bash
export ANTHROPIC_API_KEY='your-key-here'
python test_stage2_sample.py
```

### Full Run:
```bash
python -m src.normalization.stage2_llm_extraction \
    --api-key $ANTHROPIC_API_KEY \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage2_llm_extracted.json \
    --ontology ontology/groups_ontology_alliases.json
```

### With Limit (for testing):
```bash
python -m src.normalization.stage2_llm_extraction \
    --api-key $ANTHROPIC_API_KEY \
    --max-theories 100
```

## Input

Reads `unmatched_theories` from Stage 1 output:
- File: `output/stage1_fuzzy_matched.json`
- Expected: ~6,206 unmatched theories (from 7,675 total)

## Output

Creates `output/stage2_llm_extracted.json` with:

```json
{
  "metadata": {
    "stage": "stage2_llm_extraction",
    "statistics": {
      "total_processed": 6206,
      "valid_theories": 4500,
      "invalid_theories": 1706,
      "novel_theories": 200,
      "known_category_theories": 4300
    },
    "valid_count": 4500,
    "invalid_count": 1706,
    "validation_rate": 72.5
  },
  "valid_theories": [...],
  "invalid_theories": [...]
}
```

### Theory Structure:
```json
{
  "theory_id": "T000002",
  "original_name": "DNA Damage Accumulation Model...",
  "stage2_metadata": {
    "is_valid_theory": true,
    "validation_reasoning": "...",
    "primary_category": "Molecular and Cellular Damage Theories",
    "secondary_categories": ["Protein and DNA Damage Theories"],
    "is_novel": false,
    "key_players": ["DNA", "p53", "ATM", "DNA repair enzymes"],
    "pathways": ["p53", "ATM/ATR"],
    "mechanism_of_action": "Accumulation of DNA damage...",
    "level_of_explanation": "Molecular",
    "type_of_cause": "Intrinsic",
    "temporal_focus": "Lifelong",
    "adaptiveness": "Non-adaptive",
    "extraction_confidence": 0.9
  },
  "passed_stage2_validation": true
}
```

## Cost Estimation

- Model: Claude 3.5 Sonnet
- ~2000 tokens per theory (input + output)
- 6,206 theories × 2000 tokens = ~12.4M tokens
- Estimated cost: ~$37-50 (depending on exact token usage)

## Next Steps

After Stage 2:
1. Use `valid_theories` for Stage 3 (mechanism clustering)
2. Filter out `invalid_theories` (not genuine aging theories)
3. Analyze novel theories separately
4. Combine with Stage 1 matched theories for complete dataset

## Files

- `src/normalization/stage2_llm_extraction.py` - Main implementation
- `test_stage2_sample.py` - Test script (5 theories)
- `STAGE2_README.md` - This file

## Example Output

```
🚀 Starting Stage 2: LLM Extraction

📂 Loading ontology structure from ontology/groups_ontology_alliases.json...
✓ Loaded 4 primary categories
  Categories: Evolutionary Theories of Aging, Molecular and Cellular Damage Theories, Systemic/Integrative Theories, Sociological and Psychological Theories

📂 Loading unmatched theories from output/stage1_fuzzy_matched.json...
✓ Loaded 6206 unmatched theories

🤖 Starting LLM extraction...
  Progress: 10/6206
  Progress: 20/6206
  ...

✓ Extraction complete!

============================================================
STAGE 2: LLM EXTRACTION STATISTICS
============================================================
Total processed: 6206

Validation results:
  Valid theories: 4500 (72.5%)
  Invalid theories: 1706 (27.5%)

Categorization:
  Novel theories: 200
  Known category theories: 4300

Errors: 0
============================================================

✅ Stage 2 complete!
```
