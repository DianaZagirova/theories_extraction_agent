# Theory Tracker: Comprehensive Stage Tracking

## Overview

The Theory Tracker provides complete lineage tracking for every theory ID across all normalization stages. It helps you understand what happened to each theory at every step of the pipeline.

## Tracked Stages

1. **Stage 0**: Quality Filtering (`stage0_quality_filter.py`)
2. **Stage 1**: Fuzzy Matching (`stage1_fuzzy_matching.py`)
3. **Stage 1.5**: LLM-Assisted Mapping (`stage1_5_llm_assistant_mapping.py`)
4. **Stage 2**: Group Normalization (`stage2_group_normalization.py`)
5. **Stage 3**: Iterative Refinement (`stage3_iterative_refinement.py`)

## What It Tracks

For each theory ID, the tracker collects:

### Stage 0 (Quality Filtering)
- `theory_id`: Unique identifier
- `original_name`: Original theory name
- `paper_id`: Source paper
- `source`: Abstract or fulltext
- `confidence_level`: High/Medium/Low
- `stage0_status`: "passed_filter"

### Stage 1 (Fuzzy Matching)
- `stage1_status`: "matched" or "unmatched"
- `stage1_matched_name`: Ontology name (if matched)
- `stage1_match_score`: Similarity score

### Stage 1.5 (LLM-Assisted Mapping)
- `stage1_5_status`: "mapped" or "unmapped"
- `stage1_5_mapped_name`: LLM-mapped name (if mapped)

### Stage 2 (Group Normalization)
- `stage2_status`: "normalized" or "not_found"
- `stage2_normalized_name`: Standardized group name

### Stage 3 (Iterative Refinement)
- `stage3_status`: "matched_to_ontology", "refined_not_matched", or "processed"
- `stage3_final_name`: Final standardized name
- `stage3_ontology_match`: True/False (matched to ontology)

## Usage

### Generate Full Tracking Report

```bash
python src/tracking/theory_tracker.py
```

This will:
1. Load all stage outputs
2. Track each theory through all stages
3. Generate comprehensive report
4. Save JSON and CSV outputs

### Output Files

1. **JSON Report**: `output/theory_tracking_report.json`
   - Complete lineage data
   - Stage statistics
   - Metadata

2. **CSV Report**: `output/theory_tracking_report.csv`
   - Spreadsheet format for analysis
   - One row per theory
   - All stage information

### Test Tracker

```bash
python test_theory_tracker.py
```

## Report Structure

### JSON Report

```json
{
  "metadata": {
    "total_theories": 15000,
    "stages_tracked": ["stage0", "stage1", "stage1_5", "stage2", "stage3"]
  },
  "stage_statistics": {
    "stage0": {
      "passed_filter": 15000
    },
    "stage1": {
      "matched": 5000,
      "unmatched": 10000
    },
    "stage1_5": {
      "mapped": 8000,
      "unmapped": 7000
    },
    "stage2": {
      "normalized": 14500,
      "not_found": 500
    },
    "stage3": {
      "matched_to_ontology": 7000,
      "refined_not_matched": 7500
    }
  },
  "theory_lineage": {
    "theory_123": {
      "theory_id": "theory_123",
      "original_name": "Telomere Shortening Theory",
      "paper_id": "PMC123456",
      "confidence_level": "high",
      "stage0_status": "passed_filter",
      "stage1_status": "matched",
      "stage1_matched_name": "Telomere Theory",
      "stage1_match_score": 0.95,
      "stage1_5_status": "mapped",
      "stage1_5_mapped_name": "Telomere Theory",
      "stage2_status": "normalized",
      "stage2_normalized_name": "Telomere Theory",
      "stage3_status": "matched_to_ontology",
      "stage3_final_name": "Telomere Theory",
      "stage3_ontology_match": true
    },
    ...
  }
}
```

### CSV Report

| theory_id | original_name | paper_id | stage0_status | stage1_status | stage1_matched_name | stage2_normalized_name | stage3_final_name | stage3_ontology_match |
|-----------|---------------|----------|---------------|---------------|---------------------|------------------------|-------------------|----------------------|
| theory_123 | Telomere Shortening | PMC123 | passed_filter | matched | Telomere Theory | Telomere Theory | Telomere Theory | true |
| theory_456 | Novel Aging Theory | PMC456 | passed_filter | unmatched | null | Novel Aging Theory | Novel Aging Theory | false |

## Programmatic Usage

### Query Specific Theory

```python
from src.tracking.theory_tracker import TheoryTracker

tracker = TheoryTracker()
tracker.track_stage0()
tracker.track_stage1()
# ... track other stages

# Query specific theory
theory_data = tracker.query_theory('theory_123')
print(theory_data)
```

### Get Theories by Status

```python
# Get all theories matched in Stage 1
matched = tracker.get_theories_by_status('stage1', 'matched')

# Get all theories matched to ontology in Stage 3
ontology_matched = tracker.get_theories_by_status('stage3', 'matched_to_ontology')
```

### Get Final Mapping Summary

```python
summary = tracker.get_final_mapping_summary()
print(summary)
# {
#   'stage3_ontology_matched': 7000,
#   'stage3_refined': 7500,
#   'stage2_normalized': 0,
#   'stage1_5_mapped': 0,
#   'stage1_matched': 0,
#   'unmapped': 500
# }
```

## Example Output

```
================================================================================
THEORY TRACKER: Comprehensive Stage Tracking
================================================================================

================================================================================
TRACKING STAGE 0: Quality Filtering
================================================================================
📊 Found 15000 theories in Stage 0
✓ Tracked 15000 theories from Stage 0

================================================================================
TRACKING STAGE 1: Fuzzy Matching
================================================================================
📊 Matched: 5000, Unmatched: 10000
✓ Tracked Stage 1: 5000 matched, 10000 unmatched

================================================================================
TRACKING STAGE 1.5: LLM-Assisted Mapping
================================================================================
📊 Mapped: 8000, Unmapped: 7000
✓ Tracked Stage 1.5: 8000 mapped, 7000 unmapped

================================================================================
TRACKING STAGE 2: Group Normalization
================================================================================
📊 Found 14403 mappings in Stage 2
✓ Tracked Stage 2: 14500 normalized

================================================================================
TRACKING STAGE 3: Iterative Refinement
================================================================================
📊 Matched: 7000, Unmatched: 7500
✓ Tracked Stage 3: 7000 matched to ontology, 7500 refined but not matched

================================================================================
TRACKING SUMMARY
================================================================================

📊 Total theories tracked: 15000

🔹 Stage 0 (Quality Filtering):
   Passed filter: 15000

🔹 Stage 1 (Fuzzy Matching):
   Matched to ontology: 5000 (33.3%)
   Unmatched: 10000 (66.7%)

🔹 Stage 1.5 (LLM-Assisted Mapping):
   Mapped: 8000 (53.3%)
   Unmapped: 7000 (46.7%)

🔹 Stage 2 (Group Normalization):
   Normalized: 14500 (96.7%)
   Not found: 500 (3.3%)

🔹 Stage 3 (Iterative Refinement):
   Matched to ontology: 7000 (46.7%)
   Refined (not matched): 7500 (50.0%)

================================================================================
FINAL MAPPING SUMMARY
================================================================================

📊 Final Status Distribution:
   Stage 3 - Matched to ontology: 7000 (46.7%)
   Stage 3 - Refined (not matched): 7500 (50.0%)
   Stage 2 - Normalized only: 0 (0.0%)
   Stage 1.5 - Mapped only: 0 (0.0%)
   Stage 1 - Matched only: 0 (0.0%)
   Unmapped: 500 (3.3%)

✅ Tracking complete!

📁 Output files:
   - JSON: output/theory_tracking_report.json
   - CSV: output/theory_tracking_report.csv
```

## Use Cases

1. **Quality Assurance**: Verify each theory was processed correctly
2. **Debugging**: Find where theories were lost or misprocessed
3. **Analysis**: Understand pipeline effectiveness at each stage
4. **Reporting**: Generate statistics for papers/presentations
5. **Data Lineage**: Complete audit trail for each theory

## Benefits

- **Complete Visibility**: See every theory's journey through all stages
- **Easy Analysis**: CSV format for Excel/pandas analysis
- **Debugging**: Quickly identify processing issues
- **Statistics**: Comprehensive stage-by-stage metrics
- **Audit Trail**: Full lineage for reproducibility
