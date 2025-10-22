# Quick Start Guide

## Complete Pipeline Execution

### Step 1: Run All Stages

```bash
# Stage 0: Quality filtering
python src/normalization/stage0_quality_filter.py

# Stage 1: Fuzzy matching
python src/normalization/stage1_fuzzy_matching.py

# Stage 1.5: LLM mapping
python src/normalization/stage1_5_llm_assistant_mapping.py

# Stage 2: Group normalization
python src/normalization/stage2_group_normalization.py

# Stage 3: Iterative refinement
python src/normalization/stage3_iterative_refinement.py

# Generate tracking report
python src/tracking/theory_tracker.py
```

### Step 2: Check Results

```bash
# View tracking summary
cat output/theory_tracking_report.json | jq '.stage_statistics'

# View final mappings
cat output/stage3_refined_theories.json | jq '.metadata'

# Open CSV in Excel/LibreOffice
libreoffice output/theory_tracking_report.csv
```

## Resume from Checkpoint (Stage 2)

If Stage 2 is interrupted:

```bash
# Resume from last checkpoint
python src/normalization/stage2_group_normalization.py --resume
```

## Query Specific Theory

```python
from src.tracking.theory_tracker import TheoryTracker

tracker = TheoryTracker()
tracker.track_stage0()
tracker.track_stage1()
tracker.track_stage1_5()
tracker.track_stage2()
tracker.track_stage3()

# Query specific theory
theory = tracker.query_theory('T000123')
print(f"Original: {theory['original_name']}")
print(f"Final: {theory['stage3_final_name']}")
print(f"Ontology match: {theory['stage3_ontology_match']}")
```

## Key Files

| File | Description |
|------|-------------|
| `output/stage0_filtered_theories.json` | Filtered theories (15,000) |
| `output/stage1_fuzzy_matched.json` | Fuzzy matched (5,000 matched) |
| `output/stage1_5_llm_mapped.json` | LLM mapped (8,000 mapped) |
| `output/stage2_grouped_theories.json` | Normalized (14,403 mappings) |
| `output/stage3_refined_theories.json` | Final refined (7,000 ontology + 7,500 refined) |
| `output/theory_tracking_report.json` | Complete tracking report |
| `output/theory_tracking_report.csv` | Tracking report (spreadsheet) |

## Configuration

### Stage 2: Batch Size

Edit `src/normalization/stage2_group_normalization.py`:

```python
normalizer.run(
    max_batch_size=200  # Adjust batch size (100-400)
)
```

### Stage 3: Max Iterations

Edit `src/normalization/stage3_iterative_refinement.py`:

```python
refiner = Stage3IterativeRefinement(
    max_iterations=3  # Adjust iterations (1-5)
)
```

## Troubleshooting

### Stage 2 is slow
- Increase `max_batch_size` to 400
- Check API rate limits

### Stage 2 interrupted
- Use `--resume` flag to continue from checkpoint

### Missing theories in tracker
- Check that all stage outputs exist
- Verify theory_id consistency across stages

### High costs
- Reduce batch sizes
- Reduce Stage 3 iterations
- Use cheaper model (gpt-4.1-nano)

## Documentation

- **Pipeline Overview**: `PIPELINE_OVERVIEW.md`
- **Stage 2 Details**: `STAGE2_USAGE.md`
- **Stage 3 Details**: `STAGE3_USAGE.md`
- **Tracker Details**: `THEORY_TRACKER_USAGE.md`
