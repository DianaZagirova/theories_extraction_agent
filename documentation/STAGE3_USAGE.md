# Stage 3: Iterative Refinement with Ontology Alignment

## Overview

Stage 3 takes the output from Stage 2 and iteratively refines the mappings by:
1. **Matching to ontology**: Check if mapped names match the original ontology theories
2. **Fixing matches**: Lock in mappings that match ontology (these are final)
3. **Re-processing unmatched**: Send unmatched names back through LLM for refinement
4. **Iterating**: Repeat until convergence or max iterations reached

## Input

- **Stage 2 output**: `output/stage2_grouped_theories.json`
  - Contains initial mappings: `initial_name → mapped_name`
- **Ontology**: `ontology/groups_ontology_alliases.json`
  - Reference theories to align with

## Output

### Final Output
- **File**: `output/stage3_refined_theories.json`
- **Structure**:
```json
{
  "metadata": {
    "stage": "stage3_iterative_refinement",
    "status": "complete",
    "total_iterations": 3,
    "matched_to_ontology": 5000,
    "unmatched": 2000,
    "match_rate": 71.4,
    "unique_final_names": 1500
  },
  "matched_mappings": {
    "initial_name_1": "Ontology Theory Name",
    ...
  },
  "unmatched_mappings": {
    "initial_name_2": "Standardized Name",
    ...
  },
  "all_mappings": {
    "initial_name_1": "Ontology Theory Name",
    "initial_name_2": "Standardized Name",
    ...
  }
}
```

### Iteration Outputs
- **Files**: `output/stage3_refined_theories_iter1.json`, `iter2.json`, etc.
- Each iteration saves intermediate results

## How It Works

### Iteration Flow

```
Stage 2 Output (14,403 mappings)
         ↓
┌────────────────────────────────┐
│   Iteration 1                  │
│   1. Match to ontology         │
│   2. Fix matched (e.g., 5,000) │
│   3. Re-process unmatched      │
│      (9,403 names)             │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│   Iteration 2                  │
│   1. Match to ontology         │
│   2. Fix matched (e.g., +2,000)│
│   3. Re-process unmatched      │
│      (7,403 names)             │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│   Iteration 3                  │
│   1. Match to ontology         │
│   2. Fix matched (e.g., +500)  │
│   3. Remaining unmatched       │
│      (6,903 names)             │
└────────────────────────────────┘
         ↓
Final Output: 7,500 matched + 6,903 unmatched
```

### Matching Logic

1. **Exact match** (case-insensitive):
   - `"telomere theory"` → `"Telomere Theory"` (ontology)

2. **Substring match** (with length check):
   - `"Telomere Shortening"` → `"Telomere Theory"` (ontology)
   - Only if length difference < 10 characters

3. **No match**:
   - Send back to LLM for refinement

## Usage

### Basic Usage

```bash
# Run Stage 3 with default settings
python src/normalization/stage3_iterative_refinement.py
```

### Test Stage 3

```bash
# Test matching logic without full run
python test_stage3.py
```

### Configuration

Edit `src/normalization/stage3_iterative_refinement.py`:

```python
refiner = Stage3IterativeRefinement(
    stage2_path='output/stage2_grouped_theories.json',
    ontology_path='ontology/groups_ontology_alliases.json',
    max_iterations=3  # Adjust max iterations
)

refiner.run(
    output_path='output/stage3_refined_theories.json',
    batch_size=200  # Batch size for re-processing
)
```

## Expected Results

### Iteration 1
- **Input**: 14,403 mappings from Stage 2
- **Matched**: ~5,000-7,000 (35-50%)
- **Unmatched**: ~7,000-9,000 (50-65%)

### Iteration 2
- **Input**: 7,000-9,000 unmatched from Iteration 1
- **Matched**: +1,000-2,000 (additional matches)
- **Unmatched**: ~5,000-7,000 (remaining)

### Iteration 3
- **Input**: 5,000-7,000 unmatched from Iteration 2
- **Matched**: +500-1,000 (additional matches)
- **Unmatched**: ~4,000-6,000 (final unmatched)

### Final
- **Total matched**: 6,500-10,000 (45-70%)
- **Total unmatched**: 4,000-8,000 (30-55%)
- **Unique final names**: ~1,500-2,500

## Statistics

- **Processing time**: ~30-60 minutes (depends on unmatched count)
- **Cost**: ~$1-2 USD per iteration
- **Total cost**: ~$3-6 USD for 3 iterations

## Benefits

1. **Ontology alignment**: Ensures consistency with reference theories
2. **Iterative refinement**: Multiple passes improve quality
3. **Convergence tracking**: See improvement across iterations
4. **Flexible**: Adjust max iterations based on needs

## Next Steps

After Stage 3, you have:
- **Matched mappings**: Aligned with ontology (high confidence)
- **Unmatched mappings**: Standardized but not in ontology (may be novel theories)

You can:
1. Use matched mappings directly
2. Review unmatched mappings manually
3. Add high-quality unmatched names to ontology
4. Run additional iterations if needed
