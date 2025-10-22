# Stage 5 Output Consolidation

## Overview

The consolidation script creates a final mapping of all theories to their normalized names, tracking theory IDs and maintaining the complete lineage from Stage 4 through Stage 5.

## Input Files

1. **`data/clustering_data/clusters_with_paper_counts.json`**
   - Created by: `scripts/clusterization/cluster_stage4_output.py`
   - Contains: Clustered theories with paper counts
   - Used as: Input to Stage 5

2. **`output/stage5_cluster_refined_theories.json`**
   - Created by: `src/normalization/stage5_cluster_refinement.py`
   - Contains: Normalized names for rare theories (<6 papers)
   - Used as: Stage 5 output

3. **`data/clustering_data/filtered_paper_data.json`**
   - Created by: `scripts/clusterization/cluster_stage4_output.py`
   - Contains: Filtered theory data (max paper_focus per DOI)
   - Used for: Mapping theory names to theory IDs (one ID per theory name)

## Processing Logic

### For Each Theory:

#### Reference Theories (paper_count ≥ 6)
- **Final name**: Original `theory_name` (unchanged)
- **Strategy**: `reference`
- **Reasoning**: "Reference theory with ≥6 papers - retained original name"

#### Rare Theories (paper_count < 6)
- **Final name**: `normalized_name` from Stage 5 output
- **Strategy**: From Stage 5 (`assign_common`, `map`, or `retain`)
- **Reasoning**: From Stage 5 LLM decision

#### Missing from Stage 5
- **Final name**: Original `theory_name` (fallback)
- **Strategy**: `retain`
- **Reasoning**: "Missing from Stage 5 output - retained original name"

## Output Structure

### `output/stage5_consolidated_final_theories.json`

```json
{
  "metadata": {
    "source_clusters": "path/to/clusters.json",
    "source_stage5": "path/to/stage5_output.json",
    "source_filtered_data": "path/to/filtered_paper_data.json",
    "total_theories": 2763,
    "reference_theories": 240,
    "normalized_theories": 2523,
    "missing_from_stage5": 0,
    "unique_final_names": 450,
    "stage5_metadata": {...}
  },
  "final_name_summary": [
    {
      "final_name": "Mitochondrial Dysfunction Theory",
      "original_names_count": 35,
      "original_names": ["Mitochondrial Free Radical Theory", ...],
      "total_papers": 250,
      "theory_ids_count": 35,
      "theory_ids": ["theory_001", "theory_045", ...],
      "dois": ["10.1234/journal.2020.001", "10.1234/journal.2021.045", ...],
      "strategies": ["assign_common", "reference"]
    },
    ...
  ],
  "theory_mapping": [
    {
      "theory_name": "Mitochondrial Free Radical Theory",
      "final_name": "Mitochondrial Dysfunction Theory",
      "paper_count": 45,
      "cluster_id": "46",
      "strategy": "reference",
      "confidence": 1.0,
      "reasoning": "Reference theory with ≥6 papers - retained original name",
      "theory_id": "theory_001"
    },
    ...
  ]
}
```

## Usage

### Basic Usage

```bash
python scripts/consolidate_stage5_output.py
```

### Custom Paths

```bash
python scripts/consolidate_stage5_output.py \
  --clusters data/clustering_data/clusters_with_paper_counts.json \
  --stage5-output output/stage5_cluster_refined_theories.json \
  --filtered-data data/clustering_data/filtered_paper_data.json \
  --output output/stage5_consolidated_final_theories.json
```

## Output Features

### 1. Theory Mapping
- Complete mapping of every theory to its final normalized name
- Includes **one theory ID per theory** (from filtered data with max paper_focus per DOI)
- Tracks strategy and confidence for each normalization decision

### 2. Final Name Summary
- Groups theories by final name
- Shows compression ratio (original names → final names)
- Collects all theory IDs for theories grouped under each final name
- Counts total papers and theory IDs per final name
- Sorted by total paper count (most cited first)

### 3. Statistics
- Total theories processed
- Reference vs normalized theory counts
- Missing theories detection
- Compression metrics

## Validation

The script validates:
- ✅ All theories from clusters are accounted for
- ✅ Reference theories (≥6 papers) retain original names
- ✅ Rare theories (<6 papers) get normalized names from Stage 5
- ✅ Missing theories are flagged and handled with fallback
- ✅ Theory IDs are correctly mapped from filtered data (one ID per theory with max paper_focus)

## Example Output

```
================================================================================
CONSOLIDATION SUMMARY
================================================================================
Total theories processed: 2763
  - Reference theories (≥6 papers): 240
  - Normalized theories (<6 papers): 2523
  - Missing from Stage 5: 0

Unique final names: 450
  - Reduction: 2763 → 450
  - Compression ratio: 16.29%

✅ Consolidated output saved to: output/stage5_consolidated_final_theories.json

================================================================================
TOP 10 FINAL NAMES BY PAPER COUNT
================================================================================
1. Mitochondrial Dysfunction Theory
   Papers: 250 | Original names: 35 | Theory IDs: 180
2. Oxidative Stress Theory
   Papers: 180 | Original names: 28 | Theory IDs: 145
3. DNA Damage Theory
   Papers: 150 | Original names: 22 | Theory IDs: 120
...
```

## Integration with Pipeline

```
Stage 4 (theory_validation.py)
    ↓
theory_tracking_report.json
    ↓
cluster_stage4_output.py
    ↓
clusters_with_paper_counts.json
    ↓
Stage 5 (stage5_cluster_refinement.py)
    ↓
stage5_cluster_refined_theories.json
    ↓
consolidate_stage5_output.py  ← YOU ARE HERE
    ↓
stage5_consolidated_final_theories.json
```

## Next Steps

After consolidation, you can:
1. Use `final_name_summary` for high-level theory landscape analysis
2. Use `theory_mapping` to trace individual theories back to papers
3. Export to database or other formats for downstream analysis
4. Generate visualizations of theory clustering and normalization
