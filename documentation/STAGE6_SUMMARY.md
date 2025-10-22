# Stage 6: Cluster Separation - Complete Summary

## ✅ Implementation Complete

Stage 6 separates overly general theory clusters (>40 papers) into specific subclusters based on mechanistic themes.

## Key Features

### 1. Data Isolation Guarantee
- **Each LLM prompt contains theories from ONLY ONE cluster**
- Theories from different clusters are NEVER mixed
- Enforced through sequential processing and runtime validation

### 2. Simplified Output Format
LLM returns individual theory assignments:
```json
{
  "theory_assignments": [
    {"theory_id": "T000001", "subcluster_name": "ROS-Induced Cellular Senescence Theory"},
    {"theory_id": "T000002", "subcluster_name": "ROS-Induced Cellular Senescence Theory"},
    {"theory_id": "T000010", "subcluster_name": "Telomere-Associated Cellular Senescence Theory"}
  ]
}
```

### 3. Smart Batching
- Max 30 theories per prompt
- Intelligent splitting: 35 → [18, 17] not [30, 5]
- All batches from same cluster only

### 4. Batch-Level Retry
- Failed batches retried entirely (not individual theories)
- Up to 3 retry attempts per batch
- Clear logging of retry attempts

### 5. Singleton Warning Fallback
If a batch fails after all retries:
- Theories assigned to **original cluster name**
- Marked with `status: "singleton_warning"`
- Includes `warning_reason` for tracking
- **No theories are lost**

Example:
```json
{
  "subcluster_name": "Cellular Senescence Theory",
  "theory_ids": ["T000040", "T000075", ...],
  "theory_count": 30,
  "status": "singleton_warning",
  "warning_reason": "Failed to separate after 3 retries"
}
```

### 6. Automatic Consolidation
- Combines Stage 5 + Stage 6 results automatically
- Maintains full tracking with `stage5_parent` and `was_separated_in_stage6`
- Creates `output/stage6_consolidated_final_theories.json`

### 7. Validation
- Minimum subcluster size: 5 theories (configurable)
- All theory IDs must be assigned
- No duplicates allowed
- Runtime assertions for data integrity

## Files Created

1. **`src/normalization/stage6_cluster_separation.py`** - Main implementation (857 lines)
2. **`scripts/run_stage6_separation.py`** - Runner with test mode
3. **`scripts/test_scripts/test_stage6_separation.py`** - Unit tests
4. **`scripts/consolidate_stage6_results.py`** - Standalone consolidation
5. **`scripts/analyze_stage6_results.py`** - Results analysis
6. **`STAGE6_DATA_FLOW.md`** - Data flow and isolation guarantees
7. **`STAGE6_OUTPUT_FORMAT.md`** - Output format documentation

## Usage

### Test Mode (Recommended First)
```bash
# Test on 1 cluster
python scripts/run_stage6_separation.py --test --limit 1

# Test on top 3 clusters
python scripts/run_stage6_separation.py --test --limit 3
```

### Full Run
```bash
# Process all 41 clusters with >40 papers
python scripts/run_stage6_separation.py
```

**Expected:**
- Processing time: ~20-30 minutes
- Cost: ~$2-3 (using gpt-4o-mini)
- Theories processed: ~8,500
- Subclusters created: ~150-200

### Output Files
1. **`output/stage6_separated_clusters.json`** - Separation details
2. **`output/stage6_consolidated_final_theories.json`** - **USE THIS** for downstream processing

### Analyze Results
```bash
python scripts/analyze_stage6_results.py
```

## Statistics Tracked

- Total clusters processed
- Batches processed
- Subclusters created
- Successful/failed separations
- Retry attempts
- **Batches with singleton warning**
- **Theories with singleton warning**
- Token usage and cost

## Example Output Structure

```json
{
  "metadata": {
    "stage": "stage6_cluster_separation",
    "paper_threshold": 40,
    "min_subcluster_size": 5,
    "timestamp": "2025-10-21 22:00:00"
  },
  "separated_clusters": [
    {
      "original_cluster_name": "Cellular Senescence Theory",
      "original_total_papers": 1285,
      "separation_successful": true,
      "subclusters": [
        {
          "subcluster_name": "ROS-Induced Cellular Senescence Theory",
          "theory_ids": ["T000040", ...],
          "theory_count": 245
        },
        {
          "subcluster_name": "Telomere-Associated Cellular Senescence Theory",
          "theory_ids": ["T000120", ...],
          "theory_count": 312
        },
        {
          "subcluster_name": "Cellular Senescence Theory",
          "theory_ids": ["T001234", ...],
          "theory_count": 28,
          "status": "singleton_warning",
          "warning_reason": "Failed to separate after 3 retries"
        }
      ]
    }
  ],
  "statistics": {
    "clusters_to_separate": 41,
    "batches_processed": 285,
    "subclusters_created": 164,
    "batches_with_singleton_warning": 3,
    "theories_with_singleton_warning": 87,
    "total_cost": 2.45
  }
}
```

## Consolidated Output Structure

```json
{
  "metadata": {
    "source_stage5": "output/stage5_consolidated_final_theories.json",
    "source_stage6": "output/stage6_separated_clusters.json",
    "total_theory_ids": 15940,
    "unique_final_names": 1230,
    "separated_in_stage6": 159,
    "unchanged_from_stage5": 1071
  },
  "final_name_summary": [
    {
      "final_name": "ROS-Induced Cellular Senescence Theory",
      "total_papers": 245,
      "theory_ids": ["T000040", ...],
      "stage5_parent": "Cellular Senescence Theory",
      "was_separated_in_stage6": true
    },
    {
      "final_name": "Immunological Theory",
      "total_papers": 220,
      "theory_ids": ["T005678", ...],
      "stage5_parent": "Immunological Theory",
      "was_separated_in_stage6": false
    }
  ]
}
```

## Key Guarantees

✅ **No data loss** - All theories are assigned (even failed batches)  
✅ **No cross-contamination** - Each prompt contains theories from one cluster only  
✅ **Full tracking** - Can trace theory from stage0 → stage5 → stage6  
✅ **Failure handling** - Failed batches marked with singleton warning  
✅ **Validation** - Multiple checks ensure data integrity  
✅ **Reproducibility** - Original assignments preserved in output  

## Next Steps

After running Stage 6:

1. **Review singleton warnings** (if any)
   - Check `status: "singleton_warning"` entries
   - Manually review if needed
   - Consider re-running with different parameters

2. **Use consolidated output**
   - `output/stage6_consolidated_final_theories.json`
   - Contains all theories with final names
   - Ready for downstream analysis

3. **Analyze results**
   - Run `scripts/analyze_stage6_results.py`
   - Review size distributions
   - Check separation quality

## Configuration Options

```bash
# Custom paper threshold
python scripts/run_stage6_separation.py --threshold 100

# Custom min subcluster size
python scripts/run_stage6_separation.py --min-size 6

# Custom batch size
python scripts/run_stage6_separation.py --max-batch 20
```

## Troubleshooting

**High singleton warning rate?**
- Try increasing `--max-batch` (more context for LLM)
- Try decreasing `--min-size` (easier to create valid subclusters)
- Check if cluster is truly separable

**Out of memory?**
- Decrease `--max-batch`
- Process fewer clusters at once with `--test --limit N`

**High cost?**
- Use test mode first
- Adjust `--threshold` to process fewer clusters
- Consider using smaller model (though may reduce quality)
