# Stage 6: Selective Cluster Rerun

## Problem

After running Stage 6, some clusters may have issues (missing theories, high singleton rate, etc.). Rerunning the entire Stage 6 process wastes time and tokens on clusters that are already correct.

## Solution: Selective Reprocessing

Process only clusters with issues, keeping successful results.

## Workflow

### Step 1: Analyze Checkpoints

```bash
python scripts/analyze_stage6_checkpoints.py
```

**What it does:**
- Reads all `stage6_checkpoint_*.json` files
- Compares with stage5 data to find issues
- Identifies clusters with:
  - Missing theories
  - Extra theories  
  - High singleton warning rate (>30%)
  - Theory count mismatches

**Output:**
```
================================================================================
STAGE 6 CHECKPOINT ANALYSIS
================================================================================

📂 Loading stage5: output/stage5_consolidated_final_theories.json
✓ Loaded 247 clusters from stage5

🔍 Analyzing 41 checkpoint files...

================================================================================
ANALYSIS RESULTS
================================================================================

✅ 38 clusters OK (no issues)
⚠️  3 clusters WITH ISSUES

================================================================================
🚨 CRITICAL ISSUES (2 clusters)
================================================================================

📌 Sirtuin Regulation Theory
   Expected: 45 theories
   Processed: 44 theories
   Missing: 1 theories
   Issues:
      - 1 theories missing from checkpoint
      - Theory count mismatch: expected 45, got 44
   Sample missing IDs: T017865

📌 Caloric Restriction Mimetics Theory
   Expected: 38 theories
   Processed: 36 theories
   Missing: 2 theories
   Issues:
      - 2 theories missing from checkpoint
      - Theory count mismatch: expected 38, got 36
   Sample missing IDs: T014954, T014970

================================================================================
⚠️  WARNINGS (1 clusters)
================================================================================

📌 Cellular Senescence Theory
   Theories: 1,234
   Singleton warnings: 412 theories (33.4%)
   Issues:
      - 33.4% theories in singleton warnings (412/1234)

================================================================================
CLUSTERS TO RERUN
================================================================================

3 clusters need reprocessing:

  1. 🚨 Sirtuin Regulation Theory
  2. 🚨 Caloric Restriction Mimetics Theory
  3. ⚠️  Cellular Senescence Theory

💾 Saved rerun list to: output/stage6_clusters_to_rerun.json
```

### Step 2: Rerun Problem Clusters

```bash
python scripts/rerun_stage6_clusters.py
```

**What it does:**
- Loads `output/stage6_clusters_to_rerun.json`
- Filters stage5 data to only include problem clusters
- Runs Stage 6 separation on those clusters only
- Merges new results with existing successful results

**Output:**
```
================================================================================
STAGE 6: SELECTIVE CLUSTER REPROCESSING
================================================================================
Configuration:
  Paper threshold: >40
  Min subcluster size: 2
  Max theories per batch: 26
  Clusters to rerun: 3
  Output: output/stage6_separated_clusters.json
================================================================================

📋 Clusters to rerun:
  1. Sirtuin Regulation Theory
  2. Caloric Restriction Mimetics Theory
  3. Cellular Senescence Theory

📂 Loading existing results from output/stage6_separated_clusters.json
✓ Found 41 existing cluster results
✓ Keeping 38 existing results
✓ Will replace 3 cluster results

🔧 Filtering stage5 data to only include clusters to rerun...
✓ Filtered to 3 clusters

🚀 Starting reprocessing...
[... processes only 3 clusters ...]

🔄 Merging with existing results...
✓ Merged results: 41 total clusters
  - 38 kept from previous run
  - 3 newly reprocessed

✅ Selective reprocessing complete!
   Updated output: output/stage6_separated_clusters.json
```

### Step 3: Regenerate Consolidated Output

The consolidation happens automatically when Stage 6 completes, or you can run it manually:

```bash
# Consolidation happens automatically in separator.run()
# Or check the consolidated output
python scripts/validate_stage6_consolidation.py
```

## Manual Cluster Selection

You can also specify clusters manually:

```bash
python scripts/rerun_stage6_clusters.py \
    --clusters "Sirtuin Regulation Theory" "Caloric Restriction Mimetics Theory"
```

## Benefits

### Time Savings

**Full rerun:**
```
41 clusters × 50 batches/cluster × 2 min/batch = 4,100 minutes ≈ 68 hours
```

**Selective rerun:**
```
3 clusters × 50 batches/cluster × 2 min/batch = 300 minutes ≈ 5 hours
```

**Savings: 63 hours (93% reduction)**

### Token Savings

**Full rerun:**
```
41 clusters × 1,234 theories/cluster × 2,000 tokens/theory = 101M tokens
Cost: ~$100
```

**Selective rerun:**
```
3 clusters × 1,234 theories/cluster × 2,000 tokens/theory = 7.4M tokens
Cost: ~$7.40
```

**Savings: $92.60 (93% reduction)**

## Issue Types

### Critical Issues (Must Rerun)

1. **Missing Theories**
   - Theories in stage5 but not in checkpoint
   - Causes: Carry-forward bugs, batch failures
   - Impact: Data loss

2. **Extra Theories**
   - Theories in checkpoint but not in stage5
   - Causes: Duplicate processing, data corruption
   - Impact: Invalid results

3. **Theory Count Mismatch**
   - Total theories doesn't match stage5
   - Causes: Any of the above
   - Impact: Incomplete processing

### Warnings (Optional Rerun)

1. **High Singleton Warning Rate**
   - >30% of theories in singleton warnings
   - Causes: Poor separation, overly specific subclusters
   - Impact: Less useful separation
   - Decision: Rerun if you want better quality

## Files Created

### Analysis Output

**`output/stage6_clusters_to_rerun.json`**
```json
{
  "total_clusters_analyzed": 41,
  "clusters_ok": 38,
  "clusters_with_issues": 3,
  "critical_issues": 2,
  "warnings": 1,
  "clusters_to_rerun": [
    "Sirtuin Regulation Theory",
    "Caloric Restriction Mimetics Theory",
    "Cellular Senescence Theory"
  ],
  "detailed_issues": [
    {
      "cluster_name": "Sirtuin Regulation Theory",
      "severity": "critical",
      "expected_theories": 45,
      "processed_theories": 44,
      "missing_theories": 1,
      "missing_theory_ids": ["T017865"]
    }
  ]
}
```

## Advanced Options

### Custom Thresholds

```bash
# More lenient singleton warning threshold
python scripts/analyze_stage6_checkpoints.py --singleton-threshold 0.5

# Different batch size for rerun
python scripts/rerun_stage6_clusters.py --max-batch 30
```

### Dry Run

```bash
# See what would be rerun without actually doing it
python scripts/analyze_stage6_checkpoints.py --dry-run
```

## Troubleshooting

### "No checkpoint files found"

```bash
# Check if Stage 6 has been run
ls output/stage6_checkpoint_*.json

# If empty, run Stage 6 first
python scripts/run_stage6_separation.py
```

### "No matching stage5 cluster found"

Checkpoint filename doesn't match stage5 cluster name. This can happen if:
- Cluster names have special characters
- Stage5 data has changed

**Fix:** Delete checkpoint and rerun that cluster.

### "All clusters OK but validation fails"

The checkpoint analysis only checks individual clusters. Run full validation:

```bash
python scripts/validate_stage6_consolidation.py
```

## Complete Example

```bash
# 1. Run initial Stage 6
python scripts/run_stage6_separation.py

# 2. Validate results
python scripts/validate_stage6_consolidation.py
# ❌ FAIL: 3 theories missing

# 3. Analyze checkpoints to find problem clusters
python scripts/analyze_stage6_checkpoints.py
# Found 3 clusters with issues

# 4. Rerun only problem clusters
python scripts/rerun_stage6_clusters.py
# Reprocessed 3 clusters, merged with 38 existing

# 5. Validate again
python scripts/validate_stage6_consolidation.py
# ✅ PASS: All theories accounted for!
```

## Summary

✅ **Saves time** - Only reprocess clusters with issues  
✅ **Saves tokens** - Don't rerun successful clusters  
✅ **Preserves work** - Keeps successful results  
✅ **Targeted fixes** - Focus on actual problems  
✅ **Automatic merging** - Seamlessly combines old and new results  
✅ **Full validation** - Ensures no data loss  

This approach is especially valuable for large datasets where Stage 6 may take many hours and cost significant tokens.
