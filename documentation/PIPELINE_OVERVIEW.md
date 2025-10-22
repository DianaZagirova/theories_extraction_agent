# Complete Normalization Pipeline Overview

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
│  theories_abstract_per_paper.json + theories_per_paper.json         │
│                    (~20,000 raw theories)                            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0: Quality Filtering                                         │
│  File: src/normalization/stage0_quality_filter.py                   │
│  ─────────────────────────────────────────────────────────────────  │
│  • Filter by confidence level (high/medium/low)                     │
│  • Optional LLM validation for medium confidence                    │
│  • Remove duplicates and low-quality theories                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/stage0_filtered_theories.json (~15,000 theories)    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Fuzzy Matching                                            │
│  File: src/normalization/stage1_fuzzy_matching.py                   │
│  ─────────────────────────────────────────────────────────────────  │
│  • Match theories to ontology using fuzzy string matching           │
│  • Use RapidFuzz for similarity scoring                             │
│  • Threshold: 80% similarity                                        │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/stage1_fuzzy_matched.json                           │
│    • Matched: ~5,000 theories (33%)                                 │
│    • Unmatched: ~10,000 theories (67%)                              │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1.5: LLM-Assisted Mapping                                    │
│  File: src/normalization/stage1_5_llm_assistant_mapping.py          │
│  ─────────────────────────────────────────────────────────────────  │
│  • Use LLM to map unmatched theories to ontology                    │
│  • Batch processing (100 theories per batch)                        │
│  • Context-aware mapping with ontology reference                    │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/stage1_5_llm_mapped.json                            │
│    • Mapped: ~8,000 theories (53%)                                  │
│    • Unmapped: ~7,000 theories (47%)                                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Group Normalization                                       │
│  File: src/normalization/stage2_group_normalization.py              │
│  ─────────────────────────────────────────────────────────────────  │
│  • Combine mapped + unmapped theories (~14,403 unique names)        │
│  • Group similar theories and create standardized names             │
│  • Batch processing with smart batching (200-400 per batch)         │
│  • Progress tracking with checkpoints every 5 batches               │
│  • Resume from checkpoint support (--resume flag)                   │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/stage2_grouped_theories.json                        │
│    • Format: {"initial_name": "standardized_name"}                  │
│    • ~14,403 mappings → ~1,500-2,500 unique standardized names      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Iterative Refinement                                      │
│  File: src/normalization/stage3_iterative_refinement.py             │
│  ─────────────────────────────────────────────────────────────────  │
│  • Match standardized names to ontology (case-insensitive)          │
│  • Fix matched names to ontology (final mappings)                   │
│  • Re-process unmatched names through LLM                           │
│  • Iterate until convergence (max 3 iterations)                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/stage3_refined_theories.json                        │
│    • Matched to ontology: ~7,000 (46-50%)                           │
│    • Refined (not matched): ~7,500 (50-54%)                         │
│    • Iteration outputs: _iter1.json, _iter2.json, _iter3.json      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  THEORY TRACKER: Comprehensive Tracking                             │
│  File: src/tracking/theory_tracker.py                               │
│  ─────────────────────────────────────────────────────────────────  │
│  • Track each theory_id through all stages                          │
│  • Collect metadata at each step                                    │
│  • Generate lineage report (JSON + CSV)                             │
│  • Provide stage-by-stage statistics                                │
│  ─────────────────────────────────────────────────────────────────  │
│  Output: output/theory_tracking_report.json + .csv                  │
│    • Complete lineage for all ~15,000 theories                      │
│    • Status at each stage                                           │
│    • Final mapping summary                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage Summary

| Stage | Purpose | Input | Output | Match Rate |
|-------|---------|-------|--------|------------|
| **0** | Quality Filter | ~20,000 raw | ~15,000 filtered | 75% pass |
| **1** | Fuzzy Match | 15,000 theories | 5,000 matched | 33% match |
| **1.5** | LLM Map | 10,000 unmatched | 8,000 mapped | 53% map |
| **2** | Normalize | 14,403 unique | 1,500-2,500 groups | 100% process |
| **3** | Refine | 14,403 mappings | 7,000 ontology + 7,500 refined | 46% ontology |

## Key Features

### Stage 0
- ✅ Confidence-based filtering
- ✅ Optional LLM validation
- ✅ Duplicate removal

### Stage 1
- ✅ Fast fuzzy matching
- ✅ RapidFuzz algorithm
- ✅ Configurable threshold

### Stage 1.5
- ✅ LLM-powered mapping
- ✅ Batch processing
- ✅ Ontology-aware

### Stage 2
- ✅ Smart batching (combines small groups)
- ✅ Progress bar with tqdm
- ✅ Checkpoint saving (batch 1, then every 5)
- ✅ Resume from checkpoint (--resume)
- ✅ Metadata tracking per batch

### Stage 3
- ✅ Ontology alignment
- ✅ Iterative refinement
- ✅ Convergence tracking
- ✅ Iteration outputs

### Theory Tracker
- ✅ Complete lineage tracking
- ✅ Stage-by-stage metadata
- ✅ JSON + CSV reports
- ✅ Query capabilities

## Running the Pipeline

### Full Pipeline

```bash
# Stage 0: Quality filtering
python src/normalization/stage0_quality_filter.py

# Stage 1: Fuzzy matching
python src/normalization/stage1_fuzzy_matching.py

# Stage 1.5: LLM mapping
python src/normalization/stage1_5_llm_assistant_mapping.py

# Stage 2: Group normalization
python src/normalization/stage2_group_normalization.py

# Stage 2: Resume from checkpoint (if interrupted)
python src/normalization/stage2_group_normalization.py --resume

# Stage 3: Iterative refinement
python src/normalization/stage3_iterative_refinement.py

# Generate tracking report
python src/tracking/theory_tracker.py
```

### Quick Test

```bash
# Test individual stages
python test_stage2_metadata.py
python test_stage3.py
python test_theory_tracker.py
```

## Output Files

```
output/
├── stage0_filtered_theories.json          # Stage 0 output
├── stage1_fuzzy_matched.json              # Stage 1 output
├── stage1_5_llm_mapped.json               # Stage 1.5 output
├── stage2_grouped_theories.json           # Stage 2 output (with checkpoints)
├── stage3_refined_theories.json           # Stage 3 final output
├── stage3_refined_theories_iter1.json     # Stage 3 iteration 1
├── stage3_refined_theories_iter2.json     # Stage 3 iteration 2
├── stage3_refined_theories_iter3.json     # Stage 3 iteration 3
├── theory_tracking_report.json            # Tracker JSON report
└── theory_tracking_report.csv             # Tracker CSV report
```

## Cost Estimates

| Stage | API Calls | Est. Cost | Time |
|-------|-----------|-----------|------|
| Stage 0 (with validation) | ~5,000 | $0.50 | 30 min |
| Stage 1 | 0 (local) | $0.00 | 2 min |
| Stage 1.5 | ~100 batches | $1.50 | 20 min |
| Stage 2 | ~46 batches | $2.00 | 45 min |
| Stage 3 | ~150 batches (3 iter) | $4.00 | 60 min |
| **Total** | ~300 batches | **~$8.00** | **~2.5 hrs** |

## Documentation

- **Stage 2**: `STAGE2_USAGE.md` - Group normalization details
- **Stage 3**: `STAGE3_USAGE.md` - Iterative refinement details
- **Tracker**: `THEORY_TRACKER_USAGE.md` - Tracking system details
- **This file**: `PIPELINE_OVERVIEW.md` - Complete pipeline overview
