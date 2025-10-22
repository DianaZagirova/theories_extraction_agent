# Theory Normalization Pipeline - Implementation Guide

## Overview

Complete implementation of the theory normalization pipeline that processes ~14,000 aging theories into ~300-350 normalized theories with parent-child hierarchy.

## Pipeline Stages

### Stage 0: Quality Filtering
- **File:** `src/normalization/stage0_quality_filter.py`
- **Function:** Filters theories by confidence, removes false positives
- **Input:** `theories_per_paper.json`
- **Output:** `output/stage0_filtered_theories.json`

### Stage 1: Multi-Dimensional Embedding
- **File:** `src/normalization/stage1_embedding.py`
- **Function:** Generates embeddings at 3 levels (name, semantic, detailed) + concept features
- **Input:** Stage 0 output
- **Output:** `output/stage1_embeddings.json`

### Stage 2: Hierarchical Clustering
- **File:** `src/normalization/stage2_clustering.py`
- **Function:** Creates 3-level hierarchy (families → parents → children)
- **Input:** Stage 1 output
- **Output:** `output/stage2_clusters.json`

### Stage 3: LLM Validation
- **File:** `src/normalization/stage3_llm_validation.py`
- **Function:** Validates clusters, preserves distinctions, generates canonical names
- **Input:** Stage 2 output
- **Output:** `output/stage3_validated.json`

### Stage 4: Ontology Matching
- **File:** `src/normalization/stage4_ontology_matching.py`
- **Function:** Matches to known theories in `initial_ontology.json`
- **Input:** Stage 3 output
- **Output:** `output/stage4_ontology_matched.json`

## Quick Start

### 1. Run Prototype (Recommended First Step)

Test on 200 theories to validate approach:

```bash
python3 run_normalization_prototype.py --subset-size 200
```

This will:
- Create a subset of 200 theories
- Run all 5 stages
- Generate a summary report
- Output to `output/prototype/`

### 2. Tune Thresholds

Find optimal clustering thresholds:

```bash
# Quick test (3 configurations)
python3 tune_thresholds.py --quick --subset-size 200

# Full grid search (27 configurations)
python3 tune_thresholds.py --subset-size 200
```

Results saved to: `output/threshold_tuning_results.json`

### 3. Run Full Pipeline

After finding optimal thresholds, run on all theories:

```bash
# Stage 0: Quality filtering
python3 src/normalization/stage0_quality_filter.py

# Stage 1: Embeddings
python3 src/normalization/stage1_embedding.py

# Stage 2: Clustering (adjust thresholds as needed)
python3 src/normalization/stage2_clustering.py

# Stage 3: LLM validation
python3 src/normalization/stage3_llm_validation.py

# Stage 4: Ontology matching
python3 src/normalization/stage4_ontology_matching.py
```

## Configuration

### Clustering Thresholds

Adjust in `run_normalization_prototype.py` or pass as arguments:

```bash
python3 run_normalization_prototype.py \
    --family-threshold 0.7 \
    --parent-threshold 0.5 \
    --child-threshold 0.4 \
    --subset-size 200
```

**Threshold Guidelines:**
- **Higher threshold** (0.7-0.8) → Fewer, larger clusters
- **Lower threshold** (0.3-0.4) → More, smaller clusters

**Recommended starting values:**
- Family: 0.7 (broad categorization)
- Parent: 0.5 (moderate grouping)
- Child: 0.4 (preserve distinctions)

### LLM Validation

Enable/disable in Stage 0 and Stage 3:

```python
# Stage 0: Medium confidence validation
filtered = filter_engine.filter_by_confidence(
    theories, 
    validate_medium=True  # Set to False to skip LLM validation
)
```

## Expected Results

### For 200 Theory Prototype
- Input: ~200 theories
- Families: ~8-12
- Parents: ~30-40
- Children: ~40-50
- Compression: ~4:1

### For Full Dataset (14,000 theories)
- Input: ~14,000 theories
- Filtered: ~13,300 (after removing false positives)
- Families: ~30-50
- Parents: ~150-200
- Children: ~300-350
- Compression: ~38:1

## Output Structure

Final output (`stage4_ontology_matched.json`):

```json
{
  "metadata": {
    "stage": "stage4_ontology_matching",
    "statistics": {
      "total_theories": 200,
      "exact_matches": 45,
      "partial_matches": 30,
      "novel_theories": 25
    }
  },
  "theories": [...],
  "families": [
    {
      "cluster_id": "F001",
      "level": "family",
      "canonical_name": "Mitochondrial Theories",
      "theory_count": 25,
      "child_cluster_ids": ["P001", "P002", "P003"]
    }
  ],
  "parents": [
    {
      "cluster_id": "P001",
      "level": "parent",
      "canonical_name": "Mitochondrial Dysfunction Theory",
      "theory_count": 10,
      "parent_cluster_id": "F001",
      "child_cluster_ids": ["C001", "C002", "C003"]
    }
  ],
  "children": [
    {
      "cluster_id": "C001",
      "level": "child",
      "canonical_name": "Cisd2-mediated mitochondrial protection theory",
      "theory_ids": ["T000123", "T000456"],
      "alternative_names": ["Cisd2 theory...", "Cisd2-mediated..."],
      "ontology_match": "Mitochondrial Decline Theory",
      "ontology_confidence": 0.85,
      "match_type": "partial",
      "coherence_score": 0.92
    }
  ]
}
```

## Key Features

### 1. Fine-Grained Distinction Preservation

The pipeline preserves meaningful differences between similar theories:

**Example:**
- "CB1 receptor-mediated mitochondrial quality control" → Separate theory
- "p53-mediated mitochondrial stress response" → Separate theory

Both mention "mitochondrial" but have different mechanisms (CB1 vs p53).

**How it works:**
- Multi-dimensional embeddings capture nuances
- Concept features flag different mechanisms
- Combined similarity scoring: `0.6 × embedding_sim + 0.4 × feature_sim`
- LLM validation explicitly checks for over-clustering

### 2. Parent-Child Hierarchy

Automatically identifies generic (parent) vs specific (child) theories:

**Example:**
- **Parent:** "Mitochondrial Dysfunction Theory" (generic)
- **Children:**
  - "Cisd2-mediated mitochondrial protection theory" (specific)
  - "CB1 receptor-mediated mitochondrial quality control" (specific)

### 3. False Positive Removal

Multiple layers of filtering:
- Confidence-based filtering (remove low confidence)
- LLM re-validation of medium confidence
- Singleton review (theories that don't cluster)
- Cross-validation with ontology

### 4. Ontology Integration

Matches normalized theories to known theories:
- **Exact match** (>0.9 similarity): Use ontology name
- **Partial match** (0.7-0.9): Novel variant of known theory
- **No match** (<0.7): Completely novel theory

## Troubleshooting

### Issue: Too many clusters (under-clustering)

**Solution:** Increase thresholds
```bash
python3 run_normalization_prototype.py \
    --family-threshold 0.8 \
    --parent-threshold 0.6 \
    --child-threshold 0.5
```

### Issue: Too few clusters (over-clustering, losing distinctions)

**Solution:** Decrease thresholds
```bash
python3 run_normalization_prototype.py \
    --family-threshold 0.6 \
    --parent-threshold 0.4 \
    --child-threshold 0.3
```

### Issue: LLM API errors

**Solution:** Check `.env` file has correct credentials:
```
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-02-15-preview
OPENAI_MODEL=gpt-4
```

### Issue: Out of memory

**Solution:** Process in batches or reduce subset size:
```bash
python3 run_normalization_prototype.py --subset-size 100
```

## Cost Estimation

### Prototype (200 theories)
- Embeddings: $0.01
- LLM validation: ~$1-2
- **Total: ~$2**

### Full Pipeline (14,000 theories)
- Embeddings: $0.20
- LLM validation: ~$26.50
- **Total: ~$27**

## Performance

### Prototype (200 theories)
- Stage 0: ~1 minute
- Stage 1: ~2-3 minutes (with OpenAI API)
- Stage 2: ~1 minute
- Stage 3: ~5-10 minutes (LLM validation)
- Stage 4: ~1 minute
- **Total: ~10-15 minutes**

### Full Pipeline (14,000 theories)
- Stage 0: ~10 minutes
- Stage 1: ~1-2 hours (with batching)
- Stage 2: ~30 minutes
- Stage 3: ~3-4 hours (LLM validation)
- Stage 4: ~15 minutes
- **Total: ~5-7 hours**

## Next Steps

1. ✅ **Run prototype** to validate approach
2. ✅ **Tune thresholds** to find optimal configuration
3. ⏳ **Review sample results** to ensure quality
4. ⏳ **Run full pipeline** on all 14K theories
5. ⏳ **Manual review** of flagged cases
6. ⏳ **Export final results** to desired format

## Support

For issues or questions:
1. Check `REFINED_SOLUTION.md` for detailed technical documentation
2. Review `SOLUTION_SUMMARY.md` for high-level overview
3. Examine prototype output in `output/prototype/` for debugging

## Files Created

### Core Pipeline
- `src/normalization/stage0_quality_filter.py`
- `src/normalization/stage1_embedding.py`
- `src/normalization/stage2_clustering.py`
- `src/normalization/stage3_llm_validation.py`
- `src/normalization/stage4_ontology_matching.py`

### Runners & Tools
- `run_normalization_prototype.py` - Main prototype runner
- `tune_thresholds.py` - Threshold optimization tool

### Documentation
- `REFINED_SOLUTION.md` - Detailed technical specification
- `SOLUTION_SUMMARY.md` - Executive summary
- `NORMALIZATION_README.md` - This file

Good luck with your theory normalization! 🚀
