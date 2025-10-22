# Stage 5: Cluster-Based Theory Refinement

## Overview

Stage 5 refines theory names within predefined embedding-based clusters. It uses LLM to normalize rare theories (with <6 papers) by either grouping them into new distinct names, mapping them to well-established reference theories, or retaining their original names.

## Key Features

### 1. Predefined Clusters
- Uses embedding-based clusters from `data/clusters_with_paper_counts.json`
- Each cluster contains theories that are semantically similar based on embeddings
- Clusters are predefined (keys in the JSON file)

### 2. Theory Separation
For each cluster, theories are separated into:
- **REFERENCE_LIST**: Theories with ≥6 papers
  - Well-established theories with sufficient evidence
  - Key concepts extracted from 3 random samples
  - Used as reference for normalization
  
- **RARE_LIST**: Theories with <6 papers
  - Less common theories that need normalization
  - Key concepts extracted from available samples (up to 3)
  - Target of the normalization process

### 3. Concept Extraction Pipeline
```
theory_name → theory_tracking_report.json (get theory_ids)
           → stage0_filtered_theories.json (get key_concepts)
```

### 4. Normalization Strategies
The LLM is instructed to use the following priority order:

1. **PREFERRED: Group into 2-3 distinct names**
   - Create new standardized names DISTINCT from REFERENCE_LIST
   - Group similar rare theories under these new names
   - Names should be clear, generalizable, mechanism-based

2. **ALTERNATIVE: Map to REFERENCE_LIST**
   - Only if mechanisms are very similar to a reference theory
   - Requires high confidence (>0.7)
   - Uses exact reference theory name

3. **FALLBACK: Retain original**
   - Only if mechanisms are too dissimilar
   - Keeps the original theory name

## Input Files

1. **data/clusters_with_paper_counts.json**
   ```json
   {
     "214": {
       "size": 23,
       "members": [
         {
           "theory_name": "Cellular Senescence Theory",
           "paper_count": 1280
         },
         {
           "theory_name": "Negligible Senescence Theory",
           "paper_count": 14
         }
       ]
     }
   }
   ```

2. **output/theory_tracking_report.json**
   - Maps theory names to theory IDs via `theory_lineage`
   - Uses `final_name_normalized` field

3. **output/stage0_filtered_theories.json**
   - Contains key_concepts for each theory_id
   - Provides mechanistic details for comparison

## Output Format

**output/stage5_cluster_refined_theories.json**

```json
{
  "metadata": {
    "stage": "stage5_cluster_refinement",
    "total_clusters": 150,
    "total_theories": 5000,
    "reference_theories": 1200,
    "rare_theories": 3800,
    "grouped_theories": 2500,
    "mapped_to_reference": 800,
    "retained_original": 500,
    "clusters_with_missing": 15,
    "total_retries": 15,
    "theories_recovered_by_retry": 42,
    "total_input_tokens": 1000000,
    "total_output_tokens": 200000,
    "total_cost": 0.27
  },
  "clusters": [
    {
      "cluster_id": "214",
      "normalizations": [
        {
          "original_name": "Negligible Senescence Theory",
          "strategy": "group",
          "normalized_name": "Alternative Senescence Models",
          "mapping_confidence": 0.0,
          "reasoning": "Groups theories proposing alternative senescence patterns"
        },
        {
          "original_name": "Clonal Senescence Theory",
          "strategy": "map",
          "normalized_name": "Cellular Senescence Theory",
          "mapping_confidence": 0.85,
          "reasoning": "Mechanisms closely align with cellular senescence"
        }
      ],
      "error": null
    }
  ]
}
```

## Usage

### Basic Usage

```bash
# Run full Stage 5 processing
python -m src.normalization.stage5_cluster_refinement

# Resume from checkpoint
python -m src.normalization.stage5_cluster_refinement --resume

# Specify custom output path
python -m src.normalization.stage5_cluster_refinement --output output/custom_stage5.json

# Enable retry for poor grouping (when LLM creates unique names instead of grouping)
python -m src.normalization.stage5_cluster_refinement --retry-poor-grouping
```

### Test on Subset

```bash
# Test on first 3 clusters
python test_stage5.py
```

### Programmatic Usage

```python
from src.normalization.stage5_cluster_refinement import Stage5ClusterRefiner

# Initialize refiner
refiner = Stage5ClusterRefiner(
    clusters_path='data/clusters_with_paper_counts.json',
    tracker_path='output/theory_tracking_report.json',
    stage0_path='output/stage0_filtered_theories.json',
    max_concurrent=10
)

# Process all clusters
output = refiner.process_clusters(
    output_path='output/stage5_cluster_refined_theories.json',
    resume_from_checkpoint=False
)

# Save results
refiner.save_results(output, output_path='output/stage5_cluster_refined_theories.json')
```

## Configuration

### Environment Variables

- `USE_MODULE_FILTERING_LLM_STAGE5`: Set to `'openai'` or `'azure'` (default: `'azure'`)
- `OPENAI_API_KEY2`: Required if using OpenAI

### Parameters

- `max_concurrent`: Number of concurrent API calls (default: 10)
- `resume_from_checkpoint`: Resume from previous checkpoint (default: False)
- `retry_poor_grouping`: Retry when LLM creates unique names instead of grouping (default: False)
- `output_path`: Path to save results (default: 'output/stage5_cluster_refined_theories.json')
- Rate limiting: 180,000 TPM, 450 RPM (90% buffer)
- Model: `gpt-4.1-mini`
- Temperature: 0.1
- Max tokens: 4000 per response
- Checkpoint interval: Every 5 clusters

## Naming Rules

The LLM follows these rules when creating new names:

1. **Avoid excessive specificity**: Don't include too many details
2. **Generalize based on mechanisms**: Not specific diseases/organs/pathways
3. **Theory suffix**: If name ends with "Theory", don't add "of Aging"
4. **Spelling**: Use "aging" not "ageing"
5. **No composite names**: Never create names with multiple theories

## Statistics Tracked

- Total clusters processed
- Total theories (reference + rare)
- Normalization strategy distribution:
  - Grouped theories
  - Mapped to reference
  - Retained original
- Token usage and cost

## Error Handling

- **JSON parsing errors**: Returns empty normalizations with error message
- **LLM errors**: Catches exceptions and logs error details
- **Missing data**: Handles cases where concepts are unavailable
- **Rate limiting**: Automatic retry with exponential backoff
- **Missing theories validation**: 
  - Validates all input theories are present in output
  - Automatically retries for missing theories
  - Adds missing theories with 'retain' strategy if retry fails
  - Tracks statistics for missing theories and recovery rate
- **Poor grouping detection** (optional retry):
  - Detects when LLM uses 'assign_common' but creates unique names
  - With `--retry-poor-grouping` flag, retries with stronger instructions
  - Only retries if >3 theories all get unique names
  - Keeps original if retry doesn't improve grouping

## Performance Considerations

1. **Async processing**: Uses asyncio with semaphore for concurrent API calls
2. **Rate limiting**: Respects API limits with 90% buffer
3. **Batching**: Processes up to 50 clusters concurrently
4. **Checkpointing**: Saves progress every 5 clusters
5. **Resume capability**: Can resume from checkpoint after interruption
6. **Token estimation**: Estimates tokens before API calls
7. **Progress tracking**: Uses tqdm for progress bars
8. **Space optimization**: Saves only normalizations (not full cluster data with concepts) to reduce file size

## Differences from Stage 4

| Aspect | Stage 4 | Stage 5 |
|--------|---------|---------|
| **Batching** | Dynamic (5 theories per batch) | Predefined (cluster-based) |
| **Grouping** | Individual validation | Cluster-based grouping |
| **Reference** | Canonical ontology | High paper-count theories in cluster |
| **Strategy** | Map or validate | Group, map, or retain |
| **Focus** | Individual theory validation | Cluster-level refinement |
| **Output** | Theory-level mappings | Cluster-level normalizations |

## Next Steps

After Stage 5, you can:

1. **Analyze grouping patterns**: Identify common theory groups
2. **Update theory tracker**: Apply Stage 5 normalizations
3. **Validate results**: Manual review of grouped theories
4. **Export for analysis**: Create summary reports
5. **Iterate**: Adjust clustering or normalization parameters

## Example Workflow

```bash
# 1. Ensure all input files are ready
ls data/clusters_with_paper_counts.json
ls output/theory_tracking_report.json
ls output/stage0_filtered_theories.json

# 2. Test on subset first
python test_stage5.py

# 3. Review test results
cat output/test_stage5_output.json | jq '.metadata'

# 4. Run full processing
python -m src.normalization.stage5_cluster_refinement

# 5. If interrupted, resume from checkpoint
python -m src.normalization.stage5_cluster_refinement --resume

# 6. Analyze results
python -c "
import json
with open('output/stage5_cluster_refined_theories.json') as f:
    data = json.load(f)
    print(json.dumps(data['metadata'], indent=2))
"
```

## Troubleshooting

### Issue: Missing key_concepts
**Solution**: Ensure theory_tracking_report.json has correct mappings and stage0_filtered_theories.json is complete

### Issue: Rate limiting errors
**Solution**: Reduce `max_concurrent` parameter or increase wait time

### Issue: JSON parsing errors
**Solution**: Check LLM response format, may need to adjust temperature or max_tokens

### Issue: Low grouping rate
**Solution**: Review prompt instructions or adjust confidence thresholds

## Cost Estimation

Based on typical usage:
- ~150 clusters
- ~5000 theories total
- ~3800 rare theories
- Average prompt: ~3000 tokens
- Average response: ~1500 tokens

**Estimated cost**: $0.20 - $0.40 per full run (using gpt-4o-mini)
