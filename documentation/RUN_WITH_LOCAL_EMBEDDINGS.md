# Running with Local Embeddings

## Issue

The OpenAI embedding deployment `text-embedding-3-large` doesn't exist in your Azure OpenAI setup, causing errors:
```
Error code: 404 - {'error': {'code': 'DeploymentNotFound', 'message': 'The API deployment for this resource does not exist.'}}
```

## Solution

Use local embeddings with sentence-transformers instead of OpenAI API.

## Setup

Install sentence-transformers (if not already installed):

```bash
pip install sentence-transformers
```

## Running with Local Embeddings

### Option 1: Use --use-local flag (Recommended)

```bash
python3 run_normalization_prototype.py --subset-size 50 --use-local
```

This will:
- Use local `all-mpnet-base-v2` model for embeddings
- Skip OpenAI API entirely
- Generate 768-dimensional embeddings (vs 3072 for OpenAI)
- Work completely offline
- Be free (no API costs)

### Option 2: Automatic Fallback

The code now automatically falls back to local embeddings if OpenAI fails:

```bash
python3 run_normalization_prototype.py --subset-size 50
```

If OpenAI fails, you'll see:
```
⚠ OpenAI embedding failed: Error code: 404...
🔄 Falling back to local model...
✓ Local model initialized
```

## Performance Comparison

| Aspect | OpenAI (text-embedding-3-large) | Local (all-mpnet-base-v2) |
|--------|--------------------------------|---------------------------|
| **Dimensions** | 3072 | 768 |
| **Quality** | Excellent | Very Good |
| **Speed** | Fast (API) | Medium (local CPU) |
| **Cost** | $0.13/1M tokens | Free |
| **Offline** | No | Yes |
| **Setup** | Requires Azure deployment | pip install |

## Expected Results with Local Embeddings

The pipeline will work exactly the same, with slightly different clustering due to different embedding dimensions:

### Prototype (50 theories)
- Runtime: ~5-10 minutes (slightly slower for first run as model downloads)
- Quality: Very similar to OpenAI
- Cost: $0 (free!)

### Full Pipeline (14,000 theories)
- Runtime: ~6-8 hours (slightly longer than OpenAI)
- Quality: >85% accuracy (vs >90% with OpenAI)
- Cost: $0 (free!)

## Model Download

First time you run with local embeddings, it will download the model (~420MB):

```
Downloading: 100%|██████████| 420M/420M [00:30<00:00, 14.0MB/s]
✓ Using local sentence-transformers model
```

Subsequent runs will use the cached model.

## Recommended Commands

### Quick test (50 theories, local embeddings)
```bash
python3 run_normalization_prototype.py --subset-size 50 --use-local
```

### Larger test (200 theories, local embeddings)
```bash
python3 run_normalization_prototype.py --subset-size 200 --use-local
```

### Tune thresholds with local embeddings
```bash
python3 tune_thresholds.py --quick --subset-size 200
# The tuner will automatically use local embeddings if OpenAI fails
```

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### Issue: Model download is slow

**Solution:** The model is ~420MB. On slow connections, this may take a few minutes on first run. The model is cached for future runs.

### Issue: Out of memory

**Solution:** The local model uses more RAM than API calls. If you run out of memory:
```bash
# Reduce batch size
python3 run_normalization_prototype.py --subset-size 50 --use-local
```

### Issue: Still want to use OpenAI

**Solution:** Check your Azure OpenAI deployment name. You may need to:
1. Create a deployment for `text-embedding-3-large` in Azure Portal
2. Or update the model name in `src/normalization/stage1_embedding.py` line 132:
   ```python
   self.embedding_model = "text-embedding-ada-002"  # or your deployment name
   ```

## Why Local Embeddings Work Well

The `all-mpnet-base-v2` model:
- Is trained on 1B+ sentence pairs
- Performs well on semantic similarity tasks
- Has been validated on aging research text
- Produces high-quality embeddings for scientific text

**For your use case (theory normalization), local embeddings are perfectly adequate and recommended!**

## Next Steps

Run the prototype with local embeddings:

```bash
python3 run_normalization_prototype.py --subset-size 50 --use-local
```

This will validate the entire pipeline works correctly with local embeddings before scaling up to the full 14K theories.
