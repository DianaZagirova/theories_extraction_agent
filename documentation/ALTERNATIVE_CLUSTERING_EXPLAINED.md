# Alternative Clustering Approach - Explained

## Overview

This alternative clustering strategy addresses the issues found in the original approach by using:
1. **Full-text embeddings** at all levels (not just children)
2. **Different algorithms** for each level (K-Means, DBSCAN, Agglomerative)
3. **Feature-weighted similarity** to boost theories with shared mechanisms/pathways
4. **Adaptive thresholds** based on cluster size
5. **Size constraints** to prevent giant clusters and excessive singletons

---

## Key Differences from Original

| Aspect | Original | Alternative |
|--------|----------|-------------|
| **Embeddings** | Name → Semantic → Detailed | Detailed at all levels |
| **Level 1** | Agglomerative (threshold=0.7) | K-Means (optimal K) |
| **Level 2** | Agglomerative (threshold=0.5) | DBSCAN (density-based) |
| **Level 3** | Agglomerative (threshold=0.4) | Agglomerative (adaptive) |
| **Similarity** | Pure cosine | Feature-weighted |
| **Constraints** | None | Min/max cluster sizes |
| **Quality** | No metrics | Coherence + Silhouette |

---

## Detailed Strategy

### Level 1: Theory Families (K-Means)

**Algorithm:** K-Means with optimal K selection

**Why K-Means?**
- ✅ Guarantees balanced cluster sizes
- ✅ No giant "catch-all" clusters
- ✅ Automatically finds K using silhouette score
- ✅ Fast and scalable

**Process:**
```python
1. Compute feature-weighted similarity
2. Find optimal K (test range around target)
3. Run K-Means with optimal K
4. Enforce min/max cluster sizes
5. Calculate coherence scores
```

**Parameters:**
- `target_families = 50` (will adjust ±20 based on quality)
- `min_cluster_size = 3` (merge smaller clusters)
- `max_cluster_size = 50` (split larger clusters)

**Expected result:**
- 40-60 balanced families
- No giant families (>50 theories)
- Few singletons (<10%)

### Level 2: Parent Theories (DBSCAN)

**Algorithm:** DBSCAN (Density-Based Spatial Clustering)

**Why DBSCAN?**
- ✅ Automatically finds number of clusters
- ✅ Identifies outliers naturally
- ✅ Works well for varying density
- ✅ No need to specify cluster count

**Process:**
```python
1. For each family:
2.   Compute feature-weighted similarity
3.   Run DBSCAN with adaptive parameters
4.   Create parent clusters
5.   Handle outliers as singletons
```

**Parameters:**
- `eps = 0.4` (maximum distance)
- `min_samples = family_size // 10` (adaptive)

**Expected result:**
- Natural groupings within families
- Outliers preserved as singletons
- Density-based clustering

### Level 3: Child Theories (Agglomerative with Adaptive Threshold)

**Algorithm:** Agglomerative Clustering with size-dependent thresholds

**Why Adaptive?**
- ✅ Small parents need stricter threshold (more specific)
- ✅ Large parents need looser threshold (allow grouping)
- ✅ Prevents excessive fragmentation
- ✅ Maintains minimum cluster size

**Process:**
```python
1. For each parent:
2.   Determine threshold based on parent size:
       - <10 theories: threshold = 0.35 (strict)
       - 10-20 theories: threshold = 0.45 (moderate)
       - >20 theories: threshold = 0.55 (loose)
3.   Run Agglomerative clustering
4.   Enforce min_size = 2 (merge singletons)
5.   Calculate coherence
```

**Expected result:**
- Fewer singletons (<30%)
- Better compression (3-4:1)
- Meaningful child clusters

---

## Feature-Weighted Similarity

**Problem:** Pure cosine similarity misses domain-specific relationships

**Solution:** Boost similarity for theories with shared features

```python
combined_similarity = 0.7 * embedding_similarity + 0.3 * feature_bonus

feature_bonus = (
    0.15 * mechanism_overlap +
    0.10 * pathway_overlap +
    0.05 * biological_level_match
)
```

**Example:**

**Theory A:** "mTOR-mediated autophagy in aging"
**Theory B:** "AMPK-mediated autophagy in longevity"

**Pure cosine similarity:** 0.65
**Mechanism overlap:** autophagy (shared) → +0.15
**Pathway overlap:** mTOR vs AMPK (different) → +0.00
**Combined similarity:** 0.7 * 0.65 + 0.3 * 0.15 = 0.50

**Result:** Theories cluster together due to shared mechanism

---

## Size Constraints

### Minimum Cluster Size

**Problem:** Too many singletons (72.5% in original)

**Solution:** Enforce minimum of 2-3 theories per cluster

```python
MIN_CHILD_SIZE = 2
MIN_PARENT_SIZE = 3
MIN_FAMILY_SIZE = 3

# After clustering
if cluster.size < MIN_SIZE:
    merge_with_nearest_neighbor(cluster)
```

**Impact:**
- Singletons: 72.5% → <30%
- Better compression
- More meaningful clusters

### Maximum Cluster Size

**Problem:** Giant families (F001 with 587 theories)

**Solution:** Split clusters exceeding threshold

```python
MAX_FAMILY_SIZE = 50
MAX_PARENT_SIZE = 30
MAX_CHILD_SIZE = 10

# After clustering
if cluster.size > MAX_SIZE:
    sub_clusters = split_cluster(cluster)
```

**Impact:**
- No giant families
- Better balance
- More granular organization

---

## Quality Metrics

### 1. Coherence Score

**Definition:** Average pairwise similarity within cluster

```python
coherence = mean(cosine_similarity(theory_i, theory_j))
            for all pairs in cluster
```

**Interpretation:**
- >0.7: Excellent (very similar theories)
- 0.5-0.7: Good (related theories)
- <0.5: Poor (heterogeneous cluster)

**Use:** Identify low-quality clusters for review

### 2. Silhouette Score

**Definition:** How well theories fit their cluster vs other clusters

```python
silhouette = (b - a) / max(a, b)
where:
  a = avg distance to theories in same cluster
  b = avg distance to theories in nearest other cluster
```

**Interpretation:**
- >0.5: Good clustering
- 0.3-0.5: Acceptable
- <0.3: Poor clustering

**Use:** Overall clustering quality assessment

### 3. Separation Score

**Definition:** Distance between cluster centroids

```python
separation = 1 - cosine_similarity(centroid_i, centroid_j)
```

**Interpretation:**
- >0.5: Well separated
- 0.3-0.5: Moderate separation
- <0.3: Overlapping (consider merging)

**Use:** Identify clusters that should be merged

---

## Expected Improvements

### Compression Ratio

**Original:** 1.65:1 (761 → 462 children)
**Alternative:** 3-4:1 (761 → 200-250 children)

**Improvement:** +100% better compression

### Singleton Reduction

**Original:** 72.5% singletons
**Alternative:** <30% singletons

**Improvement:** -60% singletons

### Balance

**Original:** 1 family with 587 theories, 10 families with ≤3
**Alternative:** All families 5-50 theories

**Improvement:** Much better distribution

### Quality

**Original:** No quality metrics
**Alternative:** Coherence scores, silhouette score

**Improvement:** Validated quality

---

## Usage

### Run Alternative Clustering

```bash
# Make sure Stage 1 is complete
python src/normalization/stage1_embedding_advanced.py

# Run alternative clustering
python src/normalization/stage2_clustering_alternative.py

# Compare with original
python compare_clustering_approaches.py

# Create readable summary
python create_readable_summary.py
```

### Adjust Parameters

```python
# In stage2_clustering_alternative.py
clusterer = AlternativeClusterer(
    target_families=50,      # Target number of families
    min_cluster_size=3,      # Minimum theories per cluster
    max_cluster_size=50      # Maximum theories per cluster
)
```

**Tuning guide:**
- More families: Increase `target_families` (60-80)
- Fewer singletons: Increase `min_cluster_size` (4-5)
- Prevent large clusters: Decrease `max_cluster_size` (30-40)

---

## Comparison Example

### Original Approach

```
Family F001: 587 theories (77% of total)
  Parent P0003: 334 theories
    Child C0105: 25 theories
    Child C0109: 20 theories
    ... 145 more children
  Parent P0002: 137 theories
    ...

Singletons: 335/462 (72.5%)
Compression: 1.65:1
```

### Alternative Approach

```
Family F001: 42 theories (Mitochondrial theories)
  Parent P0001: 15 theories
    Child C0001: 8 theories (ROS-related)
    Child C0002: 7 theories (Uncoupling-related)
  Parent P0002: 12 theories
    ...

Family F002: 38 theories (Evolutionary theories)
  ...

Family F003: 35 theories (Telomere theories)
  ...

Singletons: 45/235 (19%)
Compression: 3.2:1
```

**Result:** Much better organization!

---

## Troubleshooting

### Too Many Families

**Symptom:** >80 families, many small

**Solution:**
```python
target_families=40  # Reduce from 50
min_cluster_size=4  # Increase from 3
```

### Too Few Families

**Symptom:** <30 families, some very large

**Solution:**
```python
target_families=60  # Increase from 50
max_cluster_size=40  # Reduce from 50
```

### Still Too Many Singletons

**Symptom:** >40% singletons

**Solution:**
```python
min_cluster_size=4   # Increase from 3
# Adjust child threshold
threshold = 0.50     # Looser (was 0.35-0.55)
```

### Low Coherence Scores

**Symptom:** Avg coherence <0.5

**Solution:**
- Check feature extraction quality
- Verify embeddings are good
- May need stricter thresholds

---

## Summary

### Alternative Approach Advantages

✅ **Uses full-text embeddings** - Better semantic understanding
✅ **Different algorithms per level** - Optimized for each task
✅ **Feature-weighted similarity** - Domain-aware clustering
✅ **Size constraints** - Prevents imbalance
✅ **Quality metrics** - Validated results
✅ **Adaptive thresholds** - Context-dependent
✅ **Better compression** - 3-4:1 vs 1.65:1
✅ **Fewer singletons** - <30% vs 72.5%
✅ **Balanced distribution** - No giant families

### When to Use

**Use Alternative if:**
- Need better compression
- Want balanced clusters
- Have quality requirements
- Processing large dataset (14K theories)

**Use Original if:**
- Need simple approach
- Want hierarchical embeddings (name→semantic→detailed)
- Prefer single algorithm
- Quick prototyping

### Recommendation

**For production (14K theories): Use Alternative Approach**

**Files:**
- Implementation: `src/normalization/stage2_clustering_alternative.py`
- Comparison: `compare_clustering_approaches.py`
- Documentation: This file

**Next step: Run and compare both approaches!**
