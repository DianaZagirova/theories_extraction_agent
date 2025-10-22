# Outlier Handling in Hierarchical Clustering - Analysis & Recommendations

## Current Behavior

### What Happens Now

```python
if label == -1:  # Noise/outliers
    continue
```

**Current strategy:** Outliers are **silently discarded** at all three clustering levels.

### Impact

**Level 1 (Families):**
- Outlier theories are **lost** - not assigned to any family
- These theories never make it to Level 2 or Level 3
- **Data loss:** Potentially valuable unique theories are excluded

**Level 2 (Parents) & Level 3 (Children):**
- Same issue - outliers within families/parents are discarded
- Compounds the data loss problem

---

## Problem Analysis

### Issue 1: Data Loss ❌

**Scenario:** A truly novel theory that doesn't fit existing categories

**Example:**
```
Theory: "Quantum coherence in microtubules affects aging"
Status: Too different from all other theories
Result: LOST - never appears in final output
```

**Impact:**
- Novel/innovative theories are excluded
- Rare but important theories disappear
- No way to identify "unique" theories

### Issue 2: No Tracking ❌

**Current code:**
```python
self.stats = {
    'total_theories': 0,
    'num_families': 0,
    'num_parents': 0,
    'num_children': 0,
    'singletons': 0  # Never updated!
}
```

**Problems:**
- No count of outliers
- No list of which theories were excluded
- No way to audit data loss

### Issue 3: Silent Failure ❌

**No warnings or logs:**
```python
if label == -1:
    continue  # Silent - user doesn't know theories were lost
```

**Impact:**
- User doesn't know data was lost
- Can't investigate why theories were excluded
- Can't adjust thresholds to recover theories

---

## Why Outliers Occur

### Reason 1: Threshold Too Strict

**Distance threshold = 0.7 (families)**

If a theory's distance to all others > 0.7:
- AgglomerativeClustering assigns label = -1
- Theory becomes outlier

**Example:**
```
Theory A: "Mitochondrial dysfunction"
Theory B: "Telomere shortening"  
Distance: 0.85 (> 0.7 threshold)
Result: Both become outliers if no other theories nearby
```

### Reason 2: Truly Unique Theories

Some theories are genuinely novel:
- New mechanisms not yet widely studied
- Cross-domain theories (e.g., quantum biology + aging)
- Highly specific theories with unique terminology

**These SHOULD be preserved, not discarded!**

### Reason 3: Embedding Quality

Poor embeddings can cause artificial outliers:
- Theory name too short/generic
- Missing key concepts
- Embedding model doesn't capture domain specifics

---

## Recommended Solutions

### Solution 1: Create "Singleton" Clusters (RECOMMENDED) ✅

**Strategy:** Treat each outlier as its own cluster

```python
def cluster_level1_families(self, theories: List[Dict], 
                            name_embeddings: np.ndarray) -> List[TheoryCluster]:
    """Level 1: Cluster into theory families."""
    print(f"\n🔄 Level 1: Clustering into theory families...")
    
    # ... clustering code ...
    
    families = []
    outlier_theories = []
    unique_labels = set(labels)
    
    for label_idx, label in enumerate(sorted(unique_labels)):
        if label == -1:  # Outliers
            # Collect outlier indices
            outlier_indices = np.where(labels == label)[0]
            outlier_theories.extend([theories[i] for i in outlier_indices])
            continue
        
        # ... normal cluster creation ...
    
    # Create singleton clusters for outliers
    for outlier in outlier_theories:
        label_idx = len(families)
        family = TheoryCluster(
            cluster_id=f"F{label_idx+1:03d}",
            level='family',
            theory_ids=[outlier['theory_id']],
            centroid=name_embeddings[theories.index(outlier)],
            is_singleton=True  # Flag as singleton
        )
        families.append(family)
        self.stats['singletons'] += 1
    
    print(f"✓ Created {len(families)} theory families")
    print(f"  Regular clusters: {len(families) - len(outlier_theories)}")
    print(f"  Singleton clusters: {len(outlier_theories)}")
    
    return families
```

**Benefits:**
- ✅ No data loss
- ✅ Outliers preserved as unique theories
- ✅ Can be analyzed separately
- ✅ Transparent - user knows about singletons

**Drawbacks:**
- ⚠️ More clusters (but that's okay)
- ⚠️ Need to handle singletons in downstream stages

---

### Solution 2: Assign to Nearest Cluster (ALTERNATIVE) ⚠️

**Strategy:** Force outliers into nearest existing cluster

```python
def cluster_level1_families(self, theories: List[Dict], 
                            name_embeddings: np.ndarray) -> List[TheoryCluster]:
    """Level 1: Cluster into theory families."""
    
    # ... clustering code ...
    
    # Handle outliers
    outlier_indices = np.where(labels == -1)[0]
    
    if len(outlier_indices) > 0:
        print(f"   Assigning {len(outlier_indices)} outliers to nearest clusters...")
        
        for outlier_idx in outlier_indices:
            # Find nearest cluster
            outlier_embedding = name_embeddings[outlier_idx]
            
            best_cluster = None
            best_distance = float('inf')
            
            for label in set(labels):
                if label == -1:
                    continue
                
                cluster_indices = np.where(labels == label)[0]
                cluster_centroid = name_embeddings[cluster_indices].mean(axis=0)
                
                distance = 1 - cosine_similarity(
                    outlier_embedding.reshape(1, -1),
                    cluster_centroid.reshape(1, -1)
                )[0][0]
                
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = label
            
            # Assign to nearest cluster
            labels[outlier_idx] = best_cluster
            print(f"     Theory {theories[outlier_idx]['theory_id']}: "
                  f"assigned to cluster {best_cluster} (distance: {best_distance:.3f})")
    
    # ... rest of cluster creation ...
```

**Benefits:**
- ✅ No data loss
- ✅ All theories assigned to clusters
- ✅ Simpler downstream handling

**Drawbacks:**
- ❌ Forces dissimilar theories together
- ❌ Can reduce cluster coherence
- ❌ Hides the fact that theories are outliers

---

### Solution 3: Create "Uncategorized" Meta-Cluster (ALTERNATIVE) ⚠️

**Strategy:** Put all outliers in one special cluster

```python
# After normal clustering
outlier_indices = np.where(labels == -1)[0]

if len(outlier_indices) > 0:
    outlier_ids = [theories[i]['theory_id'] for i in outlier_indices]
    
    uncategorized = TheoryCluster(
        cluster_id="F000",  # Special ID
        level='family',
        theory_ids=outlier_ids,
        centroid=name_embeddings[outlier_indices].mean(axis=0),
        canonical_name="Uncategorized Theories",
        is_uncategorized=True
    )
    families.insert(0, uncategorized)
    
    print(f"  Uncategorized theories: {len(outlier_ids)}")
```

**Benefits:**
- ✅ No data loss
- ✅ Easy to identify outliers
- ✅ Can review manually

**Drawbacks:**
- ❌ Heterogeneous cluster (low coherence)
- ❌ May be very large
- ❌ Doesn't help with understanding outliers

---

## Comparison of Solutions

| Solution | Data Loss | Coherence | Transparency | Complexity | Recommended |
|----------|-----------|-----------|--------------|------------|-------------|
| **Current (discard)** | ❌ High | ✅ High | ❌ Low | ✅ Low | ❌ No |
| **Singleton clusters** | ✅ None | ✅ High | ✅ High | ⚠️ Medium | ✅ **YES** |
| **Nearest cluster** | ✅ None | ❌ Medium | ⚠️ Medium | ⚠️ Medium | ⚠️ Maybe |
| **Uncategorized** | ✅ None | ❌ Low | ✅ High | ✅ Low | ⚠️ Maybe |

---

## Recommended Implementation

### Phase 1: Add Singleton Tracking (Immediate)

**Minimal change - just track outliers:**

```python
def cluster_level1_families(self, theories: List[Dict], 
                            name_embeddings: np.ndarray) -> List[TheoryCluster]:
    """Level 1: Cluster into theory families."""
    
    # ... existing clustering code ...
    
    families = []
    outlier_count = 0
    outlier_ids = []
    unique_labels = set(labels)
    
    for label_idx, label in enumerate(sorted(unique_labels)):
        if label == -1:  # Noise/outliers
            outlier_indices = np.where(labels == label)[0]
            outlier_count = len(outlier_indices)
            outlier_ids = [theories[i]['theory_id'] for i in outlier_indices]
            
            # Log warning
            print(f"   ⚠️  WARNING: {outlier_count} outlier theories detected!")
            print(f"       These theories will be excluded from clustering.")
            print(f"       Consider adjusting family_threshold (current: {self.family_threshold})")
            
            # Track in stats
            self.stats['outliers_level1'] = outlier_count
            self.stats['outlier_ids_level1'] = outlier_ids
            
            continue
        
        # ... rest of code unchanged ...
```

**Benefits:**
- ✅ Minimal code change
- ✅ User aware of data loss
- ✅ Can investigate outliers
- ✅ Can adjust thresholds

### Phase 2: Create Singleton Clusters (Recommended)

**Full solution - preserve all theories:**

```python
def cluster_level1_families(self, theories: List[Dict], 
                            name_embeddings: np.ndarray) -> List[TheoryCluster]:
    """Level 1: Cluster into theory families."""
    print(f"\n🔄 Level 1: Clustering into theory families...")
    print(f"   Threshold: {self.family_threshold}")
    
    # Compute distance matrix
    similarity_matrix = cosine_similarity(name_embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=self.family_threshold,
        linkage='average',
        metric='precomputed'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Separate regular clusters and outliers
    families = []
    outlier_indices = []
    unique_labels = set(labels)
    regular_label_count = 0
    
    # Create regular clusters
    for label in sorted(unique_labels):
        if label == -1:  # Collect outliers
            outlier_indices = np.where(labels == label)[0].tolist()
            continue
        
        theory_indices = np.where(labels == label)[0]
        theory_ids = [theories[i]['theory_id'] for i in theory_indices]
        centroid = name_embeddings[theory_indices].mean(axis=0)
        
        regular_label_count += 1
        family = TheoryCluster(
            cluster_id=f"F{regular_label_count:03d}",
            level='family',
            theory_ids=theory_ids,
            centroid=centroid,
            is_singleton=False
        )
        families.append(family)
    
    # Create singleton clusters for outliers
    singleton_count = 0
    for outlier_idx in outlier_indices:
        singleton_count += 1
        cluster_num = regular_label_count + singleton_count
        
        family = TheoryCluster(
            cluster_id=f"F{cluster_num:03d}",
            level='family',
            theory_ids=[theories[outlier_idx]['theory_id']],
            centroid=name_embeddings[outlier_idx],
            is_singleton=True,
            canonical_name=f"Singleton: {theories[outlier_idx]['name']}"
        )
        families.append(family)
    
    self.families = families
    self.stats['num_families'] = len(families)
    self.stats['num_regular_families'] = regular_label_count
    self.stats['num_singleton_families'] = singleton_count
    
    print(f"✓ Created {len(families)} theory families")
    print(f"  Regular clusters: {regular_label_count}")
    print(f"  Singleton clusters: {singleton_count}")
    if singleton_count > 0:
        print(f"  ⚠️  {singleton_count} theories are unique (no close matches)")
    print(f"  Avg theories per regular family: {sum(len(f.theory_ids) for f in families if not f.is_singleton)/max(regular_label_count, 1):.1f}")
    
    return families
```

**Also update TheoryCluster dataclass:**

```python
@dataclass
class TheoryCluster:
    """Represents a cluster of theories at any level."""
    cluster_id: str
    level: str  # 'family', 'parent', or 'child'
    theory_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    canonical_name: Optional[str] = None
    alternative_names: List[str] = field(default_factory=list)
    coherence_score: float = 0.0
    parent_cluster_id: Optional[str] = None
    child_cluster_ids: List[str] = field(default_factory=list)
    is_singleton: bool = False  # NEW: Flag for singleton clusters
    is_uncategorized: bool = False  # NEW: Flag for uncategorized cluster
```

---

## Handling Singletons in Downstream Stages

### Level 2 (Parents)

```python
def cluster_level2_parents(self, theories: List[Dict], 
                          semantic_embeddings: np.ndarray,
                          families: List[TheoryCluster]) -> List[TheoryCluster]:
    """Level 2: Cluster into parent theories within each family."""
    
    all_parents = []
    parent_counter = 0
    
    for family in families:
        # Singletons stay as single parents
        if family.is_singleton or len(family.theory_ids) == 1:
            parent_counter += 1
            parent = TheoryCluster(
                cluster_id=f"P{parent_counter:04d}",
                level='parent',
                theory_ids=family.theory_ids,
                parent_cluster_id=family.cluster_id,
                is_singleton=family.is_singleton  # Propagate flag
            )
            all_parents.append(parent)
            family.child_cluster_ids.append(parent.cluster_id)
            continue
        
        # ... normal clustering for non-singletons ...
```

### Level 3 (Children)

Same approach - singletons propagate through all levels.

---

## Testing Strategy

### Test 1: Verify No Data Loss

```python
# Before clustering
total_theories_before = len(theories)

# After clustering
total_theories_after = sum(len(f.theory_ids) for f in families)

assert total_theories_before == total_theories_after, \
    f"Data loss detected: {total_theories_before} → {total_theories_after}"
```

### Test 2: Identify Outliers

```python
# Check singleton statistics
singleton_families = [f for f in families if f.is_singleton]
print(f"\nSingleton Analysis:")
print(f"  Count: {len(singleton_families)}")
print(f"  Percentage: {len(singleton_families)/len(families)*100:.1f}%")

if len(singleton_families) > 0:
    print(f"\n  Singleton theories:")
    for family in singleton_families[:10]:  # Show first 10
        theory_id = family.theory_ids[0]
        theory = next(t for t in theories if t['theory_id'] == theory_id)
        print(f"    - {theory['name']}")
```

### Test 3: Threshold Sensitivity

```python
# Test different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    clusterer = HierarchicalClusterer(family_threshold=threshold)
    families = clusterer.cluster_level1_families(theories, name_embeddings)
    
    singletons = sum(1 for f in families if f.is_singleton)
    print(f"Threshold {threshold}: {singletons} singletons ({singletons/len(theories)*100:.1f}%)")
```

---

## Summary

### Current Problem ❌

- **Outliers are silently discarded** at all three levels
- **Data loss:** Unique/novel theories are lost
- **No tracking:** User doesn't know theories were excluded
- **No transparency:** Can't investigate or adjust

### Recommended Solution ✅

**Phase 1 (Immediate):**
- Add outlier tracking and warnings
- Log which theories are excluded
- Update stats dictionary

**Phase 2 (Recommended):**
- Create singleton clusters for outliers
- Preserve all theories
- Flag singletons for special handling
- Propagate through all levels

### Benefits

✅ **No data loss** - All theories preserved  
✅ **Transparency** - User knows about unique theories  
✅ **Flexibility** - Can analyze singletons separately  
✅ **Quality** - Can identify truly novel theories  
✅ **Debugging** - Can investigate why theories are outliers  

### Implementation Effort

- **Phase 1:** 30 minutes (add tracking)
- **Phase 2:** 2-3 hours (full singleton support)

### Next Steps

1. Implement Phase 1 (tracking) immediately
2. Test on 761 theories
3. Analyze outlier statistics
4. Decide if Phase 2 (singletons) needed based on outlier count
5. If >5% outliers, implement Phase 2

**Recommendation: Implement Phase 2 (singleton clusters) - it's the right approach for scientific data where unique theories are valuable!**
