# Stage 6: Data Flow and Isolation Guarantees

## Critical Guarantee: No Cross-Cluster Mixing

**Each LLM prompt contains theories from ONLY ONE cluster/final_name. Theories from different clusters are NEVER mixed.**

## Data Flow Diagram

```
Stage 5 Output
└── final_name_summary (1071 unique names)
    ├── Cluster 1: "Cellular Senescence Theory" (1285 papers)
    │   └── theory_ids: [T000040, T000075, T000085, ...]
    │
    ├── Cluster 2: "Mitochondrial Decline Theory" (1070 papers)
    │   └── theory_ids: [T000123, T000456, T000789, ...]
    │
    └── Cluster 3: "Disposable Soma Theory" (994 papers)
        └── theory_ids: [T001234, T001567, T001890, ...]

                    ↓ Stage 6 Processing ↓

┌─────────────────────────────────────────────────────────────┐
│ CLUSTER 1: "Cellular Senescence Theory"                     │
│ ✓ Isolated processing - NO mixing with other clusters       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Batch 1.1 (30 theories)                                    │
│  ├── T000040 → key_concepts, paper_title                    │
│  ├── T000075 → key_concepts, paper_title                    │
│  └── ... (28 more)                                          │
│  └─→ LLM Prompt #1 ─→ Creates subclusters A, B, C          │
│                                                              │
│  Batch 1.2 (30 theories)                                    │
│  ├── T000120 → key_concepts, paper_title                    │
│  ├── T000135 → key_concepts, paper_title                    │
│  └── ... (28 more)                                          │
│  └─→ LLM Prompt #2 ─→ Creates subclusters D, E, F          │
│                                                              │
│  ... (43 batches total for this cluster)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CLUSTER 2: "Mitochondrial Decline Theory"                   │
│ ✓ Isolated processing - NO mixing with other clusters       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Batch 2.1 (30 theories)                                    │
│  ├── T000123 → key_concepts, paper_title                    │
│  ├── T000456 → key_concepts, paper_title                    │
│  └── ... (28 more)                                          │
│  └─→ LLM Prompt #44 ─→ Creates subclusters G, H, I         │
│                                                              │
│  ... (36 batches total for this cluster)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

... (41 clusters total)
```

## Code-Level Guarantees

### 1. Cluster-Level Isolation

```python
async def process_all_clusters_async(self):
    """Each cluster processed independently and sequentially."""
    for cluster_info in large_clusters:
        # Process ONE cluster at a time
        result = await self._process_cluster_async(cluster_info)
```

**Guarantee**: Clusters are processed sequentially, one at a time.

### 2. Batch Creation from Single Cluster

```python
async def _process_cluster_async(self, cluster_info: Dict):
    """All batches created from this single cluster's theory_ids only."""
    cluster_name = cluster_info['final_name']
    theory_ids = cluster_info['theory_ids']  # ONLY from this cluster
    
    # Create batches - ALL from this single cluster
    batches = self._create_smart_batches(theory_ids)
```

**Guarantee**: All batches for a cluster contain only that cluster's theory_ids.

### 3. Runtime Validation

```python
# Validation: Ensure all theories in this batch are from the same cluster
assert all(tid in theory_ids for tid in batch_ids), \
    f"CRITICAL ERROR: Batch contains theories not from cluster '{cluster_name}'"
```

**Guarantee**: Runtime assertion fails if any theory from a different cluster appears.

### 4. LLM Prompt Construction

```python
async def _separate_batch_async(self, cluster_name: str, batch_theories: List[Dict], ...):
    """
    CRITICAL GUARANTEES:
    1. All theories in batch_theories are from the SAME cluster (cluster_name)
    2. The LLM prompt will ONLY contain data from these theories
    """
    # Prompt includes ONLY data from batch_theories
    prompt = self._create_separation_prompt(cluster_name, batch_theories, batch_info)
```

**Guarantee**: LLM prompt contains only theories from the specified cluster.

## Example: Processing "Cellular Senescence Theory"

### Input (from Stage 5)
- **Cluster name**: "Cellular Senescence Theory"
- **Theory IDs**: 1,285 theories [T000040, T000075, ..., T027419]
- **Original names**: ["Cellular Senescence Theory", "Asymmetric Division Limited Senescence Theory", ...]

### Processing
1. **Batch creation**: 1,285 theories → 43 batches
   - Batch 1: T000040-T000069 (30 theories)
   - Batch 2: T000070-T000099 (30 theories)
   - ...
   - Batch 43: T027390-T027419 (29 theories)

2. **LLM calls**: 43 separate prompts
   - Each prompt contains 20-30 theories
   - **ALL from "Cellular Senescence Theory" cluster**
   - **NONE from other clusters**

3. **Output**: Multiple subclusters
   - "ROS-Induced Cellular Senescence Theory" (245 theories)
   - "Telomere-Associated Cellular Senescence Theory" (312 theories)
   - "p53-Mediated Cellular Senescence Theory" (189 theories)
   - etc.

### Validation
✅ All 1,285 input theory IDs are assigned to subclusters  
✅ No theory ID appears in multiple subclusters  
✅ Each subcluster has ≥5 theories  
✅ All theories are from original "Cellular Senescence Theory" cluster

## Why This Matters

### ✅ Correct Approach (Current Implementation)
- LLM sees related theories with similar mechanisms
- Can identify meaningful mechanistic distinctions
- Creates coherent, specific subclusters
- Example: Separates "Cellular Senescence" by mechanism (ROS, telomere, p53)

### ❌ Wrong Approach (If We Mixed Clusters)
- LLM would see unrelated theories
- Example: Mixing "Cellular Senescence" + "Mitochondrial Decline" theories
- Would create confused, meaningless subclusters
- Cannot identify real mechanistic patterns

## Summary

**The code guarantees that each LLM prompt contains theories from exactly ONE cluster/final_name.**

This is enforced through:
1. Sequential cluster processing
2. Batch creation from single cluster
3. Runtime validation assertions
4. Clear documentation and logging

No theories from different clusters are ever mixed in the same LLM prompt.
