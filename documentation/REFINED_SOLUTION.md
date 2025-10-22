# Theory Normalization: Refined Solution for 14K Theories

## Critical Requirements Analysis

### Scale & Quality Constraints
- **Input:** ~14,000 theories (current: 2,608 from 2,100 papers)
- **Output:** ~300-350 normalized theories with parent-child hierarchy
- **Confidence distribution:**
  - High: 77.7% (~10,883 theories) - reliable but may have duplicates
  - Medium: 21.5% (~3,011 theories) - needs validation
  - Low: 0.7% (~104 theories) - likely false positives, filter out
- **Challenge:** Many theories share keywords but differ in subtle details

### Key Insights from Data Analysis

1. **False Positives Exist**
   - Low confidence theories should be filtered
   - Medium confidence needs LLM re-validation
   - Even "high" confidence may include edge cases

2. **Parent-Child Structure**
   - Generic: "Mitochondrial Theory of Aging" (parent)
   - Specific: "Cisd2-mediated mitochondrial protection theory of aging" (child)
   - ~199 potentially generic theories identified
   - ~171 mechanism-specific theories identified

3. **Fine-Grained Distinctions Required**
   - 53 mitochondrial-related theories exist
   - Many share keywords but describe different mechanisms
   - Example: "CB1 receptor-mediated mitochondrial quality control" vs "CEP-1/p53-mediated mitochondrial stress response"
   - **These should remain separate theories**

4. **Rich Metadata Available**
   - `key_concepts`: 3-6 concepts per theory with descriptions
   - `description`: Additional context
   - `confidence_is_theory`: Quality indicator
   - `mode`: How theory was presented in paper

---

## Refined Architecture

### Stage 0: Quality Filtering & Data Enrichment

#### 0.1 Confidence-Based Filtering
```python
def filter_by_confidence(theories):
    """Remove false positives based on confidence scores."""
    
    # Automatic filtering
    high_conf = [t for t in theories if t['confidence_is_theory'] == 'high']
    medium_conf = [t for t in theories if t['confidence_is_theory'] == 'medium']
    low_conf = [t for t in theories if t['confidence_is_theory'] == 'low']
    
    # Remove low confidence (false positives)
    filtered = high_conf.copy()
    
    # Medium confidence: LLM re-validation
    for theory in medium_conf:
        if validate_is_real_theory(theory):  # LLM check
            filtered.append(theory)
    
    print(f"Filtered: {len(high_conf)} high + {len(validated_medium)} medium")
    print(f"Removed: {len(low_conf)} low + {len(rejected_medium)} medium")
    
    return filtered
```

**Expected filtering:**
- Keep: ~10,883 high + ~2,400 validated medium = **~13,300 theories**
- Remove: ~104 low + ~600 rejected medium = **~700 false positives**

#### 0.2 Data Enrichment
```python
def enrich_theory_data(theory):
    """Create rich representation from all available fields."""
    
    # Extract key concepts
    concepts = [
        f"{c['concept']}: {c['description']}" 
        for c in theory.get('key_concepts', [])
    ]
    
    # Build comprehensive text
    enriched_text = {
        'name': theory['name'],
        'concepts': ' | '.join(concepts),
        'description': theory.get('description', ''),
        'evidence': theory.get('evidence', ''),
        'full_text': f"{theory['name']}. {' '.join(concepts)}. {theory.get('description', '')}"
    }
    
    return enriched_text
```

---

### Stage 1: Multi-Dimensional Embedding

#### 1.1 Dual Embedding Strategy

**Problem:** Single embedding may miss fine-grained distinctions

**Solution:** Create multiple embeddings at different granularities

```python
def create_multi_level_embeddings(theory):
    """Generate embeddings at different levels of detail."""
    
    # Level 1: Name-only (for finding broad similarities)
    name_embedding = embed(theory['name'])
    
    # Level 2: Name + concepts (for semantic clustering)
    semantic_embedding = embed(
        f"{theory['name']}. {theory['concepts']}"
    )
    
    # Level 3: Full context (for fine-grained distinction)
    detailed_embedding = embed(theory['full_text'])
    
    return {
        'name_emb': name_embedding,
        'semantic_emb': semantic_embedding,
        'detailed_emb': detailed_embedding
    }
```

#### 1.2 Concept-Level Features

Extract structured features for fine-grained comparison:

```python
def extract_concept_features(theory):
    """Extract structured features from key concepts."""
    
    features = {
        'mechanisms': [],  # e.g., "CB1 receptor-mediated", "p53-mediated"
        'pathways': [],    # e.g., "mTOR", "insulin/IGF-1"
        'molecules': [],   # e.g., "Cisd2", "spermidine"
        'processes': [],   # e.g., "autophagy", "mitochondrial quality control"
        'level': None      # molecular, cellular, systemic
    }
    
    # Use NER or keyword extraction on concepts
    for concept in theory['key_concepts']:
        text = f"{concept['concept']} {concept['description']}"
        features['mechanisms'].extend(extract_mechanisms(text))
        features['pathways'].extend(extract_pathways(text))
        # ... etc
    
    return features
```

---

### Stage 2: Three-Level Hierarchical Clustering

#### 2.1 Level 1: Theory Families (30-50 clusters)

**Goal:** Group into broad categories

**Method:** Cluster on `name_embedding` with high distance threshold

```python
# Examples of theory families:
families = [
    "Evolutionary Theories",
    "Mitochondrial Theories",
    "DNA Damage Theories",
    "Cellular Senescence Theories",
    "Metabolic Theories",
    "Epigenetic Theories",
    # ... 30-50 total
]
```

**Distance threshold:** 0.7 (loose clustering)

#### 2.2 Level 2: Parent Theories (150-200 clusters)

**Goal:** Generic theories that encompass specific variants

**Method:** Cluster on `semantic_embedding` within each family

```python
# Example within Mitochondrial family:
parent_theories = [
    "Mitochondrial Dysfunction Theory",
    "Mitochondrial DNA Damage Theory",
    "Mitochondrial Quality Control Theory",
    "Mitochondrial Biogenesis Theory",
    # ... etc
]
```

**Distance threshold:** 0.5 (moderate clustering)

**Identification of generic theories:**
- Short names (≤4 words)
- Lack specific mechanisms
- Appear in multiple papers with variations

#### 2.3 Level 3: Child Theories (300-350 clusters)

**Goal:** Specific mechanism-based theories

**Method:** Cluster on `detailed_embedding` + concept features

```python
# Example children of "Mitochondrial Quality Control Theory":
child_theories = [
    "CB1 receptor-mediated regulation of mitochondrial quality control",
    "PINK1/Parkin-mediated mitophagy theory",
    "Mitochondrial unfolded protein response (UPRmt) theory",
    # ... each is distinct
]
```

**Distance threshold:** 0.35-0.4 (tight clustering)

**Critical:** Use concept features to prevent over-clustering
- Theories with same keywords but different mechanisms stay separate
- Example: Both mention "mitochondrial" but one is "CB1-mediated" and other is "p53-mediated"

---

### Stage 3: LLM-Based Validation & Distinction

#### 3.1 Fine-Grained Distinction Verification

**Problem:** Similar keywords but different mechanisms should stay separate

**Solution:** LLM validates each cluster for over-clustering

```python
def verify_fine_grained_distinction(cluster_theories):
    """Check if theories in cluster are truly the same or should be split."""
    
    prompt = f"""
You are an expert in aging biology. Analyze these {len(cluster_theories)} theories:

{format_theories_with_concepts(cluster_theories)}

Questions:
1. Do ALL these theories describe the EXACT SAME underlying mechanism? (yes/no)
2. If NO, identify distinct sub-groups based on:
   - Different molecular mechanisms (e.g., CB1 vs p53 mediated)
   - Different pathways (e.g., mTOR vs AMPK)
   - Different cellular processes (e.g., autophagy vs apoptosis)
3. For each sub-group, what is the key distinguishing feature?

IMPORTANT: Preserve meaningful mechanistic distinctions. Only merge if truly identical.
"""
    
    response = llm.complete(prompt)
    
    if response.should_split:
        return split_cluster(cluster_theories, response.subgroups)
    else:
        return [cluster_theories]  # Keep as one
```

#### 3.2 Parent-Child Relationship Identification

```python
def identify_parent_child(theories_in_family):
    """Identify which theories are generic (parents) vs specific (children)."""
    
    prompt = f"""
Given these theories in the same family:

{format_theories(theories_in_family)}

Identify:
1. Which theories are GENERIC (broad, encompassing concepts)?
2. Which theories are SPECIFIC (detailed mechanisms, specific molecules)?
3. For each specific theory, which generic theory is its parent?

Example:
- Parent: "Mitochondrial Dysfunction Theory"
- Children: 
  - "Cisd2-mediated mitochondrial protection theory"
  - "CB1 receptor-mediated mitochondrial quality control"
"""
    
    return parse_hierarchy(llm.complete(prompt))
```

#### 3.3 Canonical Naming with Hierarchy

```python
def generate_canonical_name(cluster, hierarchy_level):
    """Generate appropriate name based on hierarchy level."""
    
    if hierarchy_level == 'parent':
        # Generic, broad name
        prompt = "Generate a broad, encompassing name for this theory family..."
    else:  # child
        # Specific, mechanism-focused name
        prompt = "Generate a specific name that captures the unique mechanism..."
    
    return llm.complete(prompt)
```

---

### Stage 4: False Positive Re-Validation

#### 4.1 Singleton Theory Review

**Observation:** Theories that don't cluster may be:
- Truly novel/unique theories (keep)
- False positives (remove)
- Poorly extracted (re-extract or remove)

```python
def review_singletons(singleton_theories):
    """LLM validates theories that didn't cluster."""
    
    for theory in singleton_theories:
        prompt = f"""
Theory: {theory['name']}
Key Concepts: {theory['key_concepts']}
Confidence: {theory['confidence_is_theory']}

Is this a valid, well-defined aging theory? Consider:
1. Does it explain a mechanism or cause of aging?
2. Is it generalizable beyond a single observation?
3. Is it distinct from known theories?

Answer: valid / false_positive / needs_clarification
"""
        
        result = llm.complete(prompt)
        
        if result == 'false_positive':
            mark_for_removal(theory)
        elif result == 'needs_clarification':
            flag_for_human_review(theory)
```

#### 4.2 Cross-Cluster Validation

Check for theories that might belong to multiple clusters:

```python
def detect_multi_cluster_theories(theory, all_clusters):
    """Find theories that might fit in multiple clusters."""
    
    similarities = []
    for cluster in all_clusters:
        sim = compute_similarity(theory, cluster.centroid)
        if sim > 0.6:  # High similarity to multiple clusters
            similarities.append((cluster, sim))
    
    if len(similarities) > 1:
        # Theory spans multiple concepts
        # LLM decides: which cluster or create new hybrid cluster
        return llm_resolve_multi_cluster(theory, similarities)
```

---

### Stage 5: Ontology Integration with Hierarchy

#### 5.1 Hierarchical Matching

```python
def match_to_ontology_hierarchical(normalized_theory, ontology):
    """Match considering parent-child relationships."""
    
    # First, try to match parent
    if normalized_theory.is_parent:
        parent_match = find_best_match(
            normalized_theory,
            ontology_theories
        )
        
        if parent_match.confidence > 0.85:
            # Adopt ontology name and structure
            normalized_theory.canonical_name = parent_match.name
            normalized_theory.category = parent_match.category
    
    # For children, match to ontology or mark as novel variant
    else:
        # Check if parent exists in ontology
        if normalized_theory.parent in ontology:
            # Novel child of known parent
            normalized_theory.status = 'novel_variant'
        else:
            # Completely novel theory
            normalized_theory.status = 'novel_theory'
```

---

## Handling Edge Cases

### Case 1: Very Similar Theories with Subtle Differences

**Example:**
- "Mitochondrial ROS-induced aging via Complex I dysfunction"
- "Mitochondrial ROS-induced aging via Complex III dysfunction"

**Solution:**
- Detailed embedding captures "Complex I" vs "Complex III"
- Concept features flag different mechanisms
- LLM validation confirms these are distinct
- **Result:** Two separate child theories under "Mitochondrial ROS Theory" parent

### Case 2: Same Theory, Different Naming Conventions

**Example:**
- "Oxidative Stress Theory of Aging"
- "Free Radical Theory of Aging"
- "ROS Damage Theory"

**Solution:**
- High semantic similarity (>0.9)
- LLM confirms these are synonyms
- **Result:** One normalized theory with multiple aliases

### Case 3: Hierarchical Ambiguity

**Example:**
- "mTOR Signaling Theory" - is this parent or child?

**Solution:**
- Check if more specific variants exist
- If yes → parent (e.g., "Rapamycin-induced mTOR inhibition theory" is child)
- If no → standalone theory
- **Result:** Flexible hierarchy based on data

---

## Implementation Strategy

### Phase 1: Quality Filtering (Day 1)
```python
# 1. Load all theories
theories = load_theories('theories_per_paper.json')  # 14K theories

# 2. Filter by confidence
filtered = filter_by_confidence(theories)  # ~13.3K theories

# 3. Enrich with concepts
enriched = [enrich_theory_data(t) for t in filtered]

# Save checkpoint
save_checkpoint('01_filtered_enriched.json', enriched)
```

### Phase 2: Embedding Generation (Day 1-2)
```python
# 1. Generate multi-level embeddings
for theory in enriched:
    theory['embeddings'] = create_multi_level_embeddings(theory)
    theory['features'] = extract_concept_features(theory)

# 2. Save embeddings (for reuse)
save_embeddings('embeddings.pkl', enriched)
```

### Phase 3: Hierarchical Clustering (Day 2-3)
```python
# 1. Level 1: Theory families
families = cluster_level1(enriched, threshold=0.7)  # 30-50 families

# 2. Level 2: Parent theories
parents = []
for family in families:
    family_parents = cluster_level2(family.theories, threshold=0.5)
    parents.extend(family_parents)  # 150-200 parents

# 3. Level 3: Child theories
children = []
for parent in parents:
    parent_children = cluster_level3(parent.theories, threshold=0.4)
    children.extend(parent_children)  # 300-350 children

# Save hierarchy
save_hierarchy('02_hierarchy.json', families, parents, children)
```

### Phase 4: LLM Validation (Day 3-5)
```python
# 1. Validate each cluster for over-clustering
for cluster in children:
    validated = verify_fine_grained_distinction(cluster)
    if validated.should_split:
        children = split_and_replace(children, cluster, validated.subgroups)

# 2. Identify parent-child relationships
for family in families:
    hierarchy = identify_parent_child(family.all_theories)
    family.set_hierarchy(hierarchy)

# 3. Generate canonical names
for cluster in children:
    cluster.canonical_name = generate_canonical_name(cluster, 'child')

for parent in parents:
    parent.canonical_name = generate_canonical_name(parent, 'parent')

# Save validated
save_checkpoint('03_validated.json', families, parents, children)
```

### Phase 5: Quality Assurance (Day 5-6)
```python
# 1. Review singletons
singletons = [t for t in enriched if t.cluster_id is None]
validated_singletons = review_singletons(singletons)

# 2. Cross-cluster validation
multi_cluster = detect_multi_cluster_theories(enriched, children)

# 3. Ontology matching
ontology = load_ontology('ontology/initial_ontology.json')
for theory in children + parents:
    theory.ontology_match = match_to_ontology_hierarchical(theory, ontology)

# Save final
save_final('04_normalized_theories.json', families, parents, children)
```

---

## Expected Outcomes (Revised)

### Input
- 14,000 raw theories
- 77.7% high confidence, 21.5% medium, 0.7% low

### After Filtering (Stage 0)
- ~13,300 validated theories
- ~700 false positives removed

### After Clustering (Stage 2)
- **Level 1:** 30-50 theory families
- **Level 2:** 150-200 parent theories (generic)
- **Level 3:** 300-350 child theories (specific)
- Compression ratio: ~38:1 (13,300 → 350)

### Quality Metrics
- **Clustering accuracy:** >90% (validated by LLM)
- **False positive removal:** >85% of low/medium-low confidence
- **Fine-grained preservation:** >95% of mechanistic distinctions preserved
- **Ontology coverage:** ~60-70% match to known theories, 30-40% novel

### Output Structure
```json
{
  "theory_families": [
    {
      "family_id": "F001",
      "name": "Mitochondrial Theories",
      "parent_theories": [
        {
          "parent_id": "P001",
          "canonical_name": "Mitochondrial Dysfunction Theory",
          "alternative_names": ["Mitochondrial Theory of Aging"],
          "child_theories": [
            {
              "child_id": "C001",
              "canonical_name": "Cisd2-mediated mitochondrial protection theory",
              "key_mechanism": "Cisd2 protein regulation",
              "original_names": ["Cisd2-mediated...", "Cisd2 theory..."],
              "paper_count": 3,
              "confidence": 0.95
            },
            {
              "child_id": "C002",
              "canonical_name": "CB1 receptor-mediated mitochondrial quality control theory",
              "key_mechanism": "CB1 receptor signaling",
              "original_names": ["CB1 receptor-mediated..."],
              "paper_count": 2,
              "confidence": 0.92
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Cost & Timeline (Revised for 14K theories)

### Computational Costs

**Embeddings:**
- 13,300 theories × 100 tokens avg = 1.33M tokens
- OpenAI text-embedding-3-large: $0.13/1M tokens
- Cost: ~$0.20

**LLM Validation:**
- 350 child clusters × $0.02/validation = $7
- 200 parent clusters × $0.015/validation = $3
- 50 family validations × $0.01 = $0.50
- 500 singleton reviews × $0.01 = $5
- Canonical naming: 550 theories × $0.02 = $11
- **Total LLM:** ~$26.50

**Grand Total:** ~$27 (very affordable!)

### Timeline

- **Day 1:** Quality filtering + enrichment (4 hours)
- **Day 2:** Embedding generation (6 hours with batching)
- **Day 3:** Three-level clustering (4 hours)
- **Day 4-5:** LLM validation (12 hours, can parallelize)
- **Day 6:** Quality assurance + ontology matching (6 hours)
- **Day 7:** Human review of flagged cases (4 hours)

**Total:** 7 days with single-threaded processing
**Optimized:** 3-4 days with parallelization

---

## Success Criteria (Refined)

### Must Have ✓
1. Reduce 14K → 300-350 normalized theories
2. Remove >85% of false positives (low confidence)
3. Preserve fine-grained mechanistic distinctions
4. Create parent-child hierarchy
5. Match to initial ontology where applicable

### Should Have ✓
1. <5% theories requiring manual review
2. >90% clustering accuracy (LLM-validated)
3. Traceability: raw name → normalized theory
4. Confidence scores for all normalizations

### Nice to Have
1. Interactive visualization of hierarchy
2. Export to multiple formats (JSON, CSV, Neo4j graph)
3. API for querying normalized theories

---

## Risk Mitigation (Updated)

### Risk 1: Over-Clustering (Losing Important Distinctions)
**Mitigation:**
- Use detailed embeddings + concept features
- LLM validation specifically checks for over-clustering
- Conservative distance thresholds at Level 3
- Human review of merged theories with different mechanisms

### Risk 2: Under-Clustering (Too Many Theories)
**Mitigation:**
- Three-level hierarchy naturally consolidates
- LLM identifies synonyms and variants
- Parent theories provide consolidation layer

### Risk 3: False Positive Retention
**Mitigation:**
- Aggressive filtering of low confidence
- LLM re-validation of medium confidence
- Singleton review catches outliers
- Cross-validation against ontology

### Risk 4: Computational Scale
**Mitigation:**
- Batch processing for embeddings
- Hierarchical clustering reduces pairwise comparisons
- Cache LLM responses
- Parallelize independent validations

---

## Next Steps

1. **Review this refined approach** - Does it address all your concerns?
2. **Implement Stage 0** - Quality filtering and enrichment
3. **Test on subset** - Run on 1,000 theories to validate approach
4. **Iterate** - Refine thresholds and prompts based on test results
5. **Full pipeline** - Process all 14K theories
6. **Human review** - Review flagged cases and refine

**Ready to proceed with implementation?**
