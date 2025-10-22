# Theory Normalization Solution: Comprehensive Design

## Executive Summary
A multi-stage pipeline combining semantic embeddings, hierarchical clustering, LLM-based validation, and reference ontology mapping to normalize ~14,000 aging theory names into ~300-350 canonical theories with parent-child hierarchy.

**Key Enhancements:**
- **Confidence-based filtering:** Remove false positives (low confidence theories)
- **Hierarchical structure:** Preserve parent-child relationships (generic → specific)
- **Fine-grained distinction:** Preserve meaningful differences in similar theories
- **Scale-optimized:** Handles 14K theories efficiently

---

## Architecture Overview

```
Stage 0: Quality Filtering & Preprocessing
    ├─ Confidence-based filtering (remove low confidence)
    ├─ False positive detection
    └─ Data enrichment from key_concepts

Stage 1: Feature Extraction & Embedding
    ├─ Text normalization
    ├─ Semantic embeddings (name + key_concepts + description)
    └─ Concept-level features for fine-grained distinction

Stage 2: Multi-Level Hierarchical Clustering
    ├─ Coarse clustering (high-level theory families)
    ├─ Fine-grained clustering (specific theories)
    └─ Outlier detection

Stage 3: LLM-Based Validation & Distinction
    ├─ Cluster coherence validation
    ├─ Fine-grained distinction verification (similar but different)
    ├─ Canonical name generation
    └─ Parent-child relationship identification

Stage 4: Ontology Integration
    ├─ Match to known theories (initial_ontology.json)
    ├─ Novel theory identification
    └─ Quality assurance

Stage 5: Quality Assurance & Review
    ├─ False positive re-validation
    ├─ Boundary case visualization
    ├─ Medium confidence theory review
    └─ Iterative refinement
```

---

## Stage 1: Preprocessing & Feature Extraction

### 1.1 Text Normalization
```python
# Remove common variations
- Case normalization
- Remove "Theory/Hypothesis/of Aging" suffixes
- Handle abbreviations (SASP, ROS, etc.)
- Clean punctuation inconsistencies
```

### 1.2 Semantic Embeddings
**Approach:** Use OpenAI `text-embedding-3-large` or `text-embedding-ada-002`

**Why OpenAI embeddings:**
- Superior semantic understanding for domain-specific text
- Already integrated in your Azure environment
- Better than open-source alternatives for biomedical concepts

**Alternative:** `sentence-transformers/all-mpnet-base-v2` (free, local)

**Input for embeddings:**
```
Theory name + key_concepts + description (if available)
```

### 1.3 Feature Enrichment
Extract additional features:
- Core domain keywords (mitochondrial, oxidative, DNA, etc.)
- Mechanism type (programmed, damage, evolutionary, etc.)
- Biological level (molecular, cellular, systemic)

---

## Stage 2: Multi-Level Clustering

### 2.1 Hierarchical Approach

**Why hierarchical clustering:**
- Naturally creates theory families → specific theories
- No need to pre-specify number of clusters
- Can cut at different heights for coarse/fine granularity

**Algorithm:** HDBSCAN or Agglomerative Clustering
- **HDBSCAN:** Better for noise detection, density-based
- **Agglomerative:** Better for hierarchical structure

### 2.2 Two-Pass Clustering

**Pass 1: Coarse Clustering (Theory Families)**
- Distance threshold: 0.6-0.7
- Expected: ~30-50 major families
- Examples:
  - Evolutionary theories family
  - Mitochondrial theories family
  - DNA damage theories family

**Pass 2: Fine Clustering (Within Families)**
- Distance threshold: 0.4-0.5
- Expected: ~300-350 specific theories
- Preserves meaningful distinctions

### 2.3 Similarity Scoring
```python
similarity = (
    0.6 * cosine_similarity(embedding1, embedding2) +
    0.2 * keyword_overlap(theory1, theory2) +
    0.2 * concept_similarity(concepts1, concepts2)
)
```

---

## Stage 3: LLM-Based Validation & Naming

### 3.1 Cluster Validation
For each cluster, use LLM to:
1. Verify theories share core concept
2. Identify misclustered theories
3. Suggest cluster splits/merges

**Prompt template:**
```
You are an expert in aging biology. Analyze these theory names:
[list of 5-20 theories in cluster]

Questions:
1. Do these theories describe the SAME underlying idea? (yes/no/mostly)
2. If no, which theories should be separated?
3. What is the core unifying concept?
```

### 3.2 Canonical Name Generation
For validated clusters, generate canonical name:

**Prompt template:**
```
Given these theory names describing the same aging concept:
1. [theory name 1]
2. [theory name 2]
...

Generate:
1. A canonical name that captures the core concept
2. Alternative names/aliases
3. Key distinguishing features from related theories

Format: Use established naming conventions in aging research.
Prioritize clarity over brevity.
```

### 3.3 Hierarchical Assignment
Place each normalized theory in the hierarchy:
- **Category** (e.g., "Mechanistic Theories")
- **Sub-category** (e.g., "Molecular Damage")
- **Theory name** (e.g., "Free Radical/Oxidative Stress Theory")

---

## Stage 4: Ontology Integration

### 4.1 Match to Known Theories
Compare normalized theories with `initial_ontology.json`:
- Exact matches → adopt canonical name from ontology
- Partial matches → LLM decides if same/different
- No matches → novel theory candidate

### 4.2 Confidence Scoring
```python
confidence_score = {
    "high": exact match or strong consensus (>90% similarity),
    "medium": partial match or moderate consensus (70-90%),
    "low": weak match or divergent names (<70%)
}
```

### 4.3 Novel Theory Identification
Theories not matching ontology:
- Flag for expert review
- Check if truly novel or mismatch
- Add to growing normalized ontology

---

## Stage 5: Human-in-the-Loop Review

### 5.1 Uncertainty Handling
**Auto-approve:** High confidence clusters (>90% coherence)
**Flag for review:** 
- Low confidence clusters
- Boundary cases between clusters
- Singleton theories (no cluster)

### 5.2 Review Interface
Create simple web interface:
- View cluster members
- See embedding visualization (t-SNE/UMAP)
- Accept/reject/modify canonical names
- Split/merge clusters

### 5.3 Active Learning
- Start with reviewing highest-impact clusters
- System learns from corrections
- Iteratively improves clustering

---

## Implementation Plan

### Tech Stack
- **Embeddings:** OpenAI API or sentence-transformers
- **Clustering:** scikit-learn, HDBSCAN
- **LLM validation:** Azure OpenAI (GPT-4)
- **Visualization:** plotly, matplotlib
- **Storage:** JSON, SQLite (existing theories.db)

### Pipeline Components

#### Component 1: Theory Embedder
```python
class TheoryEmbedder:
    def embed_theory(self, theory_dict):
        """Create rich embedding from theory data"""
        text = f"{theory['name']} {theory['key_concepts']}"
        return openai.embeddings.create(input=text)
```

#### Component 2: Hierarchical Clusterer
```python
class TheoryClusterer:
    def cluster(self, embeddings, method='agglomerative'):
        """Two-pass hierarchical clustering"""
        # Pass 1: Coarse families
        # Pass 2: Fine theories
        return clusters, hierarchy
```

#### Component 3: LLM Validator
```python
class ClusterValidator:
    def validate_cluster(self, theory_names):
        """Use LLM to validate cluster coherence"""
        return coherence_score, canonical_name, issues
```

#### Component 4: Ontology Matcher
```python
class OntologyMatcher:
    def match_to_ontology(self, normalized_theory):
        """Match against initial_ontology.json"""
        return best_match, confidence
```

---

## Evaluation Metrics

### Quantitative
1. **Cluster purity:** % of theories in cluster truly similar
2. **Coverage:** % of theories successfully normalized
3. **Compression ratio:** unique theories / normalized theories
4. **Ontology match rate:** % matching known theories

### Qualitative
1. **Expert validation:** Sample review by domain expert
2. **Semantic coherence:** LLM-scored cluster quality
3. **Name quality:** Canonical names follow conventions

---

## Expected Outcomes

### Quantitative Goals
- **Input:** 6,000 raw theory names
- **Output:** 300-350 normalized theories
- **Compression:** ~17:1 ratio
- **Accuracy:** >90% clustering accuracy
- **Coverage:** >95% theories assigned

### Deliverables
1. **Normalized theory database** with mappings
2. **Expanded ontology** (initial + novel theories)
3. **Confidence scores** for each normalization
4. **Review queue** for uncertain cases
5. **Visualization dashboard** for exploration

---

## Alternative Approaches Considered

### Option A: Pure LLM Approach
**Pros:** Most accurate semantic understanding
**Cons:** 
- Expensive (6000 theories × pairwise comparisons = 18M comparisons)
- Slow (~1 week runtime)
- Inconsistent across batches

### Option B: Pure Clustering
**Pros:** Fast, scalable
**Cons:**
- Needs manual threshold tuning
- No semantic validation
- Poor canonical naming

### Option C: Rule-Based + Fuzzy Matching
**Pros:** Fast, deterministic
**Cons:**
- Brittle, many edge cases
- Misses semantic equivalence
- High false positive rate

### **Recommended: Hybrid Approach**
Combines strengths:
- Embeddings for semantic understanding
- Clustering for scalability
- LLM for validation and naming
- Ontology for grounding

---

## Risk Mitigation

### Risk 1: Over-clustering
**Risk:** Too many clusters (theories too specific)
**Mitigation:** 
- Tune distance thresholds conservatively
- LLM validation catches over-splits
- Compare cluster count to ontology

### Risk 2: Under-clustering
**Risk:** Too few clusters (losing important distinctions)
**Mitigation:**
- Two-pass clustering preserves hierarchy
- LLM suggests splits for heterogeneous clusters
- Expert review of large clusters

### Risk 3: LLM Inconsistency
**Risk:** Different runs produce different results
**Mitigation:**
- Use temperature=0 for deterministic outputs
- Batch similar decisions together
- Cache LLM decisions

### Risk 4: Novel Theory Identification
**Risk:** Missing truly novel theories or flagging non-novel
**Mitigation:**
- Conservative ontology matching
- Expert review of "novel" theories
- Track provenance (which papers, concepts)

---

## Timeline Estimate

### Phase 1: Implementation (1 week)
- Days 1-2: Embedding generation
- Days 3-4: Clustering pipeline
- Days 5-6: LLM validation
- Day 7: Integration & testing

### Phase 2: Processing (2-3 days)
- Day 1: Embed all theories
- Day 2: Clustering & validation
- Day 3: Ontology matching & QA

### Phase 3: Review (ongoing)
- Initial: Review top 100 uncertain cases
- Iterative: Refine based on feedback

**Total:** ~2 weeks for full pipeline + ongoing refinement

---

## Cost Estimation

### OpenAI API Costs
- **Embeddings:** 6,000 theories × $0.00013/1K tokens × 50 tokens ≈ $0.40
- **LLM validation:** 350 clusters × $0.01/call × 2 passes ≈ $7
- **Canonical naming:** 350 clusters × $0.02/call ≈ $7
- **Total:** ~$15-20

**Very affordable compared to manual work!**

---

## Success Criteria

### Must Have
✓ Reduce 6,000 theories to 300-350 normalized theories
✓ Maintain scientific accuracy (validated by sample review)
✓ Map to initial ontology where applicable
✓ Provide confidence scores

### Should Have
✓ Hierarchical organization (family → theory)
✓ Alternative names/aliases preserved
✓ Traceability (which papers → which theory)
✓ Visualization for exploration

### Nice to Have
✓ Web interface for review
✓ Active learning from corrections
✓ Export to standard formats (JSON, CSV, Neo4j)

---

## Next Steps

1. **Approve approach** (you decide if this is the right direction)
2. **Implement embedding generation**
3. **Build clustering pipeline**
4. **Integrate LLM validation**
5. **Create review interface**
6. **Run on full dataset**
7. **Iterate based on review**

Would you like me to proceed with implementation?
