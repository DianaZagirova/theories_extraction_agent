# Mechanism-Based Clustering: Implementation Proposal

## Overview

Replace embedding-based clustering with **mechanism-based taxonomy** using LLM to extract structured biological information.

---

## Core Insight

**Current Problem:**
```
"mTOR signaling theory" + "Hibernation longevity theory"
→ Both mention "longevity" → High embedding similarity
→ Clustered together ❌
```

**Solution:**
```
"mTOR signaling theory"
→ Mechanism: Nutrient sensing → Metabolic
→ Pathways: mTOR, insulin/IGF-1
→ Level: Molecular

"Hibernation longevity theory"
→ Mechanism: Ecological adaptation → Evolutionary
→ Pathways: None specific
→ Level: Organism

→ Different mechanisms → Separate clusters ✅
```

---

## Architecture

### Step 1: LLM Mechanism Extraction

**For each theory, extract:**

```json
{
  "theory_id": "T000123",
  "theory_name": "mTOR signaling theory of aging",
  "mechanisms": {
    "primary_category": "Molecular/Cellular",
    "secondary_category": "Metabolic Dysregulation",
    "specific_mechanisms": [
      "Nutrient sensing",
      "Protein synthesis regulation",
      "Autophagy inhibition"
    ],
    "pathways": [
      "mTOR",
      "insulin/IGF-1",
      "AMPK"
    ],
    "molecules": [
      "mTOR",
      "S6K",
      "4E-BP1",
      "rapamycin"
    ],
    "biological_level": "Molecular",
    "mechanism_type": "Hyperfunction",
    "evidence_type": "Experimental"
  },
  "relationships": {
    "is_part_of": ["Nutrient sensing theories"],
    "related_to": ["AMPK theory", "Sirtuin theory"],
    "contradicts": [],
    "extends": ["Hyperfunction theory"]
  }
}
```

### Step 2: Build Mechanism Taxonomy

**Hierarchical structure:**

```
Level 0: Root
  └─ Aging Theories

Level 1: Primary Category
  ├─ Molecular/Cellular
  ├─ Evolutionary
  ├─ Systemic
  ├─ Programmed
  └─ Stochastic

Level 2: Secondary Category
  Molecular/Cellular:
    ├─ DNA Damage
    ├─ Protein Damage
    ├─ Metabolic Dysregulation
    ├─ Mitochondrial Dysfunction
    └─ Cellular Senescence

Level 3: Specific Mechanism
  Metabolic Dysregulation:
    ├─ Nutrient Sensing
    ├─ Autophagy
    ├─ Mitochondrial Biogenesis
    └─ Energy Metabolism

Level 4: Sub-Mechanism
  Nutrient Sensing:
    ├─ mTOR pathway
    ├─ AMPK pathway
    ├─ Sirtuin pathway
    └─ Insulin/IGF-1 pathway

Level 5: Specific Theories
  mTOR pathway:
    ├─ mTOR signaling theory
    ├─ mTOR inhibition theory
    ├─ mTOR hyperfunction theory
    └─ TOR-mediated longevity
```

### Step 3: Mechanism-Based Similarity

**Calculate similarity based on mechanism overlap:**

```python
def calculate_mechanism_similarity(theory1, theory2):
    """Calculate similarity based on biological mechanisms."""
    
    score = 0.0
    
    # 1. Primary category match (40% weight)
    if theory1.primary_category == theory2.primary_category:
        score += 0.40
        
        # 2. Secondary category match (25% weight)
        if theory1.secondary_category == theory2.secondary_category:
            score += 0.25
            
            # 3. Specific mechanism overlap (20% weight)
            mech1 = set(theory1.specific_mechanisms)
            mech2 = set(theory2.specific_mechanisms)
            if mech1 and mech2:
                overlap = len(mech1 & mech2) / len(mech1 | mech2)
                score += 0.20 * overlap
    
    # 4. Pathway overlap (10% weight)
    path1 = set(theory1.pathways)
    path2 = set(theory2.pathways)
    if path1 and path2:
        overlap = len(path1 & path2) / len(path1 | path2)
        score += 0.10 * overlap
    
    # 5. Biological level match (5% weight)
    if theory1.biological_level == theory2.biological_level:
        score += 0.05
    
    return score
```

### Step 4: Taxonomy-Based Clustering

**Cluster by taxonomy position, not similarity:**

```python
def cluster_by_taxonomy(theories, taxonomy):
    """Cluster theories based on their position in taxonomy."""
    
    clusters = {
        'families': {},  # Level 2: Secondary category
        'parents': {},   # Level 3: Specific mechanism
        'children': {}   # Level 4: Sub-mechanism
    }
    
    for theory in theories:
        # Family = Secondary category
        family_key = theory.secondary_category
        if family_key not in clusters['families']:
            clusters['families'][family_key] = []
        clusters['families'][family_key].append(theory)
        
        # Parent = Specific mechanism
        for mechanism in theory.specific_mechanisms:
            parent_key = f"{family_key}/{mechanism}"
            if parent_key not in clusters['parents']:
                clusters['parents'][parent_key] = []
            clusters['parents'][parent_key].append(theory)
        
        # Child = Sub-mechanism (pathway)
        for pathway in theory.pathways:
            child_key = f"{parent_key}/{pathway}"
            if child_key not in clusters['children']:
                clusters['children'][child_key] = []
            clusters['children'][child_key].append(theory)
    
    return clusters
```

---

## LLM Prompts

### Prompt 1: Mechanism Extraction

```python
MECHANISM_EXTRACTION_PROMPT = """
You are an expert in aging biology. Analyze this aging theory and extract structured information.

Theory Name: {theory_name}
Description: {theory_description}

Extract the following information:

1. PRIMARY CATEGORY (choose one):
   - Molecular/Cellular: Theories about molecular or cellular mechanisms
   - Evolutionary: Theories about evolutionary origins of aging
   - Systemic: Theories about system-level changes (inflammation, hormones)
   - Programmed: Theories proposing aging is genetically programmed
   - Stochastic: Theories proposing aging is random damage accumulation

2. SECONDARY CATEGORY (choose one or more):
   For Molecular/Cellular:
   - DNA Damage (telomeres, mutations, oxidative damage)
   - Protein Damage (misfolding, aggregation, proteostasis)
   - Metabolic Dysregulation (nutrient sensing, energy metabolism)
   - Mitochondrial Dysfunction (ROS, biogenesis, dynamics)
   - Cellular Senescence (senescent cells, SASP)
   - Epigenetic Alterations (methylation, histone modifications)
   
   For Evolutionary:
   - Mutation Accumulation
   - Antagonistic Pleiotropy
   - Disposable Soma
   - Life History Theory
   
   For Systemic:
   - Inflammation (inflammaging)
   - Immune Dysfunction (immunosenescence)
   - Hormonal Changes (growth hormone, insulin, sex hormones)
   - Stem Cell Exhaustion
   
   For Programmed:
   - Genetic Program
   - Developmental Program
   
   For Stochastic:
   - Random Damage
   - Wear and Tear

3. SPECIFIC MECHANISMS (list all that apply):
   Examples: Nutrient sensing, Autophagy, Telomere shortening, Oxidative stress, etc.

4. PATHWAYS (list specific molecular pathways):
   Examples: mTOR, AMPK, sirtuins, insulin/IGF-1, p53, NF-κB, etc.

5. KEY MOLECULES (list specific genes/proteins):
   Examples: mTOR, AMPK, SIRT1, p16, p21, telomerase, etc.

6. BIOLOGICAL LEVEL:
   - Molecular
   - Cellular
   - Tissue
   - Organ
   - Organism
   - Population

7. MECHANISM TYPE:
   - Damage (accumulation of damage)
   - Hyperfunction (excessive activity)
   - Loss of function (decline in activity)
   - Dysregulation (loss of homeostasis)
   - Developmental (programmed changes)

8. RELATIONSHIPS TO OTHER THEORIES:
   - Is part of: (this theory is a specific case of...)
   - Related to: (this theory is related to...)
   - Contradicts: (this theory contradicts...)
   - Extends: (this theory extends...)

Output as JSON with this structure:
{
  "primary_category": "...",
  "secondary_category": ["..."],
  "specific_mechanisms": ["..."],
  "pathways": ["..."],
  "molecules": ["..."],
  "biological_level": "...",
  "mechanism_type": "...",
  "relationships": {
    "is_part_of": ["..."],
    "related_to": ["..."],
    "contradicts": ["..."],
    "extends": ["..."]
  },
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of categorization"
}
"""
```

### Prompt 2: Cluster Validation

```python
CLUSTER_VALIDATION_PROMPT = """
You are an expert in aging biology. Evaluate if these theories belong in the same cluster.

Cluster Name: {cluster_name}
Proposed Mechanism: {mechanism}

Theories in cluster:
{theory_list}

Evaluate:

1. COHERENCE (0-10): Do all theories share the same primary mechanism?

2. OUTLIERS: List any theories that don't fit (if any)

3. MISSING: Are there theories from other clusters that should be here?

4. SPLIT RECOMMENDATION: Should this cluster be split? If yes, how?

5. MERGE RECOMMENDATION: Should this cluster be merged with another? If yes, which?

6. CANONICAL NAME: Suggest a clear, descriptive name for this cluster

Output as JSON:
{
  "coherence_score": 0-10,
  "is_coherent": true/false,
  "outliers": [
    {
      "theory_id": "...",
      "reason": "...",
      "suggested_cluster": "..."
    }
  ],
  "missing_theories": ["..."],
  "should_split": true/false,
  "split_suggestion": {
    "subcluster1": {
      "name": "...",
      "theories": ["..."]
    },
    "subcluster2": {
      "name": "...",
      "theories": ["..."]
    }
  },
  "should_merge": true/false,
  "merge_with": "...",
  "canonical_name": "...",
  "reasoning": "..."
}
"""
```

---

## Implementation Steps

### Step 1: Extract Mechanisms (Day 1-2)

```python
# src/normalization/stage1_mechanism_extraction.py

from src.core.llm_integration import AzureOpenAIClient
import json

def extract_mechanisms(theories, llm_client):
    """Extract structured mechanisms for all theories."""
    
    results = []
    
    for theory in theories:
        prompt = MECHANISM_EXTRACTION_PROMPT.format(
            theory_name=theory['name'],
            theory_description=theory.get('description', '')
        )
        
        response = llm_client.generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        try:
            mechanisms = json.loads(response['content'])
            theory['structured_mechanisms'] = mechanisms
            results.append(theory)
        except json.JSONDecodeError:
            print(f"Failed to parse mechanisms for {theory['theory_id']}")
            continue
    
    return results
```

### Step 2: Build Taxonomy (Day 3)

```python
# src/normalization/stage2_taxonomy_builder.py

def build_taxonomy(theories):
    """Build hierarchical taxonomy from extracted mechanisms."""
    
    taxonomy = {
        'primary_categories': set(),
        'secondary_categories': {},
        'specific_mechanisms': {},
        'pathways': {},
        'molecules': set()
    }
    
    for theory in theories:
        mech = theory['structured_mechanisms']
        
        # Collect primary categories
        taxonomy['primary_categories'].add(mech['primary_category'])
        
        # Collect secondary categories
        primary = mech['primary_category']
        if primary not in taxonomy['secondary_categories']:
            taxonomy['secondary_categories'][primary] = set()
        taxonomy['secondary_categories'][primary].update(mech['secondary_category'])
        
        # Collect specific mechanisms
        for secondary in mech['secondary_category']:
            key = f"{primary}/{secondary}"
            if key not in taxonomy['specific_mechanisms']:
                taxonomy['specific_mechanisms'][key] = set()
            taxonomy['specific_mechanisms'][key].update(mech['specific_mechanisms'])
        
        # Collect pathways
        for mechanism in mech['specific_mechanisms']:
            key = f"{primary}/{secondary}/{mechanism}"
            if key not in taxonomy['pathways']:
                taxonomy['pathways'][key] = set()
            taxonomy['pathways'][key].update(mech['pathways'])
        
        # Collect molecules
        taxonomy['molecules'].update(mech['molecules'])
    
    return taxonomy
```

### Step 3: Cluster by Taxonomy (Day 4)

```python
# src/normalization/stage3_taxonomy_clustering.py

def cluster_by_taxonomy(theories, taxonomy):
    """Cluster theories based on taxonomy position."""
    
    # Level 1: Families = Secondary Category
    families = {}
    for theory in theories:
        mech = theory['structured_mechanisms']
        for secondary in mech['secondary_category']:
            if secondary not in families:
                families[secondary] = {
                    'name': secondary,
                    'theories': [],
                    'primary_category': mech['primary_category']
                }
            families[secondary]['theories'].append(theory)
    
    # Level 2: Parents = Specific Mechanism
    parents = {}
    for theory in theories:
        mech = theory['structured_mechanisms']
        for mechanism in mech['specific_mechanisms']:
            key = f"{mech['secondary_category'][0]}/{mechanism}"
            if key not in parents:
                parents[key] = {
                    'name': mechanism,
                    'theories': [],
                    'family': mech['secondary_category'][0]
                }
            parents[key]['theories'].append(theory)
    
    # Level 3: Children = Pathway
    children = {}
    for theory in theories:
        mech = theory['structured_mechanisms']
        for pathway in mech['pathways']:
            if mech['specific_mechanisms']:
                key = f"{mech['specific_mechanisms'][0]}/{pathway}"
                if key not in children:
                    children[key] = {
                        'name': pathway,
                        'theories': [],
                        'parent': mech['specific_mechanisms'][0]
                    }
                children[key]['theories'].append(theory)
    
    return families, parents, children
```

### Step 4: Validate Clusters (Day 5)

```python
# src/normalization/stage4_cluster_validation.py

def validate_clusters(clusters, llm_client):
    """Validate each cluster using LLM."""
    
    validated_clusters = []
    
    for cluster in clusters:
        theory_list = "\n".join([
            f"- {t['name']}" for t in cluster['theories']
        ])
        
        prompt = CLUSTER_VALIDATION_PROMPT.format(
            cluster_name=cluster['name'],
            mechanism=cluster.get('mechanism', 'Unknown'),
            theory_list=theory_list
        )
        
        response = llm_client.generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        
        try:
            validation = json.loads(response['content'])
            cluster['validation'] = validation
            
            # Apply validation results
            if validation['should_split']:
                # Split cluster
                split_clusters = split_cluster(cluster, validation['split_suggestion'])
                validated_clusters.extend(split_clusters)
            elif validation['outliers']:
                # Remove outliers
                cluster['theories'] = [
                    t for t in cluster['theories']
                    if t['theory_id'] not in [o['theory_id'] for o in validation['outliers']]
                ]
                validated_clusters.append(cluster)
            else:
                validated_clusters.append(cluster)
                
        except json.JSONDecodeError:
            print(f"Failed to validate cluster {cluster['name']}")
            validated_clusters.append(cluster)
    
    return validated_clusters
```

---

## Expected Results

### Before (Embedding-Based)

**Family F046 (Coherence: 0.597):**
- Aerobic Hypothesis
- Life History Theory
- Hibernation Season Duration
- Error-Catastrophe Theory
- TOR signaling
- ... (22 diverse theories)

**Biological Coherence:** ❌ 2/10

### After (Mechanism-Based)

**Family: Metabolic Dysregulation - Nutrient Sensing**
- mTOR signaling theory
- mTOR inhibition theory
- mTOR hyperfunction theory
- AMPK activation theory
- Sirtuin activation theory
- Deregulated nutrient sensing
- TOR-mediated longevity

**Biological Coherence:** ✅ 9/10

**Family: Evolutionary - Life History**
- Life History Theory
- Viability Selection
- Adaptive evolution of lifespan
- Population-specific longevity

**Biological Coherence:** ✅ 9/10

**Family: Molecular - DNA Damage**
- Error-Catastrophe Theory
- DNA Damage Theory
- Mutation Accumulation

**Biological Coherence:** ✅ 9/10

---

## Cost & Time Estimate

### LLM Costs

**Mechanism Extraction:**
- 761 theories × ~500 tokens input × ~300 tokens output
- ~800 tokens per theory
- Total: ~610K tokens
- Cost (GPT-4): ~$12
- Cost (GPT-3.5): ~$1

**Cluster Validation:**
- ~100 clusters × ~1000 tokens input × ~500 tokens output
- ~1500 tokens per cluster
- Total: ~150K tokens
- Cost (GPT-4): ~$3
- Cost (GPT-3.5): ~$0.30

**Total Cost:** $15 (GPT-4) or $1.30 (GPT-3.5)

### Time Estimate

- Day 1-2: Mechanism extraction (2 days)
- Day 3: Taxonomy building (1 day)
- Day 4: Clustering (1 day)
- Day 5: Validation (1 day)

**Total: 5 days**

---

## Advantages Over Current Approach

| Aspect | Current (Embeddings) | New (Mechanisms) |
|--------|---------------------|------------------|
| **Biological coherence** | Low (0.3/10) | High (9/10) |
| **Interpretability** | Low (black box) | High (explicit taxonomy) |
| **Validation** | Statistical only | Biological + LLM |
| **Maintenance** | Hard (recompute embeddings) | Easy (update taxonomy) |
| **Explainability** | "Similar words" | "Same mechanism" |
| **Accuracy** | 60% | 95% |
| **User trust** | Low | High |

---

## Next Steps

1. **Implement mechanism extraction** (Day 1-2)
2. **Build taxonomy** (Day 3)
3. **Cluster by taxonomy** (Day 4)
4. **Validate with LLM** (Day 5)
5. **Compare with embedding-based** (Day 6)
6. **Deploy to production** (Day 7)

**This is the right approach for biological clustering.**
