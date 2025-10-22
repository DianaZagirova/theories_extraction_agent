# Mechanism Improvement Plan

## 📊 Current State Analysis

### Results from Stage 4
```
Total theories: 1516
Canonical groups: 39
Novel groups: 0
Unmapped theories: 0

Top 5 Groups:
1. Free Radical Theory: 290 theories (3 mechanisms, 12 key players)
2. Disposable Soma Theory: 217 theories (5 mechanisms, 17 key players)
3. Antagonistic Pleiotropy Theory: 190 theories (4 mechanisms, 15 key players)
4. Mitochondrial Decline Theory: 174 theories (4 mechanisms, 13 key players)
5. Oxidative Stress Theory: 130 theories (4 mechanisms, 14 key players)
```

### 🚨 Problems Identified

#### 1. **Over-Mapping to Canonical Theories**
- **Issue**: Stage 1.5 mapped 1499/1516 theories (98.9%) to canonical
- **Problem**: Too aggressive mapping - many papers likely discuss specific aspects but get mapped to broad theories
- **Example**: A paper about "mitochondrial ROS in neurons" → mapped to "Free Radical Theory"

#### 2. **Generic Canonical Mechanisms**
- **Issue**: Canonical mechanisms are too broad
- **Example**: Free Radical Theory mechanisms:
  ```
  - "ROS generated as byproducts of aerobic metabolism"
  - "Oxidative damage to DNA, proteins, lipids"
  - "Accumulated damage impairs cellular function"
  ```
- **Problem**: These apply to 290 different papers, losing specificity

#### 3. **No Paper-Specific Context**
- **Issue**: All theories in a group share identical canonical mechanisms
- **Problem**: Can't distinguish between:
  - Paper about ROS in brain aging
  - Paper about ROS in muscle aging
  - Paper about ROS in caloric restriction

#### 4. **Missing Granularity**
- **Issue**: Large groups (290, 217, 190 theories) are too coarse
- **Problem**: Can't identify sub-theories or variations within canonical theories

---

## 🎯 Improvement Strategies

### **Strategy 1: Add Paper-Specific Mechanism Extraction**

#### Current Flow
```
Mapped theory → Assign canonical mechanisms → Done
```

#### Improved Flow
```
Mapped theory → Assign canonical mechanisms → Extract paper-specific details → Merge
```

#### Implementation

**Add to Stage 3:**
```python
def extract_paper_specific_mechanisms(self, theory: Dict, canonical_data: Dict) -> Dict:
    """
    Extract paper-specific mechanisms for a mapped theory.
    
    Args:
        theory: Theory with canonical mapping
        canonical_data: Canonical mechanisms from ontology
        
    Returns:
        Enhanced metadata with both canonical and paper-specific mechanisms
    """
    
    # Build prompt
    prompt = f"""Extract PAPER-SPECIFIC details for this theory.

CANONICAL THEORY: {canonical_data['name']}

CANONICAL MECHANISMS (for reference):
{chr(10).join([f'- {m}' for m in canonical_data['mechanisms']])}

PAPER CONTEXT:
Title: {theory.get('paper_title', 'N/A')}
Key Concepts: {theory.get('concept_text', 'N/A')[:500]}

TASK:
Extract what is SPECIFIC or NOVEL in this paper:

1. **SPECIFIC KEY PLAYERS**: Which specific molecules/factors does THIS paper focus on?
   (e.g., "SIRT1 in hippocampal neurons", "Complex I in skeletal muscle")

2. **SPECIFIC PATHWAYS**: Which specific pathways are discussed?
   (e.g., "AMPK activation in caloric restriction", "NF-κB in inflammaging")

3. **SPECIFIC MECHANISMS**: What specific mechanisms or variations does THIS paper describe?
   (e.g., "ROS-induced mitochondrial DNA mutations in aged neurons")

4. **CONTEXT**: What is the specific context?
   - Tissue/organ: brain, muscle, liver, etc.
   - Condition: aging, caloric restriction, exercise, etc.
   - Species: human, mouse, rat, C. elegans, etc.

OUTPUT FORMAT (JSON):
{{
  "specific_key_players": ["...", "..."],
  "specific_pathways": ["...", "..."],
  "specific_mechanisms": ["...", "..."],
  "context": {{
    "tissue": "...",
    "condition": "...",
    "species": "..."
  }}
}}
"""
    
    # Call LLM
    response = self.llm.client.chat.completions.create(...)
    
    # Parse and merge with canonical
    paper_specific = parse_response(response)
    
    return {
        'canonical_mechanisms': canonical_data['mechanisms'],
        'canonical_key_players': canonical_data['key_players'],
        'canonical_pathways': canonical_data['pathways'],
        'paper_specific_mechanisms': paper_specific['specific_mechanisms'],
        'paper_specific_key_players': paper_specific['specific_key_players'],
        'paper_specific_pathways': paper_specific['specific_pathways'],
        'context': paper_specific['context'],
        'source': 'hybrid'  # Both canonical and paper-specific
    }
```

---

### **Strategy 2: Sub-Group Clustering**

#### Concept
After grouping by canonical name, cluster papers within each group by paper-specific mechanisms.

#### Implementation

**Add to Stage 4:**
```python
def create_subgroups(self, canonical_group: TheoryGroup, theories: List[Dict]) -> List[TheoryGroup]:
    """
    Create sub-groups within a canonical group based on paper-specific mechanisms.
    
    Args:
        canonical_group: Large canonical group (e.g., Free Radical Theory)
        theories: All theories in this group
        
    Returns:
        List of sub-groups
    """
    
    # Extract paper-specific signatures
    paper_sigs = []
    for theory in theories:
        metadata = theory.get('stage3_metadata', {})
        sig = {
            'theory': theory,
            'specific_mechanisms': set(metadata.get('paper_specific_mechanisms', [])),
            'specific_players': set(metadata.get('paper_specific_key_players', [])),
            'context': metadata.get('context', {})
        }
        paper_sigs.append(sig)
    
    # Cluster by similarity
    subgroups = []
    while paper_sigs:
        seed = paper_sigs.pop(0)
        cluster = [seed]
        
        remaining = []
        for sig in paper_sigs:
            # Calculate similarity based on paper-specific details
            similarity = self._calculate_paper_similarity(seed, sig)
            if similarity >= 0.5:  # Threshold
                cluster.append(sig)
            else:
                remaining.append(sig)
        
        paper_sigs = remaining
        
        # Create subgroup
        subgroup = TheoryGroup(
            group_id=f"{canonical_group.group_id}_SUB{len(subgroups)+1}",
            canonical_name=canonical_group.canonical_name,
            representative_name=f"{canonical_group.canonical_name} - {get_cluster_label(cluster)}",
            theory_ids=[s['theory']['theory_id'] for s in cluster],
            theory_count=len(cluster),
            # ... other fields
        )
        subgroups.append(subgroup)
    
    return subgroups
```

**Example Output:**
```
Free Radical Theory (290 theories) →
  - Free Radical Theory - Mitochondrial ROS in Brain (45 theories)
  - Free Radical Theory - Oxidative Stress in Muscle (38 theories)
  - Free Radical Theory - ROS and Caloric Restriction (52 theories)
  - Free Radical Theory - Antioxidant Defense (31 theories)
  - ...
```

---

### **Strategy 3: Hierarchical Mechanism Structure**

#### Concept
Organize mechanisms in a hierarchy: General → Specific → Paper-specific

#### Structure
```json
{
  "mechanisms": {
    "level_1_canonical": [
      "ROS generated as byproducts of aerobic metabolism"
    ],
    "level_2_specific": [
      "Mitochondrial Complex I produces superoxide",
      "Complex III generates ROS at Qi site"
    ],
    "level_3_paper": [
      "Complex I ROS production increases 2-fold in aged rat brain mitochondria",
      "Caloric restriction reduces Complex I ROS by 40% in skeletal muscle"
    ]
  },
  "context_tags": [
    "tissue:brain",
    "tissue:muscle",
    "condition:aging",
    "condition:caloric_restriction",
    "species:rat"
  ]
}
```

---

### **Strategy 4: Improve Stage 1.5 Mapping Criteria**

#### Current Issue
Stage 1.5 is too aggressive - maps 98.9% of theories to canonical.

#### Improvements

**1. Stricter Mapping Threshold**
```python
# Current
if mapping_confidence >= 0.7:
    mapped_theories.append(theory)

# Improved
if mapping_confidence >= 0.85:  # Higher threshold
    mapped_theories.append(theory)
```

**2. Add "Partial Match" Category**
```python
if mapping_confidence >= 0.85:
    # Strong match - use canonical mechanisms
    category = 'canonical_match'
elif mapping_confidence >= 0.6:
    # Partial match - extract paper-specific mechanisms
    category = 'partial_match'
else:
    # Novel or unmatched
    category = 'novel'
```

**3. Update Prompt to Be More Conservative**
```
When mapping, be CONSERVATIVE:
- Only map if the paper's CORE MECHANISM matches the canonical theory
- If the paper discusses a SPECIFIC ASPECT or APPLICATION, mark as partial match
- If the paper proposes a NEW MECHANISM, mark as novel

Examples:
- "ROS in aging" → Free Radical Theory (canonical match)
- "ROS in brain aging via SIRT1" → Free Radical Theory (partial match - extract specifics)
- "Novel epigenetic clock based on DNA methylation" → Novel (new mechanism)
```

---

### **Strategy 5: Add Mechanism Enrichment Tags**

#### Concept
Tag mechanisms with metadata for better filtering and grouping.

#### Implementation
```python
{
  "mechanism": "Mitochondrial ROS production increases with age",
  "tags": {
    "level": "molecular",
    "tissue": "brain",
    "species": "rat",
    "intervention": "caloric_restriction",
    "direction": "increase",
    "quantified": true,
    "value": "2-fold increase"
  }
}
```

#### Benefits
- Filter by tissue: "Show all brain aging mechanisms"
- Filter by intervention: "Show all caloric restriction mechanisms"
- Compare quantified effects: "Which interventions reduce ROS most?"

---

## 📋 Implementation Roadmap

### Phase 1: Quick Wins (2-3 hours)
- [ ] **Adjust Stage 1.5 mapping threshold** (0.7 → 0.85)
- [ ] **Add partial match category** to Stage 1.5
- [ ] **Update Stage 1.5 prompt** to be more conservative
- [ ] **Test on sample data** and verify better distribution

**Expected Impact:**
- Reduce canonical mapping from 98.9% to ~70%
- Increase novel theories from 1.1% to ~20%
- More theories get paper-specific extraction

### Phase 2: Paper-Specific Extraction (4-6 hours)
- [ ] **Add paper-specific extraction** to Stage 3
- [ ] **Create hybrid mechanism format** (canonical + paper-specific)
- [ ] **Update Stage 4** to use paper-specific mechanisms for sub-grouping
- [ ] **Test and validate** on sample data

**Expected Impact:**
- All mapped theories get paper-specific details
- Better granularity within canonical groups
- Can distinguish between different applications

### Phase 3: Sub-Grouping (3-4 hours)
- [ ] **Implement sub-group clustering** in Stage 4
- [ ] **Create cluster labels** based on common themes
- [ ] **Add hierarchical group structure**
- [ ] **Update output format** to show sub-groups

**Expected Impact:**
- Large groups (290 theories) split into 5-10 sub-groups
- Each sub-group has 20-60 theories with similar specifics
- Better organization and searchability

### Phase 4: Mechanism Enrichment (Optional, 2-3 hours)
- [ ] **Add mechanism tagging** system
- [ ] **Extract context tags** from papers
- [ ] **Create filtering API**
- [ ] **Build visualization** tools

**Expected Impact:**
- Rich metadata for each mechanism
- Advanced filtering and search
- Better analysis capabilities

---

## 🎯 Expected Outcomes

### Before Improvements
```
39 groups:
- Free Radical Theory: 290 theories (generic mechanisms)
- Disposable Soma Theory: 217 theories (generic mechanisms)
- Antagonistic Pleiotropy Theory: 190 theories (generic mechanisms)
...
```

### After Phase 1 (Stricter Mapping)
```
~60 groups:
- Free Radical Theory: 200 theories (canonical)
- Disposable Soma Theory: 150 theories (canonical)
- Novel Theory 1: 15 theories (extracted)
- Novel Theory 2: 12 theories (extracted)
...
```

### After Phase 2 (Paper-Specific Extraction)
```
~60 groups with enriched mechanisms:
- Free Radical Theory: 200 theories
  - Canonical: "ROS generated as byproducts..."
  - Paper-specific: "Complex I ROS in brain", "mtDNA mutations in muscle", etc.
```

### After Phase 3 (Sub-Grouping)
```
~150 groups (39 canonical + ~110 sub-groups):
- Free Radical Theory (parent)
  ├─ ROS in Brain Aging: 45 theories
  ├─ ROS in Muscle Aging: 38 theories
  ├─ ROS and Caloric Restriction: 52 theories
  ├─ Mitochondrial ROS: 65 theories
  └─ Antioxidant Defense: 31 theories
```

---

## 💡 Quick Start: Phase 1 Implementation

### 1. Update Stage 1.5 Mapping Threshold

**File**: `src/normalization/stage1_5_llm_mapping.py`

**Change**:
```python
# Line ~390
# OLD:
if result.is_mapped and result.canonical_name:
    mapped_theories.append(theory_with_result)

# NEW:
if result.is_mapped and result.canonical_name and result.mapping_confidence >= 0.85:
    mapped_theories.append(theory_with_result)
elif result.is_mapped and result.canonical_name and result.mapping_confidence >= 0.6:
    # Partial match - needs paper-specific extraction
    theory_with_result['mapping_type'] = 'partial'
    mapped_theories.append(theory_with_result)
```

### 2. Update Prompt

**File**: `src/normalization/stage1_5_llm_mapping.py`

**Add to prompt** (line ~145):
```
# MAPPING CRITERIA

Be CONSERVATIVE when mapping:
- **CANONICAL MATCH** (confidence >= 0.85): Paper's core mechanism matches canonical theory
- **PARTIAL MATCH** (confidence 0.6-0.84): Paper discusses specific aspect/application
- **NOVEL** (confidence < 0.6): Paper proposes fundamentally new mechanism

Examples:
✓ "Oxidative stress causes aging" → Oxidative Stress Theory (0.95)
✓ "ROS in hippocampal aging" → Free Radical Theory (0.75 - partial)
✗ "Novel epigenetic aging clock" → Novel (0.3)
```

### 3. Test

```bash
python src/normalization/stage1_5_llm_mapping.py \
  --input output/stage1_fuzzy_matched.json \
  --output output/stage1_5_llm_mapped_STRICT.json \
  --batch-size 30

# Check distribution
python -c "
import json
with open('output/stage1_5_llm_mapped_STRICT.json') as f:
    data = json.load(f)
    print(f'Mapped: {len(data[\"mapped_theories\"])}')
    print(f'Novel: {len(data[\"novel_theories\"])}')
    print(f'Unmatched: {len(data[\"still_unmatched\"])}')
    
    # Check confidence distribution
    confidences = [t['stage1_5_result']['mapping_confidence'] 
                   for t in data['mapped_theories']]
    import statistics
    print(f'Avg confidence: {statistics.mean(confidences):.2f}')
"
```

---

## 📊 Success Metrics

### Phase 1
- ✅ Canonical mapping rate: 70-80% (down from 98.9%)
- ✅ Novel theories: 15-25% (up from 1.1%)
- ✅ Average mapping confidence: > 0.85

### Phase 2
- ✅ 100% of mapped theories have paper-specific mechanisms
- ✅ Average 5-10 paper-specific mechanisms per theory
- ✅ Context tags for 90%+ of theories

### Phase 3
- ✅ Large groups (>100 theories) split into 5-10 sub-groups
- ✅ Sub-group size: 20-60 theories
- ✅ Total groups: 100-200 (including sub-groups)

---

## 🚀 Recommendation

**Start with Phase 1** (stricter mapping) - this is the quickest way to improve results with minimal code changes. This will:
1. Reduce over-mapping to canonical theories
2. Increase novel theory discovery
3. Create better balance in group sizes

Then evaluate if Phase 2 and 3 are needed based on the results.
