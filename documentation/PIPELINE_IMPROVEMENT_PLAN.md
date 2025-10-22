# Pipeline Improvement Plan

## Current Pipeline Analysis

### Stage 1: Fuzzy Matching ✅
- **Input**: Raw theories from database
- **Process**: Fuzzy string matching to canonical ontology
- **Output**: `matched_theories` + `unmatched_theories`
- **Status**: Working well

### Stage 1.5: LLM Mapping ✅ (NEW)
- **Input**: `unmatched_theories` from Stage 1
- **Process**: LLM validates + maps to canonical theories
- **Output**: 
  - `mapped_theories` (now matched to canonical)
  - `novel_theories` (valid but new)
  - `still_unmatched` (valid but couldn't map)
  - `invalid_theories` (not aging theories)
- **Status**: Just implemented with new ontology format

### Stage 3: LLM Extraction (NEEDS IMPROVEMENT)
- **Input**: Currently takes `unmatched_theories` from Stage 1
- **Process**: Extracts metadata (categories, mechanisms, key players)
- **Output**: `valid_theories` + `invalid_theories`
- **Problem**: Doesn't integrate with Stage 1.5 results!

### Stage 4: Theory Grouping
- **Input**: Stage 1 matched + Stage 2 valid
- **Process**: Groups by shared mechanisms
- **Output**: Theory groups

---

## 🚨 Critical Issues

### 1. **Stage 3 is Disconnected from Stage 1.5**
- Stage 3 still processes ALL unmatched theories
- Should only process `still_unmatched` + `novel_theories` from Stage 1.5
- Wastes LLM calls on theories already mapped in Stage 1.5

### 2. **Duplicate Validation**
- Stage 1.5 validates if theory is valid
- Stage 3 validates again (redundant!)
- Should trust Stage 1.5 validation

### 3. **Missing Mechanism Extraction for Mapped Theories**
- Theories mapped in Stage 1.5 get canonical name
- But don't get their specific mechanisms extracted
- Stage 3 should extract mechanisms for ALL valid theories

### 4. **Inconsistent Data Flow**
- Stage 1.5 outputs `mapped_theories` with canonical names
- Stage 3 doesn't know about these
- Stage 4 tries to merge but data is messy

---

## 🎯 Proposed Improvements

### **Improvement 1: Redesign Stage 3 Input**

**Current Flow:**
```
Stage 1 → unmatched → Stage 1.5 → mapped/novel/still_unmatched
                    ↓
Stage 1 → unmatched → Stage 3 → extract metadata
```

**New Flow:**
```
Stage 1 → matched → [Already have canonical name]
       ↓
       unmatched → Stage 1.5 → mapped → [Have canonical name]
                             ↓
                             novel → Stage 3 → extract metadata
                             ↓
                             still_unmatched → Stage 3 → extract metadata
                             ↓
                             invalid → [Filter out]
```

**Changes Needed:**
1. Stage 3 should accept Stage 1.5 output, not Stage 1 output
2. Process only `novel_theories` + `still_unmatched`
3. Skip validation (trust Stage 1.5)

---

### **Improvement 2: Extract Mechanisms for ALL Theories**

**Problem**: Theories matched in Stage 1 or 1.5 have canonical names but no extracted mechanisms.

**Solution**: Create a new stage or modify Stage 3 to:
1. For **mapped theories** (Stage 1 + 1.5):
   - Use canonical mechanisms from ontology as baseline
   - Optionally extract paper-specific mechanisms
   - Merge with canonical mechanisms

2. For **novel/unmatched theories**:
   - Extract mechanisms from scratch (current behavior)

**Implementation:**
```python
def process_all_theories(self, stage1_output, stage1_5_output):
    """Process ALL theories with appropriate strategy."""
    
    # 1. Mapped theories (Stage 1 + Stage 1.5)
    mapped = stage1_output['matched_theories'] + stage1_5_output['mapped_theories']
    for theory in mapped:
        canonical_name = theory['match_result']['canonical_name']
        # Get canonical mechanisms from ontology
        canonical_data = self.ontology[canonical_name]
        theory['mechanisms'] = canonical_data['mechanisms']
        theory['key_players'] = canonical_data['key_players']
        theory['pathways'] = canonical_data['pathways']
        # Optionally: extract paper-specific details
    
    # 2. Novel theories (Stage 1.5)
    novel = stage1_5_output['novel_theories']
    for theory in novel:
        # Extract mechanisms from scratch
        metadata = self.extract_metadata(theory)
        theory['stage3_metadata'] = metadata
    
    # 3. Still unmatched (Stage 1.5)
    unmatched = stage1_5_output['still_unmatched']
    for theory in unmatched:
        # Extract mechanisms from scratch
        metadata = self.extract_metadata(theory)
        theory['stage3_metadata'] = metadata
    
    return {
        'mapped_with_mechanisms': mapped,
        'novel_with_mechanisms': novel,
        'unmatched_with_mechanisms': unmatched
    }
```

---

### **Improvement 3: Update Stage 3 Prompt**

**Current Issues:**
- Asks for validation (redundant)
- Asks for primary/secondary category (should use Stage 1.5 mapping)
- Doesn't leverage canonical mechanisms for comparison

**New Prompt Strategy:**

**For Novel Theories:**
```
You are extracting metadata for a NOVEL aging theory that doesn't match our ontology.

THEORY: {name}
CONTEXT: {concept_text}

This theory was validated as a genuine aging theory but doesn't match any canonical theory.
Proposed name: {proposed_name}

Extract:
1. KEY PLAYERS (10-20 items)
2. PATHWAYS (3-10 items)
3. MECHANISMS (5-15 detailed descriptions)
4. LEVEL OF EXPLANATION
5. TYPE OF CAUSE
6. TEMPORAL FOCUS
7. ADAPTIVENESS

Be comprehensive and specific.
```

**For Mapped Theories (Optional Enhancement):**
```
You are extracting paper-specific details for a theory mapped to: {canonical_name}

CANONICAL MECHANISMS:
{canonical_mechanisms}

PAPER CONTEXT:
{concept_text}

Extract any ADDITIONAL or PAPER-SPECIFIC:
1. Key players mentioned in this paper
2. Specific pathways discussed
3. Novel mechanisms or variations

Keep extraction focused on what's NEW or SPECIFIC to this paper.
```

---

### **Improvement 4: Simplify Stage 3 Data Model**

**Remove:**
- `is_valid_theory` (trust Stage 1.5)
- `validation_reasoning` (trust Stage 1.5)
- `is_novel` (already known from Stage 1.5)
- `primary_category` / `secondary_category` (use canonical mapping)

**Keep:**
- `key_players`
- `pathways`
- `mechanisms`
- `level_of_explanation`
- `type_of_cause`
- `temporal_focus`
- `adaptiveness`
- `extraction_confidence`

**Add:**
- `source` ('canonical' vs 'extracted')
- `canonical_name` (if mapped)

---

### **Improvement 5: Update Stage 4 Grouping**

**Current Issues:**
- Groups Stage 1 matched + Stage 2 valid
- Doesn't account for Stage 1.5 mapped theories
- Mechanism comparison is weak for theories without extracted mechanisms

**New Strategy:**
```python
def group_theories(self, all_theories_with_mechanisms):
    """
    Group theories by shared mechanisms.
    
    Input: All theories with mechanisms (from improved Stage 3)
    - Mapped theories: have canonical mechanisms
    - Novel theories: have extracted mechanisms
    """
    
    # 1. Group by canonical name (if mapped)
    canonical_groups = defaultdict(list)
    for theory in all_theories_with_mechanisms:
        if theory.get('canonical_name'):
            canonical_groups[theory['canonical_name']].append(theory)
    
    # 2. Group novel/unmatched by mechanism similarity
    unmapped = [t for t in all_theories_with_mechanisms if not t.get('canonical_name')]
    mechanism_groups = self.cluster_by_mechanisms(unmapped)
    
    # 3. Try to merge novel groups with canonical groups
    for novel_group in mechanism_groups:
        best_match = self.find_best_canonical_match(novel_group, canonical_groups)
        if best_match and similarity > threshold:
            canonical_groups[best_match].extend(novel_group)
        else:
            # Create new group
            canonical_groups[f"NOVEL_{len(canonical_groups)}"] = novel_group
    
    return canonical_groups
```

---

## 📋 Implementation Checklist

### Phase 1: Fix Data Flow (HIGH PRIORITY)
- [ ] Update Stage 3 to accept Stage 1.5 output
- [ ] Skip validation in Stage 3 (trust Stage 1.5)
- [ ] Process only `novel_theories` + `still_unmatched`
- [ ] Add mechanism assignment for mapped theories

### Phase 2: Improve Extraction Quality
- [ ] Update Stage 3 prompts (separate for novel vs mapped)
- [ ] Add canonical mechanism context to prompts
- [ ] Increase key players/mechanisms extraction (more comprehensive)
- [ ] Add paper-specific extraction for mapped theories

### Phase 3: Simplify Data Model
- [ ] Remove redundant validation fields from Stage 3
- [ ] Standardize mechanism format across all stages
- [ ] Add `source` field to track mechanism origin

### Phase 4: Improve Grouping
- [ ] Update Stage 4 to use all theories with mechanisms
- [ ] Improve mechanism similarity algorithm
- [ ] Add canonical-to-novel group merging

### Phase 5: Add Quality Checks
- [ ] Validate mechanism consistency within groups
- [ ] Flag theories with low extraction confidence
- [ ] Add mechanism coverage metrics

---

## 🔄 Revised Pipeline Flow

```
┌─────────────┐
│   Stage 1   │ Fuzzy Matching
│   Matched   │ ─────────┐
└─────────────┘          │
                         ├──→ Assign canonical mechanisms from ontology
┌─────────────┐          │
│   Stage 1   │          │
│  Unmatched  │          │
└──────┬──────┘          │
       │                 │
       ↓                 │
┌─────────────┐          │
│  Stage 1.5  │ LLM      │
│   Mapping   │          │
└──────┬──────┘          │
       │                 │
       ├─→ Mapped ───────┤
       │                 │
       ├─→ Novel ────────┼──→ Extract mechanisms (Stage 3)
       │                 │
       ├─→ Unmatched ────┤
       │                 │
       └─→ Invalid (filter out)
                         │
                         ↓
                  ┌─────────────┐
                  │   Stage 3   │ All theories with mechanisms
                  │  Enriched   │
                  └──────┬──────┘
                         │
                         ↓
                  ┌─────────────┐
                  │   Stage 4   │ Group by mechanisms
                  │  Grouping   │
                  └─────────────┘
```

---

## 💡 Additional Suggestions

### 1. **Batch Processing in Stage 3**
- Stage 1.5 uses batches (efficient)
- Stage 3 processes one-by-one (slow)
- **Suggestion**: Add batch processing to Stage 3

### 2. **Caching Canonical Mechanisms**
- Load ontology mechanisms once
- Reuse for all mapped theories
- **Benefit**: Faster, consistent

### 3. **Confidence Scoring**
- Track confidence at each stage
- Aggregate confidence score
- **Use**: Filter low-confidence theories

### 4. **Mechanism Normalization**
- Mechanisms from different sources have different formats
- **Suggestion**: Normalize to consistent format
- Example: "mTOR pathway" vs "mTOR signaling" vs "mTOR"

### 5. **Add Stage 3.5: Mechanism Validation**
- Compare extracted mechanisms to canonical
- Flag inconsistencies
- Suggest corrections

---

## 🎯 Quick Wins (Implement First)

1. **Update Stage 3 input** - Accept Stage 1.5 output (30 min)
2. **Skip validation in Stage 3** - Remove redundant code (15 min)
3. **Assign canonical mechanisms** - For mapped theories (45 min)
4. **Update Stage 4 input** - Use enriched Stage 3 output (30 min)

**Total Time**: ~2 hours
**Impact**: Fixes critical data flow issues

---

## 📊 Expected Improvements

### Efficiency
- **Before**: Process ~1000 theories in Stage 3
- **After**: Process ~200 novel/unmatched theories
- **Savings**: 80% fewer LLM calls

### Quality
- **Before**: Inconsistent mechanisms across stages
- **After**: All theories have mechanisms (canonical or extracted)
- **Benefit**: Better grouping, more complete data

### Consistency
- **Before**: Validation happens twice
- **After**: Single validation in Stage 1.5
- **Benefit**: Cleaner code, faster processing
