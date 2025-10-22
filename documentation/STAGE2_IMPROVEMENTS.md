# Stage 2 Improvements - Enhanced Extraction

## Changes Made

### 1. Changed `mechanism_of_action` to `mechanisms` (list)

**Before**: 
```python
mechanism_of_action: Optional[str] = None  # Single string
```

**After**:
```python
mechanisms: List[str] = field(default_factory=list)  # List of mechanisms
```

**Rationale**: Theories often have multiple mechanisms, not just one. A list allows capturing all relevant processes.

---

### 2. Enhanced Prompt for Evolutionary/Social Theories

Added comprehensive examples for **key_players** across all theory types:

#### For MOLECULAR/CELLULAR theories:
- Specific molecules: mTOR, AMPK, SIRT1, p16, p21, p53, telomerase, FOXO, NF-κB
- Cellular components: mitochondria, telomeres, ribosomes, lysosomes, proteasomes
- Proteins/enzymes: DNA polymerase, catalase, superoxide dismutase

#### For EVOLUTIONARY theories (NEW):
- Selection pressures: predation, extrinsic mortality, reproductive success
- Evolutionary forces: natural selection, genetic drift, mutation accumulation
- Life history traits: fertility, longevity, reproductive timing, parental investment
- Population factors: population size, generation time, mortality rate

#### For SOCIAL/PSYCHOLOGICAL theories (NEW):
- Social factors: social roles, social engagement, social support, family structure
- Psychological factors: self-concept, life satisfaction, coping mechanisms
- Institutional factors: retirement, healthcare systems, social policies
- Behavioral factors: activity levels, social participation, role transitions

---

### 3. Expanded Pathways Examples

Added examples for non-molecular pathways:

- **Molecular**: mTOR, AMPK, sirtuins, insulin/IGF-1, p53, NF-κB, PI3K/AKT, autophagy
- **Evolutionary** (NEW): natural selection, sexual selection, kin selection, trade-offs
- **Social** (NEW): role transitions, social integration, disengagement processes

---

### 4. Detailed Mechanisms with Examples

Changed from single mechanism description to **list of 3-10 specific mechanisms**.

#### Examples for molecular theories:
- "Accumulation of somatic mutations in nuclear DNA"
- "Mitochondrial dysfunction leading to increased ROS production"
- "Telomere shortening triggering cellular senescence"
- "Protein misfolding and aggregation"

#### Examples for evolutionary theories (NEW):
- "Declining force of natural selection with age"
- "Trade-off between early reproduction and late-life survival"
- "Accumulation of late-acting deleterious mutations"
- "Antagonistic pleiotropy between early and late fitness"

#### Examples for social theories (NEW):
- "Withdrawal from social roles and relationships"
- "Loss of meaningful social engagement"
- "Reduction in normative expectations"
- "Decreased social interaction frequency"

---

### 5. Increased List Requirements

**Before**: "List 3-10 main actors"  
**After**: "List 5-15 main actors. Be EXTENSIVE and COMPREHENSIVE."

**Instruction added**: "Be DETAILED and COMPREHENSIVE" for all lists

---

## Updated JSON Output Format

```json
{
  "primary_category": "Evolutionary Theories of Aging",
  "secondary_category": "Programmed Evolutionary Theories",
  "is_novel": false,
  "novelty_reasoning": null,
  "key_players": [
    "natural selection",
    "extrinsic mortality",
    "predation",
    "reproductive success",
    "life history traits",
    "fertility",
    "longevity",
    "generation time",
    "population size",
    "mortality rate"
  ],
  "pathways": [
    "natural selection",
    "trade-offs",
    "life history evolution"
  ],
  "mechanisms": [
    "Declining force of natural selection with age",
    "Trade-off between early reproduction and late-life survival",
    "Extrinsic mortality shapes evolution of senescence",
    "Lower extrinsic mortality favors delayed senescence"
  ],
  "level_of_explanation": "Population",
  "type_of_cause": "Extrinsic",
  "temporal_focus": "Lifelong",
  "adaptiveness": "Non-adaptive",
  "extraction_confidence": 0.9
}
```

---

## Benefits

1. **Comprehensive Coverage**: Now extracts meaningful data for ALL theory types (molecular, evolutionary, social)
2. **Multiple Mechanisms**: Captures all relevant mechanisms, not just one
3. **Extensive Lists**: Encourages LLM to provide thorough, detailed lists
4. **Better for Evolutionary Theories**: Previously returned empty lists, now has specific guidance
5. **Better for Social Theories**: Added examples for social/psychological factors

---

## Testing

Run updated test:
```bash
python test_stage2_sample.py
```

Expected improvements:
- Evolutionary theories should now have populated `key_players` and `pathways`
- Social theories should have relevant social factors
- All theories should have multiple mechanisms (not just one)
- Lists should be more comprehensive (5-15 items vs 3-10)

---

## Files Updated

1. ✅ `src/normalization/stage2_llm_extraction.py`
   - Changed `mechanism_of_action` → `mechanisms` (list)
   - Enhanced prompt with examples for all theory types
   - Increased list size requirements
   - Added "EXTENSIVE and COMPREHENSIVE" instructions

2. ✅ `test_stage2_sample.py`
   - Updated to display mechanisms as list
   - Shows count of items in each list

3. ✅ `src/normalization/stage1_fuzzy_matching.py`
   - Fixed syntax error (missing comma after 'normal aging')

---

## Before vs After Example

### Before (Evolutionary Theory):
```json
{
  "key_players": [],
  "pathways": [],
  "mechanism_of_action": "Extrinsic mortality shapes evolution..."
}
```

### After (Evolutionary Theory):
```json
{
  "key_players": [
    "natural selection",
    "extrinsic mortality",
    "predation",
    "reproductive success",
    "life history traits",
    "fertility",
    "longevity"
  ],
  "pathways": [
    "natural selection",
    "trade-offs",
    "life history evolution"
  ],
  "mechanisms": [
    "Declining force of natural selection with age",
    "Trade-off between early reproduction and late-life survival",
    "Extrinsic mortality shapes evolution of senescence"
  ]
}
```
