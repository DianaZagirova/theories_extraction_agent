# Quick Fix: Improve Mechanism Granularity

## 🚨 Current Problem

Your Stage 4 results show:
- **39 groups** from 1516 theories
- **98.9% mapped** to canonical theories (too aggressive)
- **Largest group**: 290 theories (Free Radical Theory)
- **Generic mechanisms**: All theories in a group share identical canonical mechanisms

## 🎯 Root Cause

**Stage 1.5 is over-mapping** - it's mapping almost everything to canonical theories with low confidence thresholds.

## ⚡ Quick Fix (15 minutes)

### Step 1: Increase Mapping Threshold

Edit `src/normalization/stage1_5_llm_mapping.py` around line 390:

```python
# FIND THIS:
if result.is_mapped and result.canonical_name:
    # Add match_result for consistency with Stage 1
    theory_with_result['match_result'] = {
        'matched': True,
        'canonical_name': result.canonical_name,
        'match_type': 'llm_mapping',
        'confidence': result.mapping_confidence,
        'score': result.mapping_confidence
    }
    mapped_theories.append(theory_with_result)

# REPLACE WITH:
if result.is_mapped and result.canonical_name:
    # Only map if high confidence
    if result.mapping_confidence >= 0.85:
        # Strong canonical match
        theory_with_result['match_result'] = {
            'matched': True,
            'canonical_name': result.canonical_name,
            'match_type': 'llm_mapping',
            'confidence': result.mapping_confidence,
            'score': result.mapping_confidence
        }
        mapped_theories.append(theory_with_result)
    elif result.mapping_confidence >= 0.6:
        # Partial match - treat as novel with suggested canonical
        theory_with_result['mapping_type'] = 'partial'
        theory_with_result['suggested_canonical'] = result.canonical_name
        novel_theories.append(theory_with_result)
    else:
        # Low confidence - treat as novel
        novel_theories.append(theory_with_result)
```

### Step 2: Update Prompt to Be More Conservative

Edit `src/normalization/stage1_5_llm_mapping.py` around line 150:

```python
# ADD THIS SECTION TO THE PROMPT:
prompt = f"""Your task is to:
1. Validate if each theory is a valid aging theory
2. Map valid theories to canonical theories from ontology, when possible.

# MAPPING CRITERIA - BE CONSERVATIVE!

Only assign HIGH confidence (>= 0.85) if:
- The paper's CORE MECHANISM matches the canonical theory
- The paper explicitly discusses the canonical theory
- The mechanisms are fundamentally the same

Assign MEDIUM confidence (0.6-0.84) if:
- The paper discusses a SPECIFIC ASPECT of the canonical theory
- The paper applies the canonical theory to a specific context
- The mechanisms overlap but aren't identical

Assign LOW confidence (< 0.6) if:
- The paper proposes a NEW mechanism
- The paper combines multiple theories
- The mechanisms are only tangentially related

# INSTRUCTIONS
...
```

### Step 3: Re-run Pipeline

```bash
# Re-run Stage 1.5 with stricter mapping
python src/normalization/stage1_5_llm_mapping.py \
  --input output/stage1_fuzzy_matched.json \
  --output output/stage1_5_llm_mapped_STRICT.json \
  --batch-size 30

# Re-run Stage 3
python src/normalization/stage3_llm_extraction_improved.py \
  --stage1 output/stage1_fuzzy_matched.json \
  --stage1-5 output/stage1_5_llm_mapped_STRICT.json \
  --output output/stage3_extracted_STRICT.json

# Re-run Stage 4
python src/normalization/stage4_theory_grouping_improved.py \
  --input output/stage3_extracted_STRICT.json \
  --output output/stage4_groups_STRICT.json
```

### Step 4: Check Results

```bash
python -c "
import json

# Check Stage 1.5 distribution
with open('output/stage1_5_llm_mapped_STRICT.json') as f:
    data = json.load(f)
    print('Stage 1.5 Results:')
    print(f'  Mapped: {len(data[\"mapped_theories\"])}')
    print(f'  Novel: {len(data[\"novel_theories\"])}')
    print(f'  Unmatched: {len(data[\"still_unmatched\"])}')
    print(f'  Invalid: {len(data[\"invalid_theories\"])}')

# Check Stage 4 distribution
with open('output/stage4_groups_STRICT.json') as f:
    data = json.load(f)
    groups = data['groups']
    print(f'\nStage 4 Results:')
    print(f'  Total groups: {len(groups)}')
    
    canonical = [g for g in groups if g['source'] == 'canonical']
    novel = [g for g in groups if g['source'] == 'novel']
    
    print(f'  Canonical groups: {len(canonical)}')
    print(f'  Novel groups: {len(novel)}')
    
    # Show size distribution
    sizes = [g['theory_count'] for g in groups]
    print(f'  Largest group: {max(sizes)} theories')
    print(f'  Smallest group: {min(sizes)} theories')
    print(f'  Average group: {sum(sizes)/len(sizes):.1f} theories')
"
```

## 📊 Expected Improvement

### Before (Current)
```
Stage 1.5:
  Mapped: 1499 (98.9%)
  Novel: 17 (1.1%)

Stage 4:
  Total groups: 39
  Largest group: 290 theories
  Average: 38.9 theories/group
```

### After (Expected)
```
Stage 1.5:
  Mapped: ~1100 (72%)
  Novel: ~350 (23%)
  Unmatched: ~50 (3%)

Stage 4:
  Total groups: ~80-100
  Canonical groups: ~39
  Novel groups: ~40-60
  Largest group: ~150 theories
  Average: ~15-20 theories/group
```

## 🎯 Benefits

1. **Better Granularity**: More groups with fewer theories each
2. **Novel Discovery**: Identify truly novel theories (23% vs 1%)
3. **Higher Quality**: Only high-confidence mappings to canonical
4. **More Specific**: Novel theories get detailed mechanism extraction

## 🔄 Next Steps (Optional)

If you still want more granularity after this fix:

1. **Add Paper-Specific Extraction** - Extract specific mechanisms even for mapped theories
2. **Sub-Group Clustering** - Split large canonical groups into sub-groups
3. **Hierarchical Structure** - Organize as canonical → sub-groups → papers

See `MECHANISM_IMPROVEMENT_PLAN.md` for detailed implementation.

## ⚠️ Important Notes

- This change will **reduce** the number of theories mapped to canonical
- This will **increase** LLM costs slightly (more novel theories to extract)
- This will **improve** quality by being more conservative
- You can adjust the threshold (0.85) based on your needs

## 🧪 Testing

Test on a small batch first:
```bash
python src/normalization/stage1_5_llm_mapping.py \
  --input output/stage1_fuzzy_matched.json \
  --output output/stage1_5_TEST.json \
  --batch-size 30 \
  --max-theories 100  # Test on 100 theories first
```

Then check the confidence distribution:
```bash
python -c "
import json
with open('output/stage1_5_TEST.json') as f:
    data = json.load(f)
    mapped = data['mapped_theories']
    confidences = [t['stage1_5_result']['mapping_confidence'] for t in mapped]
    
    print('Confidence Distribution:')
    print(f'  >= 0.9: {sum(1 for c in confidences if c >= 0.9)}')
    print(f'  0.85-0.9: {sum(1 for c in confidences if 0.85 <= c < 0.9)}')
    print(f'  0.7-0.85: {sum(1 for c in confidences if 0.7 <= c < 0.85)}')
    print(f'  < 0.7: {sum(1 for c in confidences if c < 0.7)}')
"
```

If too many theories have confidence < 0.85, you can lower the threshold to 0.80 or 0.75.
