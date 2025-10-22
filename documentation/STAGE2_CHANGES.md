# Stage 2 Updates Summary

## Changes Made

### 1. Changed `secondary_categories` to `secondary_category` (singular)

**Before**: 
```python
secondary_categories: List[str] = field(default_factory=list)
```

**After**:
```python
secondary_category: Optional[str] = None
```

**Rationale**: Each theory should be assigned to ONE secondary category, not multiple.

### 2. Updated LLM Integration

- Switched from Anthropic Claude to OpenAI/Azure OpenAI
- Uses `OpenAIClient` or `AzureOpenAIClient` from `src.core.llm_integration`
- Configured via environment variables:
  - `USE_MODULE_NORMALIZATION`: 'openai' or 'azure'
  - `OPENAI_MODEL`: Model name (default: 'gpt-4.1-mini')

### 3. Simplified Validation

- Removed explicit validation step (assumes all theories from Stage 1 are valid candidates)
- Default `is_valid_theory` to `True`
- Focus on extraction and categorization

### 4. Updated Prompt Structure

**Prompt now asks for**:
1. Primary category (ONE from top-level ontology)
2. Secondary category (ONE from subcategories, or mark as NOVEL)
3. Key players (3-10 biological/molecular actors)
4. Pathways (specific molecular pathways)
5. Mechanism of action (1-2 sentences)
6. Level of explanation (Molecular, Cellular, etc.)
7. Type of cause (Intrinsic, Extrinsic, Both)
8. Temporal focus (Developmental, Lifelong, etc.)
9. Adaptiveness (Adaptive, Non-adaptive, etc.)
10. Extraction confidence (0.0-1.0)

### 5. JSON Output Format

```json
{
  "primary_category": "Molecular and Cellular Damage Theories",
  "secondary_category": "Protein and DNA Damage Theories",
  "is_novel": false,
  "novelty_reasoning": null,
  "key_players": ["DNA", "p53", "ATM"],
  "pathways": ["p53", "ATM/ATR"],
  "mechanism_of_action": "Accumulation of DNA damage...",
  "level_of_explanation": "Molecular",
  "type_of_cause": "Intrinsic",
  "temporal_focus": "Lifelong",
  "adaptiveness": "Non-adaptive",
  "extraction_confidence": 0.9
}
```

## Files Updated

1. ✅ `src/normalization/stage2_llm_extraction.py`
   - Changed `secondary_categories` → `secondary_category`
   - Updated LLM client integration
   - Fixed prompt numbering
   - Added MAX_TOKENS constant

2. ✅ `test_stage2_sample.py`
   - Removed API key requirement
   - Updated to use singular `secondary_category`

## Testing

Run test with:
```bash
python test_stage2_sample.py
```

This will process 5 theories and show the extraction results.

## Next Steps

1. Test the extraction on sample theories
2. Verify JSON output format
3. Run on full dataset (6,206 theories)
4. Proceed to Stage 3 (clustering)
