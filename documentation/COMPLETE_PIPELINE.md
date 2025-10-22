# Complete Theory Normalization Pipeline

## Overview

A 3-stage pipeline to normalize and group 7,675 theories of aging from scientific papers.

```
theories_per_paper.json (7,751 theories)
           ↓
    ┌──────────────────┐
    │   STAGE 0        │  Quality Filter
    │ Quality Filter   │  Remove low-confidence
    └──────────────────┘
           ↓
    7,675 theories
           ↓
    ┌──────────────────┐
    │   STAGE 1        │  Fuzzy Matching
    │ Fuzzy Matching   │  Match to 46 canonical theories
    └──────────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
MATCHED (1,469)  UNMATCHED (6,206)
    ↓             ↓
    │      ┌──────────────────┐
    │      │   STAGE 1.5 NEW! │  LLM Mapping ($2-3)
    │      │ LLM Mapping      │  Map to canonical + validate
    │      └──────────────────┘
    │             ↓
    │      ┌──────┴──────┬──────────┬─────────┐
    │      ↓             ↓          ↓         ↓
    │   MAPPED      NOVEL      UNMATCHED  INVALID
    │   (2,100)     (2,800)     (300)     (1,000)
    │      ↓             ↓          ↓         [X]
    └──────┤             └──────────┤
           │                        ↓
           │             ┌──────────────────┐
           │             │   STAGE 2        │  LLM Extraction ($17-20)
           │             │ LLM Extraction   │  Validate & extract metadata
           │             └──────────────────┘
           │                        ↓
           │                 VALID (~2,300)
           │                        ↓
           └────────────────────────┤
                                    ↓
                            ~5,900 theories
                                    ↓
                            ┌──────────────────┐
                            │   STAGE 3        │  Theory Grouping
                            │ Theory Grouping  │  Group by shared mechanisms
                            └──────────────────┘
                                    ↓
                            ~1,200-1,500 unique theory groups
```

---

## Stage 0: Quality Filtering

**Purpose**: Remove low-quality theory extractions

**Input**: `theories_per_paper.json` (7,751 theories)

**Actions**:
- Filter by confidence (high/medium only)
- Enrich with concept text
- Remove incomplete extractions

**Output**: `output/stage0_filtered_theories.json` (7,675 theories)

**Cost**: Free (rule-based)

**Run**:
```bash
# Included in Stage 1 script
python run_stage1_on_real_data.py
```

---

## Stage 1: Fuzzy Matching

**Purpose**: Match theories to known canonical theories using fuzzy matching

**Input**: 7,675 filtered theories

**Features**:
- Abbreviation matching: (DHAC), (MHA), (MFRTA) → 115 matches
- Exact normalized matching → 1,330 matches
- High confidence fuzzy matching → 24 matches
- Uses 46 canonical theories, 161 aliases, 12 abbreviations

**Output**: `output/stage1_fuzzy_matched.json`
- **Matched**: 1,469 theories (19.1%)
- **Unmatched**: 6,206 theories (80.9%)

**Cost**: Free (rule-based)

**Run**:
```bash
python run_stage1_on_real_data.py
```

**Time**: ~2 minutes

---

## Stage 1.5: LLM Mapping (NEW!) 🆕

**Purpose**: Map unmatched theories to canonical theories using LLM intelligence

**Input**: 6,206 unmatched theories from Stage 1

**Features**:
- Batch processing (30 theories at once)
- Validates if theory is genuine
- Maps to canonical theories from ontology
- Identifies novel theories
- Filters invalid theories

**Output**: `output/stage1_5_llm_mapped.json`
- **Mapped**: 2,100 theories (34%) → Join Stage 1 matches
- **Novel**: 2,800 theories (45%) → Go to Stage 2
- **Unmatched**: 300 theories (5%) → Go to Stage 2
- **Invalid**: 1,000 theories (16%) → Filtered out

**Model**: OpenAI GPT-4o-mini (or Azure OpenAI)

**Cost**: ~$2-3 (207 batches × ~2000 tokens)

**Run**:
```bash
# Test (50 theories, 2 batches)
python test_stage1_5_mapping.py

# Full run
python -m src.normalization.stage1_5_llm_mapping \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage1_5_llm_mapped.json \
    --batch-size 30
```

**Time**: ~15-20 minutes (full run)

**Impact**:
- Match rate: 19.1% → **46.5%** (+140%)
- Stage 2 theories: 6,206 → 3,100 (-50%)
- Stage 2 cost: $35 → $17-20 (-50%)
- Net savings: **$12-15 per run**

---

## Stage 2: LLM Extraction

**Purpose**: Validate unmatched theories and extract detailed metadata

**Input**: 6,206 unmatched theories from Stage 1

**Actions**:
1. Validate if genuine theory of aging
2. Extract metadata:
   - Primary/secondary categories
   - Key players (5-15 items)
   - Pathways (molecular/evolutionary/social)
   - Mechanisms (3-10 items)
   - Level of explanation
   - Type of cause, temporal focus, adaptiveness

**Output**: `output/stage2_llm_extracted.json`
- **Valid**: ~4,500 theories (72.5%)
- **Invalid**: ~1,700 theories (27.5%)

**Model**: OpenAI GPT-4.1-mini (or Azure OpenAI)

**Cost**: ~$30-40 (6,206 theories × ~2000 tokens)

**Run**:
```bash
# Test (50 theories)
python test_stage2_sample.py

# Full run
python -m src.normalization.stage2_llm_extraction \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage2_llm_extracted.json
```

**Time**: ~30-60 minutes (full run)

---

## Stage 3: Theory Grouping

**Purpose**: Group theories sharing the same mechanisms

**Input**: 
- 1,469 matched theories (Stage 1)
- ~4,500 valid theories (Stage 2)
- **Total**: ~5,969 theories

**Strategy**:
1. **Stage 1 theories**: Group by canonical name
2. **Stage 2 theories**: Group by mechanism similarity (≥80% Jaccard)
3. Compute shared characteristics for each group

**Similarity Calculation**:
- Mechanisms: 60% weight
- Key players: 20% weight
- Pathways: 20% weight

**Output**: `output/stage3_theory_groups.json`
- **Total groups**: ~2,500-3,000
- **Singleton groups**: ~800-1,000 (unique theories)
- **Multi-theory groups**: ~1,500-2,000
- **Avg group size**: ~2-3 theories
- **Compression**: ~50-60% reduction

**Cost**: Free (rule-based)

**Run**:
```bash
# Test (with Stage 2 test data)
python test_stage3_grouping.py

# Full run
python -m src.normalization.stage3_theory_grouping \
    --stage1 output/stage1_fuzzy_matched.json \
    --stage2 output/stage2_llm_extracted.json \
    --output output/stage3_theory_groups.json
```

**Time**: ~5 minutes

---

## Complete Pipeline Summary

### Input
- **7,751 theories** from papers

### Output
- **~2,500-3,000 unique theory groups**

### Processing Flow
1. **Stage 0**: 7,751 → 7,675 (quality filter)
2. **Stage 1**: 7,675 → 1,469 matched + 6,206 unmatched
3. **Stage 2**: 6,206 → ~4,500 valid
4. **Stage 3**: 5,969 → ~2,500 groups

### Cost Breakdown
| Stage | Method | Cost | Time |
|-------|--------|------|------|
| Stage 0 | Rule-based | $0 | ~1 min |
| Stage 1 | Fuzzy matching | $0 | ~2 min |
| **Stage 1.5** 🆕 | **LLM Mapping** | **~$2-3** | **~15-20 min** |
| Stage 2 | LLM (OpenAI) | ~$17-20 | ~25-30 min |
| Stage 3 | Similarity | $0 | ~5 min |
| **Total** | | **~$19-23** | **~48-58 min** |
| **Savings vs Old** | | **-$12-16** | **Similar time** |

### Compression
- **Input**: 7,675 theories
- **Output**: ~2,500 groups
- **Reduction**: ~67% (2.5:1 compression)

---

## Running the Complete Pipeline

### 1. Setup
```bash
# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables
export USE_MODULE_NORMALIZATION='openai'
export OPENAI_MODEL='gpt-4.1-mini'
export OPENAI_API_KEY='your-key-here'
```

### 2. Run Stages 0-1
```bash
python run_stage1_on_real_data.py
```

**Output**:
- `output/stage0_filtered_theories.json`
- `output/stage1_fuzzy_matched.json`
- `output/stage1_matching_report.txt`

### 3. Run Stage 2 (Test First)
```bash
# Test with 50 theories
python test_stage2_sample.py

# Review output/stage2_llm_extracted_TEST.json

# If good, run full
python -m src.normalization.stage2_llm_extraction \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage2_llm_extracted.json
```

**Output**:
- `output/stage2_llm_extracted.json`

### 4. Run Stage 3
```bash
# Test first
python test_stage3_grouping.py

# Review output/stage3_theory_groups_TEST.json

# If good, run full
python -m src.normalization.stage3_theory_grouping \
    --stage1 output/stage1_fuzzy_matched.json \
    --stage2 output/stage2_llm_extracted.json \
    --output output/stage3_theory_groups.json
```

**Output**:
- `output/stage3_theory_groups.json`

---

## Output Files

### Stage 0-1
- `output/stage0_filtered_theories.json` - Quality filtered theories
- `output/stage1_fuzzy_matched.json` - Matched/unmatched theories
- `output/stage1_matching_report.txt` - Human-readable report

### Stage 2
- `output/stage2_llm_extracted.json` - Valid/invalid theories with metadata
- `output/stage2_llm_extracted_TEST.json` - Test run (50 theories)

### Stage 3
- `output/stage3_theory_groups.json` - Final grouped theories
- `output/stage3_theory_groups_TEST.json` - Test run

---

## Key Features

### Stage 1 Innovations
✅ Abbreviation matching (DHAC, MHA, MFRTA)  
✅ Aggressive normalization (removes "theory", "hypothesis", "aging")  
✅ Smart quote handling  
✅ Compound name validation  

### Stage 2 Enhancements
✅ Comprehensive examples for all theory types (molecular, evolutionary, social)  
✅ Mechanisms as list (not single string)  
✅ Extensive lists (5-15 items) with detailed instructions  
✅ Separate examples for evolutionary and social theories  

### Stage 3 Approach
✅ Combines Stage 1 and Stage 2 results  
✅ Mechanism-based similarity (not embeddings)  
✅ Weighted Jaccard similarity  
✅ Shared characteristics computation  

---

## Documentation

- `PIPELINE_OVERVIEW.md` - High-level overview
- `STAGE1_SUMMARY.txt` - Stage 1 details
- `STAGE2_README.md` - Stage 2 usage
- `STAGE2_IMPROVEMENTS.md` - Stage 2 enhancements
- `STAGE3_README.md` - Stage 3 usage
- `COMPLETE_PIPELINE.md` - This file

---

## Next Steps

After completing the pipeline:

1. **Analysis**:
   - Identify most popular theories
   - Find novel theories (singletons)
   - Analyze theory distribution by category

2. **Validation**:
   - Manual review of sample groups
   - Verify mechanism matching accuracy
   - Check for false positives/negatives

3. **Export**:
   - Create final normalized dataset
   - Export to CSV for analysis
   - Generate visualizations

4. **Applications**:
   - Theory popularity analysis
   - Temporal trends (which theories are gaining traction)
   - Citation network analysis
   - Knowledge graph construction
