# Keyword Expansion - Quick Summary

## 1. Current Hardcoded Keyword Groups

### Group 1: Mechanism Keywords (40 terms)
**Categories:**
- Signaling molecules (mTOR, AMPK, insulin, IGF-1, FOXO, SIRT1, p53, NF-κB)
- Metabolites (ROS, NAD+, ATP, cAMP, cGMP)
- Cellular processes (autophagy, apoptosis, senescence, inflammation)
- PTMs (methylation, acetylation, phosphorylation, ubiquitination)
- Organelles (proteasome, lysosome, mitochondria, telomere)
- Enzymes (telomerase)
- Stress responses (DNA damage, oxidative stress, ER stress)

### Group 2: Pathway Keywords (25 terms)
**Categories:**
- Nutrient sensing (mTOR, AMPK, insulin/IGF-1, PI3K, AKT)
- Stress response (p38, ERK, JNK, NF-κB)
- Developmental (Wnt, Notch, Hedgehog, TGF-β)
- Signal transduction (JAK-STAT, PKA, PKC)
- GTPase signaling (Ras, Raf, MEK)

**Total current:** 65 terms

---

## 2. Missing Keyword Groups (10 New Categories)

### High Priority ⭐⭐⭐

1. **Receptors** (20-30 terms)
   - Growth factor receptors (EGFR, VEGFR, PDGFR, FGFR)
   - Hormone receptors (estrogen, androgen, thyroid)
   - Metabolic receptors (PPAR, LXR, FXR)
   - Pattern recognition (TLR family)

2. **Genes & Transcription Factors** (30-40 terms)
   - Longevity genes (KLOTHO, DAF-16, DAF-2)
   - Transcription factors (NRF2, HIF-1, PGC-1α)
   - Tumor suppressors (BRCA1, PTEN, RB)
   - Clock genes (CLOCK, BMAL1, PER, CRY)

### Medium Priority ⭐⭐

3. **Proteins & Enzymes** (40-50 terms)
   - Chaperones (HSP70, HSP90, HSP60)
   - Proteases (caspases, cathepsins, MMPs)
   - Antioxidants (SOD, catalase, GPX)
   - DNA repair (ATM, ATR, PARP)

4. **Organelle-Specific** (15-20 terms)
   - Mitochondrial (cristae, matrix, mitophagy)
   - ER (ERAD, UPR)
   - Nuclear (nucleolus, chromatin)

5. **Metabolic Pathways** (25-30 terms)
   - Energy (glycolysis, TCA cycle, OXPHOS)
   - Lipid (beta-oxidation, lipogenesis)
   - Amino acid (methionine restriction)

6. **Epigenetic Regulators** (25-30 terms)
   - Writers (DNMTs, HATs, HMTs)
   - Erasers (TETs, HDACs, HDMs)
   - Chromatin remodelers (polycomb, trithorax)

### Low Priority ⭐

7. **Extracellular Matrix** (15-20 terms)
8. **Immune System** (20-25 terms)
9. **Cellular Structures** (15-20 terms)
10. **Disease Terms** (20-25 terms)

**Total proposed:** 230-305 new terms

---

## 3. Expected Coverage Improvement

| Phase | Keywords Added | Coverage | Improvement |
|-------|---------------|----------|-------------|
| **Current** | 65 | 65-75% | Baseline |
| **+ High Priority** | +50-70 | 75-80% | +10-15% |
| **+ Medium Priority** | +105-130 | 85-90% | +20-25% |
| **+ Low Priority** | +70-95 | 90-95% | +25-30% |
| **Total** | 295-370 | 90-95% | **+25-30%** |

---

## 4. Implementation Plan

### Phase 1: High Priority (Week 1)
**Time:** 4-6 hours  
**Keywords:** 50-70  
**Improvement:** +10-15%

1. Run Prompt 1 (Receptors)
2. Run Prompt 2 (Genes/TFs)
3. Validate outputs
4. Add to `stage1_embedding_advanced.py`
5. Test on 761 theories
6. Measure improvement

### Phase 2: Medium Priority (Week 2)
**Time:** 6-8 hours  
**Keywords:** 105-130  
**Improvement:** +15-20%

1. Run Prompts 3-6 (Proteins, Organelles, Metabolism, Epigenetics)
2. Validate outputs
3. Add to code
4. Test and measure

### Phase 3: Low Priority (Week 3-4)
**Time:** 4-6 hours  
**Keywords:** 70-95  
**Improvement:** +5-10%

1. Run Prompts 7-10 (ECM, Immune, Structures, Diseases)
2. Validate outputs
3. Add to code
4. Final optimization

**Total time:** 14-20 hours  
**Total improvement:** +30-45% coverage

---

## 5. How to Use the Prompts

### Quick Start

1. **Open:** `LLM_KEYWORD_GENERATION_PROMPTS.md`
2. **Choose LLM:** GPT-4 or Claude 3
3. **Copy Prompt 1** (Receptors)
4. **Run in LLM** with temperature 0.3-0.5
5. **Validate output** against aging literature
6. **Test on sample** theories
7. **Add to code** if validated
8. **Repeat** for other prompts

### Example with GPT-4

```python
import openai

openai.api_key = "your-key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a biogerontology expert."},
        {"role": "user", "content": PROMPT_1_RECEPTOR_KEYWORDS}
    ],
    temperature=0.3
)

keywords = response.choices[0].message.content
print(keywords)
```

### Validation Checklist

- [ ] Keywords relevant to aging research
- [ ] No duplicates with existing keywords
- [ ] Abbreviations correct
- [ ] Grouped logically
- [ ] Tested on sample theories
- [ ] Coverage improved

---

## 6. Files Created

### Analysis Documents

1. **`KEYWORD_GROUPS_ANALYSIS.md`** (3500 words)
   - Current keyword groups breakdown
   - Gap analysis (10 missing categories)
   - Priority ranking
   - Coverage estimates
   - Implementation strategy

2. **`LLM_KEYWORD_GENERATION_PROMPTS.md`** (4000 words)
   - 10 detailed prompts (one per category)
   - Validation prompt
   - Usage instructions
   - Example code
   - Quality checklist

3. **`KEYWORD_EXPANSION_SUMMARY.md`** (This file)
   - Quick reference
   - Implementation plan
   - Expected results

---

## 7. Key Insights

### Why We Need More Keywords

1. **Current coverage:** 65-75% (good but not great)
2. **Missing categories:** Receptors, genes, proteins (critical)
3. **Standalone mentions:** "EGFR activation" vs "EGFR receptor"
4. **Domain specificity:** Aging research has unique terminology

### Why Hybrid Approach Works

```
Patterns (structured) + ML Models (contextual) + Keywords (fallback) = Best Coverage
```

- **Patterns:** Catch "X-mediated", "X receptor"
- **ML Models:** Catch contextual mentions
- **Keywords:** Catch standalone mentions

### Why Not Just ML?

- ❌ Fine-tuning expensive (requires labeled data, GPU)
- ❌ LLMs overkill (slow, expensive)
- ✅ Keywords fast (O(1) lookup)
- ✅ Keywords reliable (deterministic)
- ✅ Keywords maintainable (2-4 hours/year)

---

## 8. Next Steps

### Immediate (Today)

1. Review `LLM_KEYWORD_GENERATION_PROMPTS.md`
2. Choose LLM (GPT-4 or Claude)
3. Run Prompt 1 (Receptors)
4. Validate output
5. Test on sample theories

### Short-term (This Week)

1. Run Prompt 2 (Genes/TFs)
2. Add both to code
3. Test on 761 theories
4. Measure improvement
5. Validate +10-15% coverage gain

### Medium-term (Next 2 Weeks)

1. Run Prompts 3-6 (Medium priority)
2. Add to code
3. Test and validate
4. Achieve 85-90% coverage

### Long-term (Next Month)

1. Run Prompts 7-10 (Low priority)
2. Final optimization
3. Achieve 90-95% coverage
4. Deploy to production (14K theories)

---

## 9. Expected Results

### Before Expansion

```
Mechanisms:  65% coverage (40 keywords)
Pathways:    75% coverage (25 keywords)
Receptors:   Pattern matching only
Genes:       Limited (5 keywords)
Total:       65 keywords, 65-75% coverage
```

### After High Priority

```
Mechanisms:  70% coverage (40 keywords)
Pathways:    80% coverage (25 keywords)
Receptors:   60-70% coverage (25 keywords) ✨ NEW
Genes:       50-60% coverage (35 keywords) ✨ NEW
Total:       125 keywords, 75-80% coverage (+10-15%)
```

### After All Expansions

```
Mechanisms:  85-90% coverage (40 keywords + better patterns)
Pathways:    85-90% coverage (25 keywords + better patterns)
Receptors:   70-80% coverage (25 keywords)
Genes:       65-75% coverage (35 keywords)
Proteins:    50-60% coverage (45 keywords) ✨ NEW
Organelles:  70-80% coverage (18 keywords) ✨ NEW
Metabolism:  60-70% coverage (28 keywords) ✨ NEW
Epigenetics: 60-70% coverage (28 keywords) ✨ NEW
ECM:         50-60% coverage (18 keywords) ✨ NEW
Immune:      50-60% coverage (23 keywords) ✨ NEW
Total:       295-370 keywords, 90-95% coverage (+25-30%)
```

---

## 10. Maintenance

### Quarterly Updates (4x/year)
- Review new papers
- Add 5-10 new terms per category
- **Effort:** 2-4 hours

### Annual Review (1x/year)
- Comprehensive literature review
- Re-run prompts with updated context
- Reorganize if needed
- **Effort:** 8-12 hours

**Total annual effort:** 16-28 hours (manageable)

---

## Summary

### Current State
- **2 keyword groups:** mechanisms (40), pathways (25)
- **Coverage:** 65-75%
- **Total:** 65 terms

### Proposed State
- **12 keyword groups:** +10 new categories
- **Coverage:** 90-95%
- **Total:** 295-370 terms

### Implementation
- **Phase 1 (High):** +50-70 terms, +10-15% coverage, 4-6 hours
- **Phase 2 (Medium):** +105-130 terms, +15-20% coverage, 6-8 hours
- **Phase 3 (Low):** +70-95 terms, +5-10% coverage, 4-6 hours
- **Total:** +230-305 terms, +30-45% coverage, 14-20 hours

### Impact
- **Mechanism extraction:** 65% → 85-90% (+20-25%)
- **Pathway extraction:** 75% → 85-90% (+10-15%)
- **New categories:** Receptors, genes, proteins, organelles, etc.
- **Overall:** Comprehensive coverage of aging literature

**Ready to implement! Start with `LLM_KEYWORD_GENERATION_PROMPTS.md`** 🚀
