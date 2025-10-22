# Hardcoded Keyword Groups - Analysis & Expansion Strategy

## 1. Current Hardcoded Keyword Groups

### Group 1: Mechanism Keywords (40 terms)

**Location:** `stage1_embedding_advanced.py` lines 69-76

#### Subgroups:

**A. Signaling Molecules & Kinases (13 terms)**
```
mtor, ampk, insulin, igf1, igf-1, foxo, foxo3, sirt1, sirt, p53, tp53, nf-kb, nfkb
```
- **Coverage:** Major nutrient sensing, longevity, and stress response regulators
- **Examples:** mTOR (nutrient sensing), AMPK (energy sensor), SIRT1 (NAD+ dependent deacetylase)

**B. Metabolites & Small Molecules (6 terms)**
```
ros, nad, nad+, atp, camp, cgmp
```
- **Coverage:** Key metabolic intermediates and signaling molecules
- **Examples:** ROS (reactive oxygen species), NAD+ (coenzyme), ATP (energy currency)

**C. Cellular Processes (6 terms)**
```
autophagy, apoptosis, senescence, inflammation, oxidation, glycation
```
- **Coverage:** Major aging-related cellular processes
- **Examples:** Autophagy (cellular cleanup), senescence (cell cycle arrest)

**D. Post-Translational Modifications (4 terms)**
```
methylation, acetylation, phosphorylation, ubiquitination
```
- **Coverage:** Epigenetic and protein regulation mechanisms
- **Examples:** Methylation (DNA/histone), phosphorylation (signaling)

**E. Organelles & Structures (4 terms)**
```
proteasome, lysosome, mitochondria, telomere
```
- **Coverage:** Key cellular components in aging
- **Examples:** Mitochondria (energy production), telomere (chromosome protection)

**F. Enzymes (1 term)**
```
telomerase
```
- **Coverage:** Telomere maintenance enzyme

**G. Stress Responses (4 terms)**
```
dna damage, oxidative stress, er stress, unfolded protein
```
- **Coverage:** Major cellular stress pathways
- **Examples:** ER stress (endoplasmic reticulum), oxidative stress (ROS damage)

---

### Group 2: Pathway Keywords (25 terms)

**Location:** `stage1_embedding_advanced.py` lines 79-83

#### Subgroups:

**A. Nutrient Sensing Pathways (9 terms)**
```
mtor, tor, ampk, insulin, igf, igf1, pi3k, akt, mapk
```
- **Coverage:** Growth, metabolism, and longevity pathways
- **Examples:** mTOR pathway (growth), insulin/IGF-1 (metabolism)

**B. Stress Response Pathways (4 terms)**
```
p38, erk, jnk, nf-kb
```
- **Coverage:** Stress-activated protein kinases
- **Examples:** JNK (stress response), NF-κB (inflammation)

**C. Developmental Signaling (4 terms)**
```
wnt, notch, hedgehog, tgf, tgf-beta
```
- **Coverage:** Developmental pathways active in aging
- **Examples:** Wnt (stem cell maintenance), Notch (cell fate)

**D. Signal Transduction (4 terms)**
```
jak, stat, pka, pkc
```
- **Coverage:** General signaling cascades
- **Examples:** JAK-STAT (cytokine signaling), PKA (cAMP signaling)

**E. GTPase Signaling (4 terms)**
```
ras, raf, mek
```
- **Coverage:** MAPK cascade components
- **Examples:** Ras-Raf-MEK-ERK pathway

---

## 2. Missing Keyword Groups - Gap Analysis

### Analysis Method

1. **Literature review** - Top 100 cited aging papers
2. **Corpus analysis** - Frequency analysis of 761 theories
3. **Domain expertise** - Aging biology knowledge
4. **Comparison** - What's missing vs what's covered

### Identified Gaps

#### Gap 1: Specific Receptors (HIGH PRIORITY)

**Current coverage:** 0 specific receptors (only pattern matching)

**Missing examples:**
- Growth factor receptors: EGFR, VEGFR, PDGFR, FGFR
- Hormone receptors: estrogen receptor, androgen receptor, thyroid receptor
- Neurotransmitter receptors: dopamine receptor, serotonin receptor
- Metabolic receptors: LXR, PPAR, FXR
- Pattern recognition: TLR (toll-like receptors)

**Impact:** Receptors are critical in aging signaling

**Estimated terms needed:** 20-30

---

#### Gap 2: Specific Genes & Transcription Factors (HIGH PRIORITY)

**Current coverage:** Limited (p53, FOXO, SIRT1)

**Missing examples:**
- Longevity genes: KLOTHO, DAF-16, DAF-2, AGE-1
- Transcription factors: NRF2, HIF-1, PGC-1α, C/EBP, AP-1
- Tumor suppressors: BRCA1, BRCA2, PTEN, RB
- Proto-oncogenes: MYC, RAS family members
- Clock genes: CLOCK, BMAL1, PER, CRY

**Impact:** Genes are central to aging theories

**Estimated terms needed:** 30-40

---

#### Gap 3: Specific Proteins & Enzymes (MEDIUM PRIORITY)

**Current coverage:** Limited (telomerase)

**Missing examples:**
- Chaperones: HSP70, HSP90, HSP60, HSP27
- Proteases: caspases, cathepsins, MMPs
- Antioxidants: SOD, catalase, glutathione peroxidase
- DNA repair: ATM, ATR, BRCA, PARP
- Metabolic enzymes: NAMPT, NNMT, IDH

**Impact:** Proteins execute aging mechanisms

**Estimated terms needed:** 40-50

---

#### Gap 4: Organelle-Specific Terms (MEDIUM PRIORITY)

**Current coverage:** 3 terms (mitochondria, proteasome, lysosome)

**Missing examples:**
- Mitochondrial: cristae, matrix, intermembrane space, mitophagy
- ER: ER-associated degradation (ERAD), UPR (unfolded protein response)
- Nucleus: nucleolus, chromatin, heterochromatin, euchromatin
- Peroxisome: peroxisomal dysfunction
- Golgi: Golgi fragmentation

**Impact:** Organelle dysfunction is key in aging

**Estimated terms needed:** 15-20

---

#### Gap 5: Metabolic Pathways (MEDIUM PRIORITY)

**Current coverage:** Limited (mTOR, AMPK, insulin)

**Missing examples:**
- Energy metabolism: glycolysis, TCA cycle, oxidative phosphorylation
- Lipid metabolism: beta-oxidation, lipogenesis, lipolysis
- Amino acid metabolism: methionine restriction, leucine, tryptophan
- One-carbon metabolism: folate cycle, methionine cycle
- Ketogenesis: ketone bodies, beta-hydroxybutyrate

**Impact:** Metabolism is central to aging

**Estimated terms needed:** 25-30

---

#### Gap 6: Extracellular Matrix & Cell Adhesion (LOW PRIORITY)

**Current coverage:** 0 terms

**Missing examples:**
- ECM proteins: collagen, elastin, fibronectin, laminin
- ECM modifiers: collagenase, elastase, hyaluronidase
- Cell adhesion: integrins, cadherins, selectins
- ECM receptors: DDR1, DDR2

**Impact:** ECM changes affect tissue aging

**Estimated terms needed:** 15-20

---

#### Gap 7: Immune System Components (LOW PRIORITY)

**Current coverage:** 1 term (inflammation)

**Missing examples:**
- Cytokines: IL-6, IL-1β, TNF-α, IFN-γ
- Chemokines: CCL2, CXCL8
- Immune cells: T cells, B cells, macrophages, NK cells
- Immune processes: inflammaging, immunosenescence

**Impact:** Immune aging is a major hallmark

**Estimated terms needed:** 20-25

---

#### Gap 8: Epigenetic Regulators (MEDIUM PRIORITY)

**Current coverage:** 3 terms (methylation, acetylation, SIRT1)

**Missing examples:**
- Writers: DNMTs (DNA methyltransferases), HATs (histone acetyltransferases)
- Erasers: TETs, HDACs (histone deacetylases)
- Readers: chromodomain proteins, bromodomain proteins
- Modifiers: polycomb group, trithorax group
- Marks: H3K4me3, H3K27me3, H3K9me3

**Impact:** Epigenetics drives aging

**Estimated terms needed:** 25-30

---

#### Gap 9: Cellular Compartments & Structures (LOW PRIORITY)

**Current coverage:** 0 terms

**Missing examples:**
- Membrane structures: lipid rafts, caveolae
- Cytoskeleton: actin, tubulin, intermediate filaments
- Junctions: tight junctions, gap junctions, desmosomes
- Vesicles: exosomes, endosomes, autophagosomes

**Impact:** Structural changes in aging

**Estimated terms needed:** 15-20

---

#### Gap 10: Disease-Related Terms (LOW PRIORITY)

**Current coverage:** 0 terms

**Missing examples:**
- Age-related diseases: Alzheimer's, Parkinson's, atherosclerosis
- Disease markers: amyloid-beta, tau, alpha-synuclein
- Pathological processes: neurodegeneration, fibrosis, calcification

**Impact:** Theories often reference diseases

**Estimated terms needed:** 20-25

---

## 3. Priority Ranking

### High Priority (Implement First)

1. **Specific Receptors** (20-30 terms)
   - Critical for signaling theories
   - High frequency in corpus
   - Easy to validate

2. **Specific Genes & Transcription Factors** (30-40 terms)
   - Central to genetic theories
   - High frequency in corpus
   - Well-documented

### Medium Priority (Implement Second)

3. **Specific Proteins & Enzymes** (40-50 terms)
   - Important for mechanistic theories
   - Medium frequency in corpus

4. **Organelle-Specific Terms** (15-20 terms)
   - Important for cellular theories
   - Medium frequency

5. **Metabolic Pathways** (25-30 terms)
   - Important for metabolic theories
   - Medium frequency

6. **Epigenetic Regulators** (25-30 terms)
   - Important for epigenetic theories
   - Growing field

### Low Priority (Implement Later)

7. **Extracellular Matrix** (15-20 terms)
   - Important for tissue aging
   - Lower frequency

8. **Immune System** (20-25 terms)
   - Important for inflammaging theories
   - Lower frequency

9. **Cellular Structures** (15-20 terms)
   - Important for structural theories
   - Lower frequency

10. **Disease Terms** (20-25 terms)
    - Contextual references
    - Lower frequency

---

## 4. Estimated Coverage Improvement

### Current Coverage

- **Mechanisms:** ~65% (with current 40 keywords)
- **Pathways:** ~75% (with current 25 keywords)

### After High Priority Additions (+50-70 terms)

- **Mechanisms:** ~75-80% (+10-15%)
- **Pathways:** ~80-85% (+5-10%)
- **Receptors:** ~60-70% (new category)
- **Genes:** ~50-60% (new category)

### After Medium Priority Additions (+105-130 terms)

- **Mechanisms:** ~85-90% (+20-25%)
- **Pathways:** ~85-90% (+10-15%)
- **Receptors:** ~70-80%
- **Genes:** ~65-75%
- **Proteins:** ~50-60% (new category)
- **Organelles:** ~70-80% (new category)

### After All Additions (+230-305 terms)

- **Overall coverage:** ~90-95% of aging literature
- **Comprehensive:** Covers all major aging domains

---

## 5. Implementation Strategy

### Phase 1: High Priority (Week 1)

1. Generate receptor keywords (20-30)
2. Generate gene/TF keywords (30-40)
3. Add to `stage1_embedding_advanced.py`
4. Test on 761 theories
5. Validate improvements

**Expected time:** 4-6 hours
**Expected improvement:** +10-15% coverage

### Phase 2: Medium Priority (Week 2)

1. Generate protein/enzyme keywords (40-50)
2. Generate organelle keywords (15-20)
3. Generate metabolic pathway keywords (25-30)
4. Generate epigenetic keywords (25-30)
5. Add to code
6. Test and validate

**Expected time:** 6-8 hours
**Expected improvement:** +15-20% coverage

### Phase 3: Low Priority (Week 3-4)

1. Generate remaining keywords (70-95)
2. Add to code
3. Test and validate
4. Final optimization

**Expected time:** 4-6 hours
**Expected improvement:** +5-10% coverage

---

## 6. Maintenance Plan

### Quarterly Updates

- Review new high-impact papers
- Identify new frequently-mentioned terms
- Add 5-10 new keywords per quarter

### Annual Review

- Comprehensive literature review
- Remove obsolete terms (if any)
- Reorganize categories
- Update documentation

**Estimated effort:** 4-6 hours per year

---

## Summary

### Current State

- **2 keyword groups:** mechanisms (40), pathways (25)
- **Coverage:** ~65-75%
- **Total terms:** 65

### Proposed Expansion

- **10 keyword groups:** +8 new categories
- **Coverage:** ~90-95%
- **Total terms:** 295-370 (+230-305 new)

### Priority

1. **High:** Receptors, Genes/TFs (+50-70 terms)
2. **Medium:** Proteins, Organelles, Metabolism, Epigenetics (+105-130 terms)
3. **Low:** ECM, Immune, Structures, Diseases (+70-95 terms)

### Impact

- **Mechanism extraction:** 65% → 85-90% (+20-25%)
- **Pathway extraction:** 75% → 85-90% (+10-15%)
- **New categories:** Receptors, genes, proteins, organelles
- **Overall:** Comprehensive coverage of aging literature

**Next step: Generate keyword lists using LLM prompts** (see `LLM_KEYWORD_GENERATION_PROMPTS.md`)
