# LLM Prompts for Keyword Generation

## Overview

This document contains carefully crafted prompts for generating comprehensive keyword lists for aging research theory extraction. Use these prompts with GPT-4, Claude, or similar LLMs.

---

## General Instructions for Using These Prompts

### Before Running Prompts

1. **Choose your LLM:** GPT-4, Claude 3, or equivalent
2. **Set temperature:** 0.3-0.5 (balance creativity and accuracy)
3. **Review output:** Always validate generated keywords
4. **Deduplicate:** Remove overlaps with existing keywords

### After Getting Results

1. **Validate:** Check against aging literature
2. **Test:** Run on sample theories
3. **Measure:** Compare coverage before/after
4. **Iterate:** Refine based on results

---

## Prompt Template Structure

Each prompt follows this structure:

```
[CONTEXT] → [TASK] → [CONSTRAINTS] → [OUTPUT FORMAT]
```

---

## HIGH PRIORITY PROMPTS

### Prompt 1: Receptor Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify all commonly mentioned receptors in aging literature.

# Current Coverage
We already have pattern matching for "X receptor" (e.g., "insulin receptor"), but we need standalone receptor names that appear without the word "receptor" (e.g., "EGFR activation").

# Task
Generate a comprehensive list of receptor names that frequently appear in aging research literature. Focus on:
1. Growth factor receptors (EGFR, VEGFR, PDGFR, FGFR, etc.)
2. Hormone receptors (estrogen receptor, androgen receptor, thyroid receptor, etc.)
3. Neurotransmitter receptors (dopamine, serotonin, GABA, glutamate, etc.)
4. Metabolic receptors (PPAR, LXR, FXR, etc.)
5. Pattern recognition receptors (TLR family, NOD-like receptors, etc.)
6. Cytokine receptors (IL-6R, TNF-R, IFN-R, etc.)

# Constraints
- Include only receptors mentioned in aging/longevity research
- Include common abbreviations (e.g., "egfr" and "erbb1")
- Include variant forms (e.g., "ppar-alpha", "ppar-gamma")
- Exclude generic terms like "receptor" or "binding site"
- Aim for 25-35 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
receptor_keywords = [
    'egfr', 'erbb1',  # Epidermal growth factor receptor
    'vegfr',          # Vascular endothelial growth factor receptor
    # ... continue with comments explaining each group
]
```

# Additional Requirements
- Group related receptors together
- Add inline comments explaining receptor families
- Include both full names and abbreviations where applicable
- Sort by importance/frequency in aging research
```

---

### Prompt 2: Gene & Transcription Factor Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify genes and transcription factors frequently mentioned in aging literature.

# Current Coverage
We already have: p53, tp53, foxo, foxo3, sirt1, sirt, nf-kb, nfkb

# Task
Generate a comprehensive list of gene names and transcription factors that frequently appear in aging research literature. Focus on:
1. Longevity genes (KLOTHO, DAF-16, DAF-2, AGE-1, etc.)
2. Transcription factors (NRF2, HIF-1, PGC-1α, C/EBP, AP-1, etc.)
3. Tumor suppressors (BRCA1, BRCA2, PTEN, RB, etc.)
4. Proto-oncogenes (MYC, RAS family, etc.)
5. Clock genes (CLOCK, BMAL1, PER, CRY, etc.)
6. Sirtuins (SIRT2-7)
7. FOXO family members (FOXO1, FOXO4, etc.)

# Constraints
- EXCLUDE already covered: p53, tp53, foxo, foxo3, sirt1, sirt, nf-kb, nfkb
- Include common abbreviations and variant names
- Include both human and model organism names (e.g., DAF-16 from C. elegans)
- Exclude generic terms like "gene" or "transcription factor"
- Aim for 35-45 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
gene_tf_keywords = [
    # Longevity genes
    'klotho', 'daf-16', 'daf-2', 'age-1',
    
    # Transcription factors
    'nrf2', 'nfe2l2',  # NRF2 (alternative name)
    'hif-1', 'hif-1alpha', 'hif1a',
    
    # ... continue with grouped categories and comments
]
```

# Additional Requirements
- Group by functional category
- Add inline comments for gene families
- Include alternative names and abbreviations
- Note model organism genes (C. elegans, Drosophila, etc.)
```

---

## MEDIUM PRIORITY PROMPTS

### Prompt 3: Protein & Enzyme Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify specific proteins and enzymes frequently mentioned in aging literature.

# Current Coverage
We already have: telomerase, proteasome

# Task
Generate a comprehensive list of protein and enzyme names that frequently appear in aging research literature. Focus on:
1. Chaperones (HSP70, HSP90, HSP60, HSP27, etc.)
2. Proteases (caspases, cathepsins, MMPs, etc.)
3. Antioxidant enzymes (SOD, catalase, glutathione peroxidase, etc.)
4. DNA repair proteins (ATM, ATR, BRCA, PARP, etc.)
5. Metabolic enzymes (NAMPT, NNMT, IDH, etc.)
6. Kinases (not covered in pathways)
7. Phosphatases

# Constraints
- EXCLUDE already covered: telomerase, proteasome
- Include common abbreviations
- Include isoforms where relevant (e.g., SOD1, SOD2)
- Exclude generic terms like "protein" or "enzyme"
- Aim for 45-55 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
protein_enzyme_keywords = [
    # Chaperones
    'hsp70', 'hsp90', 'hsp60', 'hsp27', 'hsp40',
    
    # Proteases
    'caspase-3', 'caspase-9', 'cathepsin', 'mmp',
    
    # ... continue with grouped categories
]
```

# Additional Requirements
- Group by functional category
- Include isoform numbers where important
- Add comments for protein families
- Note subcellular localization if relevant (e.g., SOD1 cytoplasmic, SOD2 mitochondrial)
```

---

### Prompt 4: Organelle-Specific Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify organelle-specific terms frequently mentioned in aging literature.

# Current Coverage
We already have: mitochondria, proteasome, lysosome

# Task
Generate a comprehensive list of organelle-specific terms that frequently appear in aging research literature. Focus on:
1. Mitochondrial structures (cristae, matrix, intermembrane space, etc.)
2. Mitochondrial processes (mitophagy, mitochondrial fission/fusion, etc.)
3. ER structures and processes (ERAD, UPR, ER stress, etc.)
4. Nuclear structures (nucleolus, chromatin, heterochromatin, etc.)
5. Peroxisome-related terms
6. Golgi-related terms
7. Other organelles (ribosome, centrosome, etc.)

# Constraints
- EXCLUDE already covered: mitochondria, proteasome, lysosome
- Include both structure names and processes
- Include abbreviations (e.g., "ER" for endoplasmic reticulum)
- Exclude generic terms like "organelle"
- Aim for 20-30 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
organelle_keywords = [
    # Mitochondrial
    'cristae', 'mitochondrial matrix', 'intermembrane space',
    'mitophagy', 'mitochondrial fission', 'mitochondrial fusion',
    
    # ER
    'endoplasmic reticulum', 'er', 'erad', 'upr',
    
    # ... continue with grouped categories
]
```

# Additional Requirements
- Group by organelle type
- Include both structures and processes
- Add comments explaining abbreviations
- Note when terms are specific to aging research
```

---

### Prompt 5: Metabolic Pathway Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify metabolic pathway terms frequently mentioned in aging literature.

# Current Coverage
We already have: mtor, tor, ampk, insulin, igf, pi3k, akt (covered in signaling pathways)

# Task
Generate a comprehensive list of metabolic pathway terms that frequently appear in aging research literature. Focus on:
1. Energy metabolism (glycolysis, TCA cycle, oxidative phosphorylation, etc.)
2. Lipid metabolism (beta-oxidation, lipogenesis, lipolysis, etc.)
3. Amino acid metabolism (methionine restriction, leucine, tryptophan, etc.)
4. One-carbon metabolism (folate cycle, methionine cycle, etc.)
5. Ketogenesis (ketone bodies, beta-hydroxybutyrate, etc.)
6. NAD+ metabolism (salvage pathway, de novo synthesis, etc.)

# Constraints
- EXCLUDE signaling pathways already covered (mTOR, AMPK, insulin, etc.)
- Focus on metabolic processes, not signaling
- Include pathway names and key metabolites
- Exclude generic terms like "metabolism"
- Aim for 30-40 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
metabolic_pathway_keywords = [
    # Energy metabolism
    'glycolysis', 'tca cycle', 'krebs cycle', 'citric acid cycle',
    'oxidative phosphorylation', 'oxphos', 'electron transport chain', 'etc',
    
    # Lipid metabolism
    'beta-oxidation', 'fatty acid oxidation', 'lipogenesis', 'lipolysis',
    
    # ... continue with grouped categories
]
```

# Additional Requirements
- Group by metabolic category
- Include alternative names for pathways
- Include key metabolites where relevant
- Add comments explaining abbreviations
```

---

### Prompt 6: Epigenetic Regulator Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify epigenetic regulators frequently mentioned in aging literature.

# Current Coverage
We already have: methylation, acetylation, sirt1 (as a deacetylase)

# Task
Generate a comprehensive list of epigenetic regulator terms that frequently appear in aging research literature. Focus on:
1. DNA methylation writers (DNMTs)
2. DNA methylation erasers (TETs)
3. Histone acetyltransferases (HATs)
4. Histone deacetylases (HDACs, excluding SIRT1)
5. Histone methyltransferases (HMTs)
6. Histone demethylases (HDMs)
7. Chromatin remodelers (polycomb, trithorax, SWI/SNF, etc.)
8. Histone marks (H3K4me3, H3K27me3, etc.)

# Constraints
- EXCLUDE already covered: methylation, acetylation, sirt1
- Include enzyme families and specific members
- Include histone marks where frequently mentioned
- Include abbreviations (e.g., "DNMT" and "DNA methyltransferase")
- Aim for 30-40 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list in this exact format:

```python
epigenetic_keywords = [
    # DNA methylation
    'dnmt', 'dnmt1', 'dnmt3a', 'dnmt3b',  # Writers
    'tet', 'tet1', 'tet2',                 # Erasers
    
    # Histone acetylation
    'hat', 'p300', 'cbp',                  # Writers
    'hdac', 'hdac1', 'hdac2', 'hdac3',    # Erasers (excluding SIRT1)
    
    # ... continue with grouped categories
]
```

# Additional Requirements
- Group by modification type and function (writers/erasers/readers)
- Include family names and specific members
- Add comments explaining enzyme classes
- Note which are most relevant to aging
```

---

## LOW PRIORITY PROMPTS

### Prompt 7: Extracellular Matrix Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify extracellular matrix (ECM) terms frequently mentioned in aging literature.

# Task
Generate a comprehensive list of ECM-related terms that frequently appear in aging research literature. Focus on:
1. ECM proteins (collagen, elastin, fibronectin, laminin, etc.)
2. ECM modifiers (collagenase, elastase, hyaluronidase, MMPs, etc.)
3. Cell adhesion molecules (integrins, cadherins, selectins, etc.)
4. ECM receptors (DDR1, DDR2, etc.)
5. Glycosaminoglycans (hyaluronic acid, chondroitin sulfate, etc.)

# Constraints
- Include protein families and specific members
- Include abbreviations
- Aim for 20-25 terms total
- Prioritize by frequency in aging literature

# Output Format
Provide a Python list with grouped categories and comments.
```

---

### Prompt 8: Immune System Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify immune system terms frequently mentioned in aging literature, particularly related to inflammaging and immunosenescence.

# Current Coverage
We already have: inflammation, nf-kb

# Task
Generate a comprehensive list of immune system terms that frequently appear in aging research literature. Focus on:
1. Cytokines (IL-6, IL-1β, TNF-α, IFN-γ, etc.)
2. Chemokines (CCL2, CXCL8, etc.)
3. Immune cell types (T cells, B cells, macrophages, NK cells, etc.)
4. Immune processes (inflammaging, immunosenescence, SASP, etc.)
5. Immune receptors (not covered in receptor prompt)

# Constraints
- EXCLUDE already covered: inflammation, nf-kb
- Include abbreviations and full names
- Include aging-specific terms (inflammaging, immunosenescence)
- Aim for 25-30 terms total

# Output Format
Provide a Python list with grouped categories and comments.
```

---

### Prompt 9: Cellular Structure Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify cellular structure terms frequently mentioned in aging literature.

# Task
Generate a comprehensive list of cellular structure terms that frequently appear in aging research literature. Focus on:
1. Membrane structures (lipid rafts, caveolae, etc.)
2. Cytoskeleton (actin, tubulin, intermediate filaments, etc.)
3. Cell junctions (tight junctions, gap junctions, desmosomes, etc.)
4. Vesicles (exosomes, endosomes, autophagosomes, etc.)
5. Other structures relevant to aging

# Constraints
- Focus on structures that change with aging
- Include abbreviations where applicable
- Aim for 20-25 terms total

# Output Format
Provide a Python list with grouped categories and comments.
```

---

### Prompt 10: Disease-Related Keywords

```markdown
# Context
You are a biogerontology expert helping to build a keyword extraction system for aging research theories. We need to identify age-related disease terms frequently mentioned in aging theories.

# Task
Generate a comprehensive list of age-related disease terms that frequently appear in aging research literature. Focus on:
1. Neurodegenerative diseases (Alzheimer's, Parkinson's, etc.)
2. Cardiovascular diseases (atherosclerosis, etc.)
3. Metabolic diseases (diabetes, etc.)
4. Disease markers (amyloid-beta, tau, alpha-synuclein, etc.)
5. Pathological processes (neurodegeneration, fibrosis, calcification, etc.)

# Constraints
- Focus on diseases frequently mentioned in aging theories
- Include disease markers and pathological processes
- Aim for 25-30 terms total

# Output Format
Provide a Python list with grouped categories and comments.
```

---

## VALIDATION PROMPT

### After Generating Keywords

```markdown
# Context
I have generated the following keyword list for [CATEGORY]:

[PASTE GENERATED LIST HERE]

# Task
Please review this list and:
1. Identify any terms that are NOT commonly used in aging/longevity research
2. Suggest any critical missing terms that should be included
3. Flag any duplicates or very similar terms that should be merged
4. Verify that abbreviations are correct
5. Confirm that terms are appropriately grouped

# Output Format
Provide:
1. Terms to REMOVE (with reason)
2. Terms to ADD (with reason)
3. Terms to MERGE (with suggestion)
4. Corrected abbreviations (if any)
5. Final validated list
```

---

## POST-GENERATION WORKFLOW

### Step 1: Generate Keywords

1. Run each prompt through your chosen LLM
2. Save output to temporary file
3. Review for obvious errors

### Step 2: Validate Keywords

1. Run validation prompt for each category
2. Cross-check against aging literature
3. Test on sample theories

### Step 3: Deduplicate

```python
# Check for overlaps between categories
all_keywords = (
    mechanism_keywords + 
    pathway_keywords + 
    receptor_keywords + 
    gene_tf_keywords + 
    # ... etc
)

duplicates = [k for k in all_keywords if all_keywords.count(k) > 1]
print(f"Duplicates found: {set(duplicates)}")
```

### Step 4: Test Coverage

```python
# Run on sample theories
python src/normalization/stage1_embedding_advanced.py

# Compare before/after
python compare_embeddings.py
```

### Step 5: Measure Improvement

```
Expected improvements:
- High priority: +10-15% coverage
- Medium priority: +15-20% coverage
- Low priority: +5-10% coverage
- Total: +30-45% coverage
```

---

## EXAMPLE USAGE

### Using GPT-4

```python
import openai

# Set up
openai.api_key = "your-key"

# Run prompt
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a biogerontology expert."},
        {"role": "user", "content": PROMPT_1_RECEPTOR_KEYWORDS}
    ],
    temperature=0.3
)

# Extract keywords
keywords = response.choices[0].message.content
```

### Using Claude

```python
import anthropic

# Set up
client = anthropic.Anthropic(api_key="your-key")

# Run prompt
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2000,
    temperature=0.3,
    messages=[
        {"role": "user", "content": PROMPT_1_RECEPTOR_KEYWORDS}
    ]
)

# Extract keywords
keywords = message.content[0].text
```

---

## QUALITY CHECKLIST

Before adding keywords to production:

- [ ] Generated using appropriate prompt
- [ ] Validated against aging literature
- [ ] Tested on sample theories
- [ ] Measured coverage improvement
- [ ] Deduplicated across categories
- [ ] Grouped logically
- [ ] Commented appropriately
- [ ] Abbreviations verified
- [ ] No generic terms included
- [ ] Prioritized by frequency

---

## MAINTENANCE SCHEDULE

### Quarterly (4 times/year)
- Review new high-impact papers
- Add 5-10 new terms per category
- Remove obsolete terms (rare)

### Annually (1 time/year)
- Comprehensive literature review
- Re-run all prompts with updated context
- Reorganize categories if needed
- Update documentation

**Estimated effort:** 2-4 hours per quarter, 8-12 hours annually

---

## Summary

### Prompt Categories

1. **High Priority:** Receptors, Genes/TFs (2 prompts)
2. **Medium Priority:** Proteins, Organelles, Metabolism, Epigenetics (4 prompts)
3. **Low Priority:** ECM, Immune, Structures, Diseases (4 prompts)

### Expected Output

- **Total keywords:** 230-305 new terms
- **Coverage improvement:** +30-45%
- **Time to generate:** 4-8 hours
- **Validation time:** 2-4 hours

### Next Steps

1. Choose LLM (GPT-4 or Claude recommended)
2. Run high-priority prompts first
3. Validate and test
4. Iterate based on results
5. Expand to medium and low priority

**These prompts are production-ready and will significantly improve keyword coverage!**
