# Abstract-Only Theory Extraction Guide

## Overview

Two separate scripts handle theory extraction based on paper availability:

1. **`extract_theories_per_paper.py`** - Extracts from papers WITH full text (primary/main script)
2. **`extract_theories_abstract_per_paper.py`** - Extracts from papers WITHOUT full text (abstract-only)

## Key Differences

| Feature | Full-Text Script | Abstract-Only Script |
|---------|-----------------|---------------------|
| **Input** | Papers with `full_text` or `full_text_sections` | Papers with ONLY abstract (no full text) |
| **Output JSON** | `theories_per_paper.json` | `theories_abstract_per_paper.json` |
| **Output DB** | `theories.db` | `theories_abstract.db` |
| **Extraction Quality** | High (comprehensive) | Lower (limited to abstract) |
| **Theory Limit** | Up to 15 theories | Up to 3-5 theories |
| **Token Limits** | 40K back, 8K max tokens | 5K back, 3K max tokens |

## Safety Guarantees

**The abstract-only script will NEVER override full-text results because:**

1. ✅ Uses separate output files (`theories_abstract_per_paper.json` vs `theories_per_paper.json`)
2. ✅ Uses separate database (`theories_abstract.db` vs `theories.db`)
3. ✅ Explicitly skips papers that have full text available
4. ✅ Raises error if accidentally processing a paper with full text

## Usage

### Full-Text Extraction (Primary)
```bash
python scripts/extract_theories_per_paper.py \
  --evaluations-db /path/to/evaluations.db \
  --papers-db /path/to/papers.db \
  --output theories_per_paper.json \
  --results-db theories.db \
  --max-workers 4
```

### Abstract-Only Extraction (Supplementary)
```bash
python scripts/extract_theories_abstract_per_paper.py \
  --evaluations-db /path/to/evaluations.db \
  --papers-db /path/to/papers.db \
  --output theories_abstract_per_paper.json \
  --results-db theories_abstract.db \
  --max-workers 4
```

### Test Mode
```bash
# Test abstract extraction
python scripts/extract_theories_abstract_per_paper.py \
  --test \
  --limit 10
# Output: test_abstract_<timestamp>.json
```

## Workflow

```
Validated Papers
       |
       v
Has full text? ----YES----> extract_theories_per_paper.py
       |                           |
       NO                          v
       |                    theories_per_paper.json
       v                    theories.db
Has abstract?
       |
      YES
       |
       v
extract_theories_abstract_per_paper.py
       |
       v
theories_abstract_per_paper.json
theories_abstract.db
```

## Implementation Details

### Abstract-Only Script Changes

1. **Docstring** - Clear description of abstract-only purpose
2. **Token limits** - Reduced for shorter abstracts (5K/3K vs 40K/8K)
3. **Paper loading** - Skips papers with `full_text` or `full_text_sections`
4. **Validation** - Raises error if full text detected during processing
5. **Prompt** - Adjusted to mention "abstract only" and limit to 3-5 theories
6. **Output files** - Separate defaults to prevent conflicts
7. **Test mode** - Uses `test_abstract_<timestamp>.json` prefix

### Code Locations

**Key validation logic (lines 254-260, 614-615):**
```python
# Skip papers with full text - those are handled by extract_theories_per_paper.py
if has_full_text or has_sections:
    continue  # or raise ValueError in _load_and_process
```

**Separate outputs (lines 755, 761):**
```python
default='theories_abstract_per_paper.json'  # Not theories_per_paper.json
default='theories_abstract.db'              # Not theories.db
```

## Merging Results (Optional)

If you need combined results, merge programmatically:

```python
import json

# Load both results
with open('theories_per_paper.json') as f:
    full_text_results = json.load(f)

with open('theories_abstract_per_paper.json') as f:
    abstract_results = json.load(f)

# Merge (full-text takes precedence)
full_text_dois = {r['doi'] for r in full_text_results['results']}
combined = full_text_results['results'].copy()

for result in abstract_results['results']:
    if result['doi'] not in full_text_dois:
        combined.append(result)

print(f"Full-text: {len(full_text_results['results'])}")
print(f"Abstract-only: {len(abstract_results['results'])}")
print(f"Combined: {len(combined)}")
```

## Recommendations

1. **Run full-text extraction first** - Higher quality results
2. **Run abstract extraction second** - Supplementary coverage
3. **Keep separate databases** - Easier to track data provenance
4. **Monitor quality** - Abstract-only extractions may have lower confidence
5. **Use `--resume-from-db`** - Skip already processed papers efficiently
