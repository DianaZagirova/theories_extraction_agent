# Synonym-Based Question Filtering

## Overview

The QA pipeline now includes intelligent question filtering based on keyword/synonym matching. This feature **skips irrelevant questions** and automatically assigns them a default "No" answer, saving LLM costs and processing time.

## How It Works

1. **Before sending questions to the LLM**, the pipeline checks if the paper text contains relevant keywords for each question
2. **If no keywords are found**, the question is skipped and automatically answered with:
   - `answer: "No"`
   - `confidence: 1.0`
   - `reasoning: "Skipped: no relevant keywords found in text"`
3. **Only relevant questions** are sent to the LLM for analysis

## Configuration

Edit `data/questions_synonyms.json` to configure keyword filters:

```json
{
  "naked_mole_rat_lifespan_explanation": [
    "naked mole rat", 
    "heterocephalus glaber", 
    "mole-rat", 
    "mole rat"
  ],
  "birds_lifespan_explanation": [
    "bird", 
    "avian", 
    "aves"
  ],
  "calorie_restriction_lifespan_explanation": [
    "calorie", 
    "caloric", 
    "dietary restriction", 
    "calorie restriction"
  ]
}
```

### Rules:
- **Empty list `[]`**: No filtering, always ask the question
- **Non-empty list**: Skip question if NONE of the keywords appear in the text (case-insensitive)
- Keywords are matched using simple substring search

## Benefits

✅ **Cost Savings**: Fewer questions sent to LLM = lower API costs  
✅ **Faster Processing**: Skip irrelevant questions immediately  
✅ **Better Accuracy**: Avoid forcing LLM to answer questions about topics not mentioned in the paper  
✅ **Transparent**: Skipped questions are clearly marked in the output

## Example Output

When processing a paper without any mention of birds:

```
⚡ Skipped 1 questions, asking 8 questions
```

In the database/export:
```json
{
  "birds_lifespan_explanation": {
    "answer": "No",
    "confidence": 1.0,
    "reasoning": "Skipped: no relevant keywords found in text"
  }
}
```

## Usage

The feature is **automatically enabled** when running the pipeline:

```bash
python scripts/answer_questions_per_paper.py \
    --dois-file data/sample_from_ext.txt \
    --reset-db
```

The pipeline will:
1. Load synonyms from `data/questions_synonyms.json`
2. Show which questions have filters configured
3. Skip questions without relevant keywords
4. Report how many questions were skipped per paper

## Current Filters

As of now, the following questions have keyword filters:

- **cross_species_longevity_biomarker**: 6 keywords
- **naked_mole_rat_lifespan_explanation**: 4 keywords  
- **birds_lifespan_explanation**: 3 keywords
- **large_animals_lifespan_explanation**: 6 keywords
- **calorie_restriction_lifespan_explanation**: 7 keywords

Questions without filters (empty lists) are always sent to the LLM.
