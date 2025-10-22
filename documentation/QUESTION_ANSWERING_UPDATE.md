# Question Answering Pipeline - Update Summary

## Changes Made

### 1. **New Question Structure Support**

The script now supports the updated `questions_part2.json` structure:

```json
{
    "question_name": {
        "question": "Full question text...",
        "answers": "Option1 / Option2 / Option3"
    }
}
```

**Example:**
```json
{
    "aging_biomarker": {
        "question": "Does it suggest an aging biomarker...",
        "answers": "Yes, quantitatively shown / Yes, but not shown / No"
    }
}
```

### 2. **Enhanced Output Format**

LLM now returns structured answers with:
- **answer**: Selected option from the list
- **confidence**: Confidence score (0.0-1.0)
- **reasoning**: Brief explanation (1-2 sentences)

**Example LLM Output:**
```json
{
  "aging_biomarker": {
    "answer": "Yes, quantitatively shown",
    "confidence": 0.9,
    "reasoning": "The paper presents statistical data showing correlation between the biomarker and aging rate."
  },
  "molecular_mechanism_of_aging": {
    "answer": "No",
    "confidence": 0.8,
    "reasoning": "The paper does not describe specific molecular pathways contributing to aging."
  }
}
```

### 3. **Question Name-Based Mapping**

- Answers are now mapped by **question names** (e.g., `aging_biomarker`) instead of numeric order
- This ensures robust matching even if question order changes
- More maintainable and less error-prone

### 4. **Strict Validation (No Fallbacks)**

The pipeline now implements **strict validation** with NO default fallbacks:

#### Validation Checks:
1. ✅ All questions must be answered
2. ✅ Each answer must have the required structure (`answer`, `confidence`, `reasoning`)
3. ✅ Answer must be from allowed options (or "Not available" for abstracts)
4. ❌ If validation fails → retry (up to 3 attempts)
5. ❌ If all retries fail → mark paper as FAILED (no default answers)

#### Failed Generation Criteria:
- Missing questions in response
- Missing required fields (`answer`, `confidence`, `reasoning`)
- Answer not in allowed options
- Invalid JSON structure

### 5. **"Not Available" Option for Abstracts**

Papers **without full text** (abstract-only) can answer **"Not available"** for any question:

- Full text papers: Must choose from provided options only
- Abstract-only papers: Can choose from provided options OR "Not available"

This is automatically handled in validation and prompts.

### 6. **Updated Database Schema**

New `paper_answers` table structure:

```sql
CREATE TABLE paper_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi TEXT,
    question_name TEXT,           -- NEW: question identifier
    question_text TEXT,            -- NEW: full question text
    answer TEXT,
    confidence_score REAL,         -- NEW: confidence (0.0-1.0)
    reasoning TEXT,                -- NEW: explanation
    UNIQUE(doi, question_name),
    FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
)
```

### 7. **Enhanced Statistics Tracking**

New stats tracked:
- `papers_failed`: Count of papers that failed validation
- Breakdown of successful papers (full text vs abstract)
- Clear reporting of validation failures

**Example Output:**
```
PIPELINE SUMMARY
======================================================================
Total papers: 100
Successfully processed: 95
Failed (validation errors): 5

Successful papers breakdown:
  - With full text: 70
  - With abstract only: 25

Token usage:
  - Prompt tokens: 1,500,000
  - Completion tokens: 50,000
  - Total tokens: 1,550,000

Estimated cost: $0.68
======================================================================
```

## Usage Examples

### Command Line
```bash
python scripts/answer_questions_per_paper.py \
    --evaluations-db evaluations.db \
    --papers-db papers.db \
    --questions-file data/questions_part2.json \
    --results-db paper_answers.db \
    --output-file paper_answers.json \
    --limit 100 \
    --resume-from-db
```

### Python API
```python
from scripts.answer_questions_per_paper import QuestionAnsweringPipeline

pipeline = QuestionAnsweringPipeline(
    questions_file="data/questions_part2.json"
)

pipeline.run_pipeline(
    evaluations_db="evaluations.db",
    papers_db="papers.db",
    output_file="paper_answers.json",
    results_db="paper_answers.db",
    limit=None,
    resume_from_db=True,
    store_processed_text=True
)
```

## Output Format

### JSON Output
```json
{
  "results": [
    {
      "doi": "10.1234/example",
      "pmid": "12345678",
      "title": "Paper Title",
      "validation_result": "valid",
      "confidence_score": 10,
      "answers": {
        "aging_biomarker": {
          "answer": "Yes, quantitatively shown",
          "confidence": 0.9,
          "reasoning": "Statistical correlation shown with p<0.05",
          "question_text": "Does it suggest an aging biomarker..."
        },
        "molecular_mechanism_of_aging": {
          "answer": "Yes",
          "confidence": 0.85,
          "reasoning": "Describes mTOR pathway involvement in aging",
          "question_text": "Does it suggest a molecular mechanism..."
        }
      },
      "processed_text_length": 15234,
      "used_full_text": true,
      "timestamp": "2025-10-20T09:40:00"
    }
  ],
  "stats": {
    "total_papers": 100,
    "papers_processed": 95,
    "papers_failed": 5,
    "papers_with_full_text": 70,
    "papers_with_abstract_only": 25,
    "prompt_tokens": 1500000,
    "completion_tokens": 50000,
    "total_tokens": 1550000,
    "estimated_cost_usd": 0.68
  }
}
```

## Database Queries

### Get all answers for a paper
```sql
SELECT question_name, answer, confidence_score, reasoning
FROM paper_answers
WHERE doi = '10.1234/example';
```

### Count papers by answer for specific question
```sql
SELECT answer, COUNT(*) as count
FROM paper_answers
WHERE question_name = 'aging_biomarker'
GROUP BY answer
ORDER BY count DESC;
```

### Get high-confidence answers
```sql
SELECT doi, question_name, answer, confidence_score, reasoning
FROM paper_answers
WHERE confidence_score >= 0.8
ORDER BY confidence_score DESC;
```

### Get papers that answered "Not available"
```sql
SELECT DISTINCT doi, question_name
FROM paper_answers
WHERE answer = 'Not available';
```

### Average confidence by question
```sql
SELECT question_name, 
       AVG(confidence_score) as avg_confidence,
       COUNT(*) as total_answers
FROM paper_answers
GROUP BY question_name
ORDER BY avg_confidence DESC;
```

## Error Handling

### Validation Failures
When validation fails, the script will:
1. Print detailed error messages showing which checks failed
2. Retry up to 3 times with exponential backoff
3. If all retries fail, mark paper as failed (no default answers)
4. Continue processing remaining papers

### Example Error Messages
```
⚠️  Missing answer for question: aging_biomarker
⚠️  Invalid answer format for molecular_mechanism_of_aging: expected dict, got str
⚠️  Invalid answer for longevity_intervention_to_test: 'Maybe' not in ['Yes', 'No']
❌ Failed to get valid answers for 10.1234/example after 3 attempts: Answer validation failed
```

## Migration Notes

If you have an existing database from the old version:
1. The schema has changed - you'll need to recreate the database
2. Old results won't be compatible with the new structure
3. Consider backing up old results before running the new version

## Benefits of New Approach

1. **More Robust**: Question name mapping prevents order-dependent errors
2. **Better Validation**: Strict checks ensure data quality
3. **Richer Data**: Confidence scores and reasoning provide context
4. **Clearer Failures**: Failed papers are tracked, not hidden with defaults
5. **Flexible Options**: "Not available" for abstracts when information is unclear
6. **Better Analysis**: Confidence scores enable filtering and quality assessment
