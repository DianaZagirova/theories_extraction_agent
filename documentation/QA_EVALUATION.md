# Question Answering Evaluation

## Overview

Scripts to evaluate LLM question-answering results against a validation set with ground truth answers.

## Files

- **`scripts/evaluate_qa_results.py`** - Compare LLM answers with validation set and compute metrics
- **`scripts/export_qa_results.py`** - Export LLM answers for specific DOIs to JSON
- **`data/qa_validation_set.json`** - Ground truth validation answers for 5 papers

## Validation Set Format

```json
[
  {
    "doi": "10.1016/j.arr.2021.101557",
    "Q1: Does the paper/theory suggest an aging biomarker...": "Yes",
    "Q2: Does the paper/theory suggest a molecular mechanism of aging?": "Yes",
    ...
  }
]
```

## Question Mapping

The validation set uses different question text than our system. The mapping is:

| Validation Question | System Question Name |
|---------------------|---------------------|
| Q1: aging biomarker | `aging_biomarker` |
| Q2: molecular mechanism | `molecular_mechanism_of_aging` |
| Q3: longevity intervention | `longevity_intervention_to_test` |
| Q4: aging cannot be reversed | `aging_cannot_be_reversed` |
| Q5: cross-species biomarker | `cross_species_longevity_biomarker` |
| Q6: naked mole rat | `naked_mole_rat_lifespan_explanation` |
| Q7: birds lifespan | `birds_lifespan_explanation` |
| Q8: large animals | `large_animals_lifespan_explanation` |
| Q9: calorie restriction | `calorie_restriction_lifespan_explanation` |

## Usage

### 1. Export LLM Results

Extract answers for validation DOIs:

```bash
python scripts/export_qa_results.py \
    --results-db ./qa_results/qa_results.db \
    --validation-file data/qa_validation_set.json \
    --output-file qa_exported_results.json
```

Or use a custom DOI list:

```bash
python scripts/export_qa_results.py \
    --results-db ./qa_results/qa_results.db \
    --dois-file my_dois.txt \
    --output-file qa_exported_results.json
```

### 2. Evaluate Results

Compare LLM answers with validation set:

```bash
python scripts/evaluate_qa_results.py \
    --validation-file data/qa_validation_set.json \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_evaluation_results.json
```

## Output

### Console Output

```
======================================================================
QUESTION ANSWERING EVALUATION RESULTS
======================================================================

📊 OVERALL METRICS
Total questions: 45
Correct answers: 38
Overall accuracy: 84.44%

📋 PER-QUESTION ACCURACY
----------------------------------------------------------------------
aging_biomarker                                80.0% (4/5)
molecular_mechanism_of_aging                  100.0% (5/5)
longevity_intervention_to_test                 80.0% (4/5)
aging_cannot_be_reversed                      100.0% (5/5)
cross_species_longevity_biomarker              60.0% (3/5)
naked_mole_rat_lifespan_explanation           100.0% (5/5)
birds_lifespan_explanation                     80.0% (4/5)
large_animals_lifespan_explanation             80.0% (4/5)
calorie_restriction_lifespan_explanation       80.0% (4/5)

📄 PER-PAPER RESULTS
----------------------------------------------------------------------
10.1016/j.arr.2021.101557                    100.0% (9/9)
10.1016/j.tree.2022.08.003                    88.9% (8/9)
10.14336/AD.2025.0541                         77.8% (7/9)
10.1038/s43587-023-00527-6                    88.9% (8/9)
0.1016/j.arr.2024.102310                      66.7% (6/9)

❌ INCORRECT ANSWERS
----------------------------------------------------------------------

DOI: 10.14336/AD.2025.0541
Question: aging_biomarker
  True answer: Yes
  LLM answer:  No (confidence: 0.85)

DOI: 10.14336/AD.2025.0541
Question: cross_species_longevity_biomarker
  True answer: Yes
  LLM answer:  No (confidence: 0.90)
...
```

### JSON Output

Detailed results saved to `qa_evaluation_results.json`:

```json
{
  "overall_accuracy": 0.8444,
  "total_correct": 38,
  "total_questions": 45,
  "question_metrics": {
    "aging_biomarker": {
      "correct": 4,
      "total": 5,
      "accuracy": 0.8,
      "details": [...]
    }
  },
  "paper_results": [
    {
      "doi": "10.1016/j.arr.2021.101557",
      "correct": 9,
      "total": 9,
      "accuracy": 1.0,
      "details": [...]
    }
  ],
  "missing_papers": [],
  "missing_answers": []
}
```

## Metrics Explained

### Overall Accuracy
- Percentage of all questions answered correctly across all papers
- Formula: `correct_answers / total_questions`

### Per-Question Accuracy
- Accuracy for each specific question across all papers
- Helps identify which questions the model struggles with

### Per-Paper Accuracy
- Accuracy for each paper across all questions
- Helps identify which papers are harder to analyze

### Answer Normalization

For comparison, answers are normalized:
- `"Yes, quantitatively shown"` → `"Yes"`
- `"Yes, but not shown"` → `"Yes"`
- `"Not available"` → Treated as incorrect (should be Yes/No)

## Validation Set DOIs

The validation set includes 5 papers:

1. `10.1016/j.arr.2021.101557`
2. `10.1016/j.tree.2022.08.003`
3. `10.14336/AD.2025.0541`
4. `10.1038/s43587-023-00527-6`
5. `0.1016/j.arr.2024.102310` (Note: typo in DOI - missing '1')

## Workflow

### Complete Evaluation Workflow

```bash
# 1. Run QA on validation DOIs
python scripts/answer_questions_per_paper.py \
    --dois-file data/test_qa_dois.txt \
    --results-db ./qa_results/qa_results.db

# 2. Export results
python scripts/export_qa_results.py \
    --results-db ./qa_results/qa_results.db \
    --validation-file data/qa_validation_set.json \
    --output-file qa_exported_results.json

# 3. Evaluate against validation set
python scripts/evaluate_qa_results.py \
    --validation-file data/qa_validation_set.json \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_evaluation_results.json
```

## Interpreting Results

### High Accuracy (>90%)
- Model understands the question well
- Paper content clearly addresses the question
- Good agreement with human annotation

### Medium Accuracy (70-90%)
- Some ambiguity in question interpretation
- Paper content may be borderline
- Review incorrect answers for patterns

### Low Accuracy (<70%)
- Question may be too complex or ambiguous
- Model may need better prompting
- Consider reviewing validation answers

## Common Issues

### Missing Papers
- Paper not processed yet
- Processing failed
- DOI mismatch

### Missing Answers
- Question not in database
- Processing incomplete
- Database schema mismatch

### Answer Mismatches
- Different interpretation of question
- Borderline cases
- Model hallucination
- Insufficient context (abstract vs full text)

## Improving Accuracy

1. **Review Incorrect Answers**: Look for patterns in mistakes
2. **Check Confidence Scores**: Low confidence may indicate uncertainty
3. **Compare Full Text vs Abstract**: Abstract-only may miss details
4. **Adjust Prompts**: Refine question wording or instructions
5. **Add Examples**: Provide few-shot examples in prompts
