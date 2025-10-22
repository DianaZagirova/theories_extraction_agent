# Question Answering Pipeline

## Overview

This module answers research questions from scientific papers using LLM analysis. It processes papers (full text or abstracts) and generates structured answers to predefined questions.

## Key Features

- **Flexible Text Processing**: Uses full text when available, falls back to abstracts
- **Text Preprocessing**: Reuses the same preprocessing pipeline as theory extraction
- **Answer Validation**: Ensures answers match allowed options from questions file
- **Database Storage**: Stores answers and processed texts in SQLite database
- **Resume Capability**: Can resume from previous runs
- **Progress Tracking**: Checkpoints every 100 papers

## Architecture

### Main Components

1. **QuestionAnsweringPipeline** (`scripts/answer_questions_per_paper.py`)
   - Main pipeline class
   - Handles paper loading, preprocessing, LLM calls, and result storage

2. **Database Schema** (3 tables)
   - `paper_metadata`: Paper information and processing metadata
   - `paper_answers`: Question-answer pairs for each paper
   - `processed_texts`: Preprocessed paper texts (optional)

3. **Dependencies**
   - `TextPreprocessor`: Text cleaning and section extraction
   - `OpenAIClient`: LLM integration with rate limiting
   - Questions JSON file: Defines questions and allowed answer options

## Database Schema

### paper_metadata
```sql
CREATE TABLE paper_metadata (
    doi TEXT PRIMARY KEY,
    pmid TEXT,
    title TEXT,
    validation_result TEXT,
    confidence_score INTEGER,
    processed_text_length INTEGER,
    used_full_text BOOLEAN,
    timestamp TEXT
)
```

### paper_answers
```sql
CREATE TABLE paper_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi TEXT,
    question TEXT,
    answer TEXT,
    UNIQUE(doi, question),
    FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
)
```

### processed_texts
```sql
CREATE TABLE processed_texts (
    doi TEXT PRIMARY KEY,
    processed_text TEXT,
    FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
)
```

## Questions Format

Questions are defined in a JSON file with the format:

```json
{
    "Question text here?": "Option1 / Option2 / Option3",
    "Another question?": "Yes / No"
}
```

Example (`questions_part2.json`):
```json
{
    "Does it suggest an aging biomarker?": "Yes, quantitatively shown / Yes, but not shown / No",
    "Does it suggest a molecular mechanism of aging?": "Yes / No",
    "Does it suggest a longevity intervention to test?": "Yes / No"
}
```

## Usage

### Command Line

```bash
python scripts/answer_questions_per_paper.py \
    --evaluations-db path/to/evaluations.db \
    --papers-db path/to/papers.db \
    --questions-file /home/diana.z/hack/rag_agent/data/questions_part2.json \
    --results-db paper_answers.db \
    --output-file paper_answers.json \
    --limit 100 \
    --resume-from-db
```

### Python Script

```python
from scripts.answer_questions_per_paper import QuestionAnsweringPipeline

# Initialize pipeline
pipeline = QuestionAnsweringPipeline(
    questions_file="/home/diana.z/hack/rag_agent/data/questions_part2.json"
)

# Run pipeline
pipeline.run_pipeline(
    evaluations_db="evaluations.db",
    papers_db="papers.db",
    output_file="paper_answers.json",
    results_db="paper_answers.db",
    limit=None,  # Process all papers
    resume_from_db=True,  # Skip already processed
    store_processed_text=True  # Store texts in DB
)
```

### Quick Start Example

```bash
# Update paths in run_question_answering.py, then:
python run_question_answering.py
```

## Arguments

### Required
- `--evaluations-db`: Path to evaluations database (contains validated papers)
- `--papers-db`: Path to papers database (contains paper texts)
- `--questions-file`: Path to questions JSON file
- `--results-db`: Path to output database

### Optional
- `--output-file`: Output JSON file (default: `paper_answers.json`)
- `--limit`: Limit number of papers to process
- `--resume-from-db`: Skip papers already in results database
- `--no-store-text`: Don't store processed texts in database

## Environment Variables

Configure in `.env` file:

```bash
# LLM Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Processing Limits
BACK_LIMIT=20000          # Max chars to send to LLM
MAX_TOKENS=2000           # Max tokens in LLM response
TEMPERATURE=0.2           # LLM temperature (0-1)

# Text Preprocessing
PREPROCESS_BUDGET_CHARS=20000  # Max chars after preprocessing
```

## Output Format

### JSON Output (`paper_answers.json`)

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
        "Does it suggest an aging biomarker?": "Yes, quantitatively shown",
        "Does it suggest a molecular mechanism of aging?": "Yes",
        ...
      },
      "processed_text_length": 15234,
      "used_full_text": true,
      "timestamp": "2025-10-20T09:05:00"
    }
  ],
  "stats": {
    "total_papers": 100,
    "papers_processed": 100,
    "papers_with_full_text": 75,
    "papers_with_abstract_only": 25,
    "prompt_tokens": 1500000,
    "completion_tokens": 50000,
    "total_tokens": 1550000,
    "estimated_cost_usd": 0.68
  }
}
```

## Comparison with Theory Extraction Scripts

### Similarities
1. **Text Processing**: Uses same `TextPreprocessor` for cleaning and section extraction
2. **Database Pattern**: Similar structure with metadata + detailed results tables
3. **Paper Loading**: Same validated papers from `evaluations.db`
4. **LLM Integration**: Same `OpenAIClient` with rate limiting
5. **Resume Capability**: Can skip already processed papers

### Differences
1. **Task**: Answers predefined questions vs. extracting theories
2. **Output**: Structured Q&A pairs vs. theory objects
3. **Validation**: Validates answers against allowed options
4. **Prompt**: Question-answering prompt vs. theory extraction prompt
5. **Storage**: Stores processed text in DB (optional feature)

## Performance

- **Processing Speed**: ~5-10 papers/minute (depends on text length and API limits)
- **Cost Estimate**: ~$0.005-0.01 per paper (with gpt-4o-mini)
- **Memory Usage**: Low (papers loaded on-demand, preprocessed in batches)

## Error Handling

- **Rate Limiting**: Automatic retry with exponential backoff
- **Invalid Answers**: Falls back to conservative option (usually "No")
- **Missing Text**: Skips papers without text, logs warning
- **JSON Parse Errors**: Retries up to 3 times, then uses default answers

## Querying Results

### Get all answers for a paper
```sql
SELECT question, answer 
FROM paper_answers 
WHERE doi = '10.1234/example';
```

### Count papers by answer
```sql
SELECT answer, COUNT(*) as count
FROM paper_answers
WHERE question = 'Does it suggest an aging biomarker?'
GROUP BY answer;
```

### Get papers with full text vs abstract
```sql
SELECT used_full_text, COUNT(*) as count
FROM paper_metadata
GROUP BY used_full_text;
```

### Get processed text for a paper
```sql
SELECT processed_text
FROM processed_texts
WHERE doi = '10.1234/example';
```

## Troubleshooting

### "No papers to process"
- Check that evaluations.db contains validated papers
- Check that papers.db contains corresponding papers
- Verify database paths are correct

### "Rate limit encountered"
- Reduce processing speed by adding delays
- Use multiple API keys (comma-separated in OPENAI_API_KEY)
- Increase TEMPERATURE or reduce MAX_TOKENS

### "Invalid answer format"
- Check questions JSON file format
- Verify LLM is returning valid JSON
- Check answer validation logic in code

## Future Enhancements

- [ ] Parallel processing with async/await
- [ ] Multi-model support (GPT-4, Claude, etc.)
- [ ] Confidence scores for answers
- [ ] Answer explanations/reasoning
- [ ] Batch processing optimization
- [ ] Web interface for results exploration
