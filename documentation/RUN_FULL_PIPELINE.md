# Running QA Pipeline on All 18,000 Papers

## Pre-Flight Checklist

### 1. **Reset the Database**
Delete the existing database to start fresh:

```bash
rm -f ./qa_results/qa_results.db
```

Or use the `--reset-db` flag (recommended):

```bash
python scripts/answer_questions_per_paper.py --reset-db
```

### 2. **Verify Configuration Files**

Check that all required files exist:

```bash
# Questions configuration
ls -lh data/questions_part2.json

# Synonyms configuration (for filtering)
ls -lh data/questions_synonyms.json

# Database paths (verify these exist)
ls -lh /home/diana.z/hack/llm_judge/data/evaluations.db
ls -lh /home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db
```

### 3. **Check Disk Space**

The database will grow significantly with 18,000 papers:

```bash
df -h .
```

Estimated space needed:
- **Database**: ~2-5 GB (depends on text storage)
- **Checkpoint files**: ~500 MB - 1 GB
- **Total**: Plan for at least **10 GB free space**

### 4. **Verify Environment Variables**

Current settings (from the script):
- `BACK_LIMIT_QUESTIONS=50000` (max chars sent to LLM)
- `MAX_TOKENS=2000` (LLM response limit)
- `PREPROCESS_BUDGET_CHARS_QUESTIONS=50000` (preprocessing budget)

These are already set in the code, but you can override via environment:

```bash
export BACK_LIMIT_QUESTIONS=50000
export MAX_TOKENS=2000
export PREPROCESS_BUDGET_CHARS_QUESTIONS=50000
```

## Running the Full Pipeline

### Option 1: Process All Validated Papers (Recommended)

This will process all papers marked as valid/doubted in evaluations.db:

```bash
python scripts/answer_questions_per_paper.py \
    --reset-db \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_results_full.json
```

### Option 2: Test Run First (Recommended)

Test with a small batch first to verify everything works:

```bash
# Test with 10 papers
python scripts/answer_questions_per_paper.py \
    --reset-db \
    --limit 10 \
    --results-db ./qa_results/qa_results_test.db \
    --output-file qa_results_test.json
```

If successful, run the full pipeline:

```bash
# Full run
python scripts/answer_questions_per_paper.py \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_results_full.json
```

### Option 3: Resume from Interruption

If the pipeline is interrupted, it will automatically resume:

```bash
# Just run again - it will skip already processed papers
python scripts/answer_questions_per_paper.py \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_results_full.json
```

The `--resume-from-db` flag is automatically enabled by default.

## Monitoring Progress

### Real-time Progress

The pipeline shows:
- Progress bar with papers processed
- Questions skipped per paper (synonym filtering)
- Rate limiting status
- Token usage and cost estimates

Example output:
```
Processing papers: 45%|████████████▌              | 8100/18000 [2:15:30<1:48:20, 1.52it/s]
  ⚡ Skipped 3 questions, asking 6 questions
```

### Check Database Status

While running, you can check progress:

```bash
sqlite3 ./qa_results/qa_results.db "SELECT COUNT(*) FROM paper_metadata;"
sqlite3 ./qa_results/qa_results.db "SELECT COUNT(DISTINCT doi) FROM paper_answers;"
```

### Checkpoint Files

The pipeline saves checkpoints every 100 papers:
- `qa_results_full_checkpoint.json` - Latest checkpoint

## Estimated Runtime

**Assumptions:**
- 18,000 papers
- ~7 questions per paper (after synonym filtering)
- ~30 seconds per paper (including rate limiting)

**Total time: ~150 hours (6-7 days)**

Factors affecting speed:
- ✅ Synonym filtering reduces questions
- ✅ Papers without full text are faster (abstract only)
- ⚠️ Rate limiting (180K tokens/min, 450 requests/min)
- ⚠️ Network latency

## Cost Estimation

**Per paper (estimated):**
- Input: ~15K tokens (50K chars ≈ 12.5K tokens + prompt overhead)
- Output: ~500 tokens (7 questions × ~70 tokens each)
- Cost: ~$0.009 per paper

**Total for 18,000 papers:**
- Input tokens: ~270M tokens
- Output tokens: ~9M tokens
- **Estimated cost: ~$162**

Actual cost may be lower due to:
- Synonym filtering (fewer questions)
- Abstract-only papers (less text)
- Cached processed texts

## System Requirements

### Memory
- **Minimum**: 4 GB RAM
- **Recommended**: 8 GB RAM
- The pipeline pre-loads papers into memory for speed

### CPU
- Multi-threaded processing
- Rate limiter handles concurrency

### Network
- Stable internet connection required
- Azure OpenAI API access

## Troubleshooting

### Rate Limiting
If you see frequent rate limit warnings:
```
⚠️  Rate limit encountered. Retrying in 5 seconds...
```

The pipeline handles this automatically with exponential backoff.

### Out of Memory
If the system runs out of memory:
1. Reduce batch size (not configurable in current version)
2. Process in smaller chunks using `--limit`

### Database Locked
If you see "database is locked" errors:
- The pipeline uses WAL mode to prevent this
- If it persists, check for other processes accessing the DB

### Interrupted Run
Simply restart the command - it will resume automatically:
```bash
python scripts/answer_questions_per_paper.py \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_results_full.json
```

## Post-Processing

After completion, you can:

### 1. Export Results
```bash
python scripts/export_qa_results.py \
    --results-db ./qa_results/qa_results.db \
    --validation-file data/qa_validation_set_extended.json \
    --output-file qa_exported_results_all.json
```

### 2. Evaluate Accuracy
```bash
python scripts/evaluate_qa_results.py \
    --validation-file data/qa_validation_set_extended.json \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_evaluation_results_all.json
```

### 3. Check Statistics
```bash
sqlite3 ./qa_results/qa_results.db << EOF
SELECT 
    COUNT(*) as total_papers,
    SUM(used_full_text) as with_full_text,
    COUNT(*) - SUM(used_full_text) as abstract_only
FROM paper_metadata;

SELECT 
    question_name,
    answer,
    COUNT(*) as count
FROM paper_answers
GROUP BY question_name, answer
ORDER BY question_name, count DESC;
EOF
```

## Final Checklist

Before starting the full run:

- [ ] Database reset or clean slate confirmed
- [ ] Disk space verified (>10 GB free)
- [ ] Test run completed successfully
- [ ] Questions and synonyms configured correctly
- [ ] Network connection stable
- [ ] Budget approved (~$162 estimated cost)
- [ ] Monitoring plan in place
- [ ] Ready to let it run for ~6-7 days

## Command to Run

```bash
# Full production run
python scripts/answer_questions_per_paper.py \
    --reset-db \
    --results-db ./qa_results/qa_results.db \
    --output-file qa_results_full.json \
    --evaluations-db /home/diana.z/hack/llm_judge/data/evaluations.db \
    --papers-db /home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db \
    --questions-file data/questions_part2.json
```

Or use defaults (recommended):
```bash
python scripts/answer_questions_per_paper.py --reset-db
```
