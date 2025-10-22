# Performance Optimization Analysis for extract_theories_per_paper.py

## Current Performance Bottlenecks

### 1. **Database Access Pattern (CRITICAL - 🔴 High Impact)**
**Location:** Lines 582-599 in `_load_and_process()`

**Problem:**
- Opens a new SQLite connection for **every single paper**
- Each paper requires: connect → query → fetch → close
- With 1000s of papers, this creates massive overhead

**Impact:** ~50-100ms per paper just for DB operations

**Solution:**
```python
# BEFORE (current - slow):
def _load_and_process(row):
    conn = sqlite3.connect(papers_db)  # NEW CONNECTION EACH TIME
    cur = conn.cursor()
    cur.execute("SELECT ... WHERE doi = ?", (doi,))
    paper_data = cur.fetchone()
    conn.close()  # CLOSE EACH TIME
    
# AFTER (optimized - fast):
# Pre-load ALL paper data into memory ONCE before processing
papers_cache = {}
conn = sqlite3.connect(papers_db)
cur = conn.cursor()
cur.execute("SELECT doi, pmid, title, abstract, full_text, full_text_sections FROM papers WHERE doi IN (...)")
for row in cur.fetchall():
    papers_cache[row[0]] = row
conn.close()

def _load_and_process(row):
    doi = row[0]
    paper_data = papers_cache.get(doi)  # INSTANT LOOKUP
```

**Expected Speedup:** 5-10x faster for DB access

---

### 2. **Text Preprocessing Redundancy (🟡 Medium Impact)**
**Location:** Line 614 - `self.preprocessor.preprocess()`

**Problem:**
- JSON parsing of `full_text_sections` happens for every paper
- Unicode normalization, regex operations on large texts
- Reference removal with multiple pattern matching
- All done synchronously during LLM call

**Impact:** ~100-500ms per paper depending on text size

**Solution:**
```python
# Option A: Pre-process during data loading phase
# Do all preprocessing BEFORE entering the processing loop

# Option B: Cache preprocessed text in DB
# Add a 'preprocessed_text' column to avoid reprocessing
```

**Expected Speedup:** 2-3x faster preprocessing

---

### 3. **LLM API Call Efficiency (🟡 Medium Impact)**
**Location:** Lines 355-376 - `extract_theories_stage()`

**Problem:**
- Sequential retry logic with exponential backoff
- No batching of requests
- Rate limit handling is reactive, not proactive
- Multi-key rotation exists but could be optimized

**Current Flow:**
```
Paper 1 → LLM call (2-5s) → Process result
Paper 2 → LLM call (2-5s) → Process result
Paper 3 → LLM call (2-5s) → Process result
```

**Optimization:**
- Use async/await for concurrent API calls
- Implement request queuing with rate limiting
- Better utilize multi-key rotation

**Expected Speedup:** 2-4x with proper concurrency

---

### 4. **Prompt Size (🟢 Low-Medium Impact)**
**Location:** Lines 302-351 - Prompt construction

**Problem:**
- Sending up to 40K characters per request
- Large prompts = more input tokens = slower processing
- Verbose instructions repeated for every paper

**Current:** ~10K-15K input tokens per paper

**Optimization:**
```python
# Reduce prompt verbosity
# Use system message for static instructions
# Send only essential text (smart truncation)
```

**Expected Speedup:** 10-20% faster, 20-30% cost reduction

---

### 5. **Checkpoint Frequency (🟢 Low Impact)**
**Location:** Lines 652-653, 672-673

**Problem:**
- Saves checkpoint every 300 papers
- File I/O during processing
- Could cause brief pauses

**Solution:**
- Increase to 500-1000 papers
- Use async file writing
- Only checkpoint on errors

---

## Recommended Optimization Strategy

### Phase 1: Quick Wins (1-2 hours implementation)

**A. Pre-load Paper Data**
```python
# Add before processing loop (line ~562)
print("\n📦 Pre-loading paper data into memory...")
papers_cache = {}
conn = sqlite3.connect(papers_db)
cur = conn.cursor()

# Get all DOIs we need
dois_to_fetch = [row[0] for row in meta_rows]
placeholders = ','.join(['?'] * len(dois_to_fetch))

cur.execute(f"""
    SELECT doi, pmid, title, abstract, full_text, full_text_sections
    FROM papers
    WHERE doi IN ({placeholders})
""", dois_to_fetch)

for row in tqdm(cur.fetchall(), desc="Loading papers"):
    papers_cache[row[0]] = row
conn.close()
print(f"✓ Loaded {len(papers_cache)} papers into memory")

# Modify _load_and_process to use cache
def _load_and_process(row):
    doi = row[0]
    paper_data = papers_cache.get(doi)
    if not paper_data:
        raise ValueError("Paper not found in cache")
    # ... rest of processing
```

**Expected Impact:** 5-10x faster data loading
**Risk:** Low - simple change
**Effort:** 30 minutes

---

**B. Optimize Preprocessing**
```python
# Pre-process all papers during cache loading
print("\n⚙️ Pre-processing paper texts...")
preprocessed_cache = {}

for doi, paper_data in tqdm(papers_cache.items(), desc="Preprocessing"):
    _, pmid_db, title_db, abstract, full_text, full_text_sections = paper_data
    processed_text = self.preprocessor.preprocess(full_text, full_text_sections, abstract)
    preprocessed_cache[doi] = processed_text

# Use preprocessed_cache in _load_and_process
```

**Expected Impact:** 2-3x faster preprocessing
**Risk:** Low
**Effort:** 20 minutes

---

**C. Reduce Prompt Size**
```python
# Optimize prompt (line 302-351)
# Move static instructions to system message
# Reduce example verbosity
# Smart text truncation (keep intro + conclusion)

# BEFORE: ~15K input tokens
# AFTER: ~8K input tokens
```

**Expected Impact:** 15-20% faster, 25% cost reduction
**Risk:** Low (test on sample first)
**Effort:** 30 minutes

---

### Phase 2: Advanced Optimizations (2-4 hours)

**D. Async LLM Calls**
```python
import asyncio
from openai import AsyncOpenAI

# Convert to async processing
async def extract_theories_async(paper: Dict):
    # Async LLM call
    response = await async_client.chat.completions.create(...)
    return result

# Process in batches with concurrency limit
async def process_batch(papers, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def bounded_process(paper):
        async with semaphore:
            return await extract_theories_async(paper)
    
    tasks = [bounded_process(p) for p in papers]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Expected Impact:** 3-5x faster with proper concurrency
**Risk:** Medium (requires code restructuring)
**Effort:** 2-3 hours

---

**E. Smart Text Selection**
```python
# Instead of first 40K chars, intelligently select:
# - Abstract (always)
# - Introduction (high priority)
# - Results/Discussion sections (high priority)
# - Skip Methods (low relevance for theory extraction)

def smart_text_selection(full_text, sections, max_chars=25000):
    # Prioritize theory-relevant sections
    # This gives better quality with less tokens
    pass
```

**Expected Impact:** 20-30% faster, better extraction quality
**Risk:** Low-Medium
**Effort:** 1 hour

---

## Performance Comparison

### Current Performance (Estimated)
- **Per paper:** ~5-8 seconds
- **1000 papers:** ~1.5-2.5 hours
- **10,000 papers:** ~15-25 hours

### After Phase 1 Optimizations
- **Per paper:** ~2-3 seconds (60% faster)
- **1000 papers:** ~35-50 minutes
- **10,000 papers:** ~6-8 hours

### After Phase 2 Optimizations
- **Per paper:** ~0.8-1.2 seconds (85% faster)
- **1000 papers:** ~15-20 minutes
- **10,000 papers:** ~2.5-3.5 hours

---

## Implementation Priority

### Immediate (Do Now)
1. ✅ **Pre-load paper data** - Biggest single improvement
2. ✅ **Pre-process texts** - Significant speedup
3. ✅ **Reduce prompt size** - Cost + speed benefit

### Short-term (This Week)
4. 🔄 **Async LLM calls** - Major throughput improvement
5. 🔄 **Smart text selection** - Quality + speed

### Optional (If Needed)
6. ⭕ **Caching layer** - Store preprocessed texts in DB
7. ⭕ **Batch API** - Use OpenAI batch API for non-urgent processing
8. ⭕ **Distributed processing** - Multiple machines

---

## Monitoring & Metrics

Add timing instrumentation:
```python
import time

timings = {
    'db_load': [],
    'preprocessing': [],
    'llm_call': [],
    'total': []
}

# Track each stage
start = time.time()
# ... operation ...
timings['stage'].append(time.time() - start)

# Report at end
print("\n⏱️ Performance Metrics:")
for stage, times in timings.items():
    avg = sum(times) / len(times)
    print(f"  {stage}: {avg:.2f}s avg")
```

---

## Cost Optimization

Current costs are driven by:
1. **Input tokens:** ~10-15K per paper
2. **Output tokens:** ~2-4K per paper
3. **Total:** ~12-19K tokens per paper

**Optimizations:**
- Reduce input to 6-8K tokens → 40% cost reduction
- Use gpt-4o-mini instead of gpt-4 → 90% cost reduction
- Batch processing → 50% cost reduction

**Estimated savings:** 70-85% cost reduction with optimizations

---

## Risk Assessment

| Optimization | Risk | Reversibility | Testing Required |
|-------------|------|---------------|------------------|
| Pre-load data | Low | Easy | Minimal |
| Pre-process | Low | Easy | Minimal |
| Reduce prompt | Low-Med | Easy | Sample validation |
| Async calls | Medium | Moderate | Extensive |
| Smart selection | Medium | Easy | Quality validation |

---

## Next Steps

1. **Implement Phase 1** (Quick wins)
2. **Test on 100 papers** to validate improvements
3. **Measure actual speedup** vs estimates
4. **Decide on Phase 2** based on results
5. **Monitor quality** - ensure optimizations don't hurt extraction quality
