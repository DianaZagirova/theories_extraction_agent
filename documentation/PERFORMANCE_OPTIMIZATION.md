# Performance Optimization Guide

## Current Performance

**Sequential Processing:**
- 1 paper at a time
- ~30 seconds per paper (including rate limiting)
- **Total time: ~150 hours (6-7 days)**

## Bottlenecks

1. **Sequential Processing** - Only 1 API call at a time
2. **Rate Limiter** - 180K tokens/minute, 450 requests/minute
3. **Network Latency** - Each API call has ~1-2 second overhead

## Optimization Strategy

### 1. Parallel Processing (Biggest Impact)

**Current:** Sequential (1 paper at a time)
**Optimized:** Process 10-20 papers concurrently

**Benefits:**
- Utilize full rate limit capacity
- Hide network latency
- **10-20x speedup potential**

**Implementation:**
- Use `asyncio.gather()` to process multiple papers concurrently
- Respect rate limits with semaphore
- Batch size: 10-20 papers

**Expected time:** ~8-15 hours (instead of 150 hours)

### 2. Batch API Calls (Not Applicable)

Azure OpenAI doesn't support batch API calls, so we can't combine multiple papers into one request.

### 3. Optimize Rate Limiter

**Current bottleneck:**
- 180K tokens/minute = ~12 papers/minute (for full text papers)
- 450 requests/minute = 450 papers/minute (not a bottleneck)

**Token limit is the constraint**, not request limit.

### 4. Reduce Token Usage

**Options:**
- Enable synonym filtering (skip irrelevant questions)
- Reduce preprocessing budget (less text per paper)
- Use shorter prompts

**Trade-off:** May reduce accuracy

## Recommended Optimizations

### Option 1: Parallel Processing (Recommended)

**Changes needed:**
1. Convert sequential loop to `asyncio.gather()` with batching
2. Add semaphore to limit concurrent requests
3. Ensure thread-safe database writes

**Expected speedup:** 10-15x
**New runtime:** ~10-15 hours
**Risk:** Low (well-tested pattern)

### Option 2: Parallel + Synonym Filtering

**Changes needed:**
1. Parallel processing (as above)
2. Enable synonym filtering with keywords

**Expected speedup:** 12-18x
**New runtime:** ~8-12 hours
**Risk:** Low-Medium (may skip some relevant questions)

### Option 3: Parallel + Reduced Text Budget

**Changes needed:**
1. Parallel processing
2. Reduce `BACK_LIMIT_QUESTIONS` from 50K to 30K chars

**Expected speedup:** 15-20x
**New runtime:** ~7-10 hours
**Risk:** Medium (may lose important context)

## Implementation Plan

### Quick Win: Add Concurrency

Modify the processing loop to use `asyncio.gather()`:

```python
# Instead of:
for row in meta_rows:
    result = process_paper(row)

# Use:
async def process_batch(batch):
    tasks = [process_paper(row) for row in batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

# Process in batches of 10-20
batch_size = 15
for i in range(0, len(meta_rows), batch_size):
    batch = meta_rows[i:i+batch_size]
    results = await process_batch(batch)
```

### Rate Limit Management

Add a semaphore to control concurrency:

```python
self.semaphore = asyncio.Semaphore(15)  # Max 15 concurrent

async def process_paper_with_limit(self, paper):
    async with self.semaphore:
        return await self.process_paper(paper)
```

## Performance Comparison

| Approach | Concurrency | Runtime | Speedup | Risk |
|----------|-------------|---------|---------|------|
| **Current** | 1 | 150 hours | 1x | None |
| **Parallel (10)** | 10 | 15 hours | 10x | Low |
| **Parallel (15)** | 15 | 10 hours | 15x | Low |
| **Parallel (20)** | 20 | 8 hours | 19x | Medium |
| **Parallel + Filtering** | 15 | 8 hours | 19x | Medium |

## Monitoring

After implementing parallel processing, monitor:

1. **Rate limit hits** - Should be minimal with proper semaphore
2. **Error rate** - Should remain low (<1%)
3. **Token usage rate** - Should approach 180K/minute
4. **Database locks** - Use WAL mode (already enabled)

## Testing

Before full run:

```bash
# Test with 100 papers
python scripts/answer_questions_per_paper.py --reset-db --limit 100

# Check:
# - Time taken
# - Error rate
# - Cost accuracy
```

Expected results with parallel processing:
- 100 papers in ~40-60 minutes (vs 50 hours sequential)
- Cost: ~$0.15-0.18

## Risks & Mitigation

### Risk 1: Rate Limit Violations
**Mitigation:** 
- Use semaphore to limit concurrency
- Keep rate limiter logic
- Start with lower concurrency (10) and increase

### Risk 2: Database Contention
**Mitigation:**
- Already using WAL mode
- Batch database writes
- Use connection pooling

### Risk 3: Memory Usage
**Mitigation:**
- Process in batches (not all at once)
- Clear completed results from memory
- Monitor memory usage

### Risk 4: Error Handling
**Mitigation:**
- Use `return_exceptions=True` in gather()
- Log all errors
- Retry failed papers

## Next Steps

1. **Implement parallel processing** (highest impact)
2. **Test with 100 papers**
3. **Monitor performance metrics**
4. **Adjust concurrency based on results**
5. **Run full pipeline**

## Code Changes Required

Estimated effort: 2-3 hours
Files to modify:
- `scripts/answer_questions_per_paper.py` - Add parallel processing
- Rate limiter - Make thread-safe (already is)
- Database writes - Ensure thread-safety (already using locks)
