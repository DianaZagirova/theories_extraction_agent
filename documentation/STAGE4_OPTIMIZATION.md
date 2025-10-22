# Stage 4 Theory Validation - Performance Optimization

## Overview
Optimized `src/normalization/stage4_theory_validation.py` to process theories significantly faster using async/await patterns and concurrent processing, following the proven approach from `scripts/answer_questions_per_paper.py`.

## Key Improvements

### 1. **Async/Await with Controlled Concurrency**
- Added `asyncio` support for parallel API calls
- Implemented semaphore-based concurrency control (default: 10 concurrent requests)
- Processes multiple batches simultaneously instead of sequentially

### 2. **Rate Limiting**
- Added `RateLimiter` class to respect API limits:
  - 180K tokens per minute (TPM)
  - 450 requests per minute (RPM)
  - 10% buffer to prevent hitting limits
- Automatic throttling when approaching limits

### 3. **Batch Processing with asyncio.gather()**
- Processes 50 batches at a time concurrently
- Uses `asyncio.gather()` for parallel execution
- Maintains checkpoint saves after each group

### 4. **Thread-Safe Statistics**
- Added threading locks for token/cost tracking
- Prevents race conditions in concurrent execution
- Accurate statistics across parallel operations

### 5. **Non-Blocking LLM Calls**
- Uses `loop.run_in_executor()` to avoid blocking event loop
- Allows true parallelism for API calls
- Maintains responsiveness during processing

## Usage

### Basic Usage (10 concurrent requests - default)
```bash
python src/normalization/stage4_theory_validation.py
```

### Custom Concurrency
```bash
# More aggressive (faster but higher API load)
python src/normalization/stage4_theory_validation.py --max-concurrent 20

# Conservative (slower but safer)
python src/normalization/stage4_theory_validation.py --max-concurrent 5
```

### Resume from Checkpoint
```bash
python src/normalization/stage4_theory_validation.py --resume --max-concurrent 10
```

## Expected Performance Improvement

### Before Optimization
- Sequential processing: 1 batch at a time
- ~1-2 seconds per batch (including API latency)
- For 100 batches: ~2-3 minutes minimum

### After Optimization
- Parallel processing: 10 batches simultaneously
- Same API latency but overlapped
- For 100 batches: ~20-30 seconds (5-10x faster)

**Estimated speedup: 5-10x depending on:**
- Network latency
- API response times
- Batch sizes
- Concurrency level

## Technical Details

### Architecture Changes

1. **New Methods:**
   - `_process_batch_async()` - Async version of batch processing
   - `_process_batch_with_evidence_async()` - Async version for doubted theories
   - `_process_all_batches_async()` - Orchestrates concurrent batch processing
   - Synchronous wrappers maintained for backward compatibility

2. **New Classes:**
   - `RateLimiter` - Manages API rate limits with token/request tracking

3. **Modified Constructor:**
   - Added `max_concurrent` parameter (default: 10)
   - Initialized rate limiter and threading lock

### Rate Limiting Logic
```python
# Tracks usage in rolling 1-minute window
# Waits if current_tokens + estimated_tokens > 90% of limit
# Waits if current_requests + 1 > 90% of limit
```

### Concurrency Control
```python
# Semaphore limits concurrent operations
# Processes in groups of 50 batches
# Each group runs with max_concurrent parallelism
```

## Configuration

### Environment Variables
No new environment variables required. Uses existing:
- `USE_MODULE_FILTERING_LLM_STAGE4` - LLM provider selection

### Command Line Arguments
- `--resume` - Resume from checkpoint
- `--max-concurrent N` - Set concurrency level (default: 10)

## Monitoring

The script now shows:
```
🚀 Parallel processing enabled: 10 concurrent requests
⚡ Rate limits: 180K tokens/min, 450 requests/min
🔄 Processing 100 batches with 10 concurrent requests...
```

Progress bar updates in real-time as batches complete concurrently.

## Safety Features

1. **Rate Limiting** - Prevents API throttling
2. **Exception Handling** - Failed batches marked as "doubted" 
3. **Checkpointing** - Saves progress every 5 batches
4. **Thread Safety** - Locks protect shared statistics
5. **Graceful Degradation** - Falls back on errors

## Compatibility

- ✅ Maintains same input/output format
- ✅ Same checkpoint/resume functionality
- ✅ Same validation logic
- ✅ Backward compatible with existing code

## Notes

- **Do not increase batch size** - As requested, batch size remains unchanged
- **Do not cut prompts** - Prompts remain identical
- **Only parallelization changes** - Logic unchanged, just execution model
- Recommended concurrency: 5-10 for stability, up to 20 for maximum speed
