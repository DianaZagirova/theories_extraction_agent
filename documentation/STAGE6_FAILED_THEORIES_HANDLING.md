# Stage 6: Failed Theories Handling Strategy

## Overview

The system now implements an intelligent carry-forward and reclustering strategy for theories that fail to be properly assigned.

## Strategy

### 1. Carry Forward to Next Batch

**When:** A theory fails in batch N (small subcluster, not assigned, etc.) and N is NOT the last batch

**Action:** Carry the failed theory forward to batch N+1

**Example:**
```
Batch 1: 26 theories
- Creates 2 valid subclusters (20 theories)
- 6 theories in small subclusters → Carry forward

Batch 2: 26 theories + 6 carried forward = 32 theories total
- Process all 32 together
- LLM has more context to properly assign the 6 problematic theories
```

**Benefits:**
- ✅ Gives theories another chance with fresh context
- ✅ Larger batch size may help LLM create better groupings
- ✅ Avoids premature singleton warnings

### 2. Reclustering with Previous Batches

**When:** Last batch fails completely (returns None after all retries)

**Action:** 
1. Select 3 random subclusters from previous successful batches
2. Combine their theories with the failed theories
3. Recluster all together (with 2 retry attempts)
4. If successful, remove the 3 selected subclusters and replace with new clustering

**Example:**
```
Batch 43 (last): 29 theories - FAILED

Previous batches created:
- Subcluster A: 45 theories
- Subcluster B: 38 theories  
- Subcluster C: 52 theories
- Subcluster D: 41 theories
- ... (many more)

Reclustering:
1. Randomly select 3: [Subcluster B, Subcluster D, Subcluster F]
2. Combine: 29 (failed) + 38 + 41 + 47 = 155 theories
3. Recluster all 155 theories
4. If successful:
   - Remove Subclusters B, D, F from results
   - Add new reclustered subclusters
5. If failed:
   - Assign 29 theories to original name with singleton warning
```

**Benefits:**
- ✅ Gives failed theories one more chance
- ✅ May discover better groupings by mixing with different theories
- ✅ Avoids losing work from previous batches
- ✅ Only happens for last batch (doesn't slow down normal processing)

### 3. Final Fallback: Singleton Warning

**When:** 
- Last batch fails AND reclustering fails
- OR batch completely fails (JSON error, no valid subclusters, etc.)

**Action:** Assign all failed theories to original cluster name with `singleton_warning` status

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Batch N Processing                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ LLM Separation  │
                    └─────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────┐          ┌──────────────────┐
        │   SUCCESS    │          │      FAILED      │
        │              │          │                  │
        │ Valid        │          │ - JSON error     │
        │ subclusters  │          │ - All too small  │
        │ created      │          │ - Validation err │
        └──────────────┘          └──────────────────┘
                │                           │
                ▼                           ▼
    ┌───────────────────────┐    ┌──────────────────────┐
    │ Check for singleton   │    │ Is this last batch?  │
    │ warnings (too small)  │    └──────────────────────┘
    └───────────────────────┘              │
                │                  ┌────────┴────────┐
                │                  │                 │
                ▼                  ▼                 ▼
    ┌───────────────────────┐  ┌─────┐        ┌──────────┐
    │ Any singleton         │  │ NO  │        │   YES    │
    │ warnings?             │  └─────┘        └──────────┘
    └───────────────────────┘     │                │
                │                  │                │
        ┌───────┴───────┐          │                ▼
        │               │          │     ┌──────────────────────┐
        ▼               ▼          │     │ Try Reclustering     │
    ┌─────┐    ┌──────────────┐   │     │ with 3 random        │
    │ NO  │    │     YES      │   │     │ previous subclusters │
    └─────┘    └──────────────┘   │     └──────────────────────┘
        │               │          │                │
        │               ▼          │         ┌──────┴──────┐
        │    ┌──────────────────┐ │         │             │
        │    │ Is last batch?   │ │         ▼             ▼
        │    └──────────────────┘ │    ┌─────────┐  ┌──────────┐
        │               │          │    │SUCCESS  │  │  FAILED  │
        │        ┌──────┴──────┐   │    └─────────┘  └──────────┘
        │        │             │   │         │             │
        │        ▼             ▼   │         │             │
        │    ┌─────┐    ┌──────┐  │         ▼             ▼
        │    │ NO  │    │ YES  │  │    ┌─────────────────────┐
        │    └─────┘    └──────┘  │    │ Save reclustered    │
        │        │           │     │    │ results             │
        │        │           │     │    └─────────────────────┘
        │        ▼           ▼     │              │
        │  ┌──────────┐  ┌──────┐ │              │
        │  │ Carry    │  │Keep  │ │              │
        │  │ forward  │  │as    │ │              │
        │  │ to next  │  │singl-│ │              │
        │  │ batch    │  │eton  │ │              │
        │  └──────────┘  └──────┘ │              │
        │        │           │     │              │
        └────────┴───────────┴─────┴──────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Continue to     │
                    │ next batch      │
                    └─────────────────┘
```

## Example Scenarios

### Scenario 1: Small Subcluster in Middle Batch

```
Batch 5/43: 26 theories
Result:
- Subcluster A: 18 theories ✅
- Subcluster B: 6 theories ✅
- Subcluster C: 2 theories ❌ (too small)

Action:
→ Save Subclusters A & B
→ Carry 2 theories forward to Batch 6

Batch 6/43: 26 + 2 = 28 theories
- Process all 28 together
- LLM can now properly assign the 2 theories
```

### Scenario 2: Complete Batch Failure in Middle

```
Batch 12/43: 26 theories
Result: FAILED (JSON error)

Action:
→ Carry all 26 theories forward to Batch 13

Batch 13/43: 26 + 26 = 52 theories
- Process all 52 together
- Larger context may help
```

### Scenario 3: Last Batch Failure with Reclustering

```
Batch 43/43 (last): 29 theories
Result: FAILED

Previous successful batches created 85 subclusters

Reclustering:
1. Select 3 random: [Subcluster #23, #47, #61]
   - #23: 38 theories
   - #47: 41 theories  
   - #61: 47 theories

2. Recluster: 29 + 38 + 41 + 47 = 155 theories

3. Result: SUCCESS
   - New Subcluster X: 52 theories
   - New Subcluster Y: 61 theories
   - New Subcluster Z: 42 theories

4. Final action:
   - Remove Subclusters #23, #47, #61
   - Add Subclusters X, Y, Z
   - Net result: 85 - 3 + 3 = 85 subclusters (but better quality)
```

### Scenario 4: Everything Fails

```
Batch 43/43 (last): 29 theories
Result: FAILED

Reclustering: FAILED (after 2 attempts)

Final action:
→ Assign all 29 theories to original cluster name
→ Mark with singleton_warning
→ warning_reason: "Failed to separate after 2 retries and reclustering"
```

## Statistics Tracked

- `batches_with_singleton_warning`: Count of batches that ended with singleton warnings
- `theories_with_singleton_warning`: Total theories marked as singletons
- `theories_carried_forward`: (could add) Count of theories carried to next batch
- `reclustering_attempts`: (could add) Count of reclustering attempts
- `reclustering_successes`: (could add) Count of successful reclusterings

## Configuration

```python
separator = Stage6ClusterSeparator(
    paper_threshold=40,
    max_theories_per_batch=26,  # Affects carry-forward batch size
    min_subcluster_size=2,      # Threshold for "too small"
    max_retries=2               # Retries per batch before carry-forward/reclustering
)
```

## Benefits

1. **Maximizes Success Rate** - Multiple chances for theories to be properly assigned
2. **Minimizes Data Loss** - Very few theories end up as singletons
3. **Intelligent Recovery** - Reclustering with different context may find better groupings
4. **No Manual Intervention** - Fully automatic handling of edge cases
5. **Preserves Quality** - Only marks as singleton when truly unable to cluster

## Limitations

- Reclustering only happens for last batch (performance trade-off)
- Carried-forward theories increase batch size (may exceed max_theories_per_batch)
- Random selection of 3 subclusters for reclustering (not optimized)
