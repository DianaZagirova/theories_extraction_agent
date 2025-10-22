# Updating Stage 2 to Work with Stage 1.5

## Changes Needed

Stage 2 currently processes all unmatched theories from Stage 1. With Stage 1.5, we need to update it to process only the theories that Stage 1.5 couldn't map.

## Current Flow

```
Stage 1 → unmatched_theories (6,206)
          ↓
Stage 2 → Process all 6,206
```

## New Flow

```
Stage 1 → unmatched_theories (6,206)
          ↓
Stage 1.5 → mapped (2,100)
            novel (2,800)
            unmatched (300)
            invalid (1,000)
          ↓
Stage 2 → Process only novel + unmatched (3,100)
```

## Implementation

### Option 1: Update Stage 2 Input (Recommended)

Modify Stage 2 to accept Stage 1.5 output:

```python
# src/normalization/stage2_llm_extraction.py

def process_unmatched_theories(self, 
                               input_path: str,
                               output_path: str,
                               max_theories: Optional[int] = None,
                               use_stage1_5: bool = True):  # NEW parameter
    """Process unmatched theories."""
    
    # Load input
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Check if input is from Stage 1.5
    if use_stage1_5 and 'novel_theories' in data:
        # Stage 1.5 output
        print("📂 Loading from Stage 1.5 output...")
        theories_to_process = data.get('novel_theories', []) + data.get('still_unmatched', [])
        print(f"✓ Found {len(theories_to_process)} theories to process")
        print(f"  Novel: {len(data.get('novel_theories', []))}")
        print(f"  Unmatched: {len(data.get('still_unmatched', []))}")
    else:
        # Stage 1 output (original behavior)
        print("📂 Loading from Stage 1 output...")
        theories_to_process = data.get('unmatched_theories', [])
        print(f"✓ Found {len(theories_to_process)} unmatched theories")
    
    # Rest of the function remains the same...
```

### Option 2: Create Merged Input

Create a script to merge Stage 1 and Stage 1.5 outputs:

```python
# merge_stage1_and_1_5.py

import json

def merge_outputs(stage1_path, stage1_5_path, output_path):
    """Merge Stage 1 matched with Stage 1.5 mapped theories."""
    
    # Load Stage 1
    with open(stage1_path, 'r') as f:
        stage1 = json.load(f)
    
    # Load Stage 1.5
    with open(stage1_5_path, 'r') as f:
        stage1_5 = json.load(f)
    
    # Combine matched theories
    all_matched = stage1['matched_theories'] + stage1_5['mapped_theories']
    
    # Theories for Stage 2
    for_stage2 = stage1_5['novel_theories'] + stage1_5['still_unmatched']
    
    # Save merged output
    merged = {
        'metadata': {
            'stage1_matched': len(stage1['matched_theories']),
            'stage1_5_mapped': len(stage1_5['mapped_theories']),
            'total_matched': len(all_matched),
            'for_stage2': len(for_stage2)
        },
        'matched_theories': all_matched,
        'unmatched_theories': for_stage2
    }
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✓ Merged output saved to {output_path}")
    print(f"  Total matched: {len(all_matched)}")
    print(f"  For Stage 2: {len(for_stage2)}")

if __name__ == '__main__':
    merge_outputs(
        'output/stage1_fuzzy_matched.json',
        'output/stage1_5_llm_mapped.json',
        'output/stage1_plus_1_5_merged.json'
    )
```

## Updated Pipeline Commands

### Full Pipeline with Stage 1.5

```bash
# Step 1: Stage 0 + Stage 1
python run_stage1_on_real_data.py

# Step 2: Stage 1.5 (NEW!)
python -m src.normalization.stage1_5_llm_mapping \
    --input output/stage1_fuzzy_matched.json \
    --output output/stage1_5_llm_mapped.json \
    --batch-size 30

# Step 3: Stage 2 (updated to use Stage 1.5 output)
python -m src.normalization.stage2_llm_extraction \
    --input output/stage1_5_llm_mapped.json \
    --output output/stage2_llm_extracted.json \
    --use-stage1-5

# Step 4: Merge all matched theories
python merge_stage1_and_1_5.py

# Step 5: Stage 3
python -m src.normalization.stage3_theory_grouping \
    --stage1 output/stage1_plus_1_5_merged.json \
    --stage2 output/stage2_llm_extracted.json \
    --output output/stage3_theory_groups.json
```

### Alternative: Direct Pipeline

```bash
# All in one script
python run_complete_pipeline.py
```

## Create Complete Pipeline Script

```python
# run_complete_pipeline.py

"""
Run complete theory normalization pipeline with Stage 1.5.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    
    print(f"✅ {description} complete")

def main():
    print("🚀 Starting Complete Theory Normalization Pipeline")
    print("="*80)
    
    # Stage 0 + 1
    run_command(
        "python run_stage1_on_real_data.py",
        "Stage 0-1: Quality Filter + Fuzzy Matching"
    )
    
    # Stage 1.5
    run_command(
        "python -m src.normalization.stage1_5_llm_mapping "
        "--input output/stage1_fuzzy_matched.json "
        "--output output/stage1_5_llm_mapped.json "
        "--batch-size 30",
        "Stage 1.5: LLM Mapping to Canonical Theories"
    )
    
    # Stage 2
    run_command(
        "python -m src.normalization.stage2_llm_extraction "
        "--input output/stage1_5_llm_mapped.json "
        "--output output/stage2_llm_extracted.json",
        "Stage 2: LLM Extraction for Novel Theories"
    )
    
    # Merge
    run_command(
        "python merge_stage1_and_1_5.py",
        "Merging Stage 1 and Stage 1.5 Results"
    )
    
    # Stage 3
    run_command(
        "python -m src.normalization.stage3_theory_grouping "
        "--stage1 output/stage1_plus_1_5_merged.json "
        "--stage2 output/stage2_llm_extracted.json "
        "--output output/stage3_theory_groups.json",
        "Stage 3: Theory Grouping"
    )
    
    print("\n" + "="*80)
    print("🎉 COMPLETE PIPELINE FINISHED!")
    print("="*80)
    print("\nOutput files:")
    print("  - output/stage1_fuzzy_matched.json")
    print("  - output/stage1_5_llm_mapped.json")
    print("  - output/stage2_llm_extracted.json")
    print("  - output/stage1_plus_1_5_merged.json")
    print("  - output/stage3_theory_groups.json")

if __name__ == '__main__':
    main()
```

## Testing

### Test Stage 1.5 Integration

```bash
# 1. Run Stage 1
python run_stage1_on_real_data.py

# 2. Test Stage 1.5 with 50 theories
python test_stage1_5_mapping.py

# 3. Check output
python3 << 'EOF'
import json

with open('output/stage1_5_llm_mapped_TEST.json', 'r') as f:
    data = json.load(f)

print("Stage 1.5 Test Results:")
print(f"  Mapped: {len(data['mapped_theories'])}")
print(f"  Novel: {len(data['novel_theories'])}")
print(f"  Unmatched: {len(data['still_unmatched'])}")
print(f"  Invalid: {len(data['invalid_theories'])}")
print(f"\nFor Stage 2: {len(data['novel_theories']) + len(data['still_unmatched'])}")
EOF
```

## Summary

### Changes Required

1. **Update Stage 2** to accept Stage 1.5 output (Option 1)
   - OR create merge script (Option 2)

2. **Create complete pipeline script**
   - Runs all stages in sequence
   - Handles errors
   - Shows progress

3. **Update documentation**
   - Pipeline flow diagrams
   - Command examples
   - Cost estimates

### Benefits

- ✅ Seamless integration
- ✅ Backward compatible (can still use Stage 1 output directly)
- ✅ Clear separation of concerns
- ✅ Easy to test each stage

### Next Steps

1. Choose Option 1 or 2
2. Implement changes
3. Test with small sample
4. Run full pipeline
5. Validate results
