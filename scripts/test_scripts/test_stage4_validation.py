"""
Test Stage 4 Theory Validation

This script tests the Stage 4 validation pipeline on a small sample of theories.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage4_theory_validation import Stage4TheoryValidator


def create_test_sample(stage3_path: str, output_path: str, sample_size: int = 20):
    """
    Create a test sample from Stage 3 output.
    
    Args:
        stage3_path: Path to full Stage 3 output
        output_path: Path to save test sample
        sample_size: Number of unique names to sample
    """
    print(f"Creating test sample with {sample_size} unique names...")
    
    with open(stage3_path, 'r') as f:
        data = json.load(f)
    
    mappings = data.get('mappings', {})
    
    # Get unique names and their counts
    from collections import Counter
    unique_names_counter = Counter(mappings.values())
    
    # Sample diverse unique names (mix of high and low frequency)
    unique_names = list(unique_names_counter.keys())
    
    # Sort by frequency and take a mix
    sorted_names = sorted(unique_names, key=lambda x: unique_names_counter[x], reverse=True)
    
    # Take some high-frequency, some mid-frequency, some low-frequency
    sample_names = []
    sample_names.extend(sorted_names[:sample_size//3])  # High frequency
    sample_names.extend(sorted_names[len(sorted_names)//2:len(sorted_names)//2 + sample_size//3])  # Mid
    sample_names.extend(sorted_names[-sample_size//3:])  # Low frequency
    
    # Filter mappings to only include sampled unique names
    test_mappings = {k: v for k, v in mappings.items() if v in sample_names}
    
    # Create test output
    test_data = {
        'metadata': {
            'stage': 'stage3_iterative_refinement',
            'status': 'complete',
            'note': 'Test sample for Stage 4 validation',
            'sample_size': len(sample_names),
            'total_mappings': len(test_mappings)
        },
        'mappings': test_mappings,
        'ontology_theories': data.get('ontology_theories', [])
    }
    
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✓ Created test sample:")
    print(f"  Unique names: {len(sample_names)}")
    print(f"  Total mappings: {len(test_mappings)}")
    print(f"  Saved to: {output_path}")
    
    # Show sample statistics
    print(f"\n📊 Sample statistics:")
    for name in sample_names[:5]:
        count = unique_names_counter[name]
        print(f"  - {name}: {count} instances")
    print(f"  ...")


def main():
    """Main test function."""
    print("=" * 80)
    print("TESTING STAGE 4: THEORY VALIDATION")
    print("=" * 80)
    
    # Create test sample
    test_stage3_path = 'output/stage3_refined_theories_test.json'
    
    # Check if test sample exists, if not create it
    if not Path(test_stage3_path).exists():
        print("\n📝 Creating test sample from Stage 3 output...")
        create_test_sample(
            stage3_path='output/stage3_refined_theories.json',
            output_path=test_stage3_path,
            sample_size=20
        )
    else:
        print(f"\n✓ Using existing test sample: {test_stage3_path}")
    
    # Initialize validator with test data
    print("\n" + "=" * 80)
    print("RUNNING STAGE 4 VALIDATION ON TEST SAMPLE")
    print("=" * 80)
    
    validator = Stage4TheoryValidator(
        stage3_path=test_stage3_path,
        stage0_path='output/stage0_filtered_theories.json',
        tracker_path='output/theory_tracking_report.json',
        ontology_path='ontology/group_ontology_mechanisms.json'
    )
    
    # Run validation
    validator.run(
        output_path='output/stage4_validated_theories_test.json',
        batch_size=5,
        handle_doubted=True,
        resume_from_checkpoint=False  # Set to True to resume from checkpoint
    )
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYZING TEST RESULTS")
    print("=" * 80)
    
    with open('output/stage4_validated_theories_test.json', 'r') as f:
        results = json.load(f)
    
    validations = results.get('validations', [])
    
    print(f"\n📊 Validation Results:")
    print(f"  Total validated: {len(validations)}")
    
    # Show some examples
    print(f"\n✅ Valid theories (examples):")
    valid_theories = [v for v in validations if v['is_valid_theory'] == True]
    for v in valid_theories[:3]:
        print(f"  - {v['original_name']}")
        print(f"    Reasoning: {v['validation_reasoning'][:100]}...")
        if v['is_listed']:
            print(f"    Mapped to: {v['canonical_name']} (confidence: {v['mapping_confidence']})")
        if v['is_novel']:
            print(f"    Novel theory: {v['proposed_name']}")
    
    print(f"\n❌ Invalid theories (examples):")
    invalid_theories = [v for v in validations if v['is_valid_theory'] == False]
    for v in invalid_theories[:3]:
        print(f"  - {v['original_name']}")
        print(f"    Reasoning: {v['validation_reasoning'][:100]}...")
    
    print(f"\n⚠️  Doubted theories:")
    doubted_theories = [v for v in validations if v['is_valid_theory'] == 'doubted']
    for v in doubted_theories:
        print(f"  - {v['original_name']}")
        print(f"    Reasoning: {v['validation_reasoning'][:100]}...")
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    main()
