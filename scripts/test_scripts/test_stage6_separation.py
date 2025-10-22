"""
Test script for Stage 6 cluster separation.
Tests the logic on a single cluster before running on all.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage6_cluster_separation import Stage6ClusterSeparator


def test_batch_splitting():
    """Test the smart batch splitting logic."""
    print("="*80)
    print("TEST 1: Smart Batch Splitting")
    print("="*80)
    
    separator = Stage6ClusterSeparator()
    
    test_cases = [
        (25, "Should create 1 batch"),
        (30, "Should create 1 batch"),
        (35, "Should create 2 batches [18, 17] not [30, 5]"),
        (45, "Should create 2 batches [23, 22]"),
        (60, "Should create 2 batches [30, 30]"),
        (65, "Should create 3 batches [22, 22, 21]"),
        (90, "Should create 3 batches [30, 30, 30]"),
        (100, "Should create 4 batches [25, 25, 25, 25]"),
    ]
    
    for total, description in test_cases:
        theory_ids = [f"T{i:06d}" for i in range(total)]
        batches = separator._create_smart_batches(theory_ids)
        batch_sizes = [len(b) for b in batches]
        print(f"\n{total} theories: {batch_sizes}")
        print(f"  {description}")
        
        # Validate
        assert sum(batch_sizes) == total, f"Lost theories! {sum(batch_sizes)} != {total}"
        # Allow smaller batches if it's the only way to split, but prefer larger
        min_acceptable = 10 if len(batches) > 1 else 1
        assert all(size >= min_acceptable for size in batch_sizes), f"Batch too small: {batch_sizes}"
        assert all(size <= 30 for size in batch_sizes), f"Batch too large: {batch_sizes}"
    
    print("\n✅ All batch splitting tests passed!")


def test_validation():
    """Test the validation logic."""
    print("\n" + "="*80)
    print("TEST 2: Validation Logic")
    print("="*80)
    
    separator = Stage6ClusterSeparator()
    
    # Test case 1: Valid separation (in converted format)
    input_ids = [f"T{i:06d}" for i in range(20)]
    valid_result = {
        'subclusters': [
            {
                'subcluster_name': 'Subcluster A',
                'theory_ids': input_ids[:10],
                'theory_count': 10,
                'mechanism_focus': 'Test mechanism A'
            },
            {
                'subcluster_name': 'Subcluster B',
                'theory_ids': input_ids[10:],
                'theory_count': 10,
                'mechanism_focus': 'Test mechanism B'
            }
        ]
    }
    
    is_valid, error = separator._validate_separation(valid_result, input_ids)
    print(f"\nTest 1 - Valid separation: {is_valid}")
    assert is_valid, f"Should be valid but got: {error}"
    
    # Test case 2: Subcluster too small
    invalid_small = {
        'subclusters': [
            {
                'subcluster_name': 'Subcluster A',
                'theory_ids': input_ids[:17],
                'theory_count': 17
            },
            {
                'subcluster_name': 'Subcluster B',
                'theory_ids': input_ids[17:],  # Only 3 theories
                'theory_count': 3
            }
        ]
    }
    
    is_valid, error = separator._validate_separation(invalid_small, input_ids)
    print(f"Test 2 - Subcluster too small: {is_valid} (error: {error})")
    assert not is_valid, "Should be invalid (subcluster too small)"
    
    # Test case 3: Missing theories
    invalid_missing = {
        'subclusters': [
            {
                'subcluster_name': 'Subcluster A',
                'theory_ids': input_ids[:15],  # Only 15 out of 20
                'theory_count': 15
            }
        ]
    }
    
    is_valid, error = separator._validate_separation(invalid_missing, input_ids)
    print(f"Test 3 - Missing theories: {is_valid} (error: {error})")
    assert not is_valid, "Should be invalid (missing theories)"
    
    # Test case 4: Duplicate theories
    invalid_duplicate = {
        'subclusters': [
            {
                'subcluster_name': 'Subcluster A',
                'theory_ids': input_ids[:12] + [input_ids[0]],  # Duplicate T000000
                'theory_count': 13
            },
            {
                'subcluster_name': 'Subcluster B',
                'theory_ids': input_ids[12:],
                'theory_count': 8
            }
        ]
    }
    
    is_valid, error = separator._validate_separation(invalid_duplicate, input_ids)
    print(f"Test 4 - Duplicate theories: {is_valid} (error: {error})")
    assert not is_valid, "Should be invalid (duplicates)"
    
    print("\n✅ All validation tests passed!")


def test_single_cluster():
    """Test processing a single cluster (dry run without LLM call)."""
    print("\n" + "="*80)
    print("TEST 3: Single Cluster Structure")
    print("="*80)
    
    # Load actual data
    stage5_path = PROJECT_ROOT / 'output' / 'stage5_consolidated_final_theories.json'
    
    if not stage5_path.exists():
        print("⚠️  Stage5 file not found, skipping this test")
        return
    
    with open(stage5_path, 'r') as f:
        stage5_data = json.load(f)
    
    # Find a cluster with >40 papers
    large_cluster = None
    for cluster in stage5_data['final_name_summary']:
        if cluster['total_papers'] > 40:
            large_cluster = cluster
            break
    
    if not large_cluster:
        print("⚠️  No large clusters found, skipping this test")
        return
    
    print(f"\nFound cluster: {large_cluster['final_name']}")
    print(f"  Papers: {large_cluster['total_papers']}")
    print(f"  Theories: {len(large_cluster['theory_ids'])}")
    
    # Test batch creation
    separator = Stage6ClusterSeparator()
    batches = separator._create_smart_batches(large_cluster['theory_ids'])
    print(f"  Batches: {len(batches)} (sizes: {[len(b) for b in batches]})")
    
    # Verify all theories are included
    all_ids = []
    for batch in batches:
        all_ids.extend(batch)
    
    assert len(all_ids) == len(large_cluster['theory_ids']), "Lost theories in batching!"
    assert len(set(all_ids)) == len(all_ids), "Duplicate theories in batching!"
    
    print("\n✅ Single cluster test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("STAGE 6 CLUSTER SEPARATION - TESTS")
    print("="*80)
    
    try:
        test_batch_splitting()
        test_validation()
        test_single_cluster()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the full stage6 separation:")
        print("  python src/normalization/stage6_cluster_separation.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
