"""
Test checkpoint resume functionality for Stage 2.
"""

from src.normalization.stage2_group_normalization import Stage2GroupNormalizer

print("=" * 80)
print("TEST: Checkpoint Resume Functionality")
print("=" * 80)

# Create normalizer
normalizer = Stage2GroupNormalizer(
    stage1_5_path='output/stage1_5_llm_mapped.json',
    ontology_path='ontology/groups_ontology_alliases.json'
)

# Test loading checkpoint
checkpoint_path = 'output/stage2_grouped_theories.json'
result = normalizer._load_checkpoint(checkpoint_path)

if result[0] is not None:
    all_mappings, batch_metadata_list, batches_completed, stats = result
    
    print("\n✅ Checkpoint loaded successfully!")
    print(f"\nCheckpoint details:")
    print(f"  - Batches completed: {batches_completed}")
    print(f"  - Total mappings: {len(all_mappings)}")
    print(f"  - Unique names: {len(set(all_mappings.values()))}")
    print(f"  - Batch metadata entries: {len(batch_metadata_list)}")
    print(f"  - Total cost so far: ${stats['total_cost']:.4f}")
    
    print(f"\n📝 Sample mappings (first 5):")
    for i, (initial, mapped) in enumerate(list(all_mappings.items())[:5], 1):
        print(f"  {i}. '{initial}' → '{mapped}'")
    
    print(f"\n📊 To resume processing, run:")
    print(f"  python src/normalization/stage2_group_normalization.py --resume")
else:
    print("\n❌ No valid checkpoint found or checkpoint is already complete.")
