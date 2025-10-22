"""
Test Stage 3 Iterative Refinement.
"""

from src.normalization.stage3_iterative_refinement import Stage3IterativeRefinement

print("=" * 80)
print("TEST: Stage 3 Iterative Refinement")
print("=" * 80)

# Create refiner
refiner = Stage3IterativeRefinement(
    stage2_path='output/stage2_grouped_theories.json',
    ontology_path='ontology/groups_ontology_alliases.json',
    max_iterations=2  # Use 2 iterations for testing
)

# Load Stage 2 output
stage2_mappings = refiner._load_stage2_output()

print(f"\n📊 Stage 2 Summary:")
print(f"  Total mappings: {len(stage2_mappings)}")
print(f"  Unique mapped names: {len(set(stage2_mappings.values()))}")

# Test matching to ontology
matched, unmatched = refiner._match_to_ontology(stage2_mappings)

print(f"\n📊 Matching Results:")
print(f"  ✓ Matched to ontology: {len(matched)} ({len(matched)/len(stage2_mappings)*100:.1f}%)")
print(f"  ⚠️  Unmatched: {len(unmatched)} ({len(unmatched)/len(stage2_mappings)*100:.1f}%)")

print(f"\n📝 Sample matched mappings (first 5):")
for i, (initial, ontology_name) in enumerate(list(matched.items())[:5], 1):
    print(f"  {i}. '{initial}' → '{ontology_name}' (ontology)")

print(f"\n📝 Sample unmatched mappings (first 5):")
for i, (initial, mapped) in enumerate(list(unmatched.items())[:5], 1):
    print(f"  {i}. '{initial}' → '{mapped}' (needs refinement)")

print(f"\n✅ Test complete!")
print(f"\nTo run full Stage 3 refinement:")
print(f"  python src/normalization/stage3_iterative_refinement.py")
