"""
Test Theory Tracker functionality.
"""

from src.tracking.theory_tracker import TheoryTracker

print("=" * 80)
print("TEST: Theory Tracker")
print("=" * 80)

# Create tracker
tracker = TheoryTracker()

# Track Stage 0
tracker.track_stage0('output/stage0_filtered_theories.json')

# Show sample theory lineage
print("\n📝 Sample Theory Lineage (first 3 theories):")
for i, (theory_id, data) in enumerate(list(tracker.theory_lineage.items())[:3], 1):
    print(f"\n{i}. Theory ID: {theory_id}")
    print(f"   Original name: {data['original_name']}")
    print(f"   Paper ID: {data['paper_id']}")
    print(f"   Confidence: {data['confidence_level']}")
    print(f"   Stage 0 status: {data['stage0_status']}")

# Track Stage 1 if available
try:
    tracker.track_stage1('output/stage1_fuzzy_matched.json')
    
    # Show a matched theory
    matched_theories = tracker.get_theories_by_status('stage1', 'matched')
    if matched_theories:
        print(f"\n📝 Sample Matched Theory (Stage 1):")
        theory = matched_theories[0]
        print(f"   Theory ID: {theory['theory_id']}")
        print(f"   Original: {theory['original_name']}")
        print(f"   Matched to: {theory['stage1_matched_name']}")
        print(f"   Match score: {theory.get('stage1_match_score', 'N/A')}")
except FileNotFoundError:
    print("\n⚠️  Stage 1 output not found")

# Track Stage 1.5 if available
try:
    tracker.track_stage1_5('output/stage1_5_llm_mapped.json')
except FileNotFoundError:
    print("\n⚠️  Stage 1.5 output not found")

# Track Stage 2 if available
try:
    tracker.track_stage2('output/stage2_grouped_theories.json')
except FileNotFoundError:
    print("\n⚠️  Stage 2 output not found")

print("\n✅ Test complete!")
print(f"\nTo generate full report, run:")
print(f"  python src/tracking/theory_tracker.py")
