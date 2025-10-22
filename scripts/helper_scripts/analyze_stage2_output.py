"""
Analyze stage2_grouped_theories.json to count unique mapped names.
"""
import json

def analyze_stage2_output(file_path='output/stage2_grouped_theories.json'):
    """Analyze the stage2 output file."""
    
    print("=" * 80)
    print("ANALYZING STAGE 2 OUTPUT")
    print("=" * 80)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    mappings = data.get('mappings', {})
    
    print("\n📊 Metadata:")
    print(f"  Stage: {metadata.get('stage')}")
    print(f"  Status: {metadata.get('status')}")
    print(f"  Total mappings: {metadata.get('total_mappings')}")
    print(f"  Unique mapped names (from metadata): {metadata.get('unique_mapped_names')}")
    print(f"  Total batches: {metadata.get('total_batches')}")
    print(f"  Batches completed: {metadata.get('batches_completed')}")
    
    print("\n📊 Actual Mappings Analysis:")
    print(f"  Total mappings (keys): {len(mappings)}")
    
    # Count unique values (mapped names)
    unique_values = set(mappings.values())
    print(f"  Unique mapped names (values): {len(unique_values)}")
    
    # Show some examples
    print("\n📝 Sample mappings (first 10):")
    for i, (key, value) in enumerate(list(mappings.items())[:10], 1):
        print(f"  {i}. '{key}' -> '{value}'")
    
    # Count how many keys map to each value
    from collections import Counter
    value_counts = Counter(mappings.values())
    
    print("\n📊 Top 10 most common mapped names:")
    for name, count in value_counts.most_common(10):
        print(f"  '{name}': {count} original names")
    
    print("\n📊 Distribution:")
    single_mappings = sum(1 for count in value_counts.values() if count == 1)
    multi_mappings = sum(1 for count in value_counts.values() if count > 1)
    print(f"  Mapped names with 1 original name: {single_mappings}")
    print(f"  Mapped names with multiple original names: {multi_mappings}")
    
    # Check if status is complete
    if metadata.get('status') != 'complete':
        print("\n⚠️  WARNING: Status is not 'complete'!")
        print(f"  Status: {metadata.get('status')}")
        print(f"  Batches completed: {metadata.get('batches_completed')}/{metadata.get('total_batches')}")
        print("\n💡 This file may be an incomplete checkpoint.")
        print("   Run stage2 with --resume flag to complete it.")

if __name__ == '__main__':
    analyze_stage2_output()
