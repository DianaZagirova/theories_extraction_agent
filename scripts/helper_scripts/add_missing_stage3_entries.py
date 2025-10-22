#!/usr/bin/env python3
"""
Add missing entries to stage3 output that were incorrectly filtered out by fuzzy matching.
"""
import json
import os
from pathlib import Path

# Load stage2
print("Loading stage2 output...")
with open('output/stage2_grouped_theories.json') as f:
    stage2_data = json.load(f)
stage2_unique_values = set(stage2_data.get('mappings', {}).values())
print(f"  Stage2 unique values: {len(stage2_unique_values)}")

# Load stage3
print("\nLoading stage3 output...")
with open('output/stage3_refined_theories.json') as f:
    stage3_data = json.load(f)
stage3_keys = set(stage3_data.get('mappings', {}).keys())
print(f"  Stage3 keys: {len(stage3_keys)}")

# Load ontology
print("\nLoading ontology...")
with open('ontology/groups_ontology_alliases.json') as f:
    ont = json.load(f)
ontology_theories = set()
for cat, subcats in ont['TheoriesOfAging'].items():
    for subcat, theory_list in subcats.items():
        for theory in theory_list:
            ontology_theories.add(theory['name'])
print(f"  Ontology theories: {len(ontology_theories)}")

# Apply NEW filtering logic (exact match only) - same as fixed stage3
ontology_lower = {name.lower(): name for name in ontology_theories}
should_be_in_stage3 = []

for mapped_name in stage2_unique_values:
    mapped_lower = mapped_name.lower()
    
    # Check exact match (case-insensitive) only - no fuzzy matching
    if mapped_lower not in ontology_lower:
        should_be_in_stage3.append(mapped_name)

# Find what's missing
missing_from_stage3 = set(should_be_in_stage3) - stage3_keys

print(f"\n{'=' * 80}")
print(f"ANALYSIS")
print(f"{'=' * 80}")
print(f"Should be in stage3: {len(should_be_in_stage3)}")
print(f"Currently in stage3: {len(stage3_keys)}")
print(f"Missing entries: {len(missing_from_stage3)}")

if missing_from_stage3:
    print(f"\n{'=' * 80}")
    print(f"MISSING ENTRIES ({len(missing_from_stage3)})")
    print(f"{'=' * 80}")
    for name in sorted(missing_from_stage3):
        print(f"  - {name}")
    
    # Add missing entries as identity mappings (name -> name)
    print(f"\n{'=' * 80}")
    print(f"ADDING MISSING ENTRIES")
    print(f"{'=' * 80}")
    
    for name in missing_from_stage3:
        stage3_data['mappings'][name] = name
    
    # Update metadata
    stage3_data['metadata']['total_mappings'] = len(stage3_data['mappings'])
    stage3_data['metadata']['unique_mapped_names'] = len(set(stage3_data['mappings'].values()))
    stage3_data['metadata']['names_not_in_ontology'] = len(should_be_in_stage3)
    stage3_data['metadata']['status'] = 'complete'
    stage3_data['metadata']['note'] = 'Missing entries added after fixing fuzzy matching bug'
    
    # Save updated file
    with open('output/stage3_refined_theories.json', 'w') as f:
        json.dump(stage3_data, f, indent=2)
    
    print(f"\n✅ Added {len(missing_from_stage3)} missing entries")
    print(f"✅ Updated stage3 output saved")
    print(f"\nNew statistics:")
    print(f"  Total mappings: {stage3_data['metadata']['total_mappings']}")
    print(f"  Unique mapped names: {stage3_data['metadata']['unique_mapped_names']}")
else:
    print("\n✅ No missing entries - stage3 is complete!")
