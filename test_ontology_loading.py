"""
Test loading the new ontology format.
"""

import json
from pathlib import Path

def test_ontology_loading():
    """Test loading the updated ontology files."""
    
    print("="*80)
    print("TESTING NEW ONTOLOGY FORMAT")
    print("="*80)
    
    # Load mechanisms (new format)
    mechanisms_path = Path('ontology/group_ontology_mechanisms.json')
    print(f"\n📂 Loading mechanisms from {mechanisms_path}...")
    
    with open(mechanisms_path, 'r') as f:
        mechanisms_list = json.load(f)
    
    print(f"✓ Loaded {len(mechanisms_list)} theories")
    print(f"✓ Format: List of objects")
    
    # Check first theory
    if mechanisms_list:
        first_theory = mechanisms_list[0]
        print(f"\n📋 First theory:")
        print(f"  Name: {first_theory.get('theory_name')}")
        print(f"  Key players: {len(first_theory.get('key_players', []))} items")
        print(f"  Pathways: {len(first_theory.get('pathways', []))} items")
        print(f"  Mechanisms: {len(first_theory.get('mechanisms', []))} items")
        
        print(f"\n  Sample key players:")
        for player in first_theory.get('key_players', [])[:3]:
            print(f"    - {player}")
        
        print(f"\n  Sample mechanisms:")
        for mech in first_theory.get('mechanisms', [])[:2]:
            print(f"    - {mech[:100]}...")
    
    # Convert to dict for lookup
    mechanisms_dict = {item['theory_name']: item for item in mechanisms_list}
    
    # Test lookup
    test_theories = [
        "Free Radical Theory",
        "Telomere Theory",
        "Mutation Accumulation Theory",
        "Disengagement Theory"
    ]
    
    print(f"\n🔍 Testing lookups:")
    for theory_name in test_theories:
        if theory_name in mechanisms_dict:
            theory = mechanisms_dict[theory_name]
            print(f"  ✓ {theory_name}: {len(theory.get('mechanisms', []))} mechanisms")
        else:
            print(f"  ✗ {theory_name}: NOT FOUND")
    
    # Load aliases
    aliases_path = Path('ontology/groups_ontology_alliases.json')
    print(f"\n📂 Loading aliases from {aliases_path}...")
    
    with open(aliases_path, 'r') as f:
        aliases_data = json.load(f)
    
    # Count theories in aliases
    theory_count = 0
    for category, subcats in aliases_data['TheoriesOfAging'].items():
        for subcat, theories in subcats.items():
            theory_count += len(theories)
    
    print(f"✓ Loaded {theory_count} theories from aliases")
    
    # Check coverage
    alias_names = set()
    for category, subcats in aliases_data['TheoriesOfAging'].items():
        for subcat, theories in subcats.items():
            for theory in theories:
                alias_names.add(theory['name'])
    
    mechanism_names = set(mechanisms_dict.keys())
    
    print(f"\n📊 Coverage Analysis:")
    print(f"  Theories in aliases: {len(alias_names)}")
    print(f"  Theories with mechanisms: {len(mechanism_names)}")
    print(f"  Coverage: {len(mechanism_names)/len(alias_names)*100:.1f}%")
    
    # Find missing
    missing = alias_names - mechanism_names
    if missing:
        print(f"\n⚠️  Theories without mechanisms ({len(missing)}):")
        for name in sorted(missing)[:10]:
            print(f"    - {name}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")
    
    # Find extra
    extra = mechanism_names - alias_names
    if extra:
        print(f"\n⚠️  Mechanisms without aliases ({len(extra)}):")
        for name in sorted(extra)[:10]:
            print(f"    - {name}")
        if len(extra) > 10:
            print(f"    ... and {len(extra)-10} more")
    
    print("\n" + "="*80)
    print("✅ ONTOLOGY LOADING TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_ontology_loading()
