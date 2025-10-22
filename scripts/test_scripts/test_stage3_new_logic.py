"""
Test the new Stage 3 logic to verify it works correctly.
"""
import json

def test_stage3_logic():
    """Test that Stage 3 will process the correct data."""
    
    print("=" * 80)
    print("TESTING STAGE 3 NEW LOGIC")
    print("=" * 80)
    
    # Load Stage 2 output
    print("\n1. Loading Stage 2 output...")
    with open('output/stage2_grouped_theories.json', 'r') as f:
        stage2_data = json.load(f)
    
    mappings = stage2_data.get('mappings', {})
    print(f"   Total mappings (keys): {len(mappings)}")
    
    # Extract unique mapped names (values)
    unique_mapped_names = set(mappings.values())
    print(f"   Unique mapped names (values): {len(unique_mapped_names)}")
    
    # Load ontology
    print("\n2. Loading ontology...")
    with open('ontology/groups_ontology_alliases.json', 'r') as f:
        ontology_data = json.load(f)
    
    ontology_theories = set()
    # Navigate the nested structure: TheoriesOfAging -> categories -> subcategories -> theories
    for category, subcats in ontology_data.get('TheoriesOfAging', {}).items():
        for subcat, theory_list in subcats.items():
            for theory in theory_list:
                if isinstance(theory, dict):
                    name = theory.get('name', '')
                    if name:
                        ontology_theories.add(name)
    
    print(f"   Ontology theories: {len(ontology_theories)}")
    
    # Filter names not in ontology
    print("\n3. Filtering names not in ontology...")
    ontology_lower = {name.lower(): name for name in ontology_theories}
    
    names_not_in_ontology = []
    names_in_ontology = []
    
    for mapped_name in unique_mapped_names:
        mapped_lower = mapped_name.lower()
        
        # Check exact match (case-insensitive)
        if mapped_lower in ontology_lower:
            names_in_ontology.append(mapped_name)
        else:
            # Check if it's a close match
            found_match = False
            for ont_lower in ontology_lower.keys():
                if (mapped_lower in ont_lower or ont_lower in mapped_lower) and \
                   abs(len(mapped_lower) - len(ont_lower)) < 10:
                    names_in_ontology.append(mapped_name)
                    found_match = True
                    break
            
            if not found_match:
                names_not_in_ontology.append(mapped_name)
    
    print(f"   Names IN ontology: {len(names_in_ontology)}")
    print(f"   Names NOT in ontology: {len(names_not_in_ontology)}")
    
    # Show some examples
    print("\n4. Examples of names NOT in ontology (first 20):")
    for i, name in enumerate(sorted(names_not_in_ontology)[:20], 1):
        print(f"   {i}. {name}")
    
    # Group by first character
    print("\n5. Grouping by first character...")
    from collections import defaultdict
    groups = defaultdict(list)
    
    for name in names_not_in_ontology:
        if not name:
            continue
        first_char = name[0].upper()
        groups[first_char].append(name)
    
    sorted_groups = dict(sorted(groups.items()))
    
    print(f"   Alphabetical groups: {len(sorted_groups)}")
    for char, names_list in sorted_groups.items():
        print(f"   {char}: {len(names_list)} names")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Stage 3 will process {len(names_not_in_ontology)} unique mapped names")
    print(f"These are the VALUES from Stage 2 that are NOT in the ontology")
    print(f"They will be grouped alphabetically and processed in batches")
    print("=" * 80)

if __name__ == '__main__':
    test_stage3_logic()
