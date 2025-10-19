"""
Create a concise, human-readable summary of clustering results.
Transforms the detailed stage2_clusters.json into an easy-to-understand format.
"""

import json
from pathlib import Path
from typing import Dict, List


def create_readable_summary(input_path: str, output_path: str):
    """Create human-readable summary of clustering hierarchy."""
    
    print(f"📂 Loading clustering results from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    theories = {t['theory_id']: t for t in data['theories']}
    families = data['families']
    parents = data['parents']
    children = data['children']
    
    # Build parent -> children mapping
    parent_children_map = {}
    for child in children:
        parent_id = child['parent_cluster_id']
        if parent_id not in parent_children_map:
            parent_children_map[parent_id] = []
        parent_children_map[parent_id].append(child)
    
    # Build family -> parents mapping
    family_parents_map = {}
    for parent in parents:
        family_id = parent['parent_cluster_id']
        if family_id not in family_parents_map:
            family_parents_map[family_id] = []
        family_parents_map[family_id].append(parent)
    
    # Create readable structure
    readable_summary = {
        'metadata': {
            'total_theories': data['metadata']['statistics']['total_theories'],
            'num_families': data['metadata']['statistics']['num_families'],
            'num_parents': data['metadata']['statistics']['num_parents'],
            'num_children': data['metadata']['statistics']['num_children'],
            'singleton_families': data['metadata']['statistics'].get('singleton_families', 0),
            'singleton_parents': data['metadata']['statistics'].get('singleton_parents', 0),
            'singleton_children': data['metadata']['statistics'].get('singleton_children', 0),
            'outliers_preserved': data['metadata']['statistics'].get('outliers_preserved', 0)
        },
        'families': []
    }
    
    # Process each family
    for family in sorted(families, key=lambda x: x['cluster_id']):
        family_summary = {
            'id': family['cluster_id'],
            'name': family.get('canonical_name', f"Family {family['cluster_id']}"),
            'is_singleton': family.get('is_singleton', False),
            'theory_count': family['theory_count'],
            'parents': []
        }
        
        # Get parents in this family
        family_parents = family_parents_map.get(family['cluster_id'], [])
        
        for parent in sorted(family_parents, key=lambda x: x['cluster_id']):
            parent_summary = {
                'id': parent['cluster_id'],
                'is_singleton': parent.get('is_singleton', False),
                'theory_count': parent['theory_count'],
                'children': []
            }
            
            # Get children in this parent
            parent_children = parent_children_map.get(parent['cluster_id'], [])
            
            for child in sorted(parent_children, key=lambda x: x['cluster_id']):
                child_summary = {
                    'id': child['cluster_id'],
                    'is_singleton': child.get('is_singleton', False),
                    'theory_count': child['theory_count'],
                    'theories': []
                }
                
                # Add theory names
                for theory_id in child['theory_ids']:
                    theory = theories.get(theory_id)
                    if theory:
                        child_summary['theories'].append({
                            'id': theory_id,
                            'name': theory['name']
                        })
                
                parent_summary['children'].append(child_summary)
            
            family_summary['parents'].append(parent_summary)
        
        readable_summary['families'].append(family_summary)
    
    # Save readable summary
    print(f"💾 Saving readable summary to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(readable_summary, f, indent=2)
    
    print(f"✓ Created readable summary")
    print(f"\n📊 Summary Statistics:")
    print(f"  Total theories: {readable_summary['metadata']['total_theories']}")
    print(f"  Families: {readable_summary['metadata']['num_families']}")
    print(f"    - Regular: {readable_summary['metadata']['num_families'] - readable_summary['metadata']['singleton_families']}")
    print(f"    - Singleton: {readable_summary['metadata']['singleton_families']}")
    print(f"  Parents: {readable_summary['metadata']['num_parents']}")
    print(f"    - Regular: {readable_summary['metadata']['num_parents'] - readable_summary['metadata']['singleton_parents']}")
    print(f"    - Singleton: {readable_summary['metadata']['singleton_parents']}")
    print(f"  Children: {readable_summary['metadata']['num_children']}")
    print(f"    - Regular: {readable_summary['metadata']['num_children'] - readable_summary['metadata']['singleton_children']}")
    print(f"    - Singleton: {readable_summary['metadata']['singleton_children']}")
    if readable_summary['metadata']['outliers_preserved'] > 0:
        print(f"  Outliers preserved: {readable_summary['metadata']['outliers_preserved']}")


def create_compact_summary(input_path: str, output_path: str):
    """Create ultra-compact summary showing just the hierarchy."""
    
    print(f"\n📂 Creating compact summary...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    theories = {t['theory_id']: t for t in data['theories']}
    families = data['families']
    parents = data['parents']
    children = data['children']
    
    # Build mappings
    parent_children_map = {}
    for child in children:
        parent_id = child['parent_cluster_id']
        if parent_id not in parent_children_map:
            parent_children_map[parent_id] = []
        parent_children_map[parent_id].append(child)
    
    family_parents_map = {}
    for parent in parents:
        family_id = parent['parent_cluster_id']
        if family_id not in family_parents_map:
            family_parents_map[family_id] = []
        family_parents_map[family_id].append(parent)
    
    compact = []
    
    for family in sorted(families, key=lambda x: x['cluster_id']):
        family_item = {
            family['cluster_id']: {
                'name': family.get('canonical_name', 'Unnamed'),
                'singleton': family.get('is_singleton', False),
                'parents': {}
            }
        }
        
        family_parents = family_parents_map.get(family['cluster_id'], [])
        
        for parent in sorted(family_parents, key=lambda x: x['cluster_id']):
            parent_item = {
                'singleton': parent.get('is_singleton', False),
                'children': {}
            }
            
            parent_children = parent_children_map.get(parent['cluster_id'], [])
            
            for child in sorted(parent_children, key=lambda x: x['cluster_id']):
                theory_names = [theories[tid]['name'] for tid in child['theory_ids'] if tid in theories]
                parent_item['children'][child['cluster_id']] = {
                    'singleton': child.get('is_singleton', False),
                    'theories': theory_names
                }
            
            family_item[family['cluster_id']]['parents'][parent['cluster_id']] = parent_item
        
        compact.append(family_item)
    
    with open(output_path, 'w') as f:
        json.dump(compact, f, indent=2)
    
    print(f"✓ Created compact summary at {output_path}")


def create_flat_list(input_path: str, output_path: str):
    """Create flat list of all theories with their cluster assignments."""
    
    print(f"\n📂 Creating flat theory list...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    theories = {t['theory_id']: t for t in data['theories']}
    
    # Build reverse mappings
    theory_to_child = {}
    for child in data['children']:
        for tid in child['theory_ids']:
            theory_to_child[tid] = child
    
    child_to_parent = {c['cluster_id']: c['parent_cluster_id'] for c in data['children']}
    parent_to_family = {p['cluster_id']: p['parent_cluster_id'] for p in data['parents']}
    
    family_lookup = {f['cluster_id']: f for f in data['families']}
    parent_lookup = {p['cluster_id']: p for p in data['parents']}
    
    flat_list = []
    
    for theory_id, theory in sorted(theories.items()):
        child = theory_to_child.get(theory_id)
        if not child:
            continue
        
        parent_id = child_to_parent.get(child['cluster_id'])
        family_id = parent_to_family.get(parent_id) if parent_id else None
        
        family = family_lookup.get(family_id) if family_id else None
        parent = parent_lookup.get(parent_id) if parent_id else None
        
        flat_list.append({
            'theory_id': theory_id,
            'theory_name': theory['name'],
            'family_id': family_id,
            'family_name': family.get('canonical_name', 'Unnamed') if family else None,
            'parent_id': parent_id,
            'child_id': child['cluster_id'],
            'is_singleton': child.get('is_singleton', False)
        })
    
    with open(output_path, 'w') as f:
        json.dump(flat_list, f, indent=2)
    
    print(f"✓ Created flat list at {output_path}")
    print(f"  Total theories: {len(flat_list)}")


def main():
    PREFIX ="_alternative"
    """Generate all summary formats."""
    input_path = f'output/stage2_clusters{PREFIX}.json'
    
    if not Path(input_path).exists():
        print(f"❌ Error: {input_path} not found")
        print("   Run Stage 2 first: python src/normalization/stage2_clustering.py")
        return
    
    print("="*70)
    print("CREATING READABLE SUMMARIES FROM CLUSTERING RESULTS")
    print("="*70)
    
    # Format 1: Detailed readable summary
    create_readable_summary(
        input_path,
        f'output/clustering_summary_readable{PREFIX}.json'
    )
    
    # Format 2: Compact hierarchy
    create_compact_summary(
        input_path,
        f'output/clustering_summary_compact{PREFIX}.json'
    )
    
    # Format 3: Flat list
    create_flat_list(
        input_path,
        f'output/clustering_summary_flat{PREFIX}.json'
    )
    
    print("\n" + "="*70)
    print("✅ ALL SUMMARIES CREATED")
    print("="*70)
    print("\nGenerated files:")
    print("  1. output/clustering_summary_readable.json  - Full hierarchy with theory names")
    print("  2. output/clustering_summary_compact.json   - Compact nested structure")
    print("  3. output/clustering_summary_flat.json      - Flat list of theories with assignments")
    print("\nRecommended for overview: clustering_summary_readable.json")


if __name__ == '__main__':
    main()
