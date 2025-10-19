"""
Run the complete mechanism-based clustering pipeline.
"""

import os
import sys
import time

def run_stage(stage_name, script_path):
    """Run a pipeline stage."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {stage_name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Run the script
    exit_code = os.system(f"python {script_path}")
    
    elapsed = time.time() - start_time
    
    if exit_code != 0:
        print(f"\n❌ {stage_name} failed with exit code {exit_code}")
        return False
    
    print(f"\n✅ {stage_name} completed in {elapsed:.1f}s")
    return True


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("MECHANISM-BASED CLUSTERING PIPELINE")
    print("="*70)
    print("\nThis pipeline will:")
    print("1. Extract mechanisms using LLM (~5-10 minutes, ~$10-15)")
    print("2. Build biological taxonomy")
    print("3. Cluster by mechanisms (not embeddings)")
    print("4. Generate readable summaries")
    
    # Check if Stage 1 output exists
    if not os.path.exists('output/stage1_embeddings.json'):
        print("\n❌ Error: output/stage1_embeddings.json not found")
        print("   Please run Stage 1 first: python src/normalization/stage1_embedding_advanced.py")
        return
    
    # Confirm with user
    print("\n⚠️  This will make LLM API calls (~$10-15 cost)")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    start_time = time.time()
    
    # Stage 2: Mechanism Extraction
    if not run_stage(
        "Stage 2: Mechanism Extraction",
        "src/normalization/stage2_mechanism_extraction.py"
    ):
        return
    
    # Stage 3: Mechanism Clustering
    if not run_stage(
        "Stage 3: Mechanism-Based Clustering",
        "src/normalization/stage3_mechanism_clustering.py"
    ):
        return
    
    # Generate readable summary
    print(f"\n{'='*70}")
    print("GENERATING READABLE SUMMARY")
    print(f"{'='*70}\n")
    
    # Create summary script on the fly
    summary_script = """
import json

with open('output/stage3_mechanism_clusters.json', 'r') as f:
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

# Create readable structure
readable = {
    'metadata': data['metadata'],
    'families': []
}

for family in sorted(families, key=lambda x: x['cluster_id']):
    family_summary = {
        'id': family['cluster_id'],
        'name': family['name'],
        'theory_count': family['theory_count'],
        'mechanism_signature': family['mechanism_signature'],
        'parents': []
    }
    
    family_parents = family_parents_map.get(family['cluster_id'], [])
    
    for parent in sorted(family_parents, key=lambda x: x['cluster_id']):
        parent_summary = {
            'id': parent['cluster_id'],
            'name': parent['name'],
            'theory_count': parent['theory_count'],
            'mechanism_signature': parent['mechanism_signature'],
            'children': []
        }
        
        parent_children = parent_children_map.get(parent['cluster_id'], [])
        
        for child in sorted(parent_children, key=lambda x: x['cluster_id']):
            child_summary = {
                'id': child['cluster_id'],
                'name': child['name'],
                'theory_count': child['theory_count'],
                'mechanism_signature': child['mechanism_signature'],
                'theories': []
            }
            
            for theory_id in child['theory_ids']:
                theory = theories.get(theory_id)
                if theory:
                    child_summary['theories'].append({
                        'id': theory_id,
                        'name': theory['name']
                    })
            
            parent_summary['children'].append(child_summary)
        
        family_summary['parents'].append(parent_summary)
    
    readable['families'].append(family_summary)

with open('output/mechanism_clusters_readable.json', 'w') as f:
    json.dump(readable, f, indent=2)

print("✓ Created output/mechanism_clusters_readable.json")
"""
    
    with open('_temp_summary.py', 'w') as f:
        f.write(summary_script)
    
    os.system("python _temp_summary.py")
    os.remove('_temp_summary.py')
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print("\nGenerated files:")
    print("  1. output/stage2_mechanisms.json - Extracted mechanisms")
    print("  2. output/stage3_mechanism_clusters.json - Mechanism-based clusters")
    print("  3. output/mechanism_clusters_readable.json - Human-readable format")
    print("\nNext steps:")
    print("  1. Review mechanism_clusters_readable.json")
    print("  2. Compare with embedding-based approach")
    print("  3. Validate biological coherence")


if __name__ == '__main__':
    main()
