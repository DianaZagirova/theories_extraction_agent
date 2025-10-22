"""
Standalone script to consolidate stage5 and stage6 results.

This script combines:
- output/stage5_consolidated_final_theories.json (original consolidated names)
- output/stage6_separated_clusters.json (separation results)

Into:
- output/stage6_consolidated_final_theories.json (final consolidated output)

Usage:
    python scripts/consolidate_stage6_results.py
    
    # With custom paths
    python scripts/consolidate_stage6_results.py \
        --stage5 output/stage5_consolidated_final_theories.json \
        --stage6 output/stage6_separated_clusters.json \
        --output output/stage6_consolidated_final_theories.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def consolidate_results(stage5_path: Path, stage6_path: Path, output_path: Path):
    """
    Consolidate stage5 and stage6 results.
    
    Args:
        stage5_path: Path to stage5_consolidated_final_theories.json
        stage6_path: Path to stage6_separated_clusters.json
        output_path: Path for consolidated output
    """
    print("\n" + "="*80)
    print("CONSOLIDATING STAGE5 + STAGE6 RESULTS")
    print("="*80)
    
    # Load stage5 data
    print(f"\n📂 Loading stage5 data from {stage5_path}...")
    with open(stage5_path, 'r') as f:
        stage5_data = json.load(f)
    
    stage5_summary = stage5_data.get('final_name_summary', [])
    print(f"  ✓ Loaded {len(stage5_summary)} theory names from stage5")
    
    # Load stage6 data
    print(f"\n📂 Loading stage6 data from {stage6_path}...")
    with open(stage6_path, 'r') as f:
        stage6_data = json.load(f)
    
    separated_clusters = stage6_data.get('separated_clusters', [])
    print(f"  ✓ Loaded {len(separated_clusters)} separation results from stage6")
    
    # Build mapping: theory_id -> new subcluster name (for separated clusters)
    print("\n🔄 Building theory ID mappings...")
    theory_id_to_new_name = {}
    separated_cluster_names = set()
    successful_separations = 0
    
    for cluster_result in separated_clusters:
        if cluster_result.get('separation_successful', False):
            original_name = cluster_result['original_cluster_name']
            separated_cluster_names.add(original_name)
            successful_separations += 1
            
            for subcluster in cluster_result.get('subclusters', []):
                subcluster_name = subcluster['subcluster_name']
                for theory_id in subcluster.get('theory_ids', []):
                    theory_id_to_new_name[theory_id] = subcluster_name
    
    print(f"  ✓ Separated {len(separated_cluster_names)} clusters into {len(set(theory_id_to_new_name.values()))} subclusters")
    print(f"  ✓ Mapped {len(theory_id_to_new_name)} theory IDs to new names")
    
    # Build new final_name_summary
    print("\n🔨 Building consolidated summary...")
    name_to_theories = defaultdict(lambda: {
        'theory_ids': [],
        'original_names': set(),
        'stage5_parent': None,
        'was_separated': False
    })
    
    for cluster_info in stage5_summary:
        stage5_name = cluster_info['final_name']
        theory_ids = cluster_info['theory_ids']
        original_names = cluster_info.get('original_names', [])
        
        if stage5_name in separated_cluster_names:
            # This cluster was separated - distribute theories to subclusters
            for theory_id in theory_ids:
                new_name = theory_id_to_new_name.get(theory_id)
                if new_name:
                    name_to_theories[new_name]['theory_ids'].append(theory_id)
                    name_to_theories[new_name]['original_names'].update(original_names)
                    name_to_theories[new_name]['stage5_parent'] = stage5_name
                    name_to_theories[new_name]['was_separated'] = True
                else:
                    # Theory not found in separation (shouldn't happen)
                    print(f"  ⚠️  Warning: Theory {theory_id} from '{stage5_name}' not found in separation results")
        else:
            # This cluster was not separated - keep as is
            name_to_theories[stage5_name]['theory_ids'].extend(theory_ids)
            name_to_theories[stage5_name]['original_names'].update(original_names)
            name_to_theories[stage5_name]['stage5_parent'] = stage5_name
            name_to_theories[stage5_name]['was_separated'] = False
    
    # Convert to final summary format
    new_summary = []
    for final_name, data in name_to_theories.items():
        theory_ids = data['theory_ids']
        original_names = sorted(list(data['original_names']))
        
        summary_entry = {
            'final_name': final_name,
            'original_names_count': len(original_names),
            'original_names': original_names,
            'total_papers': len(theory_ids),
            'theory_ids_count': len(theory_ids),
            'theory_ids': theory_ids,
            'stage5_parent': data['stage5_parent'],
            'was_separated_in_stage6': data['was_separated']
        }
        
        new_summary.append(summary_entry)
    
    # Sort by paper count descending
    new_summary.sort(key=lambda x: x['total_papers'], reverse=True)
    
    print(f"  ✓ Created {len(new_summary)} final theory names")
    print(f"    - {sum(1 for s in new_summary if s['was_separated_in_stage6'])} from stage6 separation")
    print(f"    - {sum(1 for s in new_summary if not s['was_separated_in_stage6'])} from stage5 (unchanged)")
    
    # Create consolidated output
    consolidated_data = {
        'metadata': {
            'source_stage5': str(stage5_path),
            'source_stage6': str(stage6_path),
            'total_theory_ids': sum(len(s['theory_ids']) for s in new_summary),
            'unique_final_names': len(new_summary),
            'separated_in_stage6': sum(1 for s in new_summary if s['was_separated_in_stage6']),
            'unchanged_from_stage5': sum(1 for s in new_summary if not s['was_separated_in_stage6']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'final_name_summary': new_summary,
        'stage6_separation_details': stage6_data
    }
    
    # Save consolidated output
    print(f"\n💾 Saving consolidated output to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"  ✓ Saved")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CONSOLIDATION SUMMARY")
    print("="*80)
    print(f"Total theory IDs: {consolidated_data['metadata']['total_theory_ids']}")
    print(f"Unique final names: {consolidated_data['metadata']['unique_final_names']}")
    print(f"  - Separated in stage6: {consolidated_data['metadata']['separated_in_stage6']}")
    print(f"  - Unchanged from stage5: {consolidated_data['metadata']['unchanged_from_stage5']}")
    print("="*80)
    
    # Show top 10 largest clusters
    print("\nTop 10 largest theory clusters:")
    for i, entry in enumerate(new_summary[:10], 1):
        status = "✂️ separated" if entry['was_separated_in_stage6'] else "unchanged"
        parent_info = f" (from {entry['stage5_parent']})" if entry['was_separated_in_stage6'] else ""
        print(f"  {i:2d}. {entry['final_name']}: {entry['total_papers']} papers [{status}]{parent_info}")
    
    print(f"\n✅ Consolidation complete!")
    print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate stage5 and stage6 results'
    )
    parser.add_argument(
        '--stage5',
        type=str,
        default='output/stage5_consolidated_final_theories.json',
        help='Path to stage5 consolidated file'
    )
    parser.add_argument(
        '--stage6',
        type=str,
        default='output/stage6_separated_clusters.json',
        help='Path to stage6 separation file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/stage6_consolidated_final_theories.json',
        help='Output path for consolidated file'
    )
    
    args = parser.parse_args()
    
    stage5_path = Path(args.stage5)
    stage6_path = Path(args.stage6)
    output_path = Path(args.output)
    
    # Validate input files exist
    if not stage5_path.exists():
        print(f"❌ Error: Stage5 file not found: {stage5_path}")
        sys.exit(1)
    
    if not stage6_path.exists():
        print(f"❌ Error: Stage6 file not found: {stage6_path}")
        sys.exit(1)
    
    # Run consolidation
    consolidate_results(stage5_path, stage6_path, output_path)


if __name__ == '__main__':
    main()
