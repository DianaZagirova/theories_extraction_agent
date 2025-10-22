#!/usr/bin/env python3
"""
Rerun Stage 6 separation for specific clusters only.

This allows selective reprocessing of clusters with issues,
saving time and tokens by not rerunning successful clusters.
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage6_cluster_separation import Stage6ClusterSeparator

def load_clusters_to_rerun(clusters_file):
    """Load list of clusters to rerun from JSON file."""
    with open(clusters_file, 'r') as f:
        data = json.load(f)
    return data.get('clusters_to_rerun', [])

def main():
    parser = argparse.ArgumentParser(
        description='Rerun Stage 6 separation for specific clusters only'
    )
    parser.add_argument(
        '--clusters-file',
        type=str,
        default='output/stage6_clusters_to_rerun.json',
        help='JSON file with list of clusters to rerun'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        nargs='+',
        help='Specific cluster names to rerun (space-separated)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=40,
        help='Paper count threshold for separation (default: 40)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=2,
        help='Minimum subcluster size (default: 2)'
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=26,
        help='Maximum theories per batch (default: 26)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/stage6_separated_clusters.json',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    # Determine which clusters to rerun
    if args.clusters:
        clusters_to_rerun = args.clusters
        print(f"📋 Rerunning {len(clusters_to_rerun)} clusters from command line")
    elif Path(args.clusters_file).exists():
        clusters_to_rerun = load_clusters_to_rerun(args.clusters_file)
        print(f"📋 Loaded {len(clusters_to_rerun)} clusters from {args.clusters_file}")
    else:
        print(f"❌ Error: No clusters specified and {args.clusters_file} not found")
        print(f"\nUsage:")
        print(f"  1. Run analysis first: python scripts/analyze_stage6_checkpoints.py")
        print(f"  2. Then rerun: python scripts/rerun_stage6_clusters.py")
        print(f"\nOr specify clusters manually:")
        print(f"  python scripts/rerun_stage6_clusters.py --clusters 'Cluster Name 1' 'Cluster Name 2'")
        sys.exit(1)
    
    if not clusters_to_rerun:
        print("✅ No clusters to rerun!")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("STAGE 6: SELECTIVE CLUSTER REPROCESSING")
    print("="*80)
    print(f"Configuration:")
    print(f"  Paper threshold: >{args.threshold}")
    print(f"  Min subcluster size: {args.min_size}")
    print(f"  Max theories per batch: {args.max_batch}")
    print(f"  Clusters to rerun: {len(clusters_to_rerun)}")
    print(f"  Output: {args.output}")
    print("="*80)
    
    print(f"\n📋 Clusters to rerun:")
    for i, cluster_name in enumerate(clusters_to_rerun, 1):
        print(f"  {i}. {cluster_name}")
    
    # Create separator
    separator = Stage6ClusterSeparator(
        paper_threshold=args.threshold,
        max_theories_per_batch=args.max_batch,
        min_subcluster_size=args.min_size,
        output_path=args.output
    )
    
    # Load existing results if they exist
    output_path = Path(args.output)
    existing_results = []
    if output_path.exists():
        print(f"\n📂 Loading existing results from {output_path}")
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('separated_clusters', [])
        print(f"✓ Found {len(existing_results)} existing cluster results")
        
        # Remove clusters that we're rerunning
        existing_cluster_names = {r['original_cluster_name'] for r in existing_results}
        clusters_to_keep = [r for r in existing_results 
                           if r['original_cluster_name'] not in clusters_to_rerun]
        
        removed_count = len(existing_results) - len(clusters_to_keep)
        print(f"✓ Keeping {len(clusters_to_keep)} existing results")
        print(f"✓ Will replace {removed_count} cluster results")
        
        existing_results = clusters_to_keep
    
    # Filter stage5 data to only include clusters we want to rerun
    print(f"\n🔧 Filtering stage5 data to only include clusters to rerun...")
    original_stage5 = separator.stage5_data.copy()
    filtered_summary = []
    
    for cluster in original_stage5['final_name_summary']:
        if cluster['final_name'] in clusters_to_rerun:
            filtered_summary.append(cluster)
    
    separator.stage5_data['final_name_summary'] = filtered_summary
    print(f"✓ Filtered to {len(filtered_summary)} clusters")
    
    # Run separation on filtered clusters
    print(f"\n🚀 Starting reprocessing...")
    separator.run()
    
    # Merge with existing results
    if existing_results:
        print(f"\n🔄 Merging with existing results...")
        
        # Load new results
        with open(output_path, 'r') as f:
            new_data = json.load(f)
        
        new_results = new_data.get('separated_clusters', [])
        
        # Combine
        all_results = existing_results + new_results
        
        # Update output
        new_data['separated_clusters'] = all_results
        new_data['statistics']['note'] = f'Merged results: {len(existing_results)} existing + {len(new_results)} reprocessed'
        
        with open(output_path, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"✓ Merged results: {len(all_results)} total clusters")
        print(f"  - {len(existing_results)} kept from previous run")
        print(f"  - {len(new_results)} newly reprocessed")
    
    print(f"\n✅ Selective reprocessing complete!")
    print(f"   Updated output: {args.output}")
    print(f"\n💡 Next step: Regenerate consolidated output")
    print(f"   The consolidation will automatically use the updated separation results")

if __name__ == '__main__':
    main()
