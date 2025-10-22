"""
Analyze and visualize stage6 separation results.

Shows:
- Which clusters were separated and into how many subclusters
- Size distribution of subclusters
- Comparison before/after separation

Usage:
    python scripts/analyze_stage6_results.py
"""

import json
from pathlib import Path
from collections import Counter


def analyze_separation_results():
    """Analyze stage6 separation results."""
    
    # Paths
    stage5_path = Path('output/stage5_consolidated_final_theories.json')
    stage6_path = Path('output/stage6_separated_clusters.json')
    consolidated_path = Path('output/stage6_consolidated_final_theories.json')
    
    print("\n" + "="*80)
    print("STAGE6 SEPARATION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    
    if not stage5_path.exists():
        print(f"❌ Stage5 file not found: {stage5_path}")
        return
    
    if not stage6_path.exists():
        print(f"❌ Stage6 file not found: {stage6_path}")
        return
    
    with open(stage5_path, 'r') as f:
        stage5_data = json.load(f)
    
    with open(stage6_path, 'r') as f:
        stage6_data = json.load(f)
    
    # Check if consolidated exists
    has_consolidated = consolidated_path.exists()
    if has_consolidated:
        with open(consolidated_path, 'r') as f:
            consolidated_data = json.load(f)
    
    print("  ✓ Data loaded")
    
    # Analyze separations
    print("\n" + "="*80)
    print("SEPARATION DETAILS")
    print("="*80)
    
    separated_clusters = stage6_data.get('separated_clusters', [])
    successful = [c for c in separated_clusters if c.get('separation_successful', False)]
    failed = [c for c in separated_clusters if not c.get('separation_successful', False)]
    
    print(f"\nTotal clusters processed: {len(separated_clusters)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    
    if failed:
        print("\nFailed separations:")
        for cluster in failed:
            print(f"  - {cluster['original_cluster_name']}: {cluster.get('error', 'Unknown error')}")
    
    # Analyze each successful separation
    print("\n" + "="*80)
    print("SUCCESSFUL SEPARATIONS")
    print("="*80)
    
    for cluster in sorted(successful, key=lambda x: x['original_total_papers'], reverse=True):
        original_name = cluster['original_cluster_name']
        original_papers = cluster['original_total_papers']
        subclusters = cluster.get('subclusters', [])
        
        print(f"\n📦 {original_name}")
        print(f"   Original: {original_papers} papers")
        print(f"   Split into {len(subclusters)} subclusters:")
        
        for i, sc in enumerate(sorted(subclusters, key=lambda x: x['theory_count'], reverse=True), 1):
            name = sc['subcluster_name']
            count = sc['theory_count']
            percentage = (count / original_papers * 100) if original_papers > 0 else 0
            print(f"     {i}. {name}: {count} papers ({percentage:.1f}%)")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    stats = stage6_data.get('statistics', {})
    print(f"\nProcessing:")
    print(f"  Clusters to separate: {stats.get('clusters_to_separate', 0)}")
    print(f"  Theories processed: {stats.get('total_theories_in_large_clusters', 0)}")
    print(f"  Batches processed: {stats.get('total_batches_processed', 0)}")
    print(f"  Subclusters created: {stats.get('total_subclusters_created', 0)}")
    print(f"  Total retries: {stats.get('total_retries', 0)}")
    
    print(f"\nCost:")
    print(f"  Input tokens: {stats.get('total_input_tokens', 0):,}")
    print(f"  Output tokens: {stats.get('total_output_tokens', 0):,}")
    print(f"  Total cost: ${stats.get('total_cost', 0):.4f}")
    
    # Compare before/after
    if has_consolidated:
        print("\n" + "="*80)
        print("BEFORE vs AFTER COMPARISON")
        print("="*80)
        
        stage5_summary = stage5_data.get('final_name_summary', [])
        consolidated_summary = consolidated_data.get('final_name_summary', [])
        
        stage5_names = len(stage5_summary)
        consolidated_names = len(consolidated_summary)
        
        print(f"\nUnique theory names:")
        print(f"  Stage5 (before): {stage5_names}")
        print(f"  Stage6 (after):  {consolidated_names}")
        print(f"  Increase: +{consolidated_names - stage5_names} ({((consolidated_names - stage5_names) / stage5_names * 100):.1f}%)")
        
        # Size distribution
        stage5_sizes = [s['total_papers'] for s in stage5_summary]
        consolidated_sizes = [s['total_papers'] for s in consolidated_summary]
        
        print(f"\nCluster size distribution:")
        print(f"  Stage5 - Largest: {max(stage5_sizes)}, Median: {sorted(stage5_sizes)[len(stage5_sizes)//2]}, Smallest: {min(stage5_sizes)}")
        print(f"  Stage6 - Largest: {max(consolidated_sizes)}, Median: {sorted(consolidated_sizes)[len(consolidated_sizes)//2]}, Smallest: {min(consolidated_sizes)}")
        
        # Count clusters by size
        def count_by_size(sizes):
            return {
                '>100': sum(1 for s in sizes if s > 100),
                '41-100': sum(1 for s in sizes if 41 <= s <= 100),
                '21-40': sum(1 for s in sizes if 21 <= s <= 40),
                '11-20': sum(1 for s in sizes if 11 <= s <= 20),
                '6-10': sum(1 for s in sizes if 6 <= s <= 10),
                '1-5': sum(1 for s in sizes if 1 <= s <= 5)
            }
        
        stage5_dist = count_by_size(stage5_sizes)
        consolidated_dist = count_by_size(consolidated_sizes)
        
        print(f"\nClusters by size range:")
        print(f"  {'Range':<10} {'Stage5':<10} {'Stage6':<10} {'Change'}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for size_range in ['>100', '41-100', '21-40', '11-20', '6-10', '1-5']:
            s5 = stage5_dist[size_range]
            s6 = consolidated_dist[size_range]
            change = s6 - s5
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"  {size_range:<10} {s5:<10} {s6:<10} {change_str}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    analyze_separation_results()
