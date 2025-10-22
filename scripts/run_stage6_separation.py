"""
Runner script for Stage 6 cluster separation.

Usage:
    # Test on top 3 largest clusters only
    python scripts/run_stage6_separation.py --test --limit 3
    
    # Run on all clusters with >40 papers
    python scripts/run_stage6_separation.py
    
    # Run with custom threshold (e.g., >100 papers)
    python scripts/run_stage6_separation.py --threshold 100
    
    # Run with custom min subcluster size
    python scripts/run_stage6_separation.py --min-size 5
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage6_cluster_separation import Stage6ClusterSeparator


def main():
    parser = argparse.ArgumentParser(
        description='Stage 6: Separate overly general theory clusters'
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
        '--test',
        action='store_true',
        help='Test mode: only process limited number of clusters'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=3,
        help='Number of clusters to process in test mode (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/stage6_separated_clusters.json',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STAGE 6: CLUSTER SEPARATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Paper threshold: >{args.threshold}")
    print(f"  Min subcluster size: {args.min_size}")
    print(f"  Max theories per batch: {args.max_batch}")
    print(f"  Test mode: {args.test}")
    if args.test:
        print(f"  Limit: {args.limit} clusters")
    print(f"  Output: {args.output}")
    print("="*80)
    
    # Create separator
    separator = Stage6ClusterSeparator(
        paper_threshold=args.threshold,
        max_theories_per_batch=args.max_batch,
        min_subcluster_size=args.min_size,
        output_path=args.output
    )
    
    # If test mode, modify to process only limited clusters
    if args.test:
        print(f"\n⚠️  TEST MODE: Processing only top {args.limit} largest clusters")
        
        # Get large clusters
        large_clusters = separator._identify_large_clusters()
        
        if len(large_clusters) > args.limit:
            print(f"   Found {len(large_clusters)} large clusters, limiting to {args.limit}")
            # Keep only top N
            original_stage5 = separator.stage5_data.copy()
            limited_summary = []
            
            for cluster in large_clusters[:args.limit]:
                # Find in original summary
                for summary_item in original_stage5['final_name_summary']:
                    if summary_item['final_name'] == cluster['final_name']:
                        limited_summary.append(summary_item)
                        break
            
            # Replace with limited data
            separator.stage5_data['final_name_summary'] = limited_summary
            
            print(f"   Will process:")
            for i, cluster in enumerate(large_clusters[:args.limit], 1):
                print(f"     {i}. {cluster['final_name']}: {cluster['total_papers']} papers, {cluster['theory_count']} theories")
    
    # Run separation
    separator.run()
    
    print(f"\n✅ Stage 6 complete!")
    print(f"   Output saved to: {args.output}")
    
    if args.test:
        print(f"\n💡 To run on all clusters, execute:")
        print(f"   python scripts/run_stage6_separation.py")


if __name__ == '__main__':
    main()
