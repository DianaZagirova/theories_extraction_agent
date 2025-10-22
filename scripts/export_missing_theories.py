#!/usr/bin/env python3
"""
Export missing theories from Stage 6 checkpoints.

Creates detailed reports of which theories are missing from each cluster,
including theory metadata from stage0.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def export_missing_theories(output_dir='output', 
                            stage5_path='output/stage5_consolidated_final_theories.json',
                            stage0_path='output/stage0_filtered_theories.json',
                            output_file='output/stage6_missing_theories.json'):
    """Export all missing theories with metadata."""
    
    print("="*80)
    print("STAGE 6 MISSING THEORIES EXPORT")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading stage5: {stage5_path}")
    stage5_data = load_json(stage5_path)
    
    print(f"📂 Loading stage0: {stage0_path}")
    stage0_data = load_json(stage0_path)
    
    # Build stage0 lookup
    stage0_theories = {}
    for theory in stage0_data.get('theories', []):
        theory_id = theory.get('theory_id')
        if theory_id:
            stage0_theories[theory_id] = theory
    
    print(f"✓ Loaded {len(stage0_theories)} theories from stage0")
    
    # Build stage5 clusters
    stage5_clusters = {}
    for cluster in stage5_data.get('final_name_summary', []):
        cluster_name = cluster['final_name']
        stage5_clusters[cluster_name] = cluster
    
    print(f"✓ Loaded {len(stage5_clusters)} clusters from stage5")
    
    # Find all checkpoint files
    output_path = Path(output_dir)
    checkpoint_files = list(output_path.glob('stage6_checkpoint_*.json'))
    
    if not checkpoint_files:
        print(f"\n❌ No checkpoint files found in {output_dir}")
        return
    
    print(f"\n🔍 Analyzing {len(checkpoint_files)} checkpoint files...")
    
    # Analyze each checkpoint
    all_missing_theories = {}
    total_missing = 0
    clusters_with_missing = 0
    
    for checkpoint_path in sorted(checkpoint_files):
        checkpoint_data = load_json(checkpoint_path)
        cluster_name = checkpoint_data.get('cluster_name')
        batch_results = checkpoint_data.get('batch_results', [])
        
        # Find matching stage5 cluster
        stage5_cluster = None
        for s5_name, s5_info in stage5_clusters.items():
            if s5_name.replace(' ', '_').replace('/', '_').replace(':', '') == checkpoint_path.stem.replace('stage6_checkpoint_', ''):
                stage5_cluster = s5_info
                cluster_name = s5_name
                break
        
        if not stage5_cluster:
            continue
        
        # Get expected theory IDs
        expected_theory_ids = set(stage5_cluster.get('theory_ids', []))
        
        # Collect processed theory IDs
        processed_theory_ids = set()
        for batch_result in batch_results:
            for subcluster in batch_result.get('subclusters', []):
                processed_theory_ids.update(subcluster.get('theory_ids', []))
        
        # Find missing theories
        missing_theory_ids = expected_theory_ids - processed_theory_ids
        
        if missing_theory_ids:
            clusters_with_missing += 1
            total_missing += len(missing_theory_ids)
            
            # Get metadata for missing theories
            missing_theories_with_metadata = []
            for theory_id in sorted(missing_theory_ids):
                theory_metadata = stage0_theories.get(theory_id, {})
                
                missing_theories_with_metadata.append({
                    'theory_id': theory_id,
                    'name': theory_metadata.get('name', 'Unknown'),
                    'paper_title': theory_metadata.get('paper_title', 'Unknown'),
                    'doi': theory_metadata.get('doi', 'Unknown'),
                    'key_concepts': [c.get('concept', '') for c in theory_metadata.get('key_concepts', [])]
                })
            
            all_missing_theories[cluster_name] = {
                'cluster_name': cluster_name,
                'expected_theories': len(expected_theory_ids),
                'processed_theories': len(processed_theory_ids),
                'missing_count': len(missing_theory_ids),
                'missing_percentage': (len(missing_theory_ids) / len(expected_theory_ids) * 100) if expected_theory_ids else 0,
                'missing_theories': missing_theories_with_metadata
            }
            
            print(f"  ⚠️  {cluster_name}: {len(missing_theory_ids)}/{len(expected_theory_ids)} missing ({len(missing_theory_ids)/len(expected_theory_ids)*100:.1f}%)")
    
    # Save to JSON
    output_data = {
        'metadata': {
            'total_clusters_analyzed': len(checkpoint_files),
            'clusters_with_missing_theories': clusters_with_missing,
            'total_missing_theories': total_missing,
            'source_stage5': stage5_path,
            'source_stage0': stage0_path,
            'source_checkpoints': output_dir
        },
        'clusters': all_missing_theories
    }
    
    output_path = Path(output_file)
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved")
    
    # Also create a simple CSV for easy viewing
    csv_path = output_path.with_suffix('.csv')
    print(f"\n💾 Creating CSV summary: {csv_path}...")
    
    with open(csv_path, 'w') as f:
        f.write("cluster_name,theory_id,theory_name,paper_title,doi\n")
        for cluster_name, cluster_data in sorted(all_missing_theories.items()):
            for theory in cluster_data['missing_theories']:
                # Escape commas in fields
                name = theory['name'].replace(',', ';')
                title = theory['paper_title'].replace(',', ';')
                f.write(f'"{cluster_name}",{theory["theory_id"]},"{name}","{title}",{theory["doi"]}\n')
    
    print(f"✓ Saved")
    
    # Create a summary by cluster
    summary_path = output_path.parent / 'stage6_missing_theories_summary.txt'
    print(f"\n💾 Creating text summary: {summary_path}...")
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 6 MISSING THEORIES SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total clusters analyzed: {len(checkpoint_files)}\n")
        f.write(f"Clusters with missing theories: {clusters_with_missing}\n")
        f.write(f"Total missing theories: {total_missing}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CLUSTERS WITH MISSING THEORIES\n")
        f.write("="*80 + "\n\n")
        
        # Sort by missing percentage (descending)
        sorted_clusters = sorted(all_missing_theories.items(), 
                                key=lambda x: x[1]['missing_percentage'], 
                                reverse=True)
        
        for cluster_name, cluster_data in sorted_clusters:
            f.write(f"\n{cluster_name}\n")
            f.write(f"  Expected: {cluster_data['expected_theories']} theories\n")
            f.write(f"  Processed: {cluster_data['processed_theories']} theories\n")
            f.write(f"  Missing: {cluster_data['missing_count']} theories ({cluster_data['missing_percentage']:.1f}%)\n")
            f.write(f"\n  Missing Theory IDs:\n")
            
            for theory in cluster_data['missing_theories'][:20]:  # Show first 20
                f.write(f"    - {theory['theory_id']}: {theory['name'][:60]}\n")
            
            if len(cluster_data['missing_theories']) > 20:
                f.write(f"    ... and {len(cluster_data['missing_theories']) - 20} more\n")
            
            f.write("\n" + "-"*80 + "\n")
    
    print(f"✓ Saved")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Clusters analyzed: {len(checkpoint_files)}")
    print(f"Clusters with missing theories: {clusters_with_missing}")
    print(f"Total missing theories: {total_missing}")
    print(f"\nTop 5 clusters by missing percentage:")
    
    sorted_clusters = sorted(all_missing_theories.items(), 
                            key=lambda x: x[1]['missing_percentage'], 
                            reverse=True)
    
    for i, (cluster_name, cluster_data) in enumerate(sorted_clusters[:5], 1):
        print(f"  {i}. {cluster_name}: {cluster_data['missing_percentage']:.1f}% ({cluster_data['missing_count']}/{cluster_data['expected_theories']})")
    
    print(f"\n📄 Files created:")
    print(f"  - {output_file} (detailed JSON)")
    print(f"  - {csv_path} (CSV for spreadsheet)")
    print(f"  - {summary_path} (human-readable summary)")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export missing theories from Stage 6 checkpoints')
    parser.add_argument('--output-dir', default='output', help='Output directory with checkpoints')
    parser.add_argument('--stage5', default='output/stage5_consolidated_final_theories.json', 
                       help='Stage 5 consolidated file')
    parser.add_argument('--stage0', default='output/stage0_filtered_theories.json',
                       help='Stage 0 theories file')
    parser.add_argument('--output', default='output/stage6_missing_theories.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    if not Path(args.stage5).exists():
        print(f"❌ Error: {args.stage5} not found")
        sys.exit(1)
    
    if not Path(args.stage0).exists():
        print(f"❌ Error: {args.stage0} not found")
        sys.exit(1)
    
    export_missing_theories(args.output_dir, args.stage5, args.stage0, args.output)

if __name__ == '__main__':
    main()
