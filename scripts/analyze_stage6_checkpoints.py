#!/usr/bin/env python3
"""
Analyze Stage 6 checkpoint files to identify clusters with issues.

Checks for:
1. Missing theories (theories in stage5 but not in checkpoint)
2. High singleton warning rate
3. Failed batches
4. Incomplete processing

Only returns clusters that need reprocessing.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_checkpoint(checkpoint_path, stage5_cluster_info):
    """
    Analyze a single checkpoint file.
    
    Returns:
        dict with issue information, or None if no issues
    """
    checkpoint_data = load_json(checkpoint_path)
    cluster_name = checkpoint_data.get('cluster_name')
    batch_results = checkpoint_data.get('batch_results', [])
    
    if not batch_results:
        return {
            'cluster_name': cluster_name,
            'issue': 'no_batches',
            'description': 'No batch results found',
            'severity': 'critical'
        }
    
    # Get expected theory IDs from stage5
    expected_theory_ids = set(stage5_cluster_info.get('theory_ids', []))
    
    # Collect all theory IDs from checkpoint
    processed_theory_ids = set()
    singleton_warning_count = 0
    total_theories_in_singletons = 0
    failed_batches = 0
    
    for batch_result in batch_results:
        for subcluster in batch_result.get('subclusters', []):
            theory_ids = subcluster.get('theory_ids', [])
            processed_theory_ids.update(theory_ids)
            
            if subcluster.get('status') == 'singleton_warning':
                singleton_warning_count += 1
                total_theories_in_singletons += len(theory_ids)
    
    # Check for missing theories
    missing_theories = expected_theory_ids - processed_theory_ids
    extra_theories = processed_theory_ids - expected_theory_ids
    
    # Calculate singleton warning rate
    singleton_rate = total_theories_in_singletons / len(expected_theory_ids) if expected_theory_ids else 0
    
    issues = []
    severity = 'ok'
    
    if missing_theories:
        issues.append(f"{len(missing_theories)} theories missing from checkpoint")
        severity = 'critical'
    
    if extra_theories:
        issues.append(f"{len(extra_theories)} extra theories not in stage5")
        severity = 'critical'
    
    if singleton_rate > 0.3:  # More than 30% in singleton warnings
        issues.append(f"{singleton_rate*100:.1f}% theories in singleton warnings ({total_theories_in_singletons}/{len(expected_theory_ids)})")
        if severity != 'critical':
            severity = 'warning'
    
    if len(processed_theory_ids) != len(expected_theory_ids):
        issues.append(f"Theory count mismatch: expected {len(expected_theory_ids)}, got {len(processed_theory_ids)}")
        severity = 'critical'
    
    if issues:
        return {
            'cluster_name': cluster_name,
            'issues': issues,
            'severity': severity,
            'expected_theories': len(expected_theory_ids),
            'processed_theories': len(processed_theory_ids),
            'missing_theories': len(missing_theories),
            'singleton_warnings': singleton_warning_count,
            'theories_in_singletons': total_theories_in_singletons,
            'singleton_rate': singleton_rate,
            'missing_theory_ids': sorted(list(missing_theories))[:10] if missing_theories else []
        }
    
    return None

def analyze_all_checkpoints(output_dir='output', stage5_path='output/stage5_consolidated_final_theories.json'):
    """Analyze all Stage 6 checkpoint files."""
    
    print("="*80)
    print("STAGE 6 CHECKPOINT ANALYSIS")
    print("="*80)
    
    # Load stage5 data
    print(f"\n📂 Loading stage5: {stage5_path}")
    stage5_data = load_json(stage5_path)
    
    # Build mapping: cluster_name -> cluster_info
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
        return []
    
    print(f"\n🔍 Analyzing {len(checkpoint_files)} checkpoint files...")
    
    # Analyze each checkpoint
    clusters_with_issues = []
    clusters_ok = []
    
    for checkpoint_path in sorted(checkpoint_files):
        # Extract cluster name from filename
        filename = checkpoint_path.stem
        cluster_name = filename.replace('stage6_checkpoint_', '').replace('_', ' ')
        
        # Find matching stage5 cluster
        stage5_cluster = None
        for s5_name, s5_info in stage5_clusters.items():
            # Match by cleaned name
            if s5_name.replace(' ', '_').replace('/', '_').replace(':', '') == filename.replace('stage6_checkpoint_', ''):
                stage5_cluster = s5_info
                cluster_name = s5_name
                break
        
        if not stage5_cluster:
            print(f"  ⚠️  Checkpoint {checkpoint_path.name}: No matching stage5 cluster found")
            continue
        
        # Analyze checkpoint
        issue_info = analyze_checkpoint(checkpoint_path, stage5_cluster)
        
        if issue_info:
            clusters_with_issues.append(issue_info)
        else:
            clusters_ok.append(cluster_name)
    
    # Print results
    print(f"\n{'='*80}")
    print("ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    print(f"\n✅ {len(clusters_ok)} clusters OK (no issues)")
    print(f"⚠️  {len(clusters_with_issues)} clusters WITH ISSUES")
    
    if clusters_with_issues:
        # Group by severity
        critical = [c for c in clusters_with_issues if c['severity'] == 'critical']
        warnings = [c for c in clusters_with_issues if c['severity'] == 'warning']
        
        if critical:
            print(f"\n{'='*80}")
            print(f"🚨 CRITICAL ISSUES ({len(critical)} clusters)")
            print(f"{'='*80}")
            
            for cluster_info in critical:
                print(f"\n📌 {cluster_info['cluster_name']}")
                print(f"   Expected: {cluster_info['expected_theories']} theories")
                print(f"   Processed: {cluster_info['processed_theories']} theories")
                print(f"   Missing: {cluster_info['missing_theories']} theories")
                print(f"   Issues:")
                for issue in cluster_info['issues']:
                    print(f"      - {issue}")
                
                if cluster_info['missing_theory_ids']:
                    print(f"   Sample missing IDs: {', '.join(cluster_info['missing_theory_ids'][:5])}")
                    if len(cluster_info['missing_theory_ids']) > 5:
                        print(f"      ... and {len(cluster_info['missing_theory_ids']) - 5} more")
        
        if warnings:
            print(f"\n{'='*80}")
            print(f"⚠️  WARNINGS ({len(warnings)} clusters)")
            print(f"{'='*80}")
            
            for cluster_info in warnings:
                print(f"\n📌 {cluster_info['cluster_name']}")
                print(f"   Theories: {cluster_info['expected_theories']}")
                print(f"   Singleton warnings: {cluster_info['theories_in_singletons']} theories ({cluster_info['singleton_rate']*100:.1f}%)")
                print(f"   Issues:")
                for issue in cluster_info['issues']:
                    print(f"      - {issue}")
    
    # Generate rerun list
    if clusters_with_issues:
        print(f"\n{'='*80}")
        print("CLUSTERS TO RERUN")
        print(f"{'='*80}")
        
        rerun_list = [c['cluster_name'] for c in clusters_with_issues]
        print(f"\n{len(rerun_list)} clusters need reprocessing:\n")
        for i, cluster_name in enumerate(rerun_list, 1):
            severity = next(c['severity'] for c in clusters_with_issues if c['cluster_name'] == cluster_name)
            marker = "🚨" if severity == 'critical' else "⚠️"
            print(f"  {i}. {marker} {cluster_name}")
        
        # Save to file
        rerun_file = Path(output_dir) / 'stage6_clusters_to_rerun.json'
        rerun_data = {
            'total_clusters_analyzed': len(checkpoint_files),
            'clusters_ok': len(clusters_ok),
            'clusters_with_issues': len(clusters_with_issues),
            'critical_issues': len([c for c in clusters_with_issues if c['severity'] == 'critical']),
            'warnings': len([c for c in clusters_with_issues if c['severity'] == 'warning']),
            'clusters_to_rerun': rerun_list,
            'detailed_issues': clusters_with_issues
        }
        
        with open(rerun_file, 'w') as f:
            json.dump(rerun_data, f, indent=2)
        
        print(f"\n💾 Saved rerun list to: {rerun_file}")
        
        return rerun_list
    else:
        print(f"\n✅ All clusters processed successfully - no rerun needed!")
        return []

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Stage 6 checkpoint files')
    parser.add_argument('--output-dir', default='output', help='Output directory with checkpoints')
    parser.add_argument('--stage5', default='output/stage5_consolidated_final_theories.json', 
                       help='Stage 5 consolidated file')
    
    args = parser.parse_args()
    
    if not Path(args.stage5).exists():
        print(f"❌ Error: {args.stage5} not found")
        sys.exit(1)
    
    clusters_to_rerun = analyze_all_checkpoints(args.output_dir, args.stage5)
    
    if clusters_to_rerun:
        print(f"\n💡 To rerun only these clusters, use:")
        print(f"   python scripts/rerun_stage6_clusters.py --clusters-file output/stage6_clusters_to_rerun.json")
        sys.exit(1)  # Exit with error code to indicate issues found
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
