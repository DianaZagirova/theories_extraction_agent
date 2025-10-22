#!/usr/bin/env python3
"""
Validate Stage 6 consolidation - ensure no theories are lost.

Checks:
1. All theories from stage5 are present in stage6 consolidated output
2. No duplicate theory IDs
3. Theory counts match
"""

import json
import sys
from pathlib import Path
from collections import Counter

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_consolidation(stage5_path, stage6_consolidated_path):
    """Validate that all theories from stage5 are in stage6 consolidated."""
    
    print("="*80)
    print("STAGE 6 CONSOLIDATION VALIDATION")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading stage5: {stage5_path}")
    stage5_data = load_json(stage5_path)
    
    print(f"📂 Loading stage6 consolidated: {stage6_consolidated_path}")
    stage6_data = load_json(stage6_consolidated_path)
    
    # Extract all theory IDs from stage5
    stage5_theory_ids = set()
    stage5_clusters = {}  # cluster_name -> theory_ids
    
    for cluster in stage5_data.get('final_name_summary', []):
        cluster_name = cluster['final_name']
        theory_ids = cluster.get('theory_ids', [])
        stage5_theory_ids.update(theory_ids)
        stage5_clusters[cluster_name] = theory_ids
    
    print(f"\n✓ Stage 5: {len(stage5_theory_ids)} unique theory IDs across {len(stage5_clusters)} clusters")
    
    # Extract all theory IDs from stage6 consolidated
    stage6_theory_ids = set()
    stage6_clusters = {}  # cluster_name -> theory_ids
    
    for cluster in stage6_data.get('final_name_summary', []):
        cluster_name = cluster['final_name']
        theory_ids = cluster.get('theory_ids', [])
        stage6_theory_ids.update(theory_ids)
        stage6_clusters[cluster_name] = theory_ids
    
    print(f"✓ Stage 6: {len(stage6_theory_ids)} unique theory IDs across {len(stage6_clusters)} clusters")
    
    # Check for missing theories
    missing_in_stage6 = stage5_theory_ids - stage6_theory_ids
    extra_in_stage6 = stage6_theory_ids - stage5_theory_ids
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    
    if not missing_in_stage6 and not extra_in_stage6:
        print("✅ PASS: All theories accounted for!")
        print(f"   - {len(stage5_theory_ids)} theories in stage5")
        print(f"   - {len(stage6_theory_ids)} theories in stage6 consolidated")
        print(f"   - 0 missing, 0 extra")
    else:
        print("❌ FAIL: Theory mismatch detected!")
        
        if missing_in_stage6:
            print(f"\n⚠️  {len(missing_in_stage6)} theories MISSING in stage6:")
            
            # Find which stage5 clusters they came from
            missing_by_cluster = {}
            for theory_id in missing_in_stage6:
                for cluster_name, theory_ids in stage5_clusters.items():
                    if theory_id in theory_ids:
                        if cluster_name not in missing_by_cluster:
                            missing_by_cluster[cluster_name] = []
                        missing_by_cluster[cluster_name].append(theory_id)
                        break
            
            for cluster_name, theory_ids in sorted(missing_by_cluster.items()):
                print(f"\n   From '{cluster_name}': {len(theory_ids)} theories")
                for tid in sorted(theory_ids)[:10]:  # Show first 10
                    print(f"      - {tid}")
                if len(theory_ids) > 10:
                    print(f"      ... and {len(theory_ids) - 10} more")
        
        if extra_in_stage6:
            print(f"\n⚠️  {len(extra_in_stage6)} EXTRA theories in stage6 (not in stage5):")
            for tid in sorted(list(extra_in_stage6)[:20]):
                print(f"   - {tid}")
            if len(extra_in_stage6) > 20:
                print(f"   ... and {len(extra_in_stage6) - 20} more")
    
    # Check for duplicates in stage6
    all_stage6_ids = []
    for cluster in stage6_data.get('final_name_summary', []):
        all_stage6_ids.extend(cluster.get('theory_ids', []))
    
    duplicates = [tid for tid, count in Counter(all_stage6_ids).items() if count > 1]
    
    if duplicates:
        print(f"\n❌ DUPLICATE THEORIES in stage6: {len(duplicates)} theories appear multiple times")
        for tid in duplicates[:10]:
            count = Counter(all_stage6_ids)[tid]
            print(f"   - {tid}: appears {count} times")
            # Find which clusters
            clusters_with_tid = [name for name, ids in stage6_clusters.items() if tid in ids]
            for cluster_name in clusters_with_tid:
                print(f"      → {cluster_name}")
        if len(duplicates) > 10:
            print(f"   ... and {len(duplicates) - 10} more")
    else:
        print(f"\n✅ No duplicate theories in stage6")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Stage 5 theories: {len(stage5_theory_ids)}")
    print(f"Stage 6 theories: {len(stage6_theory_ids)}")
    print(f"Missing: {len(missing_in_stage6)}")
    print(f"Extra: {len(extra_in_stage6)}")
    print(f"Duplicates: {len(duplicates)}")
    
    if missing_in_stage6 or extra_in_stage6 or duplicates:
        print(f"\n❌ VALIDATION FAILED")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED")
        return True

def main():
    """Main entry point."""
    stage5_path = Path('output/stage5_consolidated_final_theories.json')
    stage6_path = Path('output/stage6_consolidated_final_theories.json')
    
    if not stage5_path.exists():
        print(f"❌ Error: {stage5_path} not found")
        sys.exit(1)
    
    if not stage6_path.exists():
        print(f"❌ Error: {stage6_path} not found")
        sys.exit(1)
    
    success = validate_consolidation(stage5_path, stage6_path)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
