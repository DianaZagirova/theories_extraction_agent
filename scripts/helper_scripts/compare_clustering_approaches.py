"""
Compare original vs alternative clustering approaches.
"""

import json
from collections import Counter


def analyze_clustering(data, approach_name):
    """Analyze clustering results."""
    meta = data['metadata']
    stats = meta['statistics']
    
    print(f"\n{'='*70}")
    print(f"{approach_name.upper()}")
    print(f"{'='*70}")
    
    # Overall stats
    print(f"\n📊 Overall Statistics:")
    print(f"  Total theories: {stats['total_theories']}")
    print(f"  Families: {stats['num_families']}")
    print(f"  Parents: {stats['num_parents']}")
    print(f"  Children: {stats['num_children']}")
    
    # Compression
    compression = stats['total_theories'] / stats['num_children']
    print(f"  Compression ratio: {compression:.2f}:1")
    
    # Singletons
    singleton_fam = stats.get('singleton_families', 0)
    singleton_par = stats.get('singleton_parents', 0)
    singleton_chi = stats.get('singleton_children', 0)
    
    print(f"\n📈 Singleton Analysis:")
    print(f"  Singleton families: {singleton_fam} ({singleton_fam/stats['num_families']*100:.1f}%)")
    print(f"  Singleton parents: {singleton_par} ({singleton_par/stats['num_parents']*100:.1f}%)")
    print(f"  Singleton children: {singleton_chi} ({singleton_chi/stats['num_children']*100:.1f}%)")
    
    # Distribution
    families = data['families']
    family_sizes = [f['theory_count'] for f in families]
    
    print(f"\n📊 Distribution:")
    print(f"  Family sizes - Min: {min(family_sizes)}, Max: {max(family_sizes)}, Avg: {sum(family_sizes)/len(family_sizes):.1f}")
    print(f"  Median family size: {sorted(family_sizes)[len(family_sizes)//2]}")
    
    # Large families
    large_families = [f for f in families if f['theory_count'] > 50]
    if large_families:
        print(f"\n⚠️  Large families (>50 theories): {len(large_families)}")
        for fam in large_families[:3]:
            fam_id = fam.get('cluster_id', fam.get('id', 'Unknown'))
            fam_name = fam.get('canonical_name', 'Unnamed')
            print(f"    - {fam_id}: {fam_name} ({fam['theory_count']} theories)")
    
    # Small families
    small_families = [f for f in families if f['theory_count'] <= 3]
    if small_families:
        print(f"  Small families (≤3 theories): {len(small_families)}")
    
    # Coherence (if available)
    if 'avg_coherence_family' in stats and stats['avg_coherence_family'] > 0:
        print(f"\n🎯 Quality Metrics:")
        print(f"  Avg family coherence: {stats['avg_coherence_family']:.3f}")
        if 'avg_coherence_parent' in stats and stats['avg_coherence_parent'] > 0:
            print(f"  Avg parent coherence: {stats['avg_coherence_parent']:.3f}")
        if 'avg_coherence_child' in stats and stats['avg_coherence_child'] > 0:
            print(f"  Avg child coherence: {stats['avg_coherence_child']:.3f}")
        if 'silhouette_score' in stats and stats['silhouette_score'] > 0:
            print(f"  Silhouette score: {stats['silhouette_score']:.3f}")
    
    return {
        'compression': compression,
        'singleton_pct': singleton_chi / stats['num_children'] * 100,
        'max_family_size': max(family_sizes),
        'num_large_families': len(large_families),
        'coherence': stats.get('avg_coherence_family', 0)
    }


def compare_approaches():
    """Compare both clustering approaches."""
    print("="*70)
    print("CLUSTERING APPROACHES COMPARISON")
    print("="*70)
    
    # Load original
    try:
        with open('output/stage2_clusters.json', 'r') as f:
            original = json.load(f)
        original_metrics = analyze_clustering(original, "Original Approach")
    except FileNotFoundError:
        print("\n❌ Original clustering not found")
        print("   Run: python src/normalization/stage2_clustering.py")
        original_metrics = None
    
    # Load alternative
    try:
        with open('output/stage2_clusters_alternative.json', 'r') as f:
            alternative = json.load(f)
        alternative_metrics = analyze_clustering(alternative, "Alternative Approach")
    except FileNotFoundError:
        print("\n❌ Alternative clustering not found")
        print("   Run: python src/normalization/stage2_clustering_alternative.py")
        alternative_metrics = None
    
    # Comparison
    if original_metrics and alternative_metrics:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<30} {'Original':<15} {'Alternative':<15} {'Winner'}")
        print("-"*70)
        
        # Compression
        comp_orig = original_metrics['compression']
        comp_alt = alternative_metrics['compression']
        winner = "Alternative ✅" if comp_alt > comp_orig else "Original ✅"
        print(f"{'Compression ratio':<30} {comp_orig:<15.2f} {comp_alt:<15.2f} {winner}")
        
        # Singletons (lower is better)
        sing_orig = original_metrics['singleton_pct']
        sing_alt = alternative_metrics['singleton_pct']
        winner = "Alternative ✅" if sing_alt < sing_orig else "Original ✅"
        print(f"{'Singleton % (lower better)':<30} {sing_orig:<15.1f} {sing_alt:<15.1f} {winner}")
        
        # Max family size (lower is better)
        max_orig = original_metrics['max_family_size']
        max_alt = alternative_metrics['max_family_size']
        winner = "Alternative ✅" if max_alt < max_orig else "Original ✅"
        print(f"{'Max family size (lower better)':<30} {max_orig:<15} {max_alt:<15} {winner}")
        
        # Large families (fewer is better)
        large_orig = original_metrics['num_large_families']
        large_alt = alternative_metrics['num_large_families']
        winner = "Alternative ✅" if large_alt < large_orig else "Original ✅"
        print(f"{'Large families (fewer better)':<30} {large_orig:<15} {large_alt:<15} {winner}")
        
        # Coherence (higher is better)
        if original_metrics['coherence'] > 0 and alternative_metrics['coherence'] > 0:
            coh_orig = original_metrics['coherence']
            coh_alt = alternative_metrics['coherence']
            winner = "Alternative ✅" if coh_alt > coh_orig else "Original ✅"
            print(f"{'Avg coherence (higher better)':<30} {coh_orig:<15.3f} {coh_alt:<15.3f} {winner}")
        
        # Overall recommendation
        print(f"\n{'='*70}")
        print("RECOMMENDATION")
        print(f"{'='*70}")
        
        alt_wins = 0
        if comp_alt > comp_orig: alt_wins += 1
        if sing_alt < sing_orig: alt_wins += 1
        if max_alt < max_orig: alt_wins += 1
        if large_alt < large_orig: alt_wins += 1
        
        if alt_wins >= 3:
            print("\n✅ RECOMMENDED: Alternative Approach")
            print("   Better compression, fewer singletons, more balanced")
        elif alt_wins >= 2:
            print("\n⚠️  MIXED: Both approaches have merits")
            print("   Consider use case and priorities")
        else:
            print("\n✅ RECOMMENDED: Original Approach")
            print("   Better overall metrics")


if __name__ == '__main__':
    compare_approaches()
