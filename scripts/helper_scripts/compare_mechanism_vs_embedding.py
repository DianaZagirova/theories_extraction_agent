"""
Compare mechanism-based clustering vs embedding-based clustering.
Validate biological coherence improvement.
"""

import json
from collections import Counter


def analyze_cluster_diversity(theories_in_cluster, all_theories, mechanisms):
    """Analyze diversity of mechanisms in a cluster."""
    mech_lookup = {m['theory_id']: m for m in mechanisms}
    
    primary_categories = []
    secondary_categories = []
    specific_mechanisms = []
    
    for tid in theories_in_cluster:
        mech = mech_lookup.get(tid)
        if mech:
            primary_categories.append(mech['primary_category'])
            secondary_categories.extend(mech['secondary_categories'])
            specific_mechanisms.extend(mech['specific_mechanisms'])
    
    # Calculate diversity scores
    primary_diversity = len(set(primary_categories)) / max(len(primary_categories), 1)
    secondary_diversity = len(set(secondary_categories)) / max(len(secondary_categories), 1)
    mechanism_diversity = len(set(specific_mechanisms)) / max(len(specific_mechanisms), 1)
    
    return {
        'primary_diversity': primary_diversity,
        'secondary_diversity': secondary_diversity,
        'mechanism_diversity': mechanism_diversity,
        'avg_diversity': (primary_diversity + secondary_diversity + mechanism_diversity) / 3,
        'primary_categories': Counter(primary_categories),
        'secondary_categories': Counter(secondary_categories),
        'specific_mechanisms': Counter(specific_mechanisms)
    }


def compare_approaches():
    """Compare mechanism-based vs embedding-based clustering."""
    
    print("="*70)
    print("MECHANISM-BASED VS EMBEDDING-BASED COMPARISON")
    print("="*70)
    
    # Load mechanism-based results
    try:
        with open('output/stage3_mechanism_clusters.json', 'r') as f:
            mech_data = json.load(f)
        print("\n✓ Loaded mechanism-based clustering")
    except FileNotFoundError:
        print("\n❌ Mechanism-based clustering not found")
        print("   Run: python run_mechanism_pipeline.py")
        return
    
    # Load embedding-based results
    try:
        with open('output/stage2_clusters_alternative.json', 'r') as f:
            emb_data = json.load(f)
        print("✓ Loaded embedding-based clustering")
    except FileNotFoundError:
        print("\n❌ Embedding-based clustering not found")
        print("   Run: python src/normalization/stage2_clustering_alternative.py")
        return
    
    # Load mechanisms for analysis
    try:
        with open('output/stage2_mechanisms.json', 'r') as f:
            mech_extract = json.load(f)
        mechanisms = mech_extract['mechanisms']
        print("✓ Loaded mechanism extractions")
    except FileNotFoundError:
        print("\n❌ Mechanism extractions not found")
        print("   Run mechanism extraction first")
        return
    
    theories = mech_data['theories']
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    # Mechanism-based stats
    mech_stats = mech_data['metadata']['statistics']
    print(f"\nMechanism-Based:")
    print(f"  Families: {mech_stats['num_families']}")
    print(f"  Parents: {mech_stats['num_parents']}")
    print(f"  Children: {mech_stats['num_children']}")
    print(f"  Compression: {mech_stats['total_theories']/mech_stats['num_children']:.2f}:1")
    
    # Embedding-based stats
    emb_stats = emb_data['metadata']['statistics']
    print(f"\nEmbedding-Based:")
    print(f"  Families: {emb_stats['num_families']}")
    print(f"  Parents: {emb_stats['num_parents']}")
    print(f"  Children: {emb_stats['num_children']}")
    print(f"  Compression: {emb_stats['total_theories']/emb_stats['num_children']:.2f}:1")
    
    print("\n" + "="*70)
    print("BIOLOGICAL COHERENCE ANALYSIS")
    print("="*70)
    
    # Analyze mechanism-based families
    print("\n🔍 Mechanism-Based Families (Sample):")
    mech_families = mech_data['families'][:5]
    mech_diversities = []
    
    for family in mech_families:
        diversity = analyze_cluster_diversity(family['theory_ids'], theories, mechanisms)
        mech_diversities.append(diversity['avg_diversity'])
        
        print(f"\n  {family['name']} ({family['theory_count']} theories)")
        print(f"    Primary diversity: {diversity['primary_diversity']:.3f}")
        print(f"    Secondary diversity: {diversity['secondary_diversity']:.3f}")
        print(f"    Mechanism diversity: {diversity['mechanism_diversity']:.3f}")
        print(f"    Avg diversity: {diversity['avg_diversity']:.3f}")
        print(f"    Primary categories: {dict(diversity['primary_categories'])}")
    
    # Analyze embedding-based families
    print("\n🔍 Embedding-Based Families (Sample):")
    emb_families = emb_data['families'][:5]
    emb_diversities = []
    
    for family in emb_families:
        diversity = analyze_cluster_diversity(family['theory_ids'], theories, mechanisms)
        emb_diversities.append(diversity['avg_diversity'])
        
        print(f"\n  {family.get('canonical_name', family['cluster_id'])} ({family['theory_count']} theories)")
        print(f"    Primary diversity: {diversity['primary_diversity']:.3f}")
        print(f"    Secondary diversity: {diversity['secondary_diversity']:.3f}")
        print(f"    Mechanism diversity: {diversity['mechanism_diversity']:.3f}")
        print(f"    Avg diversity: {diversity['avg_diversity']:.3f}")
        print(f"    Primary categories: {dict(diversity['primary_categories'])}")
    
    print("\n" + "="*70)
    print("COHERENCE COMPARISON")
    print("="*70)
    
    avg_mech_diversity = sum(mech_diversities) / len(mech_diversities)
    avg_emb_diversity = sum(emb_diversities) / len(emb_diversities)
    
    # Lower diversity = better coherence
    mech_coherence = 1 - avg_mech_diversity
    emb_coherence = 1 - avg_emb_diversity
    
    print(f"\nMechanism-Based:")
    print(f"  Avg diversity: {avg_mech_diversity:.3f}")
    print(f"  Biological coherence: {mech_coherence:.3f}")
    
    print(f"\nEmbedding-Based:")
    print(f"  Avg diversity: {avg_emb_diversity:.3f}")
    print(f"  Biological coherence: {emb_coherence:.3f}")
    
    improvement = ((mech_coherence - emb_coherence) / emb_coherence) * 100
    
    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    
    if mech_coherence > emb_coherence:
        print(f"\n✅ Mechanism-based is {improvement:.1f}% more coherent")
        print(f"   Biological coherence: {mech_coherence:.3f} vs {emb_coherence:.3f}")
    else:
        print(f"\n⚠️  Embedding-based is slightly more coherent")
        print(f"   This may indicate mechanism extraction needs improvement")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nDiversity Score Interpretation:")
    print("  0.0 = Perfect coherence (all theories same mechanism)")
    print("  0.5 = Moderate diversity")
    print("  1.0 = Maximum diversity (all different mechanisms)")
    print("\nBiological Coherence = 1 - Diversity")
    print("  >0.8 = Excellent (theories very similar)")
    print("  0.6-0.8 = Good (theories related)")
    print("  <0.6 = Poor (theories diverse)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if mech_coherence > 0.7:
        print("\n✅ Mechanism-based clustering achieves good biological coherence")
        print("   Recommended for production use")
    elif mech_coherence > emb_coherence:
        print("\n⚠️  Mechanism-based is better but could be improved")
        print("   Consider refining mechanism extraction")
    else:
        print("\n❌ Mechanism-based needs improvement")
        print("   Review mechanism extraction quality")


def detailed_family_comparison():
    """Compare specific families in detail."""
    print("\n" + "="*70)
    print("DETAILED FAMILY COMPARISON")
    print("="*70)
    
    # Load data
    with open('output/stage3_mechanism_clusters.json', 'r') as f:
        mech_data = json.load(f)
    
    with open('output/stage2_mechanisms.json', 'r') as f:
        mech_extract = json.load(f)
    mechanisms = mech_extract['mechanisms']
    
    theories = {t['theory_id']: t for t in mech_data['theories']}
    mech_lookup = {m['theory_id']: m for m in mechanisms}
    
    # Show a well-clustered family
    print("\n🎯 Example: Well-Clustered Family (Mechanism-Based)")
    
    # Find a family with low diversity
    best_family = None
    best_coherence = 0
    
    for family in mech_data['families']:
        if family['theory_count'] >= 5:
            diversity = analyze_cluster_diversity(family['theory_ids'], theories, mechanisms)
            coherence = 1 - diversity['avg_diversity']
            if coherence > best_coherence:
                best_coherence = coherence
                best_family = family
    
    if best_family:
        print(f"\nFamily: {best_family['name']}")
        print(f"Theories: {best_family['theory_count']}")
        print(f"Coherence: {best_coherence:.3f}")
        print("\nTheories in this family:")
        for i, tid in enumerate(best_family['theory_ids'][:10], 1):
            theory = theories[tid]
            mech = mech_lookup.get(tid)
            print(f"  {i}. {theory['name']}")
            if mech:
                print(f"     Mechanism: {', '.join(mech['specific_mechanisms'][:2])}")


if __name__ == '__main__':
    compare_approaches()
    detailed_family_comparison()
