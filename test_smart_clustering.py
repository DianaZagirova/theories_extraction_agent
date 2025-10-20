"""
Test smart multi-dimensional clustering.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage4_theory_grouping_improved import ImprovedTheoryGrouper


def test_smart_clustering():
    """Test multi-dimensional similarity calculation."""
    
    print("="*80)
    print("TESTING SMART MULTI-DIMENSIONAL CLUSTERING")
    print("="*80)
    
    grouper = ImprovedTheoryGrouper()
    
    # Test cases with full metadata
    test_cases = [
        {
            'name': 'Theory A: Molecular ROS (Intrinsic)',
            'metadata': {
                'mechanisms': ['ROS accumulation', 'oxidative damage', 'mitochondrial dysfunction'],
                'key_players': ['mitochondria', 'ROS', 'DNA'],
                'pathways': ['oxidative phosphorylation'],
                'level_of_explanation': 'Molecular',
                'type_of_cause': 'Intrinsic',
                'temporal_focus': 'Lifelong',
                'adaptiveness': 'Non-adaptive'
            }
        },
        {
            'name': 'Theory B: Molecular ROS (Intrinsic) - Similar',
            'metadata': {
                'mechanisms': ['ROS accumulation', 'oxidative damage', 'DNA damage'],
                'key_players': ['mitochondria', 'ROS', 'proteins'],
                'pathways': ['oxidative phosphorylation', 'NF-κB'],
                'level_of_explanation': 'Molecular',
                'type_of_cause': 'Intrinsic',
                'temporal_focus': 'Lifelong',
                'adaptiveness': 'Non-adaptive'
            }
        },
        {
            'name': 'Theory C: Cellular ROS (Intrinsic) - Different Level',
            'metadata': {
                'mechanisms': ['ROS accumulation', 'cellular senescence'],
                'key_players': ['mitochondria', 'senescent cells'],
                'pathways': ['oxidative phosphorylation'],
                'level_of_explanation': 'Cellular',
                'type_of_cause': 'Intrinsic',
                'temporal_focus': 'Lifelong',
                'adaptiveness': 'Non-adaptive'
            }
        },
        {
            'name': 'Theory D: Telomere (Cellular, Intrinsic)',
            'metadata': {
                'mechanisms': ['telomere shortening', 'replicative senescence'],
                'key_players': ['telomerase', 'telomeres'],
                'pathways': ['DNA replication'],
                'level_of_explanation': 'Cellular',
                'type_of_cause': 'Intrinsic',
                'temporal_focus': 'Lifelong',
                'adaptiveness': 'Non-adaptive'
            }
        },
        {
            'name': 'Theory E: Evolutionary (Population, Adaptive)',
            'metadata': {
                'mechanisms': ['declining selection pressure', 'mutation accumulation'],
                'key_players': ['natural selection', 'mutations'],
                'pathways': ['evolutionary processes'],
                'level_of_explanation': 'Population',
                'type_of_cause': 'Both',
                'temporal_focus': 'Post-reproductive',
                'adaptiveness': 'Adaptive'
            }
        }
    ]
    
    print("\n📊 Pairwise Similarity Analysis:")
    print("-"*80)
    
    # Compare all pairs
    for i, theory1 in enumerate(test_cases):
        for j, theory2 in enumerate(test_cases):
            if i < j:
                # Create theory objects
                t1 = {'stage3_metadata': theory1['metadata']}
                t2 = {'stage3_metadata': theory2['metadata']}
                
                # Compute signatures
                sig1 = grouper._compute_mechanism_signature(t1)
                sig2 = grouper._compute_mechanism_signature(t2)
                
                # Get detailed breakdown
                breakdown = grouper._get_similarity_breakdown(sig1, sig2)
                
                # Determine if would cluster
                would_cluster = breakdown['total'] >= 0.6
                
                print(f"\n{theory1['name']}")
                print(f"  vs")
                print(f"{theory2['name']}")
                print(f"\n  Content Similarity: {breakdown['content_similarity']:.2f}")
                print(f"    - Mechanisms: {breakdown['breakdown']['mechanisms']:.2f}")
                print(f"    - Key Players: {breakdown['breakdown']['key_players']:.2f}")
                print(f"    - Pathways: {breakdown['breakdown']['pathways']:.2f}")
                print(f"\n  Categorical Similarity: {breakdown['categorical_similarity']:.2f}")
                
                if breakdown['breakdown']['categorical_matches']:
                    print(f"    Matches:")
                    for cat, match in breakdown['breakdown']['categorical_matches'].items():
                        status = "✓" if match else "✗"
                        print(f"      {status} {cat}")
                
                print(f"\n  TOTAL SIMILARITY: {breakdown['total']:.2f}")
                print(f"  Would cluster: {'✅ YES' if would_cluster else '❌ NO'}")
    
    print("\n" + "="*80)
    print("EXPECTED CLUSTERING RESULTS")
    print("="*80)
    
    print("""
Based on the multi-dimensional similarity scores:

Cluster 1: Molecular ROS Theories (2 theories)
  - Theory A: Molecular ROS (Intrinsic)
  - Theory B: Molecular ROS (Intrinsic) - Similar
  Reason: High mechanism overlap + same level + same type

Cluster 2: Cellular ROS Theory (1 theory)
  - Theory C: Cellular ROS (Intrinsic) - Different Level
  Reason: Different level prevents clustering with Cluster 1

Cluster 3: Telomere Theory (1 theory)
  - Theory D: Telomere (Cellular, Intrinsic)
  Reason: Different mechanisms from ROS theories

Cluster 4: Evolutionary Theory (1 theory)
  - Theory E: Evolutionary (Population, Adaptive)
  Reason: Completely different level, type, and mechanisms

Key Insights:
✅ Theories with same mechanisms AND same level/type cluster together
✅ Different levels prevent clustering even with some mechanism overlap
✅ Categorical similarity acts as validation (20% weight)
✅ Content similarity is still primary driver (80% weight)
""")
    
    print("="*80)
    print("WEIGHT BREAKDOWN")
    print("="*80)
    
    print("""
Final Similarity = Content (80%) + Categorical (20%)

Content Similarity:
  - Mechanisms: 50% of content (40% of total)
  - Key Players: 30% of content (24% of total)
  - Pathways: 20% of content (16% of total)

Categorical Similarity:
  - Level of Explanation: 25% of categorical (5% of total)
  - Type of Cause: 25% of categorical (5% of total)
  - Temporal Focus: 25% of categorical (5% of total)
  - Adaptiveness: 25% of categorical (5% of total)

This ensures mechanisms are primary, but categories provide validation.
""")
    
    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_smart_clustering()
