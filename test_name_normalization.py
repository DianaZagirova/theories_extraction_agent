"""
Test name normalization for novel theories.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage4_theory_grouping_improved import ImprovedTheoryGrouper


def test_name_normalization():
    """Test name similarity calculation."""
    
    print("="*80)
    print("TESTING NAME NORMALIZATION")
    print("="*80)
    
    grouper = ImprovedTheoryGrouper()
    
    # Test cases
    test_cases = [
        # Similar names (should cluster together)
        ("Epigenetic Clock Theory", "Epigenetic Aging Clock"),
        ("Epigenetic Clock Theory", "DNA Methylation Clock Theory"),
        ("Telomere Shortening Theory", "Telomere Theory of Aging"),
        ("Gut Microbiome Aging Theory", "Microbiome Theory of Aging"),
        
        # Different names (should NOT cluster)
        ("Epigenetic Clock Theory", "Telomere Shortening Theory"),
        ("Gut Microbiome Aging Theory", "Free Radical Theory"),
        ("Cellular Senescence Theory", "Mitochondrial Decline Theory"),
    ]
    
    print("\n📊 Name Similarity Tests:")
    print("-"*80)
    
    for name1, name2 in test_cases:
        # Normalize
        norm1 = grouper._normalize_theory_name(name1)
        norm2 = grouper._normalize_theory_name(name2)
        
        # Calculate similarity
        similarity = grouper._calculate_name_similarity(name1, name2)
        
        # Determine if would cluster (threshold 0.7)
        would_cluster = similarity >= 0.7
        
        print(f"\n{name1}")
        print(f"  vs")
        print(f"{name2}")
        print(f"  Normalized: '{norm1}' vs '{norm2}'")
        print(f"  Similarity: {similarity:.2f}")
        print(f"  Would cluster: {'✅ YES' if would_cluster else '❌ NO'}")
    
    print("\n" + "="*80)
    print("TESTING COMBINED SIMILARITY (NAME + MECHANISMS)")
    print("="*80)
    
    # Simulate theories with names and mechanisms
    theories = [
        {
            'name': 'Epigenetic Clock Theory',
            'mechanisms': ['DNA methylation', 'CpG sites', 'aging biomarker', 'biological age'],
            'key_players': ['DNMT', 'TET enzymes', 'methylation sites'],
        },
        {
            'name': 'DNA Methylation Aging Theory',
            'mechanisms': ['DNA methylation', 'epigenetic changes', 'aging biomarker'],
            'key_players': ['DNMT', 'methylation patterns'],
        },
        {
            'name': 'Epigenetic Aging Clock',
            'mechanisms': ['DNA methylation patterns', 'biological age', 'epigenetic drift'],
            'key_players': ['methylation sites', 'CpG islands'],
        },
        {
            'name': 'Telomere Shortening Theory',
            'mechanisms': ['telomere attrition', 'replicative senescence', 'cellular aging'],
            'key_players': ['telomerase', 'telomeres', 'shelterin complex'],
        },
    ]
    
    print("\n📊 Combined Similarity Tests:")
    print("-"*80)
    
    for i, theory1 in enumerate(theories):
        for j, theory2 in enumerate(theories):
            if i < j:
                # Calculate name similarity
                name_sim = grouper._calculate_name_similarity(theory1['name'], theory2['name'])
                
                # Calculate mechanism similarity
                mech1 = set(m.lower() for m in theory1['mechanisms'])
                mech2 = set(m.lower() for m in theory2['mechanisms'])
                mech_sim = grouper._jaccard_similarity(mech1, mech2)
                
                # Combined (70% name, 30% mechanism) - updated to match Stage 4
                combined_sim = name_sim * 0.7 + mech_sim * 0.3
                
                # Would cluster? (threshold now 0.6)
                would_cluster = combined_sim >= 0.6
                
                print(f"\n{theory1['name']}")
                print(f"  vs")
                print(f"{theory2['name']}")
                print(f"  Name similarity: {name_sim:.2f}")
                print(f"  Mechanism similarity: {mech_sim:.2f}")
                print(f"  Combined (70/30): {combined_sim:.2f}")
                print(f"  Would cluster: {'✅ YES' if would_cluster else '❌ NO'}")
    
    print("\n" + "="*80)
    print("EXPECTED CLUSTERS")
    print("="*80)
    
    print("""
Based on the tests above, we expect:

Cluster 1: "Epigenetic Clock Theory" (3 theories)
  - Epigenetic Clock Theory
  - DNA Methylation Aging Theory
  - Epigenetic Aging Clock
  
Cluster 2: "Telomere Shortening Theory" (1 theory)
  - Telomere Shortening Theory

This demonstrates that theories with similar names AND mechanisms
will be grouped together, while different theories remain separate.
""")
    
    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_name_normalization()
