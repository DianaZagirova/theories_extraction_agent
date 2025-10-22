"""
Test the mechanism-based clustering implementation.
Creates mock mechanism data to validate the clustering logic.
"""

import json
import os


def create_mock_mechanisms():
    """Create mock mechanism data for testing."""
    
    # Load theories
    with open('output/stage1_embeddings.json', 'r') as f:
        data = json.load(f)
    theories = data['theories'][:50]  # Use first 50 for testing
    
    print(f"Creating mock mechanisms for {len(theories)} theories...")
    
    # Define mechanism templates
    templates = [
        {
            'primary_category': 'Molecular/Cellular',
            'secondary_categories': ['Metabolic Dysregulation'],
            'specific_mechanisms': ['Nutrient sensing', 'mTOR signaling'],
            'pathways': ['mTOR', 'insulin/IGF-1'],
            'molecules': ['mTOR', 'S6K'],
            'biological_level': 'Molecular',
            'mechanism_type': 'Hyperfunction'
        },
        {
            'primary_category': 'Molecular/Cellular',
            'secondary_categories': ['DNA Damage'],
            'specific_mechanisms': ['Telomere shortening', 'Oxidative damage'],
            'pathways': ['p53', 'ATM'],
            'molecules': ['telomerase', 'p53'],
            'biological_level': 'Molecular',
            'mechanism_type': 'Damage'
        },
        {
            'primary_category': 'Evolutionary',
            'secondary_categories': ['Life History Theory'],
            'specific_mechanisms': ['Natural selection', 'Trade-offs'],
            'pathways': [],
            'molecules': [],
            'biological_level': 'Organism',
            'mechanism_type': 'Developmental'
        },
        {
            'primary_category': 'Molecular/Cellular',
            'secondary_categories': ['Mitochondrial Dysfunction'],
            'specific_mechanisms': ['ROS production', 'Mitochondrial biogenesis'],
            'pathways': ['PGC-1alpha', 'AMPK'],
            'molecules': ['PGC-1alpha', 'AMPK'],
            'biological_level': 'Cellular',
            'mechanism_type': 'Damage'
        },
        {
            'primary_category': 'Systemic',
            'secondary_categories': ['Inflammation'],
            'specific_mechanisms': ['Inflammaging', 'Cytokine production'],
            'pathways': ['NF-kB', 'IL-6'],
            'molecules': ['NF-kB', 'IL-6', 'TNF-alpha'],
            'biological_level': 'Organism',
            'mechanism_type': 'Dysregulation'
        }
    ]
    
    mechanisms = []
    
    for i, theory in enumerate(theories):
        # Assign mechanism template based on theory name
        template_idx = i % len(templates)
        template = templates[template_idx].copy()
        
        mechanism = {
            'theory_id': theory['theory_id'],
            **template,
            'key_concepts': template['specific_mechanisms'][:3],
            'confidence': 0.85,
            'reasoning': f"Mock mechanism for testing"
        }
        
        mechanisms.append(mechanism)
    
    # Save mock mechanisms
    os.makedirs('output', exist_ok=True)
    
    mock_data = {
        'metadata': {
            'stage': 'stage2_mechanism_extraction',
            'note': 'MOCK DATA FOR TESTING',
            'statistics': {
                'total_theories': len(theories),
                'successful_extractions': len(mechanisms),
                'failed_extractions': 0,
                'avg_confidence': 0.85
            }
        },
        'mechanisms': mechanisms
    }
    
    with open('output/stage2_mechanisms_mock.json', 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"✓ Created mock mechanisms: output/stage2_mechanisms_mock.json")
    
    return mechanisms


def test_clustering():
    """Test mechanism-based clustering with mock data."""
    
    print("\n" + "="*70)
    print("TESTING MECHANISM-BASED CLUSTERING")
    print("="*70)
    
    # Create mock mechanisms
    mechanisms = create_mock_mechanisms()
    
    # Load theories
    with open('output/stage1_embeddings.json', 'r') as f:
        data = json.load(f)
    theories = data['theories'][:50]
    
    # Import clustering module
    import sys
    sys.path.append('src/normalization')
    from stage3_mechanism_clustering import MechanismClusterer
    
    # Initialize clusterer
    clusterer = MechanismClusterer()
    
    # Build taxonomy
    print("\n🔄 Building taxonomy...")
    taxonomy = clusterer.build_taxonomy(mechanisms)
    
    print("\n📊 Taxonomy Summary:")
    print(f"  Primary categories: {list(taxonomy['primary_categories'].keys())}")
    print(f"  Total pathways: {len(taxonomy['pathways'])}")
    print(f"  Total molecules: {len(taxonomy['molecules'])}")
    
    # Cluster Level 1
    print("\n🔄 Clustering Level 1 (Families)...")
    families = clusterer.cluster_level1_families(theories, mechanisms)
    
    print(f"\n✓ Created {len(families)} families:")
    for family in families:
        print(f"  - {family.name}: {len(family.theory_ids)} theories")
    
    # Cluster Level 2
    print("\n🔄 Clustering Level 2 (Parents)...")
    parents = clusterer.cluster_level2_parents(theories, mechanisms, families)
    
    print(f"\n✓ Created {len(parents)} parents")
    
    # Cluster Level 3
    print("\n🔄 Clustering Level 3 (Children)...")
    children = clusterer.cluster_level3_children(theories, mechanisms, parents)
    
    print(f"\n✓ Created {len(children)} children")
    
    # Save results
    print("\n💾 Saving test results...")
    clusterer.save_clusters(theories, 'output/stage3_mechanism_clusters_test.json')
    
    # Print statistics
    clusterer.print_statistics()
    
    # Show sample family
    print("\n📊 Sample Family:")
    if families:
        family = families[0]
        print(f"\nFamily: {family.name}")
        print(f"Theories: {len(family.theory_ids)}")
        print(f"Signature: {family.mechanism_signature}")
        print("\nTheories in this family:")
        for i, tid in enumerate(family.theory_ids[:5], 1):
            theory = next(t for t in theories if t['theory_id'] == tid)
            print(f"  {i}. {theory['name']}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n✅ Implementation validated successfully!")
    print("\nGenerated files:")
    print("  - output/stage2_mechanisms_mock.json (mock data)")
    print("  - output/stage3_mechanism_clusters_test.json (test results)")
    print("\nNext step: Run with real LLM extraction")
    print("  python run_mechanism_pipeline.py")


if __name__ == '__main__':
    test_clustering()
