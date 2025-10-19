"""
Test mechanism extraction with actual LLM calls on a small sample.
This validates the implementation with real data before running on full dataset.
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from normalization.stage2_mechanism_extraction import MechanismExtractor
from normalization.stage3_mechanism_clustering import MechanismClusterer
from core.llm_integration import AzureOpenAIClient


def test_small_sample(sample_size=15):
    """Test mechanism extraction and clustering on small sample."""
    
    print("="*70)
    print("MECHANISM-BASED CLUSTERING: SMALL SAMPLE TEST")
    print("="*70)
    print(f"\nSample size: {sample_size} theories")
    print("This will make actual LLM API calls")
    
    # Estimate cost
    cost_per_theory = 0.02  # ~$0.02 per theory with GPT-4
    estimated_cost = sample_size * cost_per_theory
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    # Confirm
    response = input("\nContinue? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Load theories
    print("\n" + "="*70)
    print("STEP 1: Loading theories")
    print("="*70)
    
    with open('output/stage1_embeddings.json', 'r') as f:
        data = json.load(f)
    
    # Select diverse sample
    theories = data['theories']
    
    # Get theories with different keywords for diversity
    keywords = ['mTOR', 'telomere', 'mitochondrial', 'evolutionary', 'inflammation', 
                'oxidative', 'senescence', 'autophagy', 'DNA', 'protein']
    
    sample_theories = []
    used_indices = set()
    
    # Try to get one theory per keyword
    for keyword in keywords:
        for i, theory in enumerate(theories):
            if i not in used_indices and keyword.lower() in theory['name'].lower():
                sample_theories.append(theory)
                used_indices.add(i)
                break
        if len(sample_theories) >= sample_size:
            break
    
    # Fill remaining with random theories
    if len(sample_theories) < sample_size:
        for i, theory in enumerate(theories):
            if i not in used_indices:
                sample_theories.append(theory)
                used_indices.add(i)
                if len(sample_theories) >= sample_size:
                    break
    
    print(f"✓ Selected {len(sample_theories)} diverse theories")
    print("\nSample theories:")
    for i, theory in enumerate(sample_theories[:5], 1):
        print(f"  {i}. {theory['name'][:80]}...")
    if len(sample_theories) > 5:
        print(f"  ... and {len(sample_theories)-5} more")
    
    # Initialize LLM client
    print("\n" + "="*70)
    print("STEP 2: Initializing LLM client")
    print("="*70)
    
    try:
        llm_client = AzureOpenAIClient()
        print(f"✓ Using model: {llm_client.model}")
    except Exception as e:
        print(f"❌ Error initializing LLM client: {e}")
        print("\nPlease check your .env file contains:")
        print("  AZURE_OPENAI_ENDPOINT=...")
        print("  AZURE_OPENAI_API_KEY=...")
        print("  AZURE_OPENAI_API_VERSION=...")
        print("  OPENAI_MODEL=...")
        return
    
    # Extract mechanisms
    print("\n" + "="*70)
    print("STEP 3: Extracting mechanisms with LLM")
    print("="*70)
    
    extractor = MechanismExtractor(llm_client)
    
    mechanisms = extractor.extract_mechanisms(
        sample_theories,
        batch_size=5,
        save_progress=False
    )
    
    if not mechanisms:
        print("\n❌ No mechanisms extracted. Check LLM responses.")
        return
    
    # Save mechanisms
    os.makedirs('output', exist_ok=True)
    extractor.save_results(mechanisms, 'output/stage2_mechanisms_sample.json')
    
    # Print statistics
    extractor.print_statistics()
    
    # Show sample extractions
    print("\n" + "="*70)
    print("SAMPLE MECHANISM EXTRACTIONS")
    print("="*70)
    
    for i, mech in enumerate(mechanisms[:3], 1):
        theory = next(t for t in sample_theories if t['theory_id'] == mech.theory_id)
        print(f"\n{i}. Theory: {theory['name'][:70]}...")
        print(f"   Primary: {mech.primary_category}")
        print(f"   Secondary: {', '.join(mech.secondary_categories)}")
        print(f"   Mechanisms: {', '.join(mech.specific_mechanisms[:3])}")
        if mech.pathways:
            print(f"   Pathways: {', '.join(mech.pathways[:3])}")
        print(f"   Level: {mech.biological_level}")
        print(f"   Type: {mech.mechanism_type}")
        print(f"   Confidence: {mech.confidence:.2f}")
        print(f"   Reasoning: {mech.reasoning[:100]}...")
    
    # Cluster by mechanisms
    print("\n" + "="*70)
    print("STEP 4: Clustering by mechanisms")
    print("="*70)
    
    clusterer = MechanismClusterer()
    
    # Build taxonomy
    print("\n🔄 Building taxonomy...")
    taxonomy = clusterer.build_taxonomy([m.to_dict() for m in mechanisms])
    
    print("\n📊 Taxonomy Summary:")
    print(f"  Primary categories: {list(taxonomy['primary_categories'].keys())}")
    for primary, secondaries in taxonomy['secondary_categories'].items():
        print(f"  {primary}: {list(secondaries.keys())}")
    
    # Cluster Level 1
    print("\n🔄 Clustering Level 1 (Families)...")
    families = clusterer.cluster_level1_families(
        sample_theories, 
        [m.to_dict() for m in mechanisms]
    )
    
    print(f"\n✓ Created {len(families)} families:")
    for family in families:
        print(f"  - {family.name}: {len(family.theory_ids)} theories")
    
    # Cluster Level 2
    print("\n🔄 Clustering Level 2 (Parents)...")
    parents = clusterer.cluster_level2_parents(
        sample_theories,
        [m.to_dict() for m in mechanisms],
        families
    )
    
    print(f"✓ Created {len(parents)} parents")
    
    # Cluster Level 3
    print("\n🔄 Clustering Level 3 (Children)...")
    children = clusterer.cluster_level3_children(
        sample_theories,
        [m.to_dict() for m in mechanisms],
        parents
    )
    
    print(f"✓ Created {len(children)} children")
    
    # Save results
    clusterer.save_clusters(sample_theories, 'output/stage3_mechanism_clusters_sample.json')
    
    # Print statistics
    clusterer.print_statistics()
    
    # Show detailed family breakdown
    print("\n" + "="*70)
    print("DETAILED FAMILY BREAKDOWN")
    print("="*70)
    
    for family in families:
        print(f"\n{'='*70}")
        print(f"Family: {family.name}")
        print(f"{'='*70}")
        print(f"Theories: {len(family.theory_ids)}")
        print(f"Signature: {family.mechanism_signature['secondary_category']}")
        
        if family.mechanism_signature.get('common_mechanisms'):
            print(f"Common mechanisms: {', '.join(family.mechanism_signature['common_mechanisms'])}")
        
        print("\nTheories in this family:")
        for i, tid in enumerate(family.theory_ids, 1):
            theory = next(t for t in sample_theories if t['theory_id'] == tid)
            mech = next(m for m in mechanisms if m.theory_id == tid)
            print(f"  {i}. {theory['name'][:70]}...")
            print(f"     Mechanisms: {', '.join(mech.specific_mechanisms[:2])}")
            if mech.pathways:
                print(f"     Pathways: {', '.join(mech.pathways[:2])}")
    
    # Analyze coherence
    print("\n" + "="*70)
    print("BIOLOGICAL COHERENCE ANALYSIS")
    print("="*70)
    
    mech_lookup = {m.theory_id: m for m in mechanisms}
    
    for family in families:
        # Check if all theories in family have same primary category
        primary_categories = set()
        secondary_categories = set()
        
        for tid in family.theory_ids:
            mech = mech_lookup.get(tid)
            if mech:
                primary_categories.add(mech.primary_category)
                secondary_categories.update(mech.secondary_categories)
        
        primary_coherence = 1.0 if len(primary_categories) == 1 else 0.5
        secondary_coherence = 1.0 / len(secondary_categories) if secondary_categories else 0
        
        print(f"\n{family.name}:")
        print(f"  Primary categories: {primary_categories}")
        print(f"  Secondary categories: {secondary_categories}")
        print(f"  Primary coherence: {primary_coherence:.2f}")
        print(f"  Secondary coherence: {secondary_coherence:.2f}")
        
        if primary_coherence == 1.0 and secondary_coherence > 0.8:
            print(f"  ✅ Excellent coherence!")
        elif primary_coherence == 1.0:
            print(f"  ✓ Good coherence")
        else:
            print(f"  ⚠️  Mixed categories - may need refinement")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    success_rate = extractor.stats['successful_extractions'] / extractor.stats['total_theories']
    
    print(f"\n✅ Mechanism extraction: {success_rate*100:.1f}% success rate")
    print(f"✅ Clustering: {len(families)} families, {len(parents)} parents, {len(children)} children")
    print(f"✅ Compression: {len(sample_theories)}/{len(children)} = {len(sample_theories)/len(children):.1f}:1")
    
    print("\nGenerated files:")
    print("  - output/stage2_mechanisms_sample.json")
    print("  - output/stage3_mechanism_clusters_sample.json")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review the mechanism extractions above")
    print("   - Are categories correct?")
    print("   - Are mechanisms accurate?")
    print("   - Is confidence reasonable?")
    
    print("\n2. Review the family groupings")
    print("   - Do theories in same family share mechanisms?")
    print("   - Are coherence scores high?")
    print("   - Any misclassifications?")
    
    print("\n3. If results look good, run on full dataset:")
    print("   python run_mechanism_pipeline.py")
    
    print("\n4. Compare with embedding-based approach:")
    print("   python compare_mechanism_vs_embedding.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test mechanism extraction on small sample')
    parser.add_argument('--sample-size', type=int, default=15,
                       help='Number of theories to test (default: 15)')
    
    args = parser.parse_args()
    
    test_small_sample(args.sample_size)
