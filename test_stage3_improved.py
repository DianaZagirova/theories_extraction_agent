"""
Test improved Stage 3 extraction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage3_llm_extraction_improved import ImprovedLLMExtractor


def main():
    print("="*80)
    print("STAGE 3 IMPROVED: METADATA EXTRACTION - TEST RUN")
    print("="*80)
    
    # Initialize extractor
    extractor = ImprovedLLMExtractor()
    
    # Process theories
    results = extractor.process_stage1_5_output(
        stage1_output_path='output/stage1_fuzzy_matched.json',
        stage1_5_output_path='output/stage1_5_llm_mapped_TEST.json',
        output_path='output/stage3_extracted_improved_TEST.json',
        batch_size=10  # Small batch for testing
    )
    
    # Print sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    theories = results['theories_with_mechanisms']
    
    # Show mapped theories with canonical mechanisms
    print("\n📋 Sample Mapped Theories (with canonical mechanisms):")
    print("-"*80)
    mapped = [t for t in theories if t.get('stage3_metadata', {}).get('source') == 'canonical']
    for i, theory in enumerate(mapped[:3], 1):
        theory_name = theory.get('name', theory.get('original_name', 'Unknown'))
        canonical_name = theory.get('match_result', {}).get('canonical_name')
        metadata = theory.get('stage3_metadata', {})
        
        print(f"\n{i}. {theory_name}")
        print(f"   Canonical: {canonical_name}")
        print(f"   Key players ({len(metadata.get('key_players', []))}): {', '.join(metadata.get('key_players', [])[:5])}...")
        print(f"   Mechanisms ({len(metadata.get('mechanisms', []))}): {metadata.get('mechanisms', [])[:2]}")
    
    # Show extracted theories
    print("\n\n🔬 Sample Extracted Theories (novel/unmatched):")
    print("-"*80)
    extracted = [t for t in theories if t.get('stage3_metadata', {}).get('source') == 'extracted']
    for i, theory in enumerate(extracted[:3], 1):
        theory_name = theory.get('name', theory.get('original_name', 'Unknown'))
        metadata = theory.get('stage3_metadata', {})
        
        print(f"\n{i}. {theory_name}")
        print(f"   Confidence: {metadata.get('extraction_confidence', 0):.2f}")
        print(f"   Key players ({len(metadata.get('key_players', []))}): {', '.join(metadata.get('key_players', [])[:5])}...")
        print(f"   Mechanisms ({len(metadata.get('mechanisms', []))}): {len(metadata.get('mechanisms', []))}")
        if metadata.get('mechanisms'):
            print(f"   Sample: {metadata['mechanisms'][0][:100]}...")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
