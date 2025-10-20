"""
Test Stage 2 LLM extraction on a small sample of theories.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage2_llm_extraction import LLMExtractor


def main():
    print("="*80)
    print("STAGE 2: LLM EXTRACTION - TEST RUN (5 theories)")
    print("="*80)
    
    # Initialize extractor
    extractor = LLMExtractor(
        ontology_path='ontology/groups_ontology_alliases.json'
    )
    
    # Process only 5 theories for testing
    results = extractor.process_unmatched_theories(
        input_path='output/stage1_fuzzy_matched.json',
        output_path='output/stage2_llm_extracted_TEST.json',
        max_theories=50
    )
    
    # Print statistics
    extractor.print_statistics()
    
    # Print sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        metadata = result['stage2_metadata']
        print(f"\n{i}. Theory: {result['original_name']}")
        print(f"   Valid: {metadata['is_valid_theory']}")
        print(f"   Reasoning: {metadata['validation_reasoning']}")
        
        if metadata['is_valid_theory']:
            print(f"   Primary Category: {metadata['primary_category']}")
            print(f"   Secondary: {metadata['secondary_category']}")
            print(f"   Novel: {metadata['is_novel']}")
            if metadata['key_players']:
                print(f"   Key Players ({len(metadata['key_players'])}): {', '.join(metadata['key_players'][:5])}...")
            if metadata['pathways']:
                print(f"   Pathways ({len(metadata['pathways'])}): {', '.join(metadata['pathways'][:3])}...")
            if metadata['mechanisms']:
                print(f"   Mechanisms ({len(metadata['mechanisms'])}): {metadata['mechanisms'][0][:80]}...")
            print(f"   Level: {metadata['level_of_explanation']}")
            print(f"   Confidence: {metadata['extraction_confidence']:.2f}")
    
    print("\n" + "="*80)
    print("✅ Test complete! Check output/stage2_llm_extracted_TEST.json")
    print("="*80)


if __name__ == '__main__':
    main()
