"""
Test Stage 1.5 LLM mapping on a small sample.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage1_5_llm_mapping import LLMMapper


def main():
    print("="*80)
    print("STAGE 1.5: LLM MAPPING - TEST RUN (50 theories)")
    print("="*80)
    
    # Initialize mapper
    mapper = LLMMapper(
        ontology_path='ontology/groups_ontology_alliases.json',
        mechanisms_path='ontology/group_ontology_mechanisms.json'
    )
    
    # Process first 50 unmatched theories
    results = mapper.process_unmatched_theories(
        stage1_output_path='output/stage1_fuzzy_matched.json',
        output_path='output/stage1_5_llm_mapped_TEST.json',
        batch_size=25,  # 2 batches of 25
        max_theories=50
    )
    
    print("\n" + "="*80)
    print("✅ Test complete! Check output/stage1_5_llm_mapped_TEST.json")
    print("="*80)
    
    # Show breakdown
    print("\nResults breakdown:")
    print(f"  Mapped to canonical: {len(results['mapped_theories'])}")
    print(f"  Novel theories: {len(results['novel_theories'])}")
    print(f"  Still unmatched: {len(results['still_unmatched'])}")
    print(f"  Invalid: {len(results['invalid_theories'])}")


if __name__ == '__main__':
    main()
