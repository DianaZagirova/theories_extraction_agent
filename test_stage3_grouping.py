"""
Test Stage 3 theory grouping on real data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage3_theory_grouping import TheoryGrouper


def main():
    print("="*80)
    print("STAGE 3: THEORY GROUPING - TEST RUN")
    print("="*80)
    
    # Initialize grouper
    grouper = TheoryGrouper(
        exact_match_threshold=1.0,
        high_overlap_threshold=0.8
    )
    
    # Load data
    stage1_theories, stage2_theories = grouper.load_data(
        stage1_path='output/stage1_fuzzy_matched.json',
        stage2_path='output/stage2_llm_extracted_TEST.json'  # Use test output
    )
    
    # Group theories
    groups = grouper.group_theories(stage1_theories, stage2_theories)
    
    # Save results
    grouper.save_results(
        stage1_theories, 
        stage2_theories,
        output_path='output/stage3_theory_groups_TEST.json'
    )
    
    # Print statistics
    grouper.print_statistics()
    
    # Print sample groups
    grouper.print_sample_groups(n=15)
    
    print("\n" + "="*80)
    print("✅ Test complete! Check output/stage3_theory_groups_TEST.json")
    print("="*80)


if __name__ == '__main__':
    main()
