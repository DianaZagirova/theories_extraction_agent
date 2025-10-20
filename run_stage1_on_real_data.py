"""
Run Stage 1 fuzzy matching on real theories_per_paper.json data.
Creates both human-readable output and processed JSON for next stages.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage0_quality_filter import QualityFilter
from src.normalization.stage1_fuzzy_matching import FuzzyMatcher


def main():
    print("="*80)
    print("STAGE 1: FUZZY MATCHING ON REAL DATA")
    print("="*80)
    
    # Step 1: Run Stage 0 first (quality filtering)
    print("\n📋 Step 1: Running Stage 0 - Quality Filtering")
    print("-" * 80)
    
    filter_engine = QualityFilter(llm_client=None)
    theories = filter_engine.load_theories('theories_per_paper.json')
    filtered = filter_engine.filter_by_confidence(theories, validate_medium=False)
    enriched = filter_engine.enrich_theories(filtered)
    filter_engine.save_filtered_theories(enriched, 'output/stage0_filtered_theories.json')
    filter_engine.print_statistics()
    
    # Step 2: Run Stage 1 (fuzzy matching)
    print("\n\n📋 Step 2: Running Stage 1 - Fuzzy Matching")
    print("-" * 80)
    
    matcher = FuzzyMatcher(
        ontology_path='ontology/groups_ontology_alliases.json',
        exact_threshold=100,
        high_confidence_threshold=90,
        min_token_overlap=0.8
    )
    
    results = matcher.process_theories('output/stage0_filtered_theories.json')
    matcher.save_results(results, 'output/stage1_fuzzy_matched.json')
    matcher.print_statistics()
    
    # Step 3: Create human-readable test output
    print("\n\n📋 Step 3: Creating Human-Readable Test Output")
    print("-" * 80)
    
    matched = [r for r in results if r.match_result and r.match_result.matched]
    unmatched = [r for r in results if not r.match_result or not r.match_result.matched]
    
    with open('output/stage1_matching_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 1: FUZZY MATCHING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total theories processed: {len(results)}\n")
        f.write(f"Successfully matched: {len(matched)} ({len(matched)/len(results)*100:.1f}%)\n")
        f.write(f"Unmatched (for LLM): {len(unmatched)} ({len(unmatched)/len(results)*100:.1f}%)\n\n")
        
        # Matched theories section
        f.write("="*80 + "\n")
        f.write("MATCHED THEORIES (Sample: first 100)\n")
        f.write("="*80 + "\n\n")
        
        for i, theory in enumerate(matched[:100], 1):
            f.write(f"{i}. ORIGINAL: {theory.original_name}\n")
            f.write(f"   CANONICAL: {theory.canonical_name}\n")
            f.write(f"   Match Type: {theory.match_result.match_type}\n")
            f.write(f"   Score: {theory.match_result.score:.1f}\n")
            if theory.match_result.matched_alias != theory.original_name:
                f.write(f"   Via Alias: {theory.match_result.matched_alias}\n")
            f.write(f"   Paper: {theory.paper_title[:80]}...\n")
            f.write(f"   DOI: {theory.doi}\n")
            f.write("\n")
        
        if len(matched) > 100:
            f.write(f"\n... and {len(matched) - 100} more matched theories\n\n")
        
        # Unmatched theories section
        f.write("\n" + "="*80 + "\n")
        f.write("UNMATCHED THEORIES (Sample: first 100)\n")
        f.write("="*80 + "\n\n")
        
        for i, theory in enumerate(unmatched[:100], 1):
            f.write(f"{i}. ORIGINAL: {theory.original_name}\n")
            if theory.match_result:
                f.write(f"   Best Match: {theory.match_result.canonical_name} (score: {theory.match_result.score:.1f})\n")
                f.write(f"   Reason: {theory.match_result.reasoning}\n")
            f.write(f"   Paper: {theory.paper_title[:80]}...\n")
            f.write(f"   DOI: {theory.doi}\n")
            f.write("\n")
        
        if len(unmatched) > 100:
            f.write(f"\n... and {len(unmatched) - 100} more unmatched theories\n\n")
        
        # Statistics by match type
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICS BY MATCH TYPE\n")
        f.write("="*80 + "\n\n")
        
        abbr_matches = [r for r in matched if r.match_result.match_type == 'abbreviation']
        exact_matches = [r for r in matched if r.match_result.match_type == 'exact']
        high_conf_matches = [r for r in matched if r.match_result.match_type == 'high_confidence']
        
        f.write(f"Abbreviation matches: {len(abbr_matches)} ({len(abbr_matches)/len(results)*100:.1f}%)\n")
        f.write(f"Exact matches: {len(exact_matches)} ({len(exact_matches)/len(results)*100:.1f}%)\n")
        f.write(f"High confidence fuzzy matches: {len(high_conf_matches)} ({len(high_conf_matches)/len(results)*100:.1f}%)\n")
        f.write(f"No match (needs LLM): {len(unmatched)} ({len(unmatched)/len(results)*100:.1f}%)\n")
        
        # Top canonical theories
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 MOST FREQUENTLY MATCHED CANONICAL THEORIES\n")
        f.write("="*80 + "\n\n")
        
        from collections import Counter
        canonical_counts = Counter([r.canonical_name for r in matched])
        
        for canonical, count in canonical_counts.most_common(20):
            f.write(f"{count:4d} theories → {canonical}\n")
    
    print(f"✓ Saved human-readable report to: output/stage1_matching_report.txt")
    
    # Step 4: Print sample matches
    print("\n\n📋 Step 4: Sample Matches")
    print("-" * 80)
    matcher.print_sample_matches(results, n=20)
    
    print("\n" + "="*80)
    print("✅ STAGE 1 COMPLETE!")
    print("="*80)
    print(f"\nOutput files created:")
    print(f"  1. output/stage0_filtered_theories.json - Filtered theories from Stage 0")
    print(f"  2. output/stage1_fuzzy_matched.json - Matched theories (for next stages)")
    print(f"  3. output/stage1_matching_report.txt - Human-readable report")
    print(f"\nNext step: Use 'unmatched_theories' from stage1_fuzzy_matched.json for LLM processing")


if __name__ == '__main__':
    main()
