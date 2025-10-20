"""
Test script for Stage 1 Fuzzy Matching.
Tests various matching scenarios to validate the logic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage1_fuzzy_matching import FuzzyMatcher


def test_matching_scenarios():
    """Test various matching scenarios."""
    print("🧪 Testing Fuzzy Matching Logic\n")
    
    # Initialize matcher
    matcher = FuzzyMatcher(
        ontology_path='ontology/groups_ontology_alliases.json',
        exact_threshold=100,
        high_confidence_threshold=90,
        min_token_overlap=0.8
    )
    
    # Test cases
    test_cases = [
        # (input_name, expected_match, description)
        ("Telomere Theory", True, "Should match 'Telomere Theory' in ontology"),
        ("Telomere Shortening Theory", True, "Should match via alias"),
        ("Telomere Attrition Theory", True, "Should match via alias"),
        ("Telomere hypothesis of cellular aging", True, "Should match - same as Telomere Theory"),
        ("Telomere Theory of Aging with Replicative Mosaicism", False, "Should NOT match - different mechanism"),
        ("Subtelomere-Telomere Theory of Aging", False, "Should NOT match - different theory"),
        
        ("Free Radical Theory", True, "Should match canonical name"),
        ("Free Radical Aging Theory", True, "Should match via alias"),
        ("Harman's Theory", True, "Should match via alias (smart quote in ontology)"),
        ("Free Radical-Rate of Living Theory", False, "Should NOT match - compound theory"),
        
        ("Oxidative Stress Theory", True, "Should match canonical name"),
        ("Oxidative Damage Theory", True, "Should match via alias"),
        ("ROS Theory", True, "Should match via alias"),
        ("Oxidative Stress Hypothesis of Aging", True, "Should match - suffixes removed in normalization"),
        
        ("Mitochondrial Theory of Aging", True, "Should match via alias"),
        ("Mitochondrial Decline Theory", True, "Should match canonical name"),
        ("Mitochondrial Dysfunction Theory", True, "Should match via alias"),
        
        ("Mutation Accumulation Theory", True, "Should match canonical name"),
        ("Mutation Accumulation Hypothesis", True, "Should match via alias"),
        
        ("Disposable Soma Theory", True, "Should match canonical name"),
        ("Disposable Soma Hypothesis", True, "Should match via alias"),
        
        ("Antagonistic Pleiotropy Theory", True, "Should match canonical name"),
        ("Williams' Theory", True, "Should match via alias"),
        ("AP Theory", True, "Should match via alias"),
        
        ("Completely Unknown Theory of Aging", False, "Should NOT match - not in ontology"),
        ("Novel Mechanism Theory", False, "Should NOT match - not in ontology"),
    ]
    
    print("="*80)
    print("MATCHING TEST RESULTS")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for input_name, expected_match, description in test_cases:
        result = matcher.match_theory(input_name)
        actual_match = result.matched
        
        status = "✓ PASS" if actual_match == expected_match else "✗ FAIL"
        
        if actual_match == expected_match:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}")
        print(f"  Input: {input_name}")
        print(f"  Expected: {'MATCH' if expected_match else 'NO MATCH'}")
        print(f"  Actual: {'MATCH' if actual_match else 'NO MATCH'}")
        if result.matched:
            print(f"  Canonical: {result.canonical_name}")
            print(f"  Score: {result.score:.1f}")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Description: {description}")
    
    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*80)
    
    return passed, failed


def test_token_extraction():
    """Test token extraction logic."""
    print("\n\n🧪 Testing Token Extraction\n")
    
    matcher = FuzzyMatcher(
        ontology_path='ontology/groups_ontology_alliases.json'
    )
    
    test_cases = [
        ("Telomere Theory of Aging", {"telomere"}),
        ("Free Radical Theory", {"free", "radical"}),
        ("Oxidative Stress Hypothesis", {"oxidative", "stress"}),
        ("Mitochondrial Decline Theory of Aging", {"mitochondrial", "decline"}),
        ("DNA Damage Accumulation Theory", {"dna", "damage", "accumulation"}),
    ]
    
    print("="*80)
    print("TOKEN EXTRACTION TEST")
    print("="*80)
    
    for input_name, expected_tokens in test_cases:
        actual_tokens = matcher._extract_core_tokens(input_name)
        match = actual_tokens == expected_tokens
        status = "✓" if match else "✗"
        
        print(f"\n{status} Input: {input_name}")
        print(f"  Expected: {expected_tokens}")
        print(f"  Actual: {actual_tokens}")


def test_compound_matching():
    """Test compound name matching logic."""
    print("\n\n🧪 Testing Compound Name Matching\n")
    
    matcher = FuzzyMatcher(
        ontology_path='ontology/groups_ontology_alliases.json',
        min_token_overlap=0.8
    )
    
    test_cases = [
        # (name1, name2, should_match, description)
        ("Telomere Theory", "Telomere Shortening Theory", True, "Simple extension"),
        ("Telomere Theory", "Telomere Theory with Replicative Mosaicism", False, "Complex variant"),
        ("Free Radical Theory", "Free Radical Aging Theory", True, "Simple extension"),
        ("Oxidative Stress Theory", "Oxidative Stress Theory of Aging", True, "Common suffix"),
        ("Mitochondrial Theory", "Mitochondrial-Nuclear Communication Theory", False, "Different mechanism"),
    ]
    
    print("="*80)
    print("COMPOUND NAME MATCHING TEST")
    print("="*80)
    
    for name1, name2, should_match, description in test_cases:
        is_match, overlap = matcher._is_compound_name_match(name1, name2)
        status = "✓" if is_match == should_match else "✗"
        
        print(f"\n{status} {description}")
        print(f"  Name 1: {name1}")
        print(f"  Name 2: {name2}")
        print(f"  Expected: {'MATCH' if should_match else 'NO MATCH'}")
        print(f"  Actual: {'MATCH' if is_match else 'NO MATCH'}")
        print(f"  Overlap: {overlap:.2f}")


if __name__ == '__main__':
    # Run all tests
    test_token_extraction()
    test_compound_matching()
    passed, failed = test_matching_scenarios()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
