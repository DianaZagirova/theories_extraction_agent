"""
Stage 1: Fuzzy Matching to Known Theories
Maps theory names to known theories from ontology using fuzzy matching.
Only maps when confidence is high to avoid false positives.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from rapidfuzz import fuzz, process
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class MatchResult:
    """Result of fuzzy matching."""
    matched: bool
    canonical_name: Optional[str] = None
    matched_alias: Optional[str] = None
    score: float = 0.0
    match_type: str = ""  # exact, high_confidence, no_match
    reasoning: str = ""


@dataclass
class TheoryMatch:
    """Theory with matching information."""
    theory_id: str
    original_name: str
    canonical_name: Optional[str] = None
    match_result: Optional[MatchResult] = None
    
    # Original theory data
    key_concepts: List[Dict] = field(default_factory=list)
    description: str = ""
    evidence: str = ""
    confidence_is_theory: str = "unknown"
    mode: str = ""
    criteria_reasoning: str = ""
    paper_focus: int = 0
    doi: str = ""
    pmid: str = ""
    paper_title: str = ""
    timestamp: str = ""
    enriched_text: Optional[str] = None
    concept_text: Optional[str] = None
    is_validated: bool = False
    validation_reason: str = ""
    
    def to_dict(self):
        """Convert to dictionary."""
        result = {
            'theory_id': self.theory_id,
            'original_name': self.original_name,
            'canonical_name': self.canonical_name,
            'match_result': {
                'matched': self.match_result.matched if self.match_result else False,
                'canonical_name': self.match_result.canonical_name if self.match_result else None,
                'matched_alias': self.match_result.matched_alias if self.match_result else None,
                'score': self.match_result.score if self.match_result else 0.0,
                'match_type': self.match_result.match_type if self.match_result else 'no_match',
                'reasoning': self.match_result.reasoning if self.match_result else ''
            } if self.match_result else None,
            'key_concepts': self.key_concepts,
            'description': self.description,
            'evidence': self.evidence,
            'confidence_is_theory': self.confidence_is_theory,
            'mode': self.mode,
            'criteria_reasoning': self.criteria_reasoning,
            'paper_focus': self.paper_focus,
            'doi': self.doi,
            'pmid': self.pmid,
            'paper_title': self.paper_title,
            'timestamp': self.timestamp,
            'enriched_text': self.enriched_text,
            'concept_text': self.concept_text,
            'is_validated': self.is_validated,
            'validation_reason': self.validation_reason
        }
        return result


class FuzzyMatcher:
    """
    Fuzzy matcher for theory names.
    
    Strategy:
    1. Exact match (case-insensitive, normalized)
    2. High confidence fuzzy match (>= 95 score)
    3. Token-based matching for compound names
    4. Conservative approach: only match when very confident
    """
    
    def __init__(self, ontology_path: str, 
                 exact_threshold: int = 100,
                 high_confidence_threshold: int = 95,
                 min_token_overlap: float = 0.7):
        """
        Initialize fuzzy matcher.
        
        Args:
            ontology_path: Path to ontology aliases JSON
            exact_threshold: Score for exact match (100)
            high_confidence_threshold: Minimum score for high confidence match (95)
            min_token_overlap: Minimum token overlap ratio for compound names (0.7)
        """
        self.exact_threshold = exact_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.min_token_overlap = min_token_overlap
        
        # Load ontology
        self.canonical_names = []  # List of (canonical_name, category, subcategory)
        self.alias_to_canonical = {}  # Map alias -> canonical_name
        self.all_searchable_names = []  # All names/aliases for fuzzy search
        self.abbreviation_to_canonical = {}  # Map abbreviation -> canonical_name
        
        self._load_ontology(ontology_path)
        
        self.stats = {
            'total_input': 0,
            'abbreviation_matches': 0,
            'exact_matches': 0,
            'high_confidence_matches': 0,
            'no_matches': 0,
            'total_output': 0
        }
    
    def _load_ontology(self, ontology_path: str):
        """Load ontology and build search structures."""
        print(f"📂 Loading ontology from {ontology_path}...")
        
        with open(ontology_path, 'r') as f:
            data = json.load(f)
        
        theories = data.get('TheoriesOfAging', {})
        
        for category, subcategories in theories.items():
            for subcategory, theory_list in subcategories.items():
                for theory_entry in theory_list:
                    canonical = theory_entry.get('name', '')
                    aliases = theory_entry.get('aliases', [])
                    abbreviations = theory_entry.get('abbreviations', [])
                    
                    if not canonical:
                        continue
                    
                    # Store canonical name with metadata
                    self.canonical_names.append((canonical, category, subcategory))
                    
                    # Map canonical name to itself
                    self.alias_to_canonical[self._normalize_name(canonical)] = canonical
                    self.all_searchable_names.append(canonical)
                    
                    # Map all aliases to canonical
                    for alias in aliases:
                        normalized_alias = self._normalize_name(alias)
                        self.alias_to_canonical[normalized_alias] = canonical
                        self.all_searchable_names.append(alias)
                    
                    # Map all abbreviations to canonical (case-insensitive)
                    for abbr in abbreviations:
                        abbr_upper = abbr.upper().strip()
                        self.abbreviation_to_canonical[abbr_upper] = canonical
        
        print(f"✓ Loaded {len(self.canonical_names)} canonical theories")
        print(f"✓ Total searchable names (including aliases): {len(self.all_searchable_names)}")
        print(f"✓ Total abbreviations: {len(self.abbreviation_to_canonical)}")
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize name for comparison.
        Removes common suffixes and variations to focus on core concepts.
        """
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove abbreviations in parentheses/brackets first
        # e.g., "Theory (ABBR)" -> "Theory"
        # This prevents abbreviations from affecting normalization
        normalized = re.sub(r'\s*[\(\[]([A-Z]{2,6}(?:\d*)?)[\)\]]\s*', ' ', normalized, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize punctuation and symbols
        # Replace hyphens with spaces (so "Mutation-Accumulation" = "Mutation Accumulation")
        normalized = normalized.replace('-', ' ')
        normalized = normalized.replace('_', ' ')
        
        # Remove other punctuation
        normalized = re.sub(r'[/\\()\[\]{}]', ' ', normalized)
        
        # Normalize apostrophes - handle both regular (') and smart quotes (', ')
        # U+2019 RIGHT SINGLE QUOTATION MARK, U+2018 LEFT SINGLE QUOTATION MARK
        normalized = normalized.replace('\u2019', "'")  # ' -> '
        normalized = normalized.replace('\u2018', "'")  # ' -> '
        normalized = normalized.replace(''', "'")  # Literal smart quote
        normalized = normalized.replace(''', "'")  # Literal smart quote
        
        # Remove possessives (both 's and s')
        normalized = normalized.replace("'s ", " ")
        normalized = normalized.replace("s' ", " ")
        normalized = normalized.replace("'s", "")
        normalized = normalized.replace("s'", "")
        
        # Clean up extra whitespace after symbol removal
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common theory-related words/phrases
        # Strategy: Remove generic words first, then clean up phrases
        
        # Step 1: Remove single-word generic terms from anywhere
        # Handle both singular and plural forms
        single_word_removals = [
            'hypotheses',  # plural first
            'hypothesis',
            'theories',    # plural first
            'theory',
            'models',      # plural first
            'model',
            'frameworks',  # plural first
            'framework',
            'paradigms',   # plural first
            'paradigm',
            'approaches',  # plural first
            'approach',
            'concepts',    # plural first
            'concept',
            'mechanisms',  # plural first
            'mechanism',
        ]
        
        for word in single_word_removals:
            # Remove as standalone word (with word boundaries)
            normalized = re.sub(r'\b' + word + r'\b', '', normalized)
        
        # Step 2: Remove aging/senescence/ageing
        normalized = re.sub(r'\baging\b', '', normalized)
        normalized = re.sub(r'\bageing\b', '', normalized)  # British spelling
        normalized = re.sub(r'\bsenescence\b', '', normalized)
        
        # Step 3: Clean up common phrases that remain
        phrases_to_remove = [
            'of cellular',
            'of',
            'the',
            'a',
            'an',
        ]
        
        for phrase in phrases_to_remove:
            normalized = re.sub(r'\b' + phrase + r'\b', '', normalized)
        
        # Final cleanup: extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _extract_abbreviations(self, name: str) -> list:
        """
        Extract abbreviations from theory name.
        Looks for patterns like (ABBR) or [ABBR] in the name.
        
        Examples:
            "Dysdifferentiation Hypothesis of Aging and Cancer (DHAC)" -> ["DHAC"]
            "Free Radical Theory (FRT)" -> ["FRT"]
            "Antagonistic Pleiotropy (AP) Theory" -> ["AP"]
        """
        import re
        
        # Pattern to match abbreviations in parentheses or brackets
        # Looks for 2-6 uppercase letters, possibly with numbers
        pattern = r'[\(\[]([A-Z]{2,6}(?:\d*)?)[\)\]]'
        
        matches = re.findall(pattern, name)
        return matches
    
    def _extract_core_tokens(self, name: str) -> set:
        """Extract core meaningful tokens from theory name."""
        normalized = self._normalize_name(name)
        
        # Remove common suffixes
        for suffix in ['theory', 'hypothesis', 'model', 'framework', 'paradigm', 'approach']:
            normalized = normalized.replace(suffix, '')
        
        # Tokenize
        tokens = normalized.split()
        
        # Remove stopwords
        stopwords = {'of', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        core_tokens = {t for t in tokens if t not in stopwords and len(t) > 2}
        
        return core_tokens
    
    def _is_compound_name_match(self, name1: str, name2: str) -> Tuple[bool, float]:
        """
        Check if two names match based on core token overlap.
        
        This handles cases like:
        - "Telomere Theory" vs "Telomere Shortening Theory" (should match)
        - "Telomere Theory" vs "Telomere Theory with Replicative Mosaicism" (should NOT match)
        - "Dedifferentiation" vs "Dysdifferentiation" (should NOT match - different concepts)
        
        Returns:
            (is_match, overlap_ratio)
        """
        tokens1 = self._extract_core_tokens(name1)
        tokens2 = self._extract_core_tokens(name2)
        
        if not tokens1 or not tokens2:
            return False, 0.0
        
        # Special case: Both have only 1 core token (single-word theories)
        # Skip compound validation - let fuzzy score decide
        # This allows high-scoring single-word matches while rejecting low scores
        if len(tokens1) == 1 and len(tokens2) == 1:
            # For single tokens, check if they're similar enough at character level
            # This catches typos but rejects truly different words
            token1 = list(tokens1)[0]
            token2 = list(tokens2)[0]
            
            # Use character-level similarity
            from rapidfuzz import fuzz
            char_similarity = fuzz.ratio(token1, token2)
            
            # If character similarity is very high (>= 93), consider it a match
            # This allows "mitochondrial" vs "mitochondria" (96%) but rejects "dedifferentiation" vs "dysdifferentiation" (91%)
            is_match = char_similarity >= 93
            return is_match, char_similarity / 100.0
        
        # Calculate overlap for multi-token theories
        intersection = tokens1 & tokens2
        smaller_set = min(len(tokens1), len(tokens2))
        larger_set = max(len(tokens1), len(tokens2))
        
        # Overlap ratio based on smaller set
        overlap_ratio = len(intersection) / smaller_set if smaller_set > 0 else 0.0
        
        # Additional check: if one name has significantly more tokens, it's likely a variant
        size_ratio = smaller_set / larger_set if larger_set > 0 else 0.0
        
        # Match if:
        # 1. High overlap (>= min_token_overlap)
        # 2. Size ratio is reasonable (>= 0.6) - prevents matching "X Theory" to "X Theory with Y and Z"
        is_match = (overlap_ratio >= self.min_token_overlap and size_ratio >= 0.6)
        
        return is_match, overlap_ratio
    
    def match_theory(self, theory_name: str) -> MatchResult:
        """
        Match a theory name to canonical ontology.
        
        Args:
            theory_name: Theory name to match
            
        Returns:
            MatchResult with matching information
        """
        normalized_input = self._normalize_name(theory_name)
        
        # Step 0: Check for abbreviation match
        abbreviations = self._extract_abbreviations(theory_name)
        for abbr in abbreviations:
            if abbr in self.abbreviation_to_canonical:
                canonical = self.abbreviation_to_canonical[abbr]
                return MatchResult(
                    matched=True,
                    canonical_name=canonical,
                    matched_alias=theory_name,
                    score=100.0,
                    match_type='abbreviation',
                    reasoning=f'Matched via abbreviation: {abbr}'
                )
        
        # Step 1: Check for exact match
        if normalized_input in self.alias_to_canonical:
            canonical = self.alias_to_canonical[normalized_input]
            return MatchResult(
                matched=True,
                canonical_name=canonical,
                matched_alias=theory_name,
                score=100.0,
                match_type='exact',
                reasoning='Exact match found in ontology'
            )
        
        # Step 2: Fuzzy matching using rapidfuzz
        # Try multiple scorers for better matching
        # token_sort_ratio handles word order, ratio handles exact similarity
        best_match_token = process.extractOne(
            theory_name,
            self.all_searchable_names,
            scorer=fuzz.token_sort_ratio
        )
        
        best_match_ratio = process.extractOne(
            theory_name,
            self.all_searchable_names,
            scorer=fuzz.ratio
        )
        
        # Use the best of both approaches
        best_match = best_match_token
        if best_match_ratio and best_match_ratio[1] > best_match_token[1]:
            best_match = best_match_ratio
        
        if not best_match:
            return MatchResult(
                matched=False,
                match_type='no_match',
                reasoning='No similar theory found in ontology'
            )
        
        matched_name, score, _ = best_match
        canonical = self.alias_to_canonical.get(self._normalize_name(matched_name))
        
        # Check if theory name contains an abbreviation
        # If so, lower threshold to 90% of normal (helps with "Theory (ABBR)" patterns)
        has_abbreviation = bool(self._extract_abbreviations(theory_name))
        effective_threshold = self.high_confidence_threshold * 0.9 if has_abbreviation else self.high_confidence_threshold
        
        # Step 3: High confidence threshold check
        if score >= effective_threshold:
            # Additional validation: check for compound name issues
            is_compound_match, overlap = self._is_compound_name_match(theory_name, matched_name)
            
            # If fuzzy score is very high (>= 97), trust it even if compound check fails
            # This handles cases like "free radical" vs "free-radical" (punctuation differences)
            if score >= 97:
                return MatchResult(
                    matched=True,
                    canonical_name=canonical,
                    matched_alias=matched_name,
                    score=score,
                    match_type='high_confidence',
                    reasoning=f'Very high fuzzy score ({score:.1f}) indicates strong match despite minor variations'
                )
            
            # For scores 95-97, check compound name validation
            if not is_compound_match:
                return MatchResult(
                    matched=False,
                    canonical_name=canonical,
                    matched_alias=matched_name,
                    score=score,
                    match_type='no_match',
                    reasoning=f'High fuzzy score ({score:.1f}) but failed compound name validation (overlap: {overlap:.2f}). Likely a variant theory.'
                )
            
            return MatchResult(
                matched=True,
                canonical_name=canonical,
                matched_alias=matched_name,
                score=score,
                match_type='high_confidence',
                reasoning=f'High confidence fuzzy match (score: {score:.1f})'
            )
        
        # Step 4: Below threshold - no match
        threshold_used = effective_threshold if has_abbreviation else self.high_confidence_threshold
        threshold_note = f" (lowered to {effective_threshold:.1f} due to abbreviation)" if has_abbreviation else ""
        return MatchResult(
            matched=False,
            canonical_name=canonical,
            matched_alias=matched_name,
            score=score,
            match_type='no_match',
            reasoning=f'Best match score ({score:.1f}) below threshold ({threshold_used:.1f}){threshold_note}'
        )
    
    def process_theories(self, input_path: str) -> List[TheoryMatch]:
        """
        Process theories from Stage 0 output.
        
        Args:
            input_path: Path to Stage 0 filtered theories JSON
            
        Returns:
            List of TheoryMatch objects
        """
        print(f"\n📂 Loading theories from {input_path}...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        theories = data.get('theories', [])
        self.stats['total_input'] = len(theories)
        
        print(f"✓ Loaded {len(theories)} theories")
        print(f"\n🔍 Performing fuzzy matching...")
        
        results = []
        
        for i, theory_data in enumerate(theories):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(theories)}")
            
            theory_name = theory_data.get('name', '')
            match_result = self.match_theory(theory_name)
            
            # Update stats
            if match_result.matched:
                if match_result.match_type == 'abbreviation':
                    self.stats['abbreviation_matches'] += 1
                elif match_result.match_type == 'exact':
                    self.stats['exact_matches'] += 1
                else:
                    self.stats['high_confidence_matches'] += 1
            else:
                self.stats['no_matches'] += 1
            
            # Create TheoryMatch object
            theory_match = TheoryMatch(
                theory_id=theory_data.get('theory_id', ''),
                original_name=theory_name,
                canonical_name=match_result.canonical_name if match_result.matched else None,
                match_result=match_result,
                key_concepts=theory_data.get('key_concepts', []),
                description=theory_data.get('description', ''),
                evidence=theory_data.get('evidence', ''),
                confidence_is_theory=theory_data.get('confidence_is_theory', 'unknown'),
                mode=theory_data.get('mode', ''),
                criteria_reasoning=theory_data.get('criteria_reasoning', ''),
                paper_focus=theory_data.get('paper_focus', 0),
                doi=theory_data.get('doi', ''),
                pmid=theory_data.get('pmid', ''),
                paper_title=theory_data.get('paper_title', ''),
                timestamp=theory_data.get('timestamp', ''),
                enriched_text=theory_data.get('enriched_text'),
                concept_text=theory_data.get('concept_text'),
                is_validated=theory_data.get('is_validated', False),
                validation_reason=theory_data.get('validation_reason', '')
            )
            
            results.append(theory_match)
        
        self.stats['total_output'] = len(results)
        
        print(f"\n✓ Matching complete!")
        
        return results
    
    def save_results(self, results: List[TheoryMatch], output_path: str):
        """Save matching results to JSON."""
        print(f"\n💾 Saving {len(results)} matched theories to {output_path}...")
        
        # Separate matched and unmatched
        matched = [r for r in results if r.match_result and r.match_result.matched]
        unmatched = [r for r in results if not r.match_result or not r.match_result.matched]
        
        output_data = {
            'metadata': {
                'stage': 'stage1_fuzzy_matching',
                'statistics': self.stats,
                'matched_count': len(matched),
                'unmatched_count': len(unmatched),
                'match_rate': len(matched) / len(results) * 100 if results else 0
            },
            'matched_theories': [t.to_dict() for t in matched],
            'unmatched_theories': [t.to_dict() for t in unmatched]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print matching statistics."""
        print("\n" + "="*60)
        print("STAGE 1: FUZZY MATCHING STATISTICS")
        print("="*60)
        print(f"Input theories: {self.stats['total_input']}")
        print(f"\nMatching results:")
        print(f"  Abbreviation matches: {self.stats['abbreviation_matches']} ({self.stats['abbreviation_matches']/self.stats['total_input']*100:.1f}%)")
        print(f"  Exact matches: {self.stats['exact_matches']} ({self.stats['exact_matches']/self.stats['total_input']*100:.1f}%)")
        print(f"  High confidence matches: {self.stats['high_confidence_matches']} ({self.stats['high_confidence_matches']/self.stats['total_input']*100:.1f}%)")
        print(f"  No matches: {self.stats['no_matches']} ({self.stats['no_matches']/self.stats['total_input']*100:.1f}%)")
        
        total_matched = self.stats['abbreviation_matches'] + self.stats['exact_matches'] + self.stats['high_confidence_matches']
        print(f"\nTotal matched: {total_matched} ({total_matched/self.stats['total_input']*100:.1f}%)")
        print(f"Remaining for LLM processing: {self.stats['no_matches']}")
        print("="*60)
    
    def print_sample_matches(self, results: List[TheoryMatch], n: int = 10):
        """Print sample matches for inspection."""
        print(f"\n📋 Sample Matches (first {n}):")
        print("-" * 80)
        
        matched = [r for r in results if r.match_result and r.match_result.matched][:n]
        
        for i, theory in enumerate(matched, 1):
            print(f"\n{i}. Original: {theory.original_name}")
            print(f"   Canonical: {theory.canonical_name}")
            print(f"   Match Type: {theory.match_result.match_type}")
            print(f"   Score: {theory.match_result.score:.1f}")
            print(f"   Reasoning: {theory.match_result.reasoning}")


def main():
    """Run Stage 1 fuzzy matching."""
    print("🚀 Starting Stage 1: Fuzzy Matching\n")
    
    # Paths
    ontology_path = 'ontology/groups_ontology_alliases.json'
    input_path = 'output/stage0_filtered_theories.json'
    output_path = 'output/stage1_fuzzy_matched.json'
    
    # Initialize matcher
    matcher = FuzzyMatcher(
        ontology_path=ontology_path,
        exact_threshold=100,
        high_confidence_threshold=90,
        min_token_overlap=0.8
    )
    
    # Process theories
    results = matcher.process_theories(input_path)
    
    # Save results
    matcher.save_results(results, output_path)
    
    # Print statistics
    matcher.print_statistics()
    
    # Print sample matches
    matcher.print_sample_matches(results, n=10)
    
    print("\n✅ Stage 1 complete!")


if __name__ == '__main__':
    main()
