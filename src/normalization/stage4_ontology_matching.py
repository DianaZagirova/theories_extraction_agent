"""
Stage 4: Ontology Integration
Matches normalized theories to known theories in initial_ontology.json
"""

import json
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class OntologyMatcher:
    """Matches theories to known ontology."""
    
    def __init__(self, ontology_path: str, llm_client=None):
        """Initialize with ontology file."""
        self.llm_client = llm_client
        self.ontology = self._load_ontology(ontology_path)
        self.stats = {
            'total_theories': 0,
            'exact_matches': 0,
            'partial_matches': 0,
            'novel_theories': 0
        }
    
    def _load_ontology(self, path: str) -> List[Dict]:
        """Load initial ontology."""
        print(f"📂 Loading ontology from {path}...")
        with open(path, 'r') as f:
            ontology = json.load(f)
        print(f"✓ Loaded {len(ontology)} known theories")
        return ontology
    
    def match_theories(self, children: List[Dict], theories: List[Dict]) -> List[Dict]:
        """Match child theories to ontology."""
        print(f"\n🔍 Matching {len(children)} theories to ontology...")
        
        self.stats['total_theories'] = len(children)
        
        for i, child in enumerate(children):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(children)}")
            
            canonical_name = child.get('canonical_name', '')
            
            # Find best match in ontology
            best_match, confidence = self._find_best_match(canonical_name)
            
            child['ontology_match'] = best_match
            child['ontology_confidence'] = confidence
            
            if confidence > 0.9:
                child['match_type'] = 'exact'
                self.stats['exact_matches'] += 1
            elif confidence > 0.7:
                child['match_type'] = 'partial'
                self.stats['partial_matches'] += 1
            else:
                child['match_type'] = 'novel'
                self.stats['novel_theories'] += 1
        
        print(f"✓ Matching complete")
        print(f"  Exact matches: {self.stats['exact_matches']}")
        print(f"  Partial matches: {self.stats['partial_matches']}")
        print(f"  Novel theories: {self.stats['novel_theories']}")
        
        return children
    
    def _find_best_match(self, theory_name: str) -> Tuple[str, float]:
        """Find best matching theory in ontology."""
        if not theory_name:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        theory_lower = theory_name.lower()
        
        for ont_theory in self.ontology:
            ont_name = ont_theory.get('Theory Name', '')
            if not ont_name:
                continue
            
            # Simple string similarity
            score = self._string_similarity(theory_lower, ont_name.lower())
            
            if score > best_score:
                best_score = score
                best_match = ont_name
        
        return best_match, best_score
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity."""
        # Simple word overlap
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
    
    def save_results(self, families: List[Dict], parents: List[Dict],
                    children: List[Dict], theories: List[Dict], output_path: str):
        """Save ontology matching results."""
        print(f"\n💾 Saving results to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage4_ontology_matching',
                'statistics': self.stats
            },
            'theories': theories,
            'families': families,
            'parents': parents,
            'children': children
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")


def main():
    """Run Stage 4 ontology matching."""
    print("🚀 Starting Stage 4: Ontology Matching\n")
    
    # Load validated theories from Stage 3
    print("📂 Loading validated theories from Stage 3...")
    with open('output/stage3_validated.json', 'r') as f:
        data = json.load(f)
    
    theories = data['theories']
    families = data['families']
    parents = data['parents']
    children = data['children']
    
    print(f"✓ Loaded data\n")
    
    # Initialize matcher
    matcher = OntologyMatcher('ontology/initial_ontology.json')
    
    # Match theories
    children = matcher.match_theories(children, theories)
    
    # Save results
    matcher.save_results(families, parents, children, theories,
                        'output/stage4_ontology_matched.json')
    
    print("\n✅ Stage 4 complete!")


if __name__ == '__main__':
    main()
