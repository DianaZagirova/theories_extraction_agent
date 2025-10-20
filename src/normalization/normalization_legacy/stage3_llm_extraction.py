"""
Stage 2: LLM-Based Theory Validation and Data Extraction

For theories not matched in Stage 1, use LLM to:
1. Validate if it's a genuine theory of aging
2. Extract detailed metadata and categorization
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
from src.core.llm_integration import AzureOpenAIClient, OpenAIClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Constants
MAX_TOKENS = 3000


@dataclass
class TheoryMetadata:
    """Extracted metadata for a theory."""
    # Validation
    is_valid_theory: bool
    validation_reasoning: str
    
    # Categorization
    primary_category: Optional[str] = None
    secondary_category: Optional[str] = None
    is_novel: bool = False
    novelty_reasoning: Optional[str] = None
    
    # Biological/Molecular details
    key_players: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    mechanisms: List[str] = field(default_factory=list)
    
    # Classification
    level_of_explanation: Optional[str] = None  # Molecular, Cellular, etc.
    type_of_cause: Optional[str] = None  # Intrinsic, Extrinsic, Both
    temporal_focus: Optional[str] = None  # Developmental, Reproductive, etc.
    adaptiveness: Optional[str] = None  # Adaptive, Non-adaptive, etc.
    
    # Confidence
    extraction_confidence: float = 0.0
    
    def to_dict(self):
        return {
            'is_valid_theory': self.is_valid_theory,
            'validation_reasoning': self.validation_reasoning,
            'primary_category': self.primary_category,
            'secondary_category': self.secondary_category,
            'is_novel': self.is_novel,
            'novelty_reasoning': self.novelty_reasoning,
            'key_players': self.key_players,
            'pathways': self.pathways,
            'mechanisms': self.mechanisms,
            'level_of_explanation': self.level_of_explanation,
            'type_of_cause': self.type_of_cause,
            'temporal_focus': self.temporal_focus,
            'adaptiveness': self.adaptiveness,
            'extraction_confidence': self.extraction_confidence
        }


class LLMExtractor:
    """
    LLM-based theory validation and metadata extraction.
    """
    
    def __init__(self, ontology_path: str):
        """
        Initialize LLM extractor.
        
        """
        self.use_module_norm = os.getenv('USE_MODULE_NORMALIZATION', 'openai')
        if self.use_module_norm == 'openai':
            self.llm = OpenAIClient()
        else:
            self.llm = AzureOpenAIClient()
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        
        # Load ontology categories
        self.primary_categories = []
        self.secondary_categories = {}  # primary -> [secondary]
        self._load_ontology_structure(ontology_path)
        
        self.stats = {
            'total_processed': 0,
            'valid_theories': 0,
            'invalid_theories': 0,
            'novel_theories': 0,
            'known_category_theories': 0,
            'extraction_errors': 0
        }
    
    def _load_ontology_structure(self, ontology_path: str):
        """Load primary and secondary categories from ontology."""
        print(f"📂 Loading ontology structure from {ontology_path}...")
        
        with open(ontology_path, 'r') as f:
            data = json.load(f)
        
        theories = data.get('TheoriesOfAging', {})
        
        for primary_category, subcategories in theories.items():
            self.primary_categories.append(primary_category)
            self.secondary_categories[primary_category] = list(subcategories.keys())
        
        print(f"✓ Loaded {len(self.primary_categories)} primary categories")
        print(f"  Categories: {', '.join(self.primary_categories)}")
    
    def _build_validation_prompt(self, theory_name: str, theory_data: Dict) -> str:
        """Build prompt for theory validation and extraction."""
        
        # Extract relevant context
        key_concepts = theory_data.get('key_concepts', [])
        evidence = theory_data.get('evidence', '')
        criteria_reasoning = theory_data.get('criteria_reasoning', '')
        paper_title = theory_data.get('paper_title', '')
        
        # Build concept text
        concept_text = ""
        if key_concepts:
            concept_text = "\n".join([
                f"- {c.get('concept', '')}: {c.get('description', '')}"
                for c in key_concepts[:5]  # Limit to 5 concepts
            ])
        
        prompt = f"""Extract metadata from a theory of aging. 
#TTHEORY DATA
NAME: {theory_name}
KEY CONCEPTS: {concept_text if concept_text else 'Not provided'}
EVIDENCE FROM SOURCE:
{evidence[:500] if evidence else 'Not provided'}
CRITERIA REASONING:
{criteria_reasoning[:500] if criteria_reasoning else 'Not provided'}

#TASK
Extract data on this theory:
1. PRIMARY CATEGORY: Choose ONE that best fits from:
{chr(10).join([f'- {cat}' for cat in self.primary_categories])}

2. SECONDARY CATEGORY: Choose ONE (if possible) that best fits from:
{chr(10).join([f'{primary}: {", ".join(subs)}' for primary, subs in self.secondary_categories.items()])}
Try to select secondary category. If this theory clearly doesn't fit any category, mark as NOVEL and explain why.

3. KEY PLAYERS: List 5-15 main actors relevant to this theory. Be EXTENSIVE and COMPREHENSIVE.
   
   For MOLECULAR/CELLULAR theories, list:
   - Specific molecules: mTOR, AMPK, SIRT1, p16, p21, p53, telomerase, etc
   - Cellular components: mitochondria, telomeres, etc
   - Proteins/enzymes: DNA polymerase, catalase, superoxide dismutase
   - etc
   
   For EVOLUTIONARY theories, list:
   - Selection pressures: predation, extrinsic mortality, reproductive success
   - Evolutionary forces: natural selection, genetic drift, mutation accumulation
   - Life history traits: fertility, longevity, reproductive timing, parental investment
   - Population factors: population size, generation time, mortality rate, etc
   - etc
   
   For SOCIAL/PSYCHOLOGICAL theories, list:
   - Social factors: social roles, social engagement, social support, family structure
   - Psychological factors: self-concept, life satisfaction, coping mechanisms
   - Institutional factors: retirement, healthcare systems, social policies
   - Behavioral factors: activity levels, social participation, role transitions
   - etc

4. PATHWAYS: List specific pathways/processes if mentioned (molecular, evolutionary, or social).
   
   Molecular examples: mTOR, AMPK, sirtuins, insulin/IGF-1, p53, NF-κB, PI3K/AKT, autophagy
   Evolutionary examples: natural selection, sexual selection, kin selection, trade-offs
   Social examples: role transitions, social integration, disengagement processes

5. MECHANISMS: List 3-10 specific mechanisms or processes. Be DETAILED and COMPREHENSIVE.
   
   Examples for molecular theories:
   - "Accumulation of somatic mutations in nuclear DNA"
   - "Mitochondrial dysfunction leading to increased ROS production"
   - "Telomere shortening triggering cellular senescence"
   - "Protein misfolding and aggregation"
   
   Examples for evolutionary theories:
   - "Declining force of natural selection with age"
   - "Trade-off between early reproduction and late-life survival"
   - "Accumulation of late-acting deleterious mutations"
   - "Antagonistic pleiotropy between early and late fitness"
   
   Examples for social theories:
   - "Withdrawal from social roles and relationships"
   - "Loss of meaningful social engagement"
   - "Reduction in normative expectations"
   - "Decreased social interaction frequency"

6. LEVEL OF EXPLANATION: Choose ONE primary level.
   Options: Molecular, Cellular, Tissue/Organ, Organismal, Population, Societal

7. TYPE OF CAUSE: Choose ONE.
   Options: Intrinsic, Extrinsic, Both

8. TEMPORAL FOCUS: Choose ONE.
   Options: Developmental, Reproductive, Post-reproductive, Lifelong, Late-life, Not-stated

9. ADAPTIVENESS: Choose ONE.
   Options: Adaptive, Non-adaptive, Both/Context-dependent, Not-stated

8. EXTRACTION CONFIDENCE: Rate your confidence in this extraction (0.0-1.0).

# OUTPUT FORMAT: 
Respond with ONLY valid JSON (no markdown, no extra text):

{{
  "primary_category": "...",
  "secondary_category": "...",
  "is_novel": true/false,
  "novelty_reasoning": "...",
  "key_players": ["...", "...", "..."],
  "pathways": ["...", "...", "..."],
  "mechanisms": ["...", "...", "..."],
  "level_of_explanation": "...",
  "type_of_cause": "...",
  "temporal_focus": "...",
  "adaptiveness": "...",
  "extraction_confidence": 0.0-1.0
}}"""
        
        return prompt
    
    def extract_metadata(self, theory_name: str, theory_data: Dict) -> TheoryMetadata:
        """
        Extract metadata for a single theory using LLM.
        
        Args:
            theory_name: Name of the theory
            theory_data: Theory data dictionary
            
        Returns:
            TheoryMetadata object
        """
        try:
            prompt = self._build_validation_prompt(theory_name, theory_data)

            response = self.llm.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"""You are a biologist with expertise in aging and senescence. 
                    """},
                {"role": "user", "content": prompt}
                ],
                temperature = 0.2,
                max_tokens=MAX_TOKENS
            )
            # Parse response
            response_text = response.choices[0].message.content.strip()
            

            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            # Create metadata object
            metadata = TheoryMetadata(
                is_valid_theory=data.get('is_valid_theory', True),  # Default to True since we removed validation step
                validation_reasoning=data.get('validation_reasoning', 'Processed by Stage 2'),
                primary_category=data.get('primary_category'),
                secondary_category=data.get('secondary_category'),
                is_novel=data.get('is_novel', False),
                novelty_reasoning=data.get('novelty_reasoning'),
                key_players=data.get('key_players', []),
                pathways=data.get('pathways', []),
                mechanisms=data.get('mechanisms', []),
                level_of_explanation=data.get('level_of_explanation'),
                type_of_cause=data.get('type_of_cause'),
                temporal_focus=data.get('temporal_focus'),
                adaptiveness=data.get('adaptiveness'),
                extraction_confidence=data.get('extraction_confidence', 0.0)
            )
            
            # Update stats
            self.stats['total_processed'] += 1
            if metadata.is_valid_theory:
                self.stats['valid_theories'] += 1
                if metadata.is_novel:
                    self.stats['novel_theories'] += 1
                else:
                    self.stats['known_category_theories'] += 1
            else:
                self.stats['invalid_theories'] += 1
            
            return metadata
            
        except Exception as e:
            print(f"⚠️  Error extracting metadata for '{theory_name}': {e}")
            self.stats['extraction_errors'] += 1
            
            # Return default invalid metadata
            return TheoryMetadata(
                is_valid_theory=False,
                validation_reasoning=f"Extraction error: {str(e)}"
            )
    
    def process_unmatched_theories(self, input_path: str, output_path: str, 
                                   max_theories: Optional[int] = None):
        """
        Process unmatched theories from Stage 1.
        
        Args:
            input_path: Path to stage1_fuzzy_matched.json
            output_path: Path to save stage2 results
            max_theories: Maximum number of theories to process (for testing)
        """
        print(f"\n📂 Loading unmatched theories from {input_path}...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        unmatched = data.get('unmatched_theories', [])
        
        if max_theories:
            unmatched = unmatched[:max_theories]
            print(f"⚠️  Processing only first {max_theories} theories (test mode)")
        
        print(f"✓ Loaded {len(unmatched)} unmatched theories")
        print(f"\n🤖 Starting LLM extraction...")
        
        results = []
        
        for i, theory in enumerate(unmatched, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(unmatched)}")
            
            theory_name = theory.get('original_name', '')
            
            # Extract metadata
            metadata = self.extract_metadata(theory_name, theory)
            
            # Combine original data with extracted metadata
            result = {
                **theory,
                'stage2_metadata': metadata.to_dict(),
                'passed_stage2_validation': metadata.is_valid_theory
            }
            
            results.append(result)
        
        print(f"\n✓ Extraction complete!")
        
        # Save results
        self._save_results(results, output_path)
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save extraction results."""
        print(f"\n💾 Saving {len(results)} theories to {output_path}...")
        
        # Separate valid and invalid
        valid = [r for r in results if r['passed_stage2_validation']]
        invalid = [r for r in results if not r['passed_stage2_validation']]
        
        output_data = {
            'metadata': {
                'stage': 'stage2_llm_extraction',
                'statistics': self.stats,
                'valid_count': len(valid),
                'invalid_count': len(invalid),
                'validation_rate': len(valid) / len(results) * 100 if results else 0
            },
            'valid_theories': valid,
            'invalid_theories': invalid
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*60)
        print("STAGE 2: LLM EXTRACTION STATISTICS")
        print("="*60)
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"\nValidation results:")
        print(f"  Valid theories: {self.stats['valid_theories']} ({self.stats['valid_theories']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Invalid theories: {self.stats['invalid_theories']} ({self.stats['invalid_theories']/self.stats['total_processed']*100:.1f}%)")
        print(f"\nCategorization:")
        print(f"  Novel theories: {self.stats['novel_theories']}")
        print(f"  Known category theories: {self.stats['known_category_theories']}")
        print(f"\nErrors: {self.stats['extraction_errors']}")
        print("="*60)


def main():
    """Run Stage 2 LLM extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2: LLM-based theory validation and extraction')
    parser.add_argument('--input', default='output/stage1_fuzzy_matched.json', help='Input file from Stage 1')
    parser.add_argument('--output', default='output/stage2_llm_extracted.json', help='Output file')
    parser.add_argument('--ontology', default='ontology/groups_ontology_alliases.json', help='Ontology file')
    parser.add_argument('--max-theories', type=int, help='Maximum theories to process (for testing)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Stage 2: LLM Extraction\n")
    
    # Initialize extractor
    extractor = LLMExtractor(
        ontology_path=args.ontology
    )
    
    # Process theories
    results = extractor.process_unmatched_theories(
        input_path=args.input,
        output_path=args.output,
        max_theories=args.max_theories
    )
    
    # Print statistics
    extractor.print_statistics()
    
    print("\n✅ Stage 2 complete!")
    print(f"\nOutput: {args.output}")
    print(f"Next step: Use valid_theories for Stage 3 processing")


if __name__ == '__main__':
    main()
