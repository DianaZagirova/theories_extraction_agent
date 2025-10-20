"""
Stage 3: LLM-Based Metadata Extraction (IMPROVED)

Processes theories from Stage 1.5 and extracts detailed metadata.

Key Improvements:
1. Accepts Stage 1.5 output (not Stage 1)
2. Skips validation (trusts Stage 1.5)
3. Assigns canonical mechanisms to mapped theories
4. Only extracts for novel/unmatched theories
5. Batch processing for efficiency
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient

# Constants
MAX_TOKENS = 4000
BATCH_SIZE = 20


@dataclass
class TheoryMetadata:
    """Extracted metadata for a theory."""
    # Core data
    key_players: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    mechanisms: List[str] = field(default_factory=list)
    
    # Classification
    level_of_explanation: Optional[str] = None
    type_of_cause: Optional[str] = None
    temporal_focus: Optional[str] = None
    adaptiveness: Optional[str] = None
    
    # Metadata
    source: str = 'extracted'  # 'canonical' or 'extracted'
    extraction_confidence: float = 0.0
    
    def to_dict(self):
        return {
            'key_players': self.key_players,
            'pathways': self.pathways,
            'mechanisms': self.mechanisms,
            'level_of_explanation': self.level_of_explanation,
            'type_of_cause': self.type_of_cause,
            'temporal_focus': self.temporal_focus,
            'adaptiveness': self.adaptiveness,
            'source': self.source,
            'extraction_confidence': self.extraction_confidence
        }


class ImprovedLLMExtractor:
    """
    Improved LLM-based metadata extraction.
    Integrates with Stage 1.5 output.
    """
    
    def __init__(self, 
                 ontology_path: str = 'ontology/groups_ontology_alliases.json',
                 mechanisms_path: str = 'ontology/group_ontology_mechanisms.json'):
        """Initialize extractor with ontology."""
        self.ontology_path = Path(ontology_path)
        self.mechanisms_path = Path(mechanisms_path)
        self.canonical_theories = self._load_ontology()
        
        # LLM setup
        self.use_module_norm = os.getenv('USE_MODULE_NORMALIZATION', 'openai')
        if self.use_module_norm == 'openai':
            self.llm = OpenAIClient()
        else:
            self.llm = AzureOpenAIClient()
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        self.stats = {
            'total_theories': 0,
            'mapped_theories': 0,
            'novel_theories': 0,
            'unmatched_theories': 0,
            'extraction_errors': 0,
            'batch_count': 0
        }
    
    def _load_ontology(self) -> Dict[str, Dict]:
        """Load canonical theories with mechanisms."""
        print(f"📂 Loading ontology from {self.mechanisms_path}...")
        
        # Load aliases
        with open(self.ontology_path, 'r') as f:
            aliases_data = json.load(f)
        
        # Load mechanisms (array format)
        with open(self.mechanisms_path, 'r') as f:
            mechanisms_list = json.load(f)
        
        # Convert to dict
        mechanisms_dict = {item['theory_name']: item for item in mechanisms_list}
        
        # Build canonical theories dict
        canonical = {}
        for category, subcats in aliases_data['TheoriesOfAging'].items():
            for subcat, theories in subcats.items():
                for theory in theories:
                    name = theory['name']
                    mech_data = mechanisms_dict.get(name, {})
                    
                    canonical[name] = {
                        'name': name,
                        'category': category,
                        'subcategory': subcat,
                        'mechanisms': mech_data.get('mechanisms', []),
                        'key_players': mech_data.get('key_players', []),
                        'pathways': mech_data.get('pathways', [])
                    }
        
        print(f"✓ Loaded {len(canonical)} canonical theories")
        return canonical
    
    def assign_canonical_mechanisms(self, theory: Dict) -> Dict:
        """
        Assign canonical mechanisms to a mapped theory.
        
        Args:
            theory: Theory with canonical_name from Stage 1 or 1.5
            
        Returns:
            Theory with mechanisms added
        """
        canonical_name = theory.get('match_result', {}).get('canonical_name')
        
        if not canonical_name or canonical_name not in self.canonical_theories:
            print(f"⚠️  Warning: Canonical name '{canonical_name}' not in ontology")
            return theory
        
        canonical_data = self.canonical_theories[canonical_name]
        
        # Create metadata from canonical data
        metadata = TheoryMetadata(
            key_players=canonical_data['key_players'],
            pathways=canonical_data['pathways'],
            mechanisms=canonical_data['mechanisms'],
            source='canonical',
            extraction_confidence=1.0
        )
        
        theory['stage3_metadata'] = metadata.to_dict()
        theory['has_mechanisms'] = True
        
        return theory
    
    def _build_extraction_prompt(self, theories: List[Dict]) -> str:
        """Build batch extraction prompt for novel/unmatched theories."""
        
        theories_text = []
        for theory in theories:
            theory_name = theory.get('name', theory.get('original_name', 'Unknown'))
            concept_text = theory.get('concept_text', 'N/A')
            if len(concept_text) > 300:
                concept_text = concept_text[:300] + "..."
            
            # Check if novel (has proposed name)
            stage1_5_result = theory.get('stage1_5_result', {})
            proposed_name = stage1_5_result.get('proposed_name')
            
            theory_info = f"""
Theory ID: {theory['theory_id']}
Original Name: {theory_name}
{f"Proposed Name: {proposed_name}" if proposed_name else ""}
Context: {concept_text}
"""
            theories_text.append(theory_info)
        
        prompt = f"""Extract comprehensive metadata for these aging theories.

These theories were validated as genuine aging theories but don't match canonical theories in our ontology.

# THEORIES TO PROCESS
{chr(10).join(theories_text)}

# EXTRACTION TASK
For EACH theory, extract:

1. **KEY PLAYERS** (10-20 items): Main actors/components/factors
   - For molecular theories: molecules, proteins, enzymes, cellular components
   - For evolutionary theories: selection pressures, life history traits, population factors
   - For social theories: social factors, psychological factors, institutions
   Be COMPREHENSIVE and SPECIFIC.

2. **PATHWAYS** (3-10 items): Specific biological/evolutionary/social pathways
   - Molecular: mTOR, AMPK, insulin/IGF-1, autophagy, etc.
   - Evolutionary: natural selection, sexual selection, trade-offs, etc.
   - Social: role transitions, social integration, etc.

3. **MECHANISMS** (5-15 items): Detailed causal mechanisms
   - Describe HOW aging occurs according to this theory
   - Be specific and detailed
   - Examples:
     * "Accumulation of somatic mutations in nuclear DNA leads to cellular dysfunction"
     * "Declining force of natural selection with age allows deleterious mutations to accumulate"
     * "Withdrawal from social roles reduces normative expectations and social integration"

4. **LEVEL OF EXPLANATION**: Choose ONE
   Options: Molecular, Cellular, Tissue/Organ, Organismal, Population, Societal

5. **TYPE OF CAUSE**: Choose ONE
   Options: Intrinsic, Extrinsic, Both

6. **TEMPORAL FOCUS**: Choose ONE
   Options: Developmental, Reproductive, Post-reproductive, Lifelong, Late-life, Not-stated

7. **ADAPTIVENESS**: Choose ONE
   Options: Adaptive, Non-adaptive, Both/Context-dependent, Not-stated

8. **EXTRACTION CONFIDENCE**: Your confidence (0.0-1.0)

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown):

{{
  "extractions": [
    {{
      "theory_id": "T000001",
      "key_players": ["...", "...", ...],
      "pathways": ["...", "...", ...],
      "mechanisms": ["...", "...", ...],
      "level_of_explanation": "...",
      "type_of_cause": "...",
      "temporal_focus": "...",
      "adaptiveness": "...",
      "extraction_confidence": 0.0-1.0
    }},
    ...
  ]
}}

Include ALL {len(theories)} theories in output."""
        
        return prompt
    
    def _parse_extraction_response(self, response: str, theories: List[Dict]) -> List[TheoryMetadata]:
        """Parse LLM extraction response."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            extractions = data.get('extractions', [])
            
            # Convert to TheoryMetadata objects
            results = []
            for extraction in extractions:
                metadata = TheoryMetadata(
                    key_players=extraction.get('key_players', []),
                    pathways=extraction.get('pathways', []),
                    mechanisms=extraction.get('mechanisms', []),
                    level_of_explanation=extraction.get('level_of_explanation'),
                    type_of_cause=extraction.get('type_of_cause'),
                    temporal_focus=extraction.get('temporal_focus'),
                    adaptiveness=extraction.get('adaptiveness'),
                    source='extracted',
                    extraction_confidence=extraction.get('extraction_confidence', 0.0)
                )
                results.append(metadata)
            
            # Verify all theories processed
            if len(results) != len(theories):
                print(f"⚠️  Warning: Expected {len(theories)} extractions, got {len(results)}")
                # Pad with empty metadata
                while len(results) < len(theories):
                    results.append(TheoryMetadata(
                        source='extracted',
                        extraction_confidence=0.0
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ Error parsing extraction response: {e}")
            # Return empty metadata for all theories
            return [TheoryMetadata(source='extracted', extraction_confidence=0.0) 
                    for _ in theories]
    
    def extract_batch(self, theories: List[Dict]) -> List[TheoryMetadata]:
        """Extract metadata for a batch of theories."""
        try:
            prompt = self._build_extraction_prompt(theories)
            
            response = self.llm.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a biologist expert in aging theories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=MAX_TOKENS
            )
            
            response_text = response.choices[0].message.content.strip()
            results = self._parse_extraction_response(response_text, theories)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in batch extraction: {e}")
            self.stats['extraction_errors'] += 1
            return [TheoryMetadata(source='extracted', extraction_confidence=0.0) 
                    for _ in theories]
    
    def process_stage1_5_output(self, 
                                stage1_output_path: str,
                                stage1_5_output_path: str,
                                output_path: str,
                                batch_size: int = BATCH_SIZE) -> Dict:
        """
        Process theories from Stage 1 and Stage 1.5.
        
        Strategy:
        1. Mapped theories (Stage 1 + 1.5) → Assign canonical mechanisms
        2. Novel theories (Stage 1.5) → Extract mechanisms
        3. Unmatched theories (Stage 1.5) → Extract mechanisms
        4. Invalid theories (Stage 1.5) → Skip
        
        Args:
            stage1_output_path: Stage 1 output
            stage1_5_output_path: Stage 1.5 output
            output_path: Where to save results
            batch_size: Batch size for extraction
        """
        print("🚀 Starting Stage 3: Improved Metadata Extraction\n")
        
        # Load Stage 1 output
        print(f"📂 Loading Stage 1 output from {stage1_output_path}...")
        with open(stage1_output_path, 'r') as f:
            stage1_data = json.load(f)
        
        stage1_matched = stage1_data.get('matched_theories', [])
        print(f"✓ Loaded {len(stage1_matched)} matched theories from Stage 1")
        
        # Load Stage 1.5 output
        print(f"📂 Loading Stage 1.5 output from {stage1_5_output_path}...")
        with open(stage1_5_output_path, 'r') as f:
            stage1_5_data = json.load(f)
        
        stage1_5_mapped = stage1_5_data.get('mapped_theories', [])
        stage1_5_novel = stage1_5_data.get('novel_theories', [])
        stage1_5_unmatched = stage1_5_data.get('still_unmatched', [])
        
        print(f"✓ Loaded from Stage 1.5:")
        print(f"  - {len(stage1_5_mapped)} mapped theories")
        print(f"  - {len(stage1_5_novel)} novel theories")
        print(f"  - {len(stage1_5_unmatched)} still unmatched theories")
        
        # Update stats
        self.stats['total_theories'] = (len(stage1_matched) + len(stage1_5_mapped) + 
                                        len(stage1_5_novel) + len(stage1_5_unmatched))
        self.stats['mapped_theories'] = len(stage1_matched) + len(stage1_5_mapped)
        self.stats['novel_theories'] = len(stage1_5_novel)
        self.stats['unmatched_theories'] = len(stage1_5_unmatched)
        
        # Step 1: Assign canonical mechanisms to mapped theories
        print(f"\n📋 Step 1: Assigning canonical mechanisms to {self.stats['mapped_theories']} mapped theories...")
        
        mapped_with_mechanisms = []
        for theory in stage1_matched + stage1_5_mapped:
            theory_with_mech = self.assign_canonical_mechanisms(theory)
            mapped_with_mechanisms.append(theory_with_mech)
        
        print(f"✓ Assigned canonical mechanisms to {len(mapped_with_mechanisms)} theories")
        
        # Step 2: Extract mechanisms for novel + unmatched theories
        to_extract = stage1_5_novel + stage1_5_unmatched
        print(f"\n🤖 Step 2: Extracting mechanisms for {len(to_extract)} novel/unmatched theories...")
        
        extracted_theories = []
        if to_extract:
            batches = [to_extract[i:i+batch_size] for i in range(0, len(to_extract), batch_size)]
            print(f"  Processing {len(batches)} batches (batch_size={batch_size})...")
            
            for batch in tqdm(batches, desc="Extracting"):
                self.stats['batch_count'] += 1
                metadata_list = self.extract_batch(batch)
                
                for theory, metadata in zip(batch, metadata_list):
                    theory['stage3_metadata'] = metadata.to_dict()
                    theory['has_mechanisms'] = True
                    extracted_theories.append(theory)
        
        print(f"✓ Extracted mechanisms for {len(extracted_theories)} theories")
        
        # Combine all theories
        all_theories = mapped_with_mechanisms + extracted_theories
        
        # Save results
        print(f"\n💾 Saving {len(all_theories)} theories to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_data = {
            'metadata': {
                'stage': 'stage3_improved_extraction',
                'statistics': self.stats,
                'batch_size': batch_size
            },
            'theories_with_mechanisms': all_theories,
            'summary': {
                'total': len(all_theories),
                'with_canonical_mechanisms': len(mapped_with_mechanisms),
                'with_extracted_mechanisms': len(extracted_theories)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
        
        # Print statistics
        self.print_statistics()
        
        return output_data
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*80)
        print("STAGE 3: IMPROVED EXTRACTION STATISTICS")
        print("="*80)
        print(f"Total theories processed: {self.stats['total_theories']}")
        print(f"\nMechanism assignment:")
        print(f"  Canonical mechanisms: {self.stats['mapped_theories']} ({self.stats['mapped_theories']/self.stats['total_theories']*100:.1f}%)")
        print(f"  Extracted mechanisms: {self.stats['novel_theories'] + self.stats['unmatched_theories']} ({(self.stats['novel_theories'] + self.stats['unmatched_theories'])/self.stats['total_theories']*100:.1f}%)")
        print(f"\nBreakdown:")
        print(f"  Novel theories: {self.stats['novel_theories']}")
        print(f"  Unmatched theories: {self.stats['unmatched_theories']}")
        print(f"\nProcessing:")
        print(f"  Batches processed: {self.stats['batch_count']}")
        print(f"  Extraction errors: {self.stats['extraction_errors']}")
        print("="*80)


def main():
    """Run improved Stage 3 extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 3: Improved metadata extraction')
    parser.add_argument('--stage1', default='output/stage1_fuzzy_matched.json', help='Stage 1 output')
    parser.add_argument('--stage1-5', default='output/stage1_5_llm_mapped.json', help='Stage 1.5 output')
    parser.add_argument('--output', default='output/stage3_extracted_improved.json', help='Output file')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ImprovedLLMExtractor()
    
    # Process theories
    extractor.process_stage1_5_output(
        stage1_output_path=args.stage1,
        stage1_5_output_path=args.stage1_5,
        output_path=args.output,
        batch_size=args.batch_size
    )
    
    print("\n✅ Stage 3 complete!")
    print(f"\nOutput: {args.output}")
    print(f"Next step: Run Stage 4 grouping on theories_with_mechanisms")


if __name__ == '__main__':
    main()
