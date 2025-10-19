"""
Stage 2: Mechanism-Based Extraction
Extract structured biological mechanisms from theories using LLM.
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from src.core.llm_integration import AzureOpenAIClient


MECHANISM_EXTRACTION_PROMPT = """You are an expert in aging biology and gerontology. Analyze this aging theory and extract structured biological information.

Theory Name: {theory_name}
Description: {theory_description}

Extract the following information:

1. PRIMARY CATEGORY (choose ONE that best fits):
   - Molecular/Cellular: Theories about molecular or cellular mechanisms (DNA, proteins, metabolism, mitochondria, senescence)
   - Evolutionary: Theories about evolutionary origins of aging (mutation accumulation, pleiotropy, disposable soma, life history)
   - Systemic: Theories about system-level changes (inflammation, immune system, hormones, intercellular communication)
   - Programmed: Theories proposing aging is genetically programmed or developmentally regulated
   - Stochastic: Theories proposing aging is primarily random damage accumulation

2. SECONDARY CATEGORIES (list all that apply):
   For Molecular/Cellular:
   - DNA Damage, Protein Damage, Metabolic Dysregulation, Mitochondrial Dysfunction, Cellular Senescence, Epigenetic Alterations
   
   For Evolutionary:
   - Mutation Accumulation, Antagonistic Pleiotropy, Disposable Soma, Life History Theory, Natural Selection
   
   For Systemic:
   - Inflammation, Immune Dysfunction, Hormonal Changes, Stem Cell Exhaustion, Intercellular Communication
   
   For Programmed:
   - Genetic Program, Developmental Program
   
   For Stochastic:
   - Random Damage, Wear and Tear, Error Accumulation

3. SPECIFIC MECHANISMS (list 2-5 key mechanisms):
   Examples: Nutrient sensing, Autophagy, Telomere shortening, Oxidative stress, Protein misfolding, etc.

4. PATHWAYS (list specific molecular pathways if mentioned):
   Examples: mTOR, AMPK, sirtuins, insulin/IGF-1, p53, NF-κB, PI3K/AKT, etc.

5. KEY MOLECULES (list specific genes/proteins if mentioned):
   Examples: mTOR, AMPK, SIRT1, p16, p21, telomerase, FOXO, etc.

6. BIOLOGICAL LEVEL (choose ONE primary level):
   - Molecular, Cellular, Tissue, Organ, Organism, Population

7. MECHANISM TYPE (choose ONE that best fits):
   - Damage: Accumulation of molecular/cellular damage
   - Hyperfunction: Excessive or continued activity of beneficial processes
   - Loss_of_function: Decline in beneficial activity
   - Dysregulation: Loss of homeostatic control
   - Developmental: Programmed developmental changes

8. KEY CONCEPTS (list 3-5 key biological concepts):
   Examples: Oxidative stress, Proteostasis, Autophagy, Senescence, Inflammation, etc.

Output ONLY valid JSON with this exact structure (no additional text):
{{
  "primary_category": "...",
  "secondary_categories": ["..."],
  "specific_mechanisms": ["...", "..."],
  "pathways": ["..."],
  "molecules": ["..."],
  "biological_level": "...",
  "mechanism_type": "...",
  "key_concepts": ["...", "..."],
  "confidence": 0.0-1.0,
  "reasoning": "Brief 1-2 sentence explanation"
}}"""


@dataclass
class MechanismData:
    """Structured mechanism data for a theory."""
    theory_id: str
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    specific_mechanisms: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)
    molecules: List[str] = field(default_factory=list)
    biological_level: str = ""
    mechanism_type: str = ""
    key_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    
    def to_dict(self):
        return {
            'theory_id': self.theory_id,
            'primary_category': self.primary_category,
            'secondary_categories': self.secondary_categories,
            'specific_mechanisms': self.specific_mechanisms,
            'pathways': self.pathways,
            'molecules': self.molecules,
            'biological_level': self.biological_level,
            'mechanism_type': self.mechanism_type,
            'key_concepts': self.key_concepts,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


class MechanismExtractor:
    """Extract structured mechanisms from theories using LLM."""
    
    def __init__(self, llm_client: AzureOpenAIClient):
        self.llm_client = llm_client
        self.stats = {
            'total_theories': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_confidence': 0.0
        }
    
    def extract_mechanisms(self, theories: List[Dict], 
                          batch_size: int = 10,
                          save_progress: bool = True) -> List[MechanismData]:
        """
        Extract mechanisms for all theories.
        
        Args:
            theories: List of theory dictionaries
            batch_size: Save progress every N theories
            save_progress: Whether to save intermediate results
        """
        print(f"🔄 Extracting mechanisms for {len(theories)} theories...")
        print(f"   Using LLM: {self.llm_client.model}")
        
        self.stats['total_theories'] = len(theories)
        results = []
        confidences = []
        
        for i, theory in enumerate(theories, 1):
            print(f"\r   Progress: {i}/{len(theories)} ({i/len(theories)*100:.1f}%)", end='', flush=True)
            
            mechanism = self._extract_single_theory(theory)
            
            if mechanism:
                results.append(mechanism)
                confidences.append(mechanism.confidence)
                self.stats['successful_extractions'] += 1
            else:
                self.stats['failed_extractions'] += 1
            
            # Save progress periodically
            if save_progress and i % batch_size == 0:
                self._save_progress(results, f'output/mechanisms_progress_{i}.json')
        
        print()  # New line after progress
        
        if confidences:
            self.stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        print(f"✓ Extracted mechanisms for {self.stats['successful_extractions']}/{len(theories)} theories")
        print(f"  Avg confidence: {self.stats['avg_confidence']:.3f}")
        print(f"  Failed: {self.stats['failed_extractions']}")
        
        return results
    
    def _extract_single_theory(self, theory: Dict) -> Optional[MechanismData]:
        """Extract mechanism for a single theory."""
        
        # Prepare description
        description = theory.get('description', '')
        if not description:
            # Use key concepts as fallback
            concepts = theory.get('key_concepts', [])
            # Handle both string and dict formats
            if concepts:
                concept_strs = []
                for c in concepts[:5]:
                    if isinstance(c, dict):
                        concept_strs.append(c.get('concept', str(c)))
                    else:
                        concept_strs.append(str(c))
                description = f"Key concepts: {', '.join(concept_strs)}"
            else:
                description = "No description available"
        
        # Truncate if too long
        if len(description) > 2000:
            description = description[:2000] + "..."
        
        prompt = MECHANISM_EXTRACTION_PROMPT.format(
            theory_name=theory['name'],
            theory_description=description
        )
        
        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "You are an expert in aging biology. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            if response.get('error'):
                print(f"\n   Error for {theory['theory_id']}: {response['error']}")
                return None
            
            # Parse JSON response
            content = response['content'].strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            mechanism_dict = json.loads(content)
            
            # Create MechanismData object
            mechanism = MechanismData(
                theory_id=theory['theory_id'],
                primary_category=mechanism_dict.get('primary_category', 'Unknown'),
                secondary_categories=mechanism_dict.get('secondary_categories', []),
                specific_mechanisms=mechanism_dict.get('specific_mechanisms', []),
                pathways=mechanism_dict.get('pathways', []),
                molecules=mechanism_dict.get('molecules', []),
                biological_level=mechanism_dict.get('biological_level', 'Unknown'),
                mechanism_type=mechanism_dict.get('mechanism_type', 'Unknown'),
                key_concepts=mechanism_dict.get('key_concepts', []),
                confidence=mechanism_dict.get('confidence', 0.5),
                reasoning=mechanism_dict.get('reasoning', '')
            )
            
            return mechanism
            
        except json.JSONDecodeError as e:
            print(f"\n   JSON parse error for {theory['theory_id']}: {e}")
            print(f"   Response: {content[:200]}...")
            return None
        except Exception as e:
            print(f"\n   Unexpected error for {theory['theory_id']}: {e}")
            return None
    
    def _save_progress(self, results: List[MechanismData], filepath: str):
        """Save intermediate results."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump([m.to_dict() for m in results], f, indent=2)
    
    def save_results(self, results: List[MechanismData], output_path: str):
        """Save final results with statistics."""
        print(f"\n💾 Saving mechanisms to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage2_mechanism_extraction',
                'statistics': self.stats,
                'llm_model': self.llm_client.model
            },
            'mechanisms': [m.to_dict() for m in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(results)} mechanism extractions")
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*60)
        print("MECHANISM EXTRACTION STATISTICS")
        print("="*60)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"Successful extractions: {self.stats['successful_extractions']}")
        print(f"Failed extractions: {self.stats['failed_extractions']}")
        print(f"Success rate: {self.stats['successful_extractions']/self.stats['total_theories']*100:.1f}%")
        print(f"Average confidence: {self.stats['avg_confidence']:.3f}")
        print("="*60)


def main():
    """Run mechanism extraction."""
    print("🚀 Starting Stage 2: Mechanism Extraction\n")
    
    # Load theories from Stage 1
    print("📂 Loading theories from Stage 1...")
    with open('output/stage1_embeddings.json', 'r') as f:
        data = json.load(f)
    
    theories = data['theories']
    print(f"✓ Loaded {len(theories)} theories")
    
    # Initialize LLM client
    print("\n🔧 Initializing LLM client...")
    llm_client = AzureOpenAIClient()
    print(f"✓ Using model: {llm_client.model}")
    
    # Initialize extractor
    extractor = MechanismExtractor(llm_client)
    
    # Extract mechanisms
    mechanisms = extractor.extract_mechanisms(
        theories,
        batch_size=10,
        save_progress=True
    )
    
    # Save results
    extractor.save_results(mechanisms, 'output/stage2_mechanisms.json')
    
    # Print statistics
    extractor.print_statistics()
    
    # Print sample results
    print("\n📊 Sample Mechanism Extractions:")
    for i, mech in enumerate(mechanisms[:3], 1):
        print(f"\n{i}. Theory: {next(t['name'] for t in theories if t['theory_id'] == mech.theory_id)}")
        print(f"   Primary: {mech.primary_category}")
        print(f"   Secondary: {', '.join(mech.secondary_categories)}")
        print(f"   Mechanisms: {', '.join(mech.specific_mechanisms[:3])}")
        print(f"   Confidence: {mech.confidence:.2f}")
    
    print("\n✅ Stage 2 (Mechanism Extraction) complete!")
    print("\nNext step: Run stage3_mechanism_clustering.py")


if __name__ == '__main__':
    main()
