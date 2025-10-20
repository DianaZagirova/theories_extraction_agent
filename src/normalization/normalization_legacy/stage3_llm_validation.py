"""
Stage 3: LLM Validation and Distinction Preservation
Validates clusters, preserves fine-grained distinctions, generates canonical names.
"""

import json
from typing import List, Dict, Tuple
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class LLMValidator:
    """Validates clusters and generates canonical names using LLM."""
    
    def __init__(self, llm_client):
        """Initialize with LLM client."""
        if not hasattr(llm_client, 'generate_response'):
            raise ValueError("LLM client must provide a generate_response() method")
        
        self.llm_client = llm_client
        self.system_prompt = (
            "You are an expert in aging biology who provides clear, structured analyses. "
            "Always respond using the requested format."
        )
        self.stats = {
            'clusters_validated': 0,
            'clusters_split': 0,
            'canonical_names_generated': 0
        }
    
    def validate_and_name_clusters(self, clusters: List[Dict], 
                                   theories: List[Dict],
                                   level: str) -> List[Dict]:
        """
        Validate clusters and generate canonical names.
        
        Args:
            clusters: List of cluster dictionaries
            theories: List of theory dictionaries
            level: 'family', 'parent', or 'child'
        
        Returns:
            Updated clusters with validation and names
        """
        print(f"\n🤖 Validating {len(clusters)} {level} clusters with LLM...")
        
        theory_lookup = {t['theory_id']: t for t in theories}
        
        for i, cluster in enumerate(clusters):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(clusters)}")
            
            # Get theories in cluster
            cluster_theories = [theory_lookup[tid] for tid in cluster['theory_ids']]
            
            # Validate coherence
            is_coherent, should_split, subgroups = self._validate_cluster_coherence(
                cluster_theories, level
            )
            
            cluster['coherence_score'] = 0.9 if is_coherent else 0.5
            
            if should_split and subgroups:
                # Mark for splitting (would need re-clustering in practice)
                cluster['needs_split'] = True
                cluster['suggested_subgroups'] = subgroups
                self.stats['clusters_split'] += 1
            
            # Generate canonical name
            canonical_name = self._generate_canonical_name(cluster_theories, level)
            cluster['canonical_name'] = canonical_name
            
            # Extract alternative names
            cluster['alternative_names'] = list(set([t['name'] for t in cluster_theories]))
            
            self.stats['clusters_validated'] += 1
            self.stats['canonical_names_generated'] += 1
        
        print(f"✓ Validated {len(clusters)} clusters")
        print(f"  Clusters needing split: {self.stats['clusters_split']}")
        
        return clusters
    
    def _validate_cluster_coherence(self, theories: List[Dict], 
                                    level: str) -> Tuple[bool, bool, List]:
        """
        Validate if theories in cluster are coherent.
        Returns: (is_coherent, should_split, subgroups)
        """
        if len(theories) == 1:
            return True, False, []
        
        # Format theories for LLM
        theory_text = self._format_theories_for_llm(theories)
        
        prompt = f"""You are an expert in aging biology. Analyze these {len(theories)} theories to determine if they describe the SAME underlying concept or if they should be separated.

{theory_text}

Questions:
1. Do ALL these theories describe the EXACT SAME mechanism/concept? (yes/no)
2. If NO, identify distinct sub-groups based on:
   - Different molecular mechanisms (e.g., CB1 vs p53 mediated)
   - Different pathways (e.g., mTOR vs AMPK)
   - Different cellular processes (e.g., autophagy vs apoptosis)

IMPORTANT: Preserve meaningful mechanistic distinctions. Only group if truly identical.

Answer format:
COHERENT: [yes/no]
SHOULD_SPLIT: [yes/no]
SUBGROUPS: [if split needed, list subgroup descriptions]
"""
        
        try:
            response = self._call_llm(
                prompt,
                temperature=0,
                max_tokens=700
            )
            
            if response.get('error'):
                raise RuntimeError(response['error'])
            
            result = response['content'].strip()
            
            # Parse response
            is_coherent = 'COHERENT: yes' in result.lower()
            should_split = 'SHOULD_SPLIT: yes' in result.lower()
            
            # Extract subgroups if mentioned
            subgroups = []
            if should_split and 'SUBGROUPS:' in result:
                subgroup_text = result.split('SUBGROUPS:')[1].strip()
                subgroups = [s.strip() for s in subgroup_text.split('\n') if s.strip()]
            
            return is_coherent, should_split, subgroups
            
        except Exception as e:
            print(f"   Warning: Validation failed: {e}")
            return True, False, []  # Default to coherent
    
    def _generate_canonical_name(self, theories: List[Dict], level: str) -> str:
        """Generate canonical name for cluster."""
        if len(theories) == 1:
            return theories[0]['name']
        
        # Format theories
        theory_names = [t['name'] for t in theories[:10]]  # Limit to 10 for prompt
        names_text = "\n".join([f"{i+1}. {name}" for i, name in enumerate(theory_names)])
        
        if level == 'family':
            instruction = "Generate a BROAD family name that encompasses all these theories (e.g., 'Mitochondrial Theories', 'DNA Damage Theories')."
        elif level == 'parent':
            instruction = "Generate a GENERIC theory name that captures the core concept (e.g., 'Mitochondrial Dysfunction Theory')."
        else:  # child
            instruction = "Generate a SPECIFIC theory name that captures the unique mechanism (e.g., 'CB1 receptor-mediated mitochondrial quality control theory')."
        
        prompt = f"""You are an expert in aging biology. Generate a canonical name for this group of theories.

Theory names:
{names_text}

{instruction}

Requirements:
- Use established naming conventions in aging research
- Be clear and descriptive
- Avoid redundancy
- For specific theories, include the key mechanism

Canonical name:"""
        
        try:
            response = self._call_llm(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            if response.get('error'):
                raise RuntimeError(response['error'])
            
            canonical_name = response['content'].strip()
            # Clean up
            canonical_name = canonical_name.replace('"', '').replace("'", "")
            if canonical_name.startswith('Canonical name:'):
                canonical_name = canonical_name.split(':', 1)[1].strip()
            
            return canonical_name
            
        except Exception as e:
            print(f"   Warning: Name generation failed: {e}")
            # Fallback: use most common words from theory names
            return self._generate_fallback_name(theories)

    def _call_llm(self, user_prompt: str, temperature: float, max_tokens: int) -> Dict:
        """Helper to call the LLM client with standard system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.llm_client.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _generate_fallback_name(self, theories: List[Dict]) -> str:
        """Generate fallback name from theory names."""
        # Simple heuristic: take first theory name
        return theories[0]['name']
    
    def _format_theories_for_llm(self, theories: List[Dict]) -> str:
        """Format theories for LLM prompt."""
        formatted = []
        for i, theory in enumerate(theories[:10], 1):  # Limit to 10
            concepts = theory.get('key_concepts', [])
            concept_text = "; ".join([c.get('concept', '') for c in concepts[:3]])
            
            formatted.append(f"{i}. {theory['name']}")
            if concept_text:
                formatted.append(f"   Key concepts: {concept_text}")
        
        return "\n".join(formatted)
    
    def save_results(self, families: List[Dict], parents: List[Dict], 
                    children: List[Dict], theories: List[Dict], output_path: str):
        """Save validation results."""
        print(f"\n💾 Saving validation results to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage3_llm_validation',
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
    """Run Stage 3 LLM validation."""
    from src.core.llm_integration import AzureOpenAIClient
    
    print("🚀 Starting Stage 3: LLM Validation\n")
    
    # Initialize LLM client
    try:
        llm_client = AzureOpenAIClient()
        print("✓ LLM client initialized\n")
    except Exception as e:
        print(f"❌ Error: LLM client required for Stage 3: {e}")
        return
    
    # Load clusters from Stage 2
    print("📂 Loading clusters from Stage 2...")
    with open('output/stage2_clusters.json', 'r') as f:
        data = json.load(f)
    
    theories = data['theories']
    families = data['families']
    parents = data['parents']
    children = data['children']
    
    print(f"✓ Loaded {len(families)} families, {len(parents)} parents, {len(children)} children\n")
    
    # Initialize validator
    validator = LLMValidator(llm_client)
    
    # Validate and name each level
    print("Validating families...")
    families = validator.validate_and_name_clusters(families, theories, 'family')
    
    print("\nValidating parents...")
    parents = validator.validate_and_name_clusters(parents, theories, 'parent')
    
    print("\nValidating children...")
    children = validator.validate_and_name_clusters(children, theories, 'child')
    
    # Save results
    validator.save_results(families, parents, children, theories, 
                          'output/stage3_validated.json')
    
    print("\n✅ Stage 3 complete!")
    print(f"   Validated: {validator.stats['clusters_validated']} clusters")
    print(f"   Generated: {validator.stats['canonical_names_generated']} canonical names")


if __name__ == '__main__':
    main()
