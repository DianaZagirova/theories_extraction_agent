"""
LLM-Powered Theory Validation and Naming System
Uses Claude/GPT to validate clusters and generate canonical names
"""

import json
from typing import List, Dict
from dataclasses import dataclass


# ============================================================================
# LLM VALIDATION PROMPTS
# ============================================================================

class TheoryValidationPrompts:
    """Carefully crafted prompts for theory normalization"""
    
    @staticmethod
    def cluster_validation_prompt(theories: List[Dict]) -> str:
        """Validate if theories should be grouped together"""
        
        theories_text = "\n\n".join([
            f"Theory {i+1}: {t['name']}\n"
            f"Key Concepts: {', '.join([c['concept'] for c in t['key_concepts'][:3]])}\n"
            f"Description: {t['key_concepts'][0]['description']}"
            for i, t in enumerate(theories[:10])  # Limit to avoid context overflow
        ])
        
        return f"""You are an expert in aging biology and theory classification. Analyze whether these theories describe the SAME fundamental mechanism or should be considered DISTINCT theories.

THEORIES TO ANALYZE:
{theories_text}

ANALYSIS FRAMEWORK:
1. Core Mechanism: Do they share the same biological mechanism?
2. Level of Analysis: Are they at the same level (molecular/cellular/organismal)?
3. Causal Chain: Are they different points in the same causal pathway?
4. Semantic Equivalence: Are these just different phrasings of the same idea?

DECISION RULES:
- MERGE if: Same mechanism with different terminology
- MERGE if: One is a specific instance of the other's general principle
- SPLIT if: Different mechanisms even if related
- SPLIT if: Different levels of biological organization
- SPLIT if: One is upstream/downstream of the other in a pathway

RESPOND IN JSON:
{{
  "should_merge": true/false,
  "confidence": 0-10,
  "reasoning": "Brief explanation",
  "suggested_canonical_name": "Most accurate standardized name",
  "sub_clusters": [
    {{
      "indices": [0, 1, 3],
      "name": "Sub-cluster name if split is needed"
    }}
  ],
  "relationships": "If split, describe how they relate"
}}"""

    @staticmethod
    def canonical_naming_prompt(theories: List[Dict], 
                                ontology_theories: List[str]) -> str:
        """Generate standardized canonical name"""
        
        theories_summary = "\n".join([
            f"- {t['name']}" for t in theories[:15]
        ])
        
        ontology_context = "\n".join([
            f"- {name}" for name in ontology_theories[:20]
        ])
        
        return f"""You are standardizing aging theory nomenclature. Given these variant names for the same theory, generate ONE canonical name.

VARIANT NAMES:
{theories_summary}

EXISTING ONTOLOGY (for reference):
{ontology_context}

NAMING PRINCIPLES:
1. Use established terminology from the field
2. Be specific but concise (3-6 words ideal)
3. Lead with the PRIMARY mechanism/phenomenon
4. Avoid redundant words like "theory of", "hypothesis"
5. Use consistent formatting (e.g., "Mitochondrial Dysfunction Theory" not "Theory of Mitochondrial Dysfunction")
6. Prefer biological mechanism over symptom description

EXAMPLES OF GOOD CANONICAL NAMES:
- "Mitochondrial Free Radical Theory"
- "Antagonistic Pleiotropy Theory"
- "Telomere Attrition Theory"
- "Inflammaging Theory"
- "Protein Homeostasis Collapse Theory"

RESPOND IN JSON:
{{
  "canonical_name": "The standardized name",
  "alternative_names": ["Other acceptable names"],
  "rationale": "Why this name is most appropriate",
  "mechanism_category": "Molecular/Cellular/Systems/Evolutionary",
  "key_mechanism": "One sentence description of core mechanism"
}}"""

    @staticmethod
    def hierarchical_relationship_prompt(theory1: Dict, theory2: Dict) -> str:
        """Determine if theories have parent-child relationship"""
        
        return f"""Determine the relationship between these two theories:

THEORY A: {theory1['name']}
Key Concepts: {theory1['key_concepts'][:2]}

THEORY B: {theory2['name']}
Key Concepts: {theory2['key_concepts'][:2]}

RELATIONSHIP TYPES:
1. IDENTICAL: Same theory, different phrasing
2. PARENT-CHILD: One is a specific instance/mechanism of the other
3. OVERLAPPING: Share some mechanisms but distinct theories
4. DISTINCT: Separate theories (may be related but fundamentally different)

RESPOND IN JSON:
{{
  "relationship": "IDENTICAL/PARENT-CHILD/OVERLAPPING/DISTINCT",
  "parent": "A/B/neither",
  "confidence": 0-10,
  "explanation": "Brief reasoning"
}}"""

    @staticmethod
    def batch_equivalence_prompt(theory_names: List[str], 
                                target_theory: str) -> str:
        """Check which theories are equivalent to a target"""
        
        names_list = "\n".join([f"{i}. {name}" for i, name in enumerate(theory_names)])
        
        return f"""Given this TARGET theory, identify which theories from the list are semantically EQUIVALENT (same core idea, different wording).

TARGET THEORY: {target_theory}

CANDIDATE THEORIES:
{names_list}

Mark each as:
- EQUIVALENT: Same theory mechanism
- RELATED: Connected but distinct
- UNRELATED: Different mechanism

RESPOND IN JSON:
{{
  "equivalent_indices": [list of indices],
  "related_indices": [list of indices],
  "confidence_scores": {{index: score}}
}}"""


# ============================================================================
# LLM VALIDATION ENGINE
# ============================================================================

class LLMValidationEngine:
    """Orchestrates LLM calls for theory validation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.prompts = TheoryValidationPrompts()
    
    def call_llm(self, prompt: str) -> Dict:
        """
        Call LLM API (Claude/GPT)
        In production, use proper API client
        """
        # Pseudo-code for API call
        # import anthropic
        # client = anthropic.Anthropic(api_key=self.api_key)
        # response = client.messages.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return json.loads(response.content[0].text)
        
        # Placeholder response
        return {
            "should_merge": True,
            "confidence": 8,
            "reasoning": "Theories share core mechanism",
            "suggested_canonical_name": "Example Theory"
        }
    
    def validate_cluster(self, theories: List[Dict]) -> Dict:
        """Validate if a cluster of theories should be merged"""
        prompt = self.prompts.cluster_validation_prompt(theories)
        return self.call_llm(prompt)
    
    def generate_canonical_name(self, theories: List[Dict], 
                               ontology: List[str]) -> Dict:
        """Generate standardized name for theory cluster"""
        prompt = self.prompts.canonical_naming_prompt(theories, ontology)
        return self.call_llm(prompt)
    
    def determine_relationship(self, theory1: Dict, theory2: Dict) -> Dict:
        """Determine hierarchical relationship between two theories"""
        prompt = self.prompts.hierarchical_relationship_prompt(theory1, theory2)
        return self.call_llm(prompt)
    
    def batch_equivalence_check(self, theory_names: List[str], 
                                target: str) -> Dict:
        """Check which theories are equivalent to target (efficient for large sets)"""
        prompt = self.prompts.batch_equivalence_prompt(theory_names, target)
        return self.call_llm(prompt)


# ============================================================================
# HYBRID PIPELINE: ML + LLM
# ============================================================================

class HybridNormalizationPipeline:
    """Combines ML clustering with LLM validation"""
    
    def __init__(self, llm_engine: LLMValidationEngine):
        self.llm = llm_engine
        self.validation_threshold = 7  # Only validate uncertain clusters
    
    def smart_validation_strategy(self, cluster_size: int, 
                                  similarity_scores: List[float]) -> str:
        """Decide when to use LLM validation"""
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Skip validation for obvious cases
        if cluster_size == 1:
            return "skip"
        
        if cluster_size == 2 and avg_similarity > 0.9:
            return "skip"  # Very similar, trust clustering
        
        if cluster_size > 20:
            return "llm_required"  # Large cluster needs validation
        
        if avg_similarity < 0.75:
            return "llm_required"  # Low similarity needs checking
        
        if cluster_size >= 5 and cluster_size <= 20:
            return "llm_optional"  # Worth checking but not critical
        
        return "skip"
    
    def iterative_refinement(self, initial_clusters: List[List[Dict]]) -> List[Dict]:
        """
        Iteratively refine clusters using LLM feedback
        
        Process:
        1. Start with ML clusters
        2. For uncertain clusters, ask LLM to validate
        3. If LLM suggests split, apply and re-cluster
        4. Generate final canonical names with LLM
        """
        
        refined_clusters = []
        
        for cluster in initial_clusters:
            if len(cluster) == 1:
                # Singleton - use as-is
                refined_clusters.append({
                    'theories': cluster,
                    'canonical_name': cluster[0]['name'],
                    'validation': 'singleton'
                })
                continue
            
            # Calculate similarity metrics for validation decision
            similarities = [0.8] * len(cluster)  # Placeholder
            
            strategy = self.smart_validation_strategy(len(cluster), similarities)
            
            if strategy == "skip":
                # Trust ML clustering
                refined_clusters.append({
                    'theories': cluster,
                    'canonical_name': self._generate_name_from_cluster(cluster),
                    'validation': 'ml_only'
                })
            
            else:
                # Use LLM validation
                validation_result = self.llm.validate_cluster(cluster)
                
                if validation_result['should_merge']:
                    # LLM confirms merge
                    canonical_name = validation_result['suggested_canonical_name']
                    refined_clusters.append({
                        'theories': cluster,
                        'canonical_name': canonical_name,
                        'validation': 'llm_confirmed',
                        'confidence': validation_result['confidence']
                    })
                
                else:
                    # LLM suggests split
                    if 'sub_clusters' in validation_result:
                        for sub_cluster_info in validation_result['sub_clusters']:
                            sub_theories = [cluster[i] for i in sub_cluster_info['indices']]
                            refined_clusters.append({
                                'theories': sub_theories,
                                'canonical_name': sub_cluster_info['name'],
                                'validation': 'llm_split'
                            })
        
        return refined_clusters
    
    def _generate_name_from_cluster(self, theories: List[Dict]) -> str:
        """Generate name without LLM (for skipped validations)"""
        # Use most frequent terms
        from collections import Counter
        all_words = []
        for theory in theories:
            all_words.extend(theory['name'].lower().split())
        
        common_words = Counter(all_words).most_common(3)

        