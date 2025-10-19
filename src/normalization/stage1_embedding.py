"""
Stage 1: Multi-Dimensional Embedding Generation
Creates embeddings at multiple levels of granularity for fine-grained distinction.
"""

import json
import numpy as np
from typing import List, Dict, Optional
import pickle
import os
import re
from dataclasses import dataclass, field
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TheoryEmbeddings:
    """Container for multi-level embeddings of a theory."""
    theory_id: str
    name_embedding: Optional[np.ndarray] = None
    semantic_embedding: Optional[np.ndarray] = None
    detailed_embedding: Optional[np.ndarray] = None
    concept_features: Dict = field(default_factory=dict)


class ConceptFeatureExtractor:
    """Extracts structured features from theory concepts."""
    
    def __init__(self):
        # Common patterns in aging theories
        self.mechanism_patterns = [
            r'(\w+)-mediated', r'(\w+)-induced', r'(\w+)-dependent',
            r'(\w+) receptor', r'(\w+) pathway', r'(\w+) signaling'
        ]
        
        self.pathway_keywords = [
            'mtor', 'ampk', 'insulin', 'igf-1', 'sirtuin', 'tor', 'foxo',
            'nf-kb', 'p53', 'akt', 'mapk', 'jak-stat', 'wnt', 'notch'
        ]
        
        self.process_keywords = [
            'autophagy', 'apoptosis', 'senescence', 'inflammation', 'oxidation',
            'glycation', 'methylation', 'acetylation', 'phosphorylation',
            'mitophagy', 'proteostasis', 'dna repair', 'telomere'
        ]
        
        self.molecule_keywords = [
            'ros', 'nad', 'atp', 'glucose', 'insulin', 'collagen', 'elastin',
            'mitochondria', 'dna', 'rna', 'protein', 'lipid'
        ]
        
        self.level_keywords = {
            'molecular': ['gene', 'protein', 'dna', 'rna', 'molecule', 'enzyme'],
            'cellular': ['cell', 'mitochondria', 'nucleus', 'membrane', 'organelle'],
            'tissue': ['tissue', 'organ', 'muscle', 'brain', 'heart', 'liver'],
            'systemic': ['system', 'organism', 'body', 'physiological', 'metabolic']
        }
    
    def extract_features(self, theory_dict: Dict) -> Dict:
        """Extract structured features from theory."""
        text = f"{theory_dict.get('name', '')} {theory_dict.get('concept_text', '')} {theory_dict.get('description', '')}"
        text_lower = text.lower()
        
        features = {
            'mechanisms': self._extract_mechanisms(text),
            'pathways': self._extract_keywords(text_lower, self.pathway_keywords),
            'processes': self._extract_keywords(text_lower, self.process_keywords),
            'molecules': self._extract_keywords(text_lower, self.molecule_keywords),
            'biological_level': self._determine_level(text_lower),
            'word_count': len(theory_dict.get('name', '').split()),
            'has_specific_mechanism': self._has_specific_mechanism(text)
        }
        
        return features
    
    def _extract_mechanisms(self, text: str) -> List[str]:
        """Extract mechanism patterns (e.g., 'CB1-mediated', 'p53-induced')."""
        mechanisms = []
        for pattern in self.mechanism_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mechanisms.extend([m.lower() for m in matches])
        return list(set(mechanisms))
    
    def _extract_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract keywords present in text."""
        found = []
        for keyword in keywords:
            if keyword in text:
                found.append(keyword)
        return found
    
    def _determine_level(self, text: str) -> str:
        """Determine biological level of theory."""
        level_scores = {}
        for level, keywords in self.level_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            level_scores[level] = score
        
        if not level_scores or max(level_scores.values()) == 0:
            return 'unknown'
        
        return max(level_scores, key=level_scores.get)
    
    def _has_specific_mechanism(self, text: str) -> bool:
        """Check if theory mentions specific mechanisms."""
        specific_indicators = [
            '-mediated', '-induced', '-dependent', 'receptor', 
            'specific', 'particular', 'via', 'through'
        ]
        return any(indicator in text.lower() for indicator in specific_indicators)


class EmbeddingGenerator:
    """Generates multi-level embeddings for theories."""
    
    def __init__(self, use_openai: bool = True, llm_client=None):
        """
        Initialize embedding generator.
        
        Args:
            use_openai: If True, use OpenAI embeddings. Otherwise use local model.
            llm_client: Azure OpenAI client instance
        """
        self.use_openai = use_openai
        self.llm_client = llm_client
        self.feature_extractor = ConceptFeatureExtractor()
        
        if use_openai:
            if not llm_client:
                raise ValueError("LLM client required for OpenAI embeddings")
            self.embedding_model = "text-embedding-3-large"
            self.embedding_dim = 3072
        else:
            # Fallback to local model
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-mpnet-base-v2')
                self.embedding_dim = 768
                print("✓ Using local sentence-transformers model")
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def generate_embeddings(self, theories: List[Dict], 
                          batch_size: int = 100) -> List[TheoryEmbeddings]:
        """
        Generate multi-level embeddings for all theories.
        
        Args:
            theories: List of theory dictionaries
            batch_size: Batch size for API calls
        
        Returns:
            List of TheoryEmbeddings objects
        """
        print(f"\n🔄 Generating multi-level embeddings for {len(theories)} theories...")
        print(f"   Using: {'OpenAI API' if self.use_openai else 'Local model'}")
        
        all_embeddings = []
        
        for i in range(0, len(theories), batch_size):
            batch = theories[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(theories)-1)//batch_size + 1} ({len(batch)} theories)")
            
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting for API
            if self.use_openai and i + batch_size < len(theories):
                time.sleep(1)
        
        print(f"✓ Generated embeddings for {len(all_embeddings)} theories")
        return all_embeddings
    
    def _process_batch(self, batch: List[Dict]) -> List[TheoryEmbeddings]:
        """Process a batch of theories."""
        batch_embeddings = []
        
        for theory in batch:
            theory_id = theory.get('theory_id', 'unknown')
            
            # Extract features
            features = self.feature_extractor.extract_features(theory)
            
            # Generate three levels of embeddings
            name_emb = self._embed_text(theory.get('name', ''))
            
            semantic_text = f"{theory.get('name', '')}. {theory.get('concept_text', '')}"
            semantic_emb = self._embed_text(semantic_text)
            
            detailed_text = theory.get('enriched_text', semantic_text)
            detailed_emb = self._embed_text(detailed_text)
            
            theory_emb = TheoryEmbeddings(
                theory_id=theory_id,
                name_embedding=name_emb,
                semantic_embedding=semantic_emb,
                detailed_embedding=detailed_emb,
                concept_features=features
            )
            
            batch_embeddings.append(theory_emb)
        
        return batch_embeddings
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        if self.use_openai:
            return self._embed_openai(text)
        else:
            return self._embed_local(text)
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            response = self.llm_client.client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]  # Truncate to token limit
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            # If OpenAI fails, fall back to local model
            if not hasattr(self, '_fallback_warned'):
                print(f"   ⚠ OpenAI embedding failed: {e}")
                print(f"   🔄 Falling back to local model...")
                self._fallback_warned = True
                # Initialize local model
                try:
                    from sentence_transformers import SentenceTransformer
                    self.local_model = SentenceTransformer('all-mpnet-base-v2')
                    self.use_openai = False  # Switch to local
                    self.embedding_dim = 768
                    print(f"   ✓ Local model initialized")
                except ImportError:
                    print(f"   ❌ Local model not available. Install: pip install sentence-transformers")
                    return np.zeros(self.embedding_dim)
            
            # Use local model
            if hasattr(self, 'local_model'):
                return self._embed_local(text)
            else:
                return np.zeros(self.embedding_dim)
    
    def _embed_local(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        try:
            # Use local_model if it exists (fallback), otherwise use self.model
            model = getattr(self, 'local_model', None) or getattr(self, 'model', None)
            if model:
                return model.encode(text, convert_to_numpy=True)
            else:
                return np.zeros(self.embedding_dim)
        except Exception as e:
            print(f"   Warning: Local embedding failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def save_embeddings(self, embeddings: List[TheoryEmbeddings], 
                       theories: List[Dict], output_path: str):
        """Save embeddings and theories to file."""
        print(f"\n💾 Saving embeddings to {output_path}...")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data
        data = {
            'metadata': {
                'stage': 'stage1_embedding',
                'total_theories': len(theories),
                'embedding_model': self.embedding_model if self.use_openai else 'all-mpnet-base-v2',
                'embedding_dim': self.embedding_dim
            },
            'theories': theories,
            'embeddings': []
        }
        
        # Convert embeddings to serializable format
        for emb in embeddings:
            emb_dict = {
                'theory_id': emb.theory_id,
                'name_embedding': emb.name_embedding.tolist() if emb.name_embedding is not None else None,
                'semantic_embedding': emb.semantic_embedding.tolist() if emb.semantic_embedding is not None else None,
                'detailed_embedding': emb.detailed_embedding.tolist() if emb.detailed_embedding is not None else None,
                'concept_features': emb.concept_features
            }
            data['embeddings'].append(emb_dict)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save as pickle for faster loading
        pickle_path = output_path.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved to {output_path}")
        print(f"✓ Saved to {pickle_path} (pickle format)")


def main():
    """Run Stage 1 embedding generation."""
    from src.core.llm_integration import AzureOpenAIClient
    
    print("🚀 Starting Stage 1: Embedding Generation\n")
    
    # Load filtered theories from Stage 0
    print("📂 Loading filtered theories from Stage 0...")
    with open('output/stage0_filtered_theories.json', 'r') as f:
        data = json.load(f)
    
    theories = data['theories']
    print(f"✓ Loaded {len(theories)} theories\n")
    
    # Initialize LLM client for OpenAI embeddings
    try:
        llm_client = AzureOpenAIClient()
        use_openai = True
        print("✓ Using OpenAI embeddings\n")
    except Exception as e:
        print(f"⚠ OpenAI not available: {e}")
        print("  Falling back to local embeddings\n")
        llm_client = None
        use_openai = False
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(use_openai=use_openai, llm_client=llm_client)
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(theories, batch_size=100)
    
    # Save results
    generator.save_embeddings(
        embeddings, 
        theories, 
        'output/stage1_embeddings.json'
    )
    
    print("\n✅ Stage 1 complete!")


if __name__ == '__main__':
    main()
