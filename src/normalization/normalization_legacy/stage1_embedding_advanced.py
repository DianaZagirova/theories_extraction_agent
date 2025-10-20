"""
Stage 1: Advanced Multi-Dimensional Embedding Generation (Production-Ready)

Key improvements:
1. Advanced feature extraction using NER, KeyBERT, and domain-specific models
2. Multiple specialized embedding models for different text properties
3. Optimized model loading with caching
4. Hierarchical relationship detection
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
import os
import re
from dataclasses import dataclass, field
import time
from pathlib import Path


@dataclass
class TheoryEmbeddings:
    """Container for multi-level embeddings of a theory."""
    theory_id: str
    name_embedding: Optional[np.ndarray] = None
    semantic_embedding: Optional[np.ndarray] = None
    detailed_embedding: Optional[np.ndarray] = None
    biomedical_embedding: Optional[np.ndarray] = None
    concept_features: Dict = field(default_factory=dict)
    hierarchical_features: Dict = field(default_factory=dict)


class AdvancedConceptFeatureExtractor:
    """Production-ready feature extractor using advanced NLP."""
    
    def __init__(self, cache_dir: str = ".cache/nlp_models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._ner_model = None
        self._keybert_model = None
        self._spacy_model = None
        
        self._init_patterns()
        print("✓ Advanced feature extractor initialized")
    
    def _init_patterns(self):
        """Initialize domain-specific regex patterns and fallback keywords."""
        # Regex patterns for structured extraction
        self.mechanism_patterns = {
            'mediated': re.compile(r'(\w+(?:-\w+)?)-mediated', re.IGNORECASE),
            'induced': re.compile(r'(\w+(?:-\w+)?)-induced', re.IGNORECASE),
            'dependent': re.compile(r'(\w+(?:-\w+)?)-dependent', re.IGNORECASE),
            'activated': re.compile(r'(\w+(?:-\w+)?)-activated', re.IGNORECASE),
        }
        
        self.receptor_pattern = re.compile(r'(\w+(?:-\w+)?)\s+receptor', re.IGNORECASE)
        self.pathway_pattern = re.compile(r'(\w+(?:-\w+)?(?:/\w+)?)\s+(?:pathway|signaling)', re.IGNORECASE)
        self.process_pattern = re.compile(
            r'(autophagy|apoptosis|senescence|inflammation|oxidation|'
            r'glycation|methylation|acetylation|phosphorylation|mitophagy|proteostasis)',
            re.IGNORECASE
        )
        
        # Fallback keywords for common mechanisms (when patterns don't match)
        # These are high-frequency terms in aging research that are often mentioned
        # without explicit "-mediated" or "-induced" modifiers
        self.mechanism_keywords = [
            'mtor', 'ampk', 'insulin', 'igf1', 'igf-1', 'foxo', 'foxo3', 'sirt1', 'sirt',
            'p53', 'tp53', 'nf-kb', 'nfkb', 'ros', 'nad', 'nad+', 'atp', 'camp', 'cgmp',
            'autophagy', 'apoptosis', 'senescence', 'inflammation', 'oxidation',
            'glycation', 'methylation', 'acetylation', 'phosphorylation', 'ubiquitination',
            'proteasome', 'lysosome', 'mitochondria', 'telomere', 'telomerase',
            'dna damage', 'oxidative stress', 'er stress', 'unfolded protein'
        ]
        
        # Fallback keywords for common pathways
        self.pathway_keywords = [
            'mtor', 'tor', 'ampk', 'insulin', 'igf', 'igf1', 'pi3k', 'akt', 'mapk',
            'jak', 'stat', 'wnt', 'notch', 'hedgehog', 'tgf', 'tgf-beta', 'nf-kb',
            'p38', 'erk', 'jnk', 'pka', 'pkc', 'ras', 'raf', 'mek'
        ]
    
    @property
    def ner_model(self):
        """Lazy load biomedical NER model."""
        if self._ner_model is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
                
                model_name = "d4data/biomedical-ner-all"
                cache_path = self.cache_dir / "ner_model"
                
                print("   Loading biomedical NER model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path))
                model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=str(cache_path))
                
                # Use device=-1 for CPU or 0 for GPU, batch_size for efficiency
                import torch
                device = 0 if torch.cuda.is_available() else -1
                self._ner_model = pipeline(
                    "ner", 
                    model=model, 
                    tokenizer=tokenizer, 
                    aggregation_strategy="simple",
                    device=device,
                    batch_size=8  # Process 8 texts at once
                )
                print(f"   ✓ NER model loaded (device: {'GPU' if device == 0 else 'CPU'})")
            except Exception as e:
                print(f"   ⚠ NER model not available: {e}")
                self._ner_model = None
        
        return self._ner_model
    
    @property
    def keybert_model(self):
        """Lazy load KeyBERT for keyword extraction."""
        if self._keybert_model is None:
            try:
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                
                cache_path = self.cache_dir / "keybert_model"
                print("   Loading KeyBERT model...")
                
                embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', cache_folder=str(cache_path))
                self._keybert_model = KeyBERT(model=embedding_model)
                print("   ✓ KeyBERT model loaded")
            except Exception as e:
                print(f"   ⚠ KeyBERT not available: {e}")
                self._keybert_model = None
        
        return self._keybert_model
    
    @property
    def spacy_model(self):
        """Lazy load spaCy for linguistic analysis."""
        if self._spacy_model is None:
            try:
                import spacy
                print("   Loading spaCy model...")
                try:
                    self._spacy_model = spacy.load("en_core_web_sm")
                except OSError:
                    print("   Downloading spaCy model...")
                    os.system("python -m spacy download en_core_web_sm")
                    self._spacy_model = spacy.load("en_core_web_sm")
                print("   ✓ spaCy model loaded")
            except Exception as e:
                print(f"   ⚠ spaCy not available: {e}")
                self._spacy_model = None
        
        return self._spacy_model
    
    def extract_features(self, theory_dict: Dict) -> Dict:
        """Extract comprehensive features from theory."""
        name = theory_dict.get('name', '')
        concept_text = theory_dict.get('concept_text', '')
        description = theory_dict.get('description', '')
        
        full_text = f"{name}. {concept_text}. {description}"
        
        features = {
            'mechanisms': self._extract_mechanisms(full_text),
            'receptors': self._extract_receptors(full_text),
            'pathways': self._extract_pathways(full_text),
            'processes': self._extract_processes(full_text),
            'entities': self._extract_entities(full_text),
            'keywords': self._extract_keywords(full_text),
            'specificity_score': self._calculate_specificity(name, full_text),
            'name_length': len(name.split()),
            'has_mechanism_modifier': bool(self._extract_mechanisms(name)),
            'biological_level': self._determine_biological_level(full_text),
            'linguistic_complexity': self._analyze_linguistic_complexity(name)
        }
        
        return features
    
    def _extract_mechanisms(self, text: str) -> List[Dict]:
        """Extract mechanism entities with pattern matching + fallback keywords."""
        mechanisms = []
        found_entities = set()
        
        # 1. Pattern-based extraction (structured: X-mediated, X-induced, etc.)
        for mech_type, pattern in self.mechanism_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                entity = match.lower()
                if entity not in found_entities:
                    mechanisms.append({'entity': entity, 'type': mech_type})
                    found_entities.add(entity)
        
        # 2. Fallback: keyword matching for common mechanisms
        # This catches standalone mentions like "mTOR regulates" without "-mediated"
        text_lower = text.lower()
        for keyword in self.mechanism_keywords:
            if keyword in text_lower and keyword not in found_entities:
                mechanisms.append({'entity': keyword, 'type': 'keyword'})
                found_entities.add(keyword)
        
        return mechanisms
    
    def _extract_receptors(self, text: str) -> List[str]:
        return [m.lower() for m in self.receptor_pattern.findall(text)]
    
    def _extract_pathways(self, text: str) -> List[str]:
        """Extract pathways with pattern matching + fallback keywords."""
        pathways = set()
        
        # 1. Pattern-based extraction (requires "pathway" or "signaling" suffix)
        pathways.update([m.lower() for m in self.pathway_pattern.findall(text)])
        
        # 2. Fallback: keyword matching for common pathways
        # This catches standalone mentions like "mTOR activation" without "pathway"
        text_lower = text.lower()
        for keyword in self.pathway_keywords:
            if keyword in text_lower:
                pathways.add(keyword)
        
        return list(pathways)
    
    def _extract_processes(self, text: str) -> List[str]:
        return [m.lower() for m in set(self.process_pattern.findall(text))]
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities using NER (single text) with noise filtering."""
        if self.ner_model is None or len(text) > 2000:
            return {}
        
        try:
            entities = self.ner_model(text[:1000])
            entity_dict = {}
            for ent in entities:
                ent_type = ent['entity_group']
                word = ent['word'].lower().strip()
                
                # Filter out noise (punctuation, artifacts, stopwords)
                if self._is_valid_entity(word):
                    if ent_type not in entity_dict:
                        entity_dict[ent_type] = []
                    entity_dict[ent_type].append(word)
            
            return entity_dict
        except:
            return {}
    
    def _is_valid_entity(self, word: str) -> bool:
        """Check if entity is valid (not noise/artifact)."""
        # Filter out punctuation and special characters
        if word in ['|', ',', '.', ';', ':', '-', '##', 'so', '(', ')', '[', ']', '{', '}']:
            return False
        
        # Filter out very short words (likely artifacts)
        if len(word) <= 1:
            return False
        
        # Filter out tokenization artifacts (subword tokens)
        if word.startswith('##'):
            return False
        
        # Filter out common stopwords misclassified as entities
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        if word in stopwords:
            return False
        
        # Filter out pure numbers (unless part of gene names like p53)
        if word.isdigit():
            return False
        
        return True
    
    def _extract_entities_batch(self, texts: List[str]) -> List[Dict]:
        """Extract named entities using NER (batch processing - more efficient) with noise filtering."""
        if self.ner_model is None:
            return [{} for _ in texts]
        
        try:
            # Truncate texts and process in batch
            truncated_texts = [text[:1000] for text in texts if len(text) <= 2000]
            
            if not truncated_texts:
                return [{} for _ in texts]
            
            # Batch processing - much faster on GPU
            batch_results = self.ner_model(truncated_texts)
            
            # Convert to dict format with noise filtering
            entity_dicts = []
            for entities in batch_results:
                entity_dict = {}
                for ent in entities:
                    ent_type = ent['entity_group']
                    word = ent['word'].lower().strip()
                    
                    # Filter out noise
                    if self._is_valid_entity(word):
                        if ent_type not in entity_dict:
                            entity_dict[ent_type] = []
                        entity_dict[ent_type].append(word)
                
                entity_dicts.append(entity_dict)
            
            return entity_dicts
        except Exception as e:
            return [{} for _ in texts]
    
    def _extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using hybrid approach (YAKE + spaCy + patterns)."""
        if len(text) < 20:
            return []
        
        keywords = {}
        
        # 1. YAKE - Fast statistical extraction
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=15)
            yake_kw = kw_extractor.extract_keywords(text[:1000])
            
            for kw, score in yake_kw:
                # Invert score and add to dict
                keywords[kw.lower()] = max(keywords.get(kw.lower(), 0), 1.0 - min(score, 1.0))
        except:
            pass
        
        # 2. spaCy entities - Domain-specific terms
        if self.spacy_model:
            try:
                doc = self.spacy_model(text[:1000])
                for ent in doc.ents:
                    # Prioritize scientific entities
                    if ent.label_ in ['DISEASE', 'CHEMICAL', 'PROTEIN', 'GPE', 'ORG']:
                        keywords[ent.text.lower()] = max(keywords.get(ent.text.lower(), 0), 0.85)
            except:
                pass
        
        # 3. Mechanism patterns - High-value terms
        try:
            mechanisms = self._extract_mechanisms(text)
            for mech in mechanisms:
                entity = mech.get('entity', '')
                if entity:
                    keywords[entity] = 0.95  # High score for mechanism terms
        except:
            pass
        
        # 4. Pathway/receptor patterns
        try:
            receptors = self._extract_receptors(text)
            for receptor in receptors:
                keywords[receptor] = 0.90
            
            pathways = self._extract_pathways(text)
            for pathway in pathways:
                keywords[pathway] = 0.90
        except:
            pass
        
        # Sort by score and return top 10
        sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_kw[:10]
    
    def _calculate_specificity(self, name: str, full_text: str) -> float:
        """Calculate specificity score (0=generic, 1=specific)."""
        score = 0.5
        
        # Check for specific indicators
        if re.search(r'-mediated|-induced|-dependent', name, re.IGNORECASE):
            score += 0.2
        if re.search(r'\btheory\b|\bhypothesis\b', name, re.IGNORECASE):
            score -= 0.15
        
        word_count = len(name.split())
        if word_count > 8:
            score += 0.2
        elif word_count < 4:
            score -= 0.1
        
        if self._extract_mechanisms(name):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _determine_biological_level(self, text: str) -> str:
        """Determine biological level."""
        if self.spacy_model is None:
            return 'unknown'
        
        try:
            doc = self.spacy_model(text[:500])
            
            level_counts = {'molecular': 0, 'cellular': 0, 'tissue': 0, 'systemic': 0}
            molecular_terms = {'gene', 'protein', 'dna', 'rna', 'molecule', 'enzyme', 'receptor'}
            cellular_terms = {'cell', 'mitochondria', 'nucleus', 'membrane', 'organelle'}
            tissue_terms = {'tissue', 'organ', 'muscle', 'brain', 'heart', 'liver'}
            systemic_terms = {'system', 'organism', 'body', 'physiological', 'metabolic'}
            
            for token in doc:
                lemma = token.lemma_.lower()
                if lemma in molecular_terms:
                    level_counts['molecular'] += 1
                elif lemma in cellular_terms:
                    level_counts['cellular'] += 1
                elif lemma in tissue_terms:
                    level_counts['tissue'] += 1
                elif lemma in systemic_terms:
                    level_counts['systemic'] += 1
            
            if sum(level_counts.values()) == 0:
                return 'unknown'
            
            non_zero = [k for k, v in level_counts.items() if v > 0]
            if len(non_zero) > 2:
                return 'multi-level'
            
            return max(level_counts, key=level_counts.get)
        except:
            return 'unknown'
    
    def _analyze_linguistic_complexity(self, text: str) -> Dict:
        """Analyze linguistic complexity."""
        if self.spacy_model is None:
            return {'complexity_score': 0.5}
        
        try:
            doc = self.spacy_model(text)
            avg_word_length = np.mean([len(token.text) for token in doc if token.is_alpha])
            num_entities = len(doc.ents)
            dependency_depth = max([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0
            
            complexity = (avg_word_length / 10) * 0.3 + (num_entities / max(len(doc), 1)) * 0.3 + (dependency_depth / 5) * 0.4
            
            return {
                'complexity_score': min(1.0, complexity),
                'avg_word_length': avg_word_length,
                'num_entities': num_entities,
                'dependency_depth': dependency_depth
            }
        except:
            return {'complexity_score': 0.5}


class MultiModelEmbeddingGenerator:
    """Production-ready embedding generator using multiple specialized models."""
    
    def __init__(self, use_openai_embeddings: bool = True, llm_client=None, cache_dir: str = ".cache/embedding_models", use_biomedical: bool = True):
        self.use_openai_embeddings = use_openai_embeddings
        self.llm_client = llm_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_biomedical = use_biomedical
        
        self.feature_extractor = AdvancedConceptFeatureExtractor(cache_dir=str(self.cache_dir / "nlp_models"))
        
        self._general_model = None
        self._biomedical_model = None
        
        if use_openai_embeddings:
            if not llm_client:
                raise ValueError("LLM client required for OpenAI embeddings")
            self.embedding_model = "text-embedding-3-large"
            self.embedding_dim = 3072
            print("✓ Using OpenAI embeddings (3072-dim)")
        else:
            self.embedding_dim = 768
            print("✓ Using local embeddings (768-dim)")
    
    @property
    def general_model(self):
        """Lazy load general-purpose embedding model."""
        if self._general_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cache_path = self.cache_dir / "general_model"
                print("   Loading general embedding model...")
                self._general_model = SentenceTransformer('all-mpnet-base-v2', cache_folder=str(cache_path))
                print("   ✓ General model loaded")
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        return self._general_model
    
    @property
    def biomedical_model(self):
        """Lazy load biomedical-specific embedding model."""
        if self._biomedical_model is None and self.use_biomedical:
            try:
                from sentence_transformers import SentenceTransformer
                cache_path = self.cache_dir / "biomedical_model"
                print("   Loading biomedical embedding model...")
                self._biomedical_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', cache_folder=str(cache_path))
                print("   ✓ Biomedical model loaded")
            except Exception as e:
                print(f"   ⚠ Biomedical model not available: {e}")
                self._biomedical_model = None
        return self._biomedical_model
    
    def generate_embeddings(self, theories: List[Dict], batch_size: int = 32) -> List[TheoryEmbeddings]:
        """Generate multi-dimensional embeddings."""
        print(f"\n🔄 Generating advanced embeddings for {len(theories)} theories...")
        print(f"   Using: {'OpenAI API' if self.use_openai_embeddings else 'Local models'}")
        
        all_embeddings = []
        
        for i in range(0, len(theories), batch_size):
            batch = theories[i:i+batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(theories)-1)//batch_size + 1}")
            
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            if self.use_openai_embeddings and i + batch_size < len(theories):
                time.sleep(1)
        
        print(f"✓ Generated embeddings for {len(all_embeddings)} theories")
        return all_embeddings
    
    def _process_batch(self, batch: List[Dict]) -> List[TheoryEmbeddings]:
        """Process a batch of theories."""
        names = [t.get('name', '') for t in batch]
        semantic_texts = [f"{t.get('name', '')}. {t.get('concept_text', '')}" for t in batch]
        detailed_texts = [t.get('enriched_text', semantic_texts[i]) for i, t in enumerate(batch)]
        
        if self.use_openai_embeddings:
            name_embs = [self._embed_openai(name) for name in names]
            semantic_embs = [self._embed_openai(text) for text in semantic_texts]
            detailed_embs = [self._embed_openai(text) for text in detailed_texts]
            biomedical_embs = [np.zeros(self.embedding_dim) for _ in batch]
        else:
            name_embs = self._embed_batch_local(names, self.general_model)
            semantic_embs = self._embed_batch_local(semantic_texts, self.general_model)
            detailed_embs = self._embed_batch_local(detailed_texts, self.general_model)
            
            if self.biomedical_model:
                biomedical_embs = self._embed_batch_local(semantic_texts, self.biomedical_model)
            else:
                biomedical_embs = [np.zeros(self.embedding_dim) for _ in batch]
        
        batch_embeddings = []
        for i, theory in enumerate(batch):
            features = self.feature_extractor.extract_features(theory)
            hierarchical_features = self._detect_hierarchical_features(theory, features)
            
            theory_emb = TheoryEmbeddings(
                theory_id=theory.get('theory_id', 'unknown'),
                name_embedding=name_embs[i],
                semantic_embedding=semantic_embs[i],
                detailed_embedding=detailed_embs[i],
                biomedical_embedding=biomedical_embs[i],
                concept_features=features,
                hierarchical_features=hierarchical_features
            )
            batch_embeddings.append(theory_emb)
        
        return batch_embeddings
    
    def _embed_batch_local(self, texts: List[str], model) -> List[np.ndarray]:
        """Embed batch using local model."""
        try:
            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            valid_texts = [texts[i] for i in valid_indices]
            
            if not valid_texts:
                return [np.zeros(self.embedding_dim) for _ in texts]
            
            embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
            
            result = []
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result.append(np.zeros(self.embedding_dim))
            return result
        except Exception as e:
            print(f"   Warning: Batch embedding failed: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            response = self.llm_client.client.embeddings.create(model=self.embedding_model, input=text[:8000])
            return np.array(response.data[0].embedding)
        except Exception as e:
            if not hasattr(self, '_fallback_warned'):
                print(f"   ⚠ OpenAI embedding failed, using local fallback")
                self._fallback_warned = True
            return self.general_model.encode(text, convert_to_numpy=True)
    
    def _detect_hierarchical_features(self, theory: Dict, features: Dict) -> Dict:
        """Detect hierarchical relationship features."""
        name = theory.get('name', '')
        specificity = features.get('specificity_score', 0.5)
        
        is_parent = specificity < 0.4 and len(name.split()) < 6 and not features.get('has_mechanism_modifier', False)
        is_child = specificity > 0.6 and len(name.split()) >= 5 and features.get('has_mechanism_modifier', False)
        
        return {
            'is_parent_candidate': is_parent,
            'is_child_candidate': is_child,
            'specificity_score': specificity
        }
    
    def save_embeddings(self, embeddings: List[TheoryEmbeddings], theories: List[Dict], output_path: str):
        """Save embeddings with metadata."""
        print(f"\n💾 Saving advanced embeddings to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage1_embedding_advanced',
                'total_theories': len(theories),
                'embedding_model': self.embedding_model if self.use_openai_embeddings else 'multi-model',
                'embedding_dim': self.embedding_dim,
                'feature_extraction': 'advanced_nlp'
            },
            'theories': theories,
            'embeddings': []
        }
        
        for emb in embeddings:
            emb_dict = {
                'theory_id': emb.theory_id,
                'name_embedding': emb.name_embedding.tolist() if emb.name_embedding is not None else None,
                'semantic_embedding': emb.semantic_embedding.tolist() if emb.semantic_embedding is not None else None,
                'detailed_embedding': emb.detailed_embedding.tolist() if emb.detailed_embedding is not None else None,
                'biomedical_embedding': emb.biomedical_embedding.tolist() if emb.biomedical_embedding is not None else None,
                'concept_features': emb.concept_features,
                'hierarchical_features': emb.hierarchical_features
            }
            data['embeddings'].append(emb_dict)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        pickle_path = output_path.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved to {output_path}")
        print(f"✓ Saved to {pickle_path}")


def main():
    """Run Stage 1 with advanced embeddings."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print("🚀 Starting Stage 1: Advanced Embedding Generation\n")
    
    # Try to import LLM client
    try:
        from src.core.llm_integration import AzureOpenAIClient
        llm_client = AzureOpenAIClient()
        use_openai = True
        use_openai_embeddings = False
        print("✓ Using OpenAI embeddings\n")
    except Exception as e:
        print(f"⚠ OpenAI not available: {e}")
        print("  Using local embeddings\n")
        llm_client = None
        use_openai = False
        use_openai_embeddings = False
    
    # Load theories
    input_path = project_root / 'output' / 'stage0_filtered_theories.json'
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        print(f"   Run Stage 0 first: python src/normalization/stage0_quality_filter.py")
        sys.exit(1)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    theories = data['theories']
    print(f"✓ Loaded {len(theories)} theories\n")
    
    # Generate embeddings
    generator = MultiModelEmbeddingGenerator(use_openai=use_openai_embeddings, llm_client=llm_client, use_biomedical=True)
    embeddings = generator.generate_embeddings(theories, batch_size=32)
    
    # Save results
    output_path = project_root / 'output' / 'stage1_embeddings_advanced.json'
    generator.save_embeddings(embeddings, theories, str(output_path))
    
    print("\n✅ Stage 1 complete!")


if __name__ == '__main__':
    main()
