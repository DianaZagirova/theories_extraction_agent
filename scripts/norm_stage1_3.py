"""
Complete Implementation: Stages 1-3 of Theory Normalization Pipeline
Ready to run with your data
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: THEORY REPRESENTATION & EMBEDDING
# ============================================================================

@dataclass
class Theory:
    """Structured representation of an extracted theory"""
    original_name: str
    key_concepts: List[Dict]
    paper_doi: str
    paper_focus: int
    confidence_score: int
    evidence: str
    mode: str
    seed_keywords: List[str] = field(default_factory=list)
    
    # Computed fields
    embedding: Optional[np.ndarray] = None
    core_terms: Set[str] = field(default_factory=set)
    canonical_form: str = ""
    
    def to_embedding_text(self, keyword_weight: int = 1) -> str:
        """
        Create rich text representation for embedding
        
        Structure optimizes for semantic similarity detection:
        - Name: Primary identifier
        - Concepts: Core mechanisms
        - Evidence: Context clues
        - Keywords: Terminology variants
        """
        name_text = self.original_name
        
        # Concatenate all concept information
        concepts_text = " | ".join([
            f"{c.get('concept', '')}: {c.get('description', '')}" 
            for c in self.key_concepts
        ])
        
        # Evidence snippet (avoid overwhelming the embedding)
        evidence_snippet = self.evidence[:300] if self.evidence else ""
        
        # Keywords with controlled weight
        keywords_text = ""
        if self.seed_keywords:
            # Repeat keywords to increase embedding weight
            keywords_text = (" ".join(self.seed_keywords) + " ") * keyword_weight
        
        # Structured format helps embedding model
        full_text = f"""Theory: {name_text}
Mechanisms: {concepts_text}
Evidence: {evidence_snippet}
Terms: {keywords_text}"""
        
        return full_text
    
    def extract_core_terms(self) -> Set[str]:
        """
        Extract meaningful biological/technical terms
        Uses simple heuristics - can be enhanced with biomedical NER
        """
        if self.core_terms:
            return self.core_terms
        
        # Combine all text sources
        text = f"{self.original_name} "
        text += " ".join([c.get('concept', '') for c in self.key_concepts])
        text += " " + " ".join(self.seed_keywords[:10])  # Limit keywords
        
        # Tokenize and clean
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {'theory', 'hypothesis', 'model', 'aging', 'ageing', 
                    'related', 'associated', 'mechanism', 'process'}
        
        # Keep meaningful terms
        meaningful = [w for w in words if w not in stopwords and len(w) > 3]
        
        self.core_terms = set(meaningful)
        return self.core_terms


class EmbeddingGenerator:
    """Generate semantic embeddings for theories"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Recommended models:
        - "all-MiniLM-L6-v2": Fast, good quality (384 dim)
        - "all-mpnet-base-v2": Better quality (768 dim)
        - "biobert-base-cased-v1.1": Domain-specific for biology
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, theories: List[Theory], 
                          batch_size: int = 32,
                          keyword_weight: int = 1) -> np.ndarray:
        """
        Generate embeddings for all theories efficiently
        
        Args:
            theories: List of Theory objects
            batch_size: Process in batches for efficiency
            keyword_weight: How much to weight keywords (1-3 recommended)
        
        Returns:
            numpy array of shape (n_theories, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(theories)} theories...")
        
        # Prepare texts
        texts = [t.to_embedding_text(keyword_weight) for t in theories]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store embeddings in theory objects
        for theory, embedding in zip(theories, embeddings):
            theory.embedding = embedding
        
        logger.info(f"Generated embeddings with dimension {embeddings.shape[1]}")
        return embeddings


# ============================================================================
# STAGE 2: INTELLIGENT PREPROCESSING
# ============================================================================

class TheoryPreprocessor:
    """Clean and standardize theory names before clustering"""
    
    # Patterns to remove
    REMOVAL_PATTERNS = [
        r'\btheory of\b',
        r'\btheory\b',
        r'\bhypothesis\b',
        r'\bmodel\b',
        r'\b-related\b',
        r'\brelated to\b',
        r'\bassociated with\b',
        r'\bthe\b'
    ]
    
    # Standardize synonyms (expand this based on your domain)
    STANDARDIZATION_MAP = {
        "mitochondrial": ["mitochondria", "mitochondrion", "mitochondrial"],
        "ROS": ["reactive oxygen species", "oxidative stress", "free radical"],
        "telomere": ["telomeric", "telomerase", "telomere"],
        "senescence": ["senescent", "cellular aging", "cell aging"],
        "inflammation": ["inflammatory", "inflammaging", "chronic inflammation"],
        "DNA damage": ["genomic instability", "genetic damage", "genome damage"],
        "proteostasis": ["protein homeostasis", "proteome maintenance"],
        "autophagy": ["autophagic", "autophagosome"],
        "apoptosis": ["apoptotic", "programmed cell death"],
        "stem cell": ["stem cells", "progenitor cell"],
        "NAD": ["nad+", "nicotinamide"],
        "mTOR": ["mtor", "mechanistic target of rapamycin"],
        "AMPK": ["ampk", "amp-activated protein kinase"]
    }
    
    def __init__(self):
        self.cache = {}
    
    def normalize_name(self, name: str) -> str:
        """
        Standardize theory name syntax
        
        Example:
        "Theory of Mitochondrial ROS Accumulation Related to Aging"
        → "mitochondrial ros accumulation"
        """
        if name in self.cache:
            return self.cache[name]
        
        original = name
        name = name.lower().strip()
        
        # Remove common patterns
        for pattern in self.REMOVAL_PATTERNS:
            name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
        
        # Standardize synonyms
        for standard, variants in self.STANDARDIZATION_MAP.items():
            for variant in variants:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                name = re.sub(pattern, standard, name, flags=re.IGNORECASE)
        
        # Clean up whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        self.cache[original] = name
        return name
    
    def extract_canonical_form(self, theory: Theory) -> str:
        """
        Generate canonical representation focusing on mechanism
        
        Strategy:
        1. Take primary concept (usually most important)
        2. Combine with normalized name
        3. Standardize terminology
        """
        # Get primary mechanism from first key concept
        primary_concept = ""
        if theory.key_concepts:
            primary_concept = theory.key_concepts[0].get('concept', '')
        
        # Combine with theory name
        combined = f"{primary_concept} {theory.original_name}"
        
        # Normalize
        canonical = self.normalize_name(combined)
        
        theory.canonical_form = canonical
        return canonical
    
    def preprocess_batch(self, theories: List[Theory]) -> List[Theory]:
        """Preprocess all theories"""
        logger.info(f"Preprocessing {len(theories)} theories...")
        
        for theory in theories:
            theory.extract_core_terms()
            self.extract_canonical_form(theory)
        
        logger.info("Preprocessing complete")
        return theories


# ============================================================================
# STAGE 3: MULTI-DIMENSIONAL SIMILARITY ANALYSIS
# ============================================================================

class TheorySimilarityEngine:
    """Calculate similarity using multiple signals"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with custom weights
        
        Default weights (tune based on your data):
        - embedding: 0.40 (semantic meaning)
        - terms: 0.30 (keyword overlap)
        - concepts: 0.30 (structured concept overlap)
        """
        self.weights = weights or {
            'embedding': 0.40,
            'terms': 0.30,
            'concepts': 0.30
        }
        
        # Validate weights sum to 1
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            for k in self.weights:
                self.weights[k] /= total
    
    def embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def term_overlap(self, theory1: Theory, theory2: Theory) -> float:
        """
        Jaccard similarity of core terms
        
        Jaccard = |A ∩ B| / |A ∪ B|
        """
        terms1 = theory1.core_terms or theory1.extract_core_terms()
        terms2 = theory2.core_terms or theory2.extract_core_terms()
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return intersection / union if union > 0 else 0.0
    
    def concept_overlap(self, theory1: Theory, theory2: Theory) -> float:
        """
        Overlap in key concepts (structured data)
        
        More reliable than term overlap as it uses extracted concepts
        """
        concepts1 = {c.get('concept', '').lower() for c in theory1.key_concepts}
        concepts2 = {c.get('concept', '').lower() for c in theory2.key_concepts}
        
        # Remove empty strings
        concepts1 = {c for c in concepts1 if c}
        concepts2 = {c for c in concepts2 if c}
        
        if not concepts1 or not concepts2:
            return 0.0
        
        intersection = len(concepts1 & concepts2)
        union = len(concepts1 | concepts2)
        
        return intersection / union if union > 0 else 0.0
    
    def detect_hierarchical_relationship(self, theory1: Theory, 
                                        theory2: Theory) -> Dict:
        """
        Detect if one theory is a specialization of another
        
        Returns:
            {
                'relationship': 'parent-child' | 'peer' | 'identical',
                'parent': 1 | 2 | None,
                'confidence': float
            }
        """
        name1 = theory1.canonical_form.lower()
        name2 = theory2.canonical_form.lower()
        
        # Check name containment
        if name1 == name2:
            return {'relationship': 'identical', 'parent': None, 'confidence': 1.0}
        
        if name1 in name2 and len(name1) < len(name2) * 0.7:
            # name1 is shorter and contained in name2 -> likely parent
            return {'relationship': 'parent-child', 'parent': 1, 'confidence': 0.8}
        
        if name2 in name1 and len(name2) < len(name1) * 0.7:
            return {'relationship': 'parent-child', 'parent': 2, 'confidence': 0.8}
        
        # Check concept subsumption
        concepts1 = {c.get('concept', '').lower() for c in theory1.key_concepts}
        concepts2 = {c.get('concept', '').lower() for c in theory2.key_concepts}
        
        if concepts1 and concepts2:
            if concepts1.issubset(concepts2) and len(concepts1) < len(concepts2):
                return {'relationship': 'parent-child', 'parent': 1, 'confidence': 0.7}
            
            if concepts2.issubset(concepts1) and len(concepts2) < len(concepts1):
                return {'relationship': 'parent-child', 'parent': 2, 'confidence': 0.7}
        
        return {'relationship': 'peer', 'parent': None, 'confidence': 0.5}
    
    def combined_similarity(self, theory1: Theory, theory2: Theory) -> Dict:
        """
        Multi-signal similarity score with breakdown
        
        Returns detailed similarity metrics for analysis
        """
        # Calculate individual similarities
        emb_sim = self.embedding_similarity(theory1.embedding, theory2.embedding)
        term_sim = self.term_overlap(theory1, theory2)
        concept_sim = self.concept_overlap(theory1, theory2)
        hierarchy = self.detect_hierarchical_relationship(theory1, theory2)
        
        # Weighted combination
        combined = (
            self.weights['embedding'] * emb_sim +
            self.weights['terms'] * term_sim +
            self.weights['concepts'] * concept_sim
        )
        
        # Boost for hierarchical relationships
        if hierarchy['relationship'] == 'parent-child':
            combined *= 1.1  # 10% boost
        elif hierarchy['relationship'] == 'identical':
            combined = 0.95  # Force high similarity
        
        # Clip to [0, 1]
        combined = min(1.0, max(0.0, combined))
        
        return {
            'combined': combined,
            'embedding': emb_sim,
            'terms': term_sim,
            'concepts': concept_sim,
            'hierarchy': hierarchy
        }
    
    def compute_similarity_matrix(self, theories: List[Theory]) -> np.ndarray:
        """
        Compute full pairwise similarity matrix
        
        Returns:
            NxN matrix where element [i,j] = similarity(theory_i, theory_j)
        """
        n = len(theories)
        similarity_matrix = np.zeros((n, n))
        
        logger.info(f"Computing {n}x{n} similarity matrix...")
        
        # Compute upper triangle (matrix is symmetric)
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # Self-similarity
            
            for j in range(i + 1, n):
                sim = self.combined_similarity(theories[i], theories[j])
                similarity_matrix[i, j] = sim['combined']
                similarity_matrix[j, i] = sim['combined']
        
        logger.info("Similarity matrix computed")
        return similarity_matrix


# ============================================================================
# HELPER: LOAD DATA FROM YOUR JSON FORMAT
# ============================================================================

def load_theories_from_json(json_path: str) -> List[Theory]:
    """
    Load theories from your extracted JSON format
    
    Expects format from your document
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    theories = []
    
    for paper in data:
        doi = paper.get('doi', '')
        confidence = paper.get('confidence_score', 0)
        
        for theory_data in paper.get('theories', []):
            theory = Theory(
                original_name=theory_data.get('name', ''),
                key_concepts=theory_data.get('key_concepts', []),
                paper_doi=doi,
                paper_focus=theory_data.get('paper_focus', 0),
                confidence_score=confidence,
                evidence=theory_data.get('evidence', ''),
                mode=theory_data.get('mode', ''),
                seed_keywords=[]  # Add if available in your data
            )
            theories.append(theory)
    
    logger.info(f"Loaded {len(theories)} theories from {len(data)} papers")
    return theories


# ============================================================================
# MAIN EXECUTION: STAGES 1-3
# ============================================================================

def run_stages_1_to_3(json_path: str, 
                      output_path: str = "similarity_matrix.npz",
                      embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Execute Stages 1-3 of normalization pipeline
    
    Args:
        json_path: Path to extracted theories JSON
        output_path: Where to save similarity matrix
        embedding_model: Which sentence-transformer model to use
    
    Returns:
        Tuple of (theories, similarity_matrix, embeddings)
    """
    
    # STAGE 1: Load and embed
    theories = load_theories_from_json(json_path)
    
    embedder = EmbeddingGenerator(embedding_model)
    embeddings = embedder.generate_embeddings(theories, keyword_weight=1)
    
    # STAGE 2: Preprocess
    preprocessor = TheoryPreprocessor()
    theories = preprocessor.preprocess_batch(theories)
    
    # STAGE 3: Compute similarities
    similarity_engine = TheorySimilarityEngine()
    similarity_matrix = similarity_engine.compute_similarity_matrix(theories)
    
    # Save results
    np.savez_compressed(
        output_path,
        similarity_matrix=similarity_matrix,
        embeddings=embeddings
    )
    
    logger.info(f"Saved similarity matrix to {output_path}")
    logger.info(f"Matrix shape: {similarity_matrix.shape}")
    logger.info(f"Mean similarity: {similarity_matrix.mean():.3f}")
    logger.info(f"Max similarity (excluding diagonal): {np.max(similarity_matrix - np.eye(len(theories))):.3f}")
    
    return theories, similarity_matrix, embeddings


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Run on your data
    theories, sim_matrix, embeddings = run_stages_1_to_3(
        json_path="extracted_theories.json",
        output_path="similarity_matrix.npz",
        embedding_model="all-MiniLM-L6-v2"  # Fast and good
    )
    
    # Analyze results
    print(f"\nProcessed {len(theories)} theories")
    print(f"Similarity matrix: {sim_matrix.shape}")
    print(f"High similarity pairs (>0.85): {(sim_matrix > 0.85).sum() // 2}")