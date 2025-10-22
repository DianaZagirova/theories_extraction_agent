# Pipeline Improvement Roadmap

## Priority Matrix

| Priority | Task | Impact | Effort | ROI |
|----------|------|--------|--------|-----|
| 🔴 P0 | Integrate ontology files | High | Low | **Very High** |
| 🔴 P0 | Add semantic matching (Stage 1) | High | Medium | **High** |
| 🔴 P0 | Redesign Stage 3 (ontology-first) | Very High | High | **Very High** |
| 🟡 P1 | Normalize Stage 2 extractions | Medium | Low | **High** |
| 🟡 P1 | Add validation metrics | Medium | Medium | **Medium** |
| 🟢 P2 | Add caching/batching | Low | Low | **Medium** |
| 🟢 P2 | Optimize costs | Low | Medium | **Low** |

---

## P0: Critical Improvements (Do First)

### 1. Integrate Ontology Files 🔴

**Problem**: Pipeline ignores existing ontology files with canonical theories and mechanisms.

**Files to integrate**:
- `ontology/groups_ontology_alliases.json` - 46 canonical theories with aliases
- `ontology/group_ontology_mechanisms.json` - Canonical mechanisms for each theory

**Implementation**:

```python
# src/normalization/ontology_loader.py
"""Load and parse ontology files."""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CanonicalTheory:
    """Represents a canonical theory from ontology."""
    name: str
    aliases: List[str]
    abbreviations: List[str]
    category: str
    subcategory: str
    key_players: List[str]
    pathways: List[str]
    mechanisms: List[str]
    level_of_explanation: str
    type_of_cause: str
    temporal_focus: str
    adaptiveness: str

class OntologyLoader:
    """Load canonical theories from ontology files."""
    
    def __init__(self, 
                 aliases_path: str = 'ontology/groups_ontology_alliases.json',
                 mechanisms_path: str = 'ontology/group_ontology_mechanisms.json'):
        self.aliases_path = Path(aliases_path)
        self.mechanisms_path = Path(mechanisms_path)
        self.canonical_theories = []
        self.load_ontology()
    
    def load_ontology(self):
        """Load both ontology files and merge."""
        # Load aliases
        with open(self.aliases_path, 'r') as f:
            aliases_data = json.load(f)
        
        # Load mechanisms
        with open(self.mechanisms_path, 'r') as f:
            mechanisms_data = json.load(f)
        
        # Parse and merge
        for category, subcats in aliases_data['TheoriesOfAging'].items():
            for subcat, theories in subcats.items():
                for theory in theories:
                    name = theory['name']
                    
                    # Get mechanisms if available
                    mech_data = mechanisms_data.get(name, {})
                    
                    canonical = CanonicalTheory(
                        name=name,
                        aliases=theory.get('aliases', []),
                        abbreviations=theory.get('abbreviations', []),
                        category=category,
                        subcategory=subcat,
                        key_players=mech_data.get('KEY PLAYERS', []),
                        pathways=mech_data.get('PATHWAYS', []),
                        mechanisms=mech_data.get('MECHANISMS', []),
                        level_of_explanation=mech_data.get('LEVEL OF EXPLANATION', ''),
                        type_of_cause=mech_data.get('TYPE OF CAUSE', ''),
                        temporal_focus=mech_data.get('TEMPORAL FOCUS', ''),
                        adaptiveness=mech_data.get('ADAPTIVENESS', '')
                    )
                    self.canonical_theories.append(canonical)
        
        print(f"✓ Loaded {len(self.canonical_theories)} canonical theories from ontology")
    
    def get_all_names(self) -> List[str]:
        """Get all canonical names."""
        return [t.name for t in self.canonical_theories]
    
    def get_all_aliases(self) -> Dict[str, str]:
        """Get mapping of alias -> canonical name."""
        alias_map = {}
        for theory in self.canonical_theories:
            for alias in theory.aliases:
                alias_map[alias] = theory.name
        return alias_map
    
    def get_all_abbreviations(self) -> Dict[str, str]:
        """Get mapping of abbreviation -> canonical name."""
        abbrev_map = {}
        for theory in self.canonical_theories:
            for abbrev in theory.abbreviations:
                abbrev_map[abbrev] = theory.name
        return abbrev_map
    
    def get_theory(self, name: str) -> CanonicalTheory:
        """Get canonical theory by name."""
        for theory in self.canonical_theories:
            if theory.name == name:
                return theory
        return None
    
    def get_mechanisms_dict(self) -> Dict[str, List[str]]:
        """Get mapping of theory name -> mechanisms."""
        return {t.name: t.mechanisms for t in self.canonical_theories}
```

**Update Stage 1**:

```python
# src/normalization/stage1_fuzzy_matching.py

from .ontology_loader import OntologyLoader

class FuzzyMatcher:
    def __init__(self):
        # Load from ontology instead of hardcoded
        self.ontology = OntologyLoader()
        self.canonical_theories = self.ontology.get_all_names()
        self.aliases = self.ontology.get_all_aliases()
        self.abbreviations = self.ontology.get_all_abbreviations()
        
        # Rest of initialization...
```

**Expected Impact**:
- ✅ Use all 46 canonical theories from ontology
- ✅ Automatic updates when ontology changes
- ✅ Access to canonical mechanisms for validation
- ✅ Foundation for Stage 3 improvements

---

### 2. Add Semantic Matching (Stage 1) 🔴

**Problem**: Only string matching, misses semantically equivalent theories.

**Implementation**:

```python
# src/normalization/semantic_matcher.py
"""Semantic matching using embeddings."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import pickle
from pathlib import Path

class SemanticMatcher:
    """Match theories using semantic embeddings."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_path: str = 'cache/embeddings_cache.pkl'):
        self.model = SentenceTransformer(model_name)
        self.cache_path = Path(cache_path)
        self.embeddings_cache = self.load_cache()
    
    def load_cache(self) -> dict:
        """Load cached embeddings."""
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_cache(self):
        """Save embeddings cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding with caching."""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = self.model.encode(text)
        self.embeddings_cache[text] = embedding
        return embedding
    
    def compute_canonical_embeddings(self, canonical_theories: List[CanonicalTheory]):
        """Precompute embeddings for canonical theories."""
        embeddings = []
        for theory in canonical_theories:
            # Combine name, aliases, and mechanisms for rich representation
            text_parts = [theory.name] + theory.aliases
            if theory.mechanisms:
                text_parts.extend(theory.mechanisms[:3])  # Top 3 mechanisms
            
            combined_text = ". ".join(text_parts)
            embedding = self.encode(combined_text)
            embeddings.append(embedding)
        
        self.save_cache()
        return np.array(embeddings)
    
    def find_best_match(self, 
                       query: str, 
                       canonical_embeddings: np.ndarray,
                       canonical_names: List[str],
                       threshold: float = 0.75) -> Tuple[Optional[str], float]:
        """Find best semantic match."""
        query_emb = self.encode(query)
        
        # Compute cosine similarities
        similarities = np.dot(canonical_embeddings, query_emb) / (
            np.linalg.norm(canonical_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return canonical_names[best_idx], float(best_score)
        
        return None, 0.0
```

**Update Stage 1 to use semantic matching**:

```python
# src/normalization/stage1_fuzzy_matching.py

class EnhancedFuzzyMatcher:
    def __init__(self):
        self.ontology = OntologyLoader()
        self.semantic_matcher = SemanticMatcher()
        
        # Precompute canonical embeddings
        self.canonical_embeddings = self.semantic_matcher.compute_canonical_embeddings(
            self.ontology.canonical_theories
        )
        self.canonical_names = self.ontology.get_all_names()
    
    def match_theory(self, theory_name: str, concept_text: str) -> MatchResult:
        """Enhanced matching with semantic layer."""
        
        # Step 1: Abbreviation match (highest confidence)
        abbrev_match = self.match_abbreviation(theory_name)
        if abbrev_match:
            return MatchResult(
                matched=True,
                canonical_name=abbrev_match,
                match_type='abbreviation',
                confidence=0.95,
                score=1.0
            )
        
        # Step 2: Exact normalized match
        exact_match = self.match_exact_normalized(theory_name)
        if exact_match:
            return MatchResult(
                matched=True,
                canonical_name=exact_match,
                match_type='exact_normalized',
                confidence=0.90,
                score=1.0
            )
        
        # Step 3: High confidence fuzzy match
        fuzzy_match, score = self.match_fuzzy(theory_name)
        if score >= 90:
            return MatchResult(
                matched=True,
                canonical_name=fuzzy_match,
                match_type='high_confidence_fuzzy',
                confidence=0.85,
                score=score/100
            )
        
        # Step 4: Semantic match (NEW!)
        semantic_match, sim_score = self.semantic_matcher.find_best_match(
            query=f"{theory_name}. {concept_text}",
            canonical_embeddings=self.canonical_embeddings,
            canonical_names=self.canonical_names,
            threshold=0.75
        )
        if semantic_match:
            return MatchResult(
                matched=True,
                canonical_name=semantic_match,
                match_type='semantic',
                confidence=0.80,
                score=sim_score
            )
        
        # No match
        return MatchResult(matched=False)
```

**Expected Impact**:
- Match rate: 19.1% → **35-40%**
- Cost savings: **$15-20** (fewer theories to Stage 2)
- Better quality: Semantic understanding vs string matching

---

### 3. Redesign Stage 3 (Ontology-First) 🔴

**Problem**: Current Stage 3 uses string-based Jaccard similarity, which fails to group semantically equivalent theories.

**New Approach**: Match to ontology first, then cluster remainder.

**Implementation**:

```python
# src/normalization/stage3_ontology_grouping.py
"""
Stage 3: Ontology-First Theory Grouping

Strategy:
1. Match Stage 2 theories to canonical theories (ontology)
2. Cluster remaining theories by semantic similarity
3. Create hierarchical structure
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

from .ontology_loader import OntologyLoader
from .semantic_matcher import SemanticMatcher

@dataclass
class TheoryGroup:
    """Enhanced theory group with ontology linkage."""
    group_id: str
    canonical_name: Optional[str]  # From ontology
    representative_name: str
    theory_ids: List[str] = field(default_factory=list)
    theory_count: int = 0
    
    # Ontology data
    ontology_category: Optional[str] = None
    ontology_subcategory: Optional[str] = None
    canonical_mechanisms: List[str] = field(default_factory=list)
    canonical_key_players: List[str] = field(default_factory=list)
    canonical_pathways: List[str] = field(default_factory=list)
    
    # Grouping metadata
    match_type: str = 'unknown'  # 'ontology', 'semantic_cluster', 'singleton'
    confidence: float = 0.0
    
    def to_dict(self):
        return {
            'group_id': self.group_id,
            'canonical_name': self.canonical_name,
            'representative_name': self.representative_name,
            'theory_ids': self.theory_ids,
            'theory_count': self.theory_count,
            'ontology_category': self.ontology_category,
            'ontology_subcategory': self.ontology_subcategory,
            'canonical_mechanisms': self.canonical_mechanisms,
            'canonical_key_players': self.canonical_key_players,
            'canonical_pathways': self.canonical_pathways,
            'match_type': self.match_type,
            'confidence': self.confidence
        }

class OntologyFirstGrouper:
    """Group theories using ontology-first approach."""
    
    def __init__(self, semantic_threshold: float = 0.80):
        self.ontology = OntologyLoader()
        self.semantic_matcher = SemanticMatcher()
        self.semantic_threshold = semantic_threshold
        
        # Precompute canonical embeddings
        self.canonical_embeddings = self._compute_canonical_embeddings()
        
        self.groups = []
        self.stats = {
            'total_theories': 0,
            'ontology_matched': 0,
            'semantic_clustered': 0,
            'singletons': 0,
            'total_groups': 0
        }
    
    def _compute_canonical_embeddings(self) -> np.ndarray:
        """Compute embeddings for canonical theories."""
        embeddings = []
        for theory in self.ontology.canonical_theories:
            # Rich representation: name + mechanisms
            text = f"{theory.name}. " + ". ".join(theory.mechanisms[:5])
            emb = self.semantic_matcher.encode(text)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def _compute_theory_embedding(self, theory: Dict) -> np.ndarray:
        """Compute embedding for a theory."""
        metadata = theory.get('stage2_metadata', {})
        
        # Combine all semantic content
        parts = [theory['original_name']]
        parts.extend(metadata.get('mechanisms', [])[:5])
        parts.extend(metadata.get('key_players', [])[:5])
        
        text = ". ".join(parts)
        return self.semantic_matcher.encode(text)
    
    def match_to_ontology(self, theory: Dict) -> Tuple[Optional[str], float]:
        """Match theory to canonical theory."""
        theory_emb = self._compute_theory_embedding(theory)
        
        # Compute similarities to all canonical theories
        similarities = np.dot(self.canonical_embeddings, theory_emb) / (
            np.linalg.norm(self.canonical_embeddings, axis=1) * np.linalg.norm(theory_emb)
        )
        
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score >= self.semantic_threshold:
            canonical_name = self.ontology.canonical_theories[best_idx].name
            return canonical_name, float(best_score)
        
        return None, 0.0
    
    def group_theories(self, stage1_theories: List[Dict], stage2_theories: List[Dict]):
        """Group theories using ontology-first approach."""
        print(f"\n🔄 Grouping theories (ontology-first approach)...")
        
        all_theories = stage1_theories + stage2_theories
        self.stats['total_theories'] = len(all_theories)
        
        # Step 1: Group Stage 1 theories (already matched to canonical)
        print(f"  Step 1: Grouping Stage 1 theories by canonical name...")
        ontology_groups = defaultdict(list)
        
        for theory in stage1_theories:
            match_result = theory.get('match_result', {})
            if match_result.get('matched'):
                canonical_name = match_result['canonical_name']
                ontology_groups[canonical_name].append(theory)
        
        print(f"    Created {len(ontology_groups)} groups from Stage 1")
        
        # Step 2: Match Stage 2 theories to ontology
        print(f"  Step 2: Matching Stage 2 theories to ontology...")
        unmatched_stage2 = []
        
        for theory in stage2_theories:
            canonical_name, score = self.match_to_ontology(theory)
            if canonical_name:
                ontology_groups[canonical_name].append(theory)
                self.stats['ontology_matched'] += 1
            else:
                unmatched_stage2.append(theory)
        
        print(f"    Matched {self.stats['ontology_matched']} Stage 2 theories to ontology")
        print(f"    Remaining unmatched: {len(unmatched_stage2)}")
        
        # Step 3: Create groups from ontology matches
        group_counter = 0
        for canonical_name, theories in ontology_groups.items():
            group_counter += 1
            
            # Get canonical theory data
            canonical_theory = self.ontology.get_theory(canonical_name)
            
            group = TheoryGroup(
                group_id=f"G{group_counter:04d}",
                canonical_name=canonical_name,
                representative_name=canonical_name,
                theory_ids=[t['theory_id'] for t in theories],
                theory_count=len(theories),
                ontology_category=canonical_theory.category if canonical_theory else None,
                ontology_subcategory=canonical_theory.subcategory if canonical_theory else None,
                canonical_mechanisms=canonical_theory.mechanisms if canonical_theory else [],
                canonical_key_players=canonical_theory.key_players if canonical_theory else [],
                canonical_pathways=canonical_theory.pathways if canonical_theory else [],
                match_type='ontology',
                confidence=0.90
            )
            self.groups.append(group)
        
        # Step 4: Cluster remaining theories by semantic similarity
        print(f"  Step 3: Clustering {len(unmatched_stage2)} unmatched theories...")
        
        if unmatched_stage2:
            clusters = self._cluster_theories(unmatched_stage2)
            
            for cluster_theories in clusters:
                group_counter += 1
                
                # Get representative name (most common or first)
                names = [t['original_name'] for t in cluster_theories]
                rep_name = Counter(names).most_common(1)[0][0]
                
                # Get primary category
                categories = [t.get('stage2_metadata', {}).get('primary_category') 
                            for t in cluster_theories]
                primary_cat = Counter(categories).most_common(1)[0][0] if categories else None
                
                group = TheoryGroup(
                    group_id=f"G{group_counter:04d}",
                    canonical_name=None,
                    representative_name=rep_name,
                    theory_ids=[t['theory_id'] for t in cluster_theories],
                    theory_count=len(cluster_theories),
                    ontology_category=primary_cat,
                    match_type='semantic_cluster' if len(cluster_theories) > 1 else 'singleton',
                    confidence=0.70
                )
                self.groups.append(group)
                
                if len(cluster_theories) == 1:
                    self.stats['singletons'] += 1
                else:
                    self.stats['semantic_clustered'] += len(cluster_theories)
        
        self.stats['total_groups'] = len(self.groups)
        
        print(f"✓ Created {len(self.groups)} theory groups")
        print(f"  Ontology-matched: {len(ontology_groups)} groups")
        print(f"  Semantic clusters: {self.stats['semantic_clustered']} theories")
        print(f"  Singletons: {self.stats['singletons']} theories")
        
        return self.groups
    
    def _cluster_theories(self, theories: List[Dict]) -> List[List[Dict]]:
        """Cluster theories using semantic similarity."""
        if not theories:
            return []
        
        # Compute embeddings
        embeddings = np.array([self._compute_theory_embedding(t) for t in theories])
        
        # Use DBSCAN for clustering
        from sklearn.cluster import DBSCAN
        
        # Compute distance matrix
        n = len(embeddings)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                distances[i, j] = distances[j, i] = 1 - sim
        
        # Cluster
        clustering = DBSCAN(eps=1-self.semantic_threshold, min_samples=1, metric='precomputed')
        labels = clustering.fit_predict(distances)
        
        # Group by label
        clusters = defaultdict(list)
        for theory, label in zip(theories, labels):
            clusters[label].append(theory)
        
        return list(clusters.values())
    
    def print_statistics(self):
        """Print grouping statistics."""
        print("\n" + "="*80)
        print("STAGE 3: ONTOLOGY-FIRST GROUPING STATISTICS")
        print("="*80)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"\nGrouping results:")
        print(f"  Ontology-matched: {self.stats['ontology_matched']} theories")
        print(f"  Semantic clusters: {self.stats['semantic_clustered']} theories")
        print(f"  Singletons: {self.stats['singletons']} theories")
        print(f"  Total groups: {self.stats['total_groups']}")
        
        # Compression
        compression = (1 - self.stats['total_groups'] / self.stats['total_theories']) * 100
        print(f"\nCompression: {compression:.1f}% ({self.stats['total_theories']} → {self.stats['total_groups']})")
        
        # Ontology coverage
        ontology_coverage = self.stats['ontology_matched'] / self.stats['total_theories'] * 100
        print(f"Ontology coverage: {ontology_coverage:.1f}%")
        print("="*80)
```

**Expected Impact**:
- Accuracy: 60-70% → **85-90%**
- Ontology alignment: 0% → **70-80%**
- Compression: 3:1 → **5:1**
- Validated groups using expert knowledge

---

## P1: Important Improvements (Do Next)

### 4. Normalize Stage 2 Extractions 🟡

Add term normalization and validation:

```python
# src/normalization/term_normalizer.py

import re
from typing import List, Set
from difflib import get_close_matches

class TermNormalizer:
    """Normalize extracted terms for consistent comparison."""
    
    def __init__(self, ontology_terms: Dict[str, List[str]]):
        self.ontology_terms = ontology_terms
        self.synonyms = self._load_synonyms()
    
    def _load_synonyms(self) -> Dict[str, str]:
        """Load common synonyms."""
        return {
            'ros': 'reactive oxygen species',
            'free radicals': 'reactive oxygen species',
            'mtdna': 'mitochondrial dna',
            'atp': 'adenosine triphosphate',
            # Add more...
        }
    
    def normalize(self, term: str) -> str:
        """Normalize a single term."""
        # Lowercase
        term = term.lower().strip()
        
        # Remove articles
        term = re.sub(r'\b(the|a|an)\b', '', term).strip()
        
        # Replace synonyms
        if term in self.synonyms:
            term = self.synonyms[term]
        
        # Singular form (simple)
        if term.endswith('s') and not term.endswith('ss'):
            term = term[:-1]
        
        return term
    
    def normalize_list(self, terms: List[str]) -> List[str]:
        """Normalize list of terms."""
        normalized = [self.normalize(t) for t in terms]
        return list(set(normalized))  # Remove duplicates
    
    def validate_against_ontology(self, terms: List[str], 
                                  ontology_key: str) -> List[str]:
        """Match terms to canonical ontology terms."""
        canonical_terms = self.ontology_terms.get(ontology_key, [])
        canonical_normalized = [self.normalize(t) for t in canonical_terms]
        
        validated = []
        for term in terms:
            norm_term = self.normalize(term)
            
            # Try exact match
            if norm_term in canonical_normalized:
                idx = canonical_normalized.index(norm_term)
                validated.append(canonical_terms[idx])
            else:
                # Try fuzzy match
                matches = get_close_matches(norm_term, canonical_normalized, n=1, cutoff=0.8)
                if matches:
                    idx = canonical_normalized.index(matches[0])
                    validated.append(canonical_terms[idx])
                else:
                    validated.append(term)  # Keep original
        
        return validated
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Day 1-2: Create `ontology_loader.py` and integrate
- [ ] Day 3-4: Add semantic matching to Stage 1
- [ ] Day 5: Test and validate improvements

### Week 2: Core Redesign
- [ ] Day 1-3: Implement `stage3_ontology_grouping.py`
- [ ] Day 4: Add term normalization
- [ ] Day 5: Integration testing

### Week 3: Polish
- [ ] Day 1-2: Add validation metrics
- [ ] Day 3: Optimize performance
- [ ] Day 4-5: Documentation and testing

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Stage 1 Match Rate | 19.1% | 35-40% | % matched theories |
| Stage 2 Cost | $35 | $15-20 | API costs |
| Stage 3 Accuracy | ~60% | 85-90% | Manual validation |
| Ontology Coverage | 0% | 70-80% | % matched to ontology |
| Compression Ratio | 3:1 | 5:1 | theories/groups |
| Processing Time | 53 min | 30 min | End-to-end time |

---

## Next Steps

1. **Review this roadmap** with stakeholders
2. **Prioritize P0 tasks** for immediate implementation
3. **Set up development branch** for improvements
4. **Create test suite** for validation
5. **Begin implementation** following timeline
