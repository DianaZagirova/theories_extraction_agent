"""
Stage 2: Alternative Hierarchical Clustering Strategy
Uses full-text embeddings and different approaches for each level.

Key differences from original:
1. Uses detailed embeddings for all levels (not just children)
2. Dynamic cluster count (not fixed thresholds)
3. Feature-weighted similarity at all levels
4. Adaptive threshold selection
"""

import json
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os


@dataclass
class TheoryCluster:
    """Represents a cluster of theories at any level."""
    cluster_id: str
    level: str  # 'family', 'parent', or 'child'
    theory_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    canonical_name: Optional[str] = None
    alternative_names: List[str] = field(default_factory=list)
    coherence_score: float = 0.0
    parent_cluster_id: Optional[str] = None
    child_cluster_ids: List[str] = field(default_factory=list)
    is_singleton: bool = False
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'cluster_id': self.cluster_id,
            'level': self.level,
            'theory_ids': self.theory_ids,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'canonical_name': self.canonical_name,
            'alternative_names': self.alternative_names,
            'coherence_score': self.coherence_score,
            'parent_cluster_id': self.parent_cluster_id,
            'child_cluster_ids': self.child_cluster_ids,
            'theory_count': len(self.theory_ids),
            'is_singleton': self.is_singleton
        }


class AlternativeClusterer:
    """Alternative clustering approach with different strategies per level."""
    
    def __init__(self, 
                 target_families: int = 50,
                 min_cluster_size: int = 3,
                 max_cluster_size: int = 50):
        """
        Initialize alternative clusterer.
        
        Args:
            target_families: Target number of families (will adjust dynamically)
            min_cluster_size: Minimum theories per cluster
            max_cluster_size: Maximum theories per cluster (will split if exceeded)
        """
        self.target_families = target_families
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        self.families = []
        self.parents = []
        self.children = []
        
        self.stats = {
            'total_theories': 0,
            'num_families': 0,
            'num_parents': 0,
            'num_children': 0,
            'singleton_families': 0,
            'singleton_parents': 0,
            'singleton_children': 0,
            'outliers_preserved': 0,
            'avg_coherence_family': 0.0,
            'avg_coherence_parent': 0.0,
            'avg_coherence_child': 0.0,
            'silhouette_score': 0.0
        }
    
    def load_embeddings(self, embeddings_path: str) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
        """Load embeddings from Stage 1."""
        print(f"📂 Loading embeddings from {embeddings_path}...")
        
        # Try pickle first (faster)
        pickle_path = embeddings_path.replace('.json', '.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(embeddings_path, 'r') as f:
                data = json.load(f)
        
        theories = data['theories']
        embeddings_list = data['embeddings']
        
        # Extract detailed embeddings (full text)
        detailed_embeddings = np.array([emb['detailed_embedding'] for emb in embeddings_list])
        
        print(f"✓ Loaded {len(theories)} theories")
        print(f"  Embedding dimension: {detailed_embeddings.shape[1]}")
        
        self.stats['total_theories'] = len(theories)
        
        return theories, detailed_embeddings, embeddings_list
    
    def _compute_feature_weighted_similarity(self, embeddings: np.ndarray, 
                                            features: List[Dict]) -> np.ndarray:
        """
        Compute similarity matrix with feature-based weighting.
        Theories with shared mechanisms/pathways get bonus similarity.
        """
        # Base similarity from embeddings
        base_similarity = cosine_similarity(embeddings)
        
        # Feature-based bonus
        n = len(features)
        feature_bonus = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                bonus = 0.0
                
                feat_i = features[i]
                feat_j = features[j]
                
                # Mechanism overlap
                mech_i = set(m['entity'] if isinstance(m, dict) else m 
                           for m in feat_i.get('mechanisms', []))
                mech_j = set(m['entity'] if isinstance(m, dict) else m 
                           for m in feat_j.get('mechanisms', []))
                
                if mech_i and mech_j:
                    overlap = len(mech_i & mech_j) / len(mech_i | mech_j)
                    bonus += 0.15 * overlap
                
                # Pathway overlap
                path_i = set(feat_i.get('pathways', []))
                path_j = set(feat_j.get('pathways', []))
                
                if path_i and path_j:
                    overlap = len(path_i & path_j) / len(path_i | path_j)
                    bonus += 0.10 * overlap
                
                # Biological level match
                if feat_i.get('biological_level') == feat_j.get('biological_level'):
                    bonus += 0.05
                
                feature_bonus[i, j] = bonus
                feature_bonus[j, i] = bonus
        
        # Combine with weight
        combined_similarity = 0.7 * base_similarity + 0.3 * feature_bonus
        
        return np.clip(combined_similarity, 0, 1)
    
    def cluster_level1_families(self, theories: List[Dict], 
                                embeddings: np.ndarray,
                                features: List[Dict]) -> List[TheoryCluster]:
        """
        Level 1: Cluster into families using K-Means with optimal K.
        Uses full-text embeddings + feature weighting.
        """
        print(f"\n🔄 Level 1: Clustering into theory families...")
        print(f"   Strategy: K-Means with optimal K selection")
        print(f"   Target families: {self.target_families}")
        
        # Compute feature-weighted similarity
        similarity_matrix = self._compute_feature_weighted_similarity(embeddings, features)
        
        # Find optimal number of clusters using elbow method + silhouette
        optimal_k = self._find_optimal_k(
            embeddings, 
            min_k=max(10, self.target_families - 20),
            max_k=min(100, self.target_families + 20)
        )
        
        print(f"   Optimal K: {optimal_k}")
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette = silhouette_score(embeddings, labels)
        self.stats['silhouette_score'] = silhouette
        print(f"   Silhouette score: {silhouette:.3f}")
        
        # Create clusters
        families = []
        family_counter = 0
        
        for label in sorted(set(labels)):
            theory_indices = np.where(labels == label)[0]
            theory_ids = [theories[i]['theory_id'] for i in theory_indices]
            
            # Skip if too small
            if len(theory_ids) < self.min_cluster_size:
                # Mark as singletons
                for idx in theory_indices:
                    family_counter += 1
                    family = TheoryCluster(
                        cluster_id=f"F{family_counter:03d}",
                        level='family',
                        theory_ids=[theories[idx]['theory_id']],
                        centroid=embeddings[idx],
                        is_singleton=True,
                        canonical_name=f"Singleton: {theories[idx]['name']}"
                    )
                    families.append(family)
                    self.stats['singleton_families'] += 1
                continue
            
            # Check if too large - split if needed
            if len(theory_ids) > self.max_cluster_size:
                # Re-cluster with more clusters
                sub_families = self._split_large_cluster(
                    theories, embeddings, features, theory_indices, 'family', family_counter
                )
                families.extend(sub_families)
                family_counter += len(sub_families)
            else:
                # Normal cluster
                family_counter += 1
                centroid = embeddings[theory_indices].mean(axis=0)
                
                # Calculate coherence
                coherence = self._calculate_coherence(embeddings[theory_indices])
                
                family = TheoryCluster(
                    cluster_id=f"F{family_counter:03d}",
                    level='family',
                    theory_ids=theory_ids,
                    centroid=centroid,
                    coherence_score=coherence,
                    is_singleton=False
                )
                families.append(family)
        
        self.families = families
        self.stats['num_families'] = len(families)
        self.stats['avg_coherence_family'] = np.mean([f.coherence_score for f in families if not f.is_singleton])
        
        print(f"✓ Created {len(families)} theory families")
        print(f"  Regular clusters: {len(families) - self.stats['singleton_families']}")
        print(f"  Singleton clusters: {self.stats['singleton_families']}")
        print(f"  Avg coherence: {self.stats['avg_coherence_family']:.3f}")
        
        return families
    
    def cluster_level2_parents(self, theories: List[Dict], 
                              embeddings: np.ndarray,
                              features: List[Dict],
                              families: List[TheoryCluster]) -> List[TheoryCluster]:
        """
        Level 2: Cluster into parents using Agglomerative clustering.
        Uses moderate threshold for balanced grouping.
        """
        print(f"\n🔄 Level 2: Clustering into parent theories...")
        print(f"   Strategy: Agglomerative (threshold=0.55)")
        
        all_parents = []
        parent_counter = 0
        theory_id_to_idx = {t['theory_id']: i for i, t in enumerate(theories)}
        
        for family in families:
            # Singletons stay as single parents
            if family.is_singleton or len(family.theory_ids) < self.min_cluster_size:
                parent_counter += 1
                parent = TheoryCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    theory_ids=family.theory_ids,
                    parent_cluster_id=family.cluster_id,
                    is_singleton=family.is_singleton,
                    coherence_score=family.coherence_score
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
                if family.is_singleton:
                    self.stats['singleton_parents'] += 1
                continue
            
            # Get embeddings for this family
            family_indices = [theory_id_to_idx[tid] for tid in family.theory_ids]
            family_embeddings = embeddings[family_indices]
            family_features = [features[i] for i in family_indices]
            
            # Use Agglomerative clustering (DBSCAN was too sensitive)
            # Compute pairwise distances
            similarity = self._compute_feature_weighted_similarity(family_embeddings, family_features)
            distance = 1 - similarity
            
            # Agglomerative with moderate threshold
            threshold = 0.55  # Moderate threshold for parent level
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='average',
                metric='precomputed'
            )
            labels = clustering.fit_predict(distance)
            
            # Create parent clusters
            outlier_indices = []
            for label in sorted(set(labels)):
                if label == -1:
                    outlier_indices = np.where(labels == label)[0].tolist()
                    continue
                
                parent_counter += 1
                cluster_indices = np.where(labels == label)[0]
                parent_theory_ids = [family.theory_ids[i] for i in cluster_indices]
                
                centroid = family_embeddings[cluster_indices].mean(axis=0)
                coherence = self._calculate_coherence(family_embeddings[cluster_indices])
                
                parent = TheoryCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    theory_ids=parent_theory_ids,
                    centroid=centroid,
                    parent_cluster_id=family.cluster_id,
                    coherence_score=coherence,
                    is_singleton=False
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
            
            # Handle outliers as singletons
            for outlier_idx in outlier_indices:
                parent_counter += 1
                parent_theory_id = family.theory_ids[outlier_idx]
                
                parent = TheoryCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    theory_ids=[parent_theory_id],
                    centroid=family_embeddings[outlier_idx],
                    parent_cluster_id=family.cluster_id,
                    is_singleton=True
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
                self.stats['singleton_parents'] += 1
                self.stats['outliers_preserved'] += 1
        
        self.parents = all_parents
        self.stats['num_parents'] = len(all_parents)
        self.stats['avg_coherence_parent'] = np.mean([p.coherence_score for p in all_parents if not p.is_singleton and p.coherence_score > 0])
        
        print(f"✓ Created {len(all_parents)} parent theories")
        print(f"  Regular clusters: {len(all_parents) - self.stats['singleton_parents']}")
        print(f"  Singleton clusters: {self.stats['singleton_parents']}")
        if self.stats['avg_coherence_parent'] > 0:
            print(f"  Avg coherence: {self.stats['avg_coherence_parent']:.3f}")
        
        return all_parents
    
    def cluster_level3_children(self, theories: List[Dict],
                               embeddings: np.ndarray,
                               features: List[Dict],
                               parents: List[TheoryCluster]) -> List[TheoryCluster]:
        """
        Level 3: Cluster into children using Agglomerative with adaptive threshold.
        Uses feature-weighted similarity for fine-grained distinction.
        """
        print(f"\n🔄 Level 3: Clustering into child theories...")
        print(f"   Strategy: Agglomerative with adaptive threshold")
        
        all_children = []
        child_counter = 0
        theory_id_to_idx = {t['theory_id']: i for i, t in enumerate(theories)}
        
        for parent in parents:
            # Singletons and small parents stay as single children
            if parent.is_singleton or len(parent.theory_ids) < self.min_cluster_size:
                child_counter += 1
                child = TheoryCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    theory_ids=parent.theory_ids,
                    parent_cluster_id=parent.cluster_id,
                    is_singleton=parent.is_singleton,
                    coherence_score=parent.coherence_score
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
                if parent.is_singleton:
                    self.stats['singleton_children'] += 1
                continue
            
            # Get embeddings for this parent
            parent_indices = [theory_id_to_idx[tid] for tid in parent.theory_ids]
            parent_embeddings = embeddings[parent_indices]
            parent_features = [features[i] for i in parent_indices]
            
            # Compute feature-weighted similarity
            similarity = self._compute_feature_weighted_similarity(parent_embeddings, parent_features)
            distance = 1 - similarity
            
            # Adaptive threshold based on parent size
            # Smaller parents -> stricter threshold (more specific)
            # Larger parents -> looser threshold (allow more grouping)
            if len(parent.theory_ids) < 10:
                threshold = 0.45  # Moderate (was 0.35 - too strict)
            elif len(parent.theory_ids) < 20:
                threshold = 0.50  # Moderate-loose (was 0.45)
            else:
                threshold = 0.60  # Loose (was 0.55)
            
            # Agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                linkage='average',
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance)
            
            # Create child clusters
            outlier_indices = []
            for label in sorted(set(labels)):
                if label == -1:
                    outlier_indices = np.where(labels == label)[0].tolist()
                    continue
                
                child_counter += 1
                cluster_indices = np.where(labels == label)[0]
                child_theory_ids = [parent.theory_ids[i] for i in cluster_indices]
                
                # Allow single-theory children (removed min size enforcement)
                centroid = parent_embeddings[cluster_indices].mean(axis=0)
                coherence = self._calculate_coherence(parent_embeddings[cluster_indices])
                
                child = TheoryCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    theory_ids=child_theory_ids,
                    centroid=centroid,
                    parent_cluster_id=parent.cluster_id,
                    coherence_score=coherence,
                    is_singleton=(len(child_theory_ids) == 1)
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
                if len(child_theory_ids) == 1:
                    self.stats['singleton_children'] += 1
            
            # Handle outliers/small clusters
            for outlier_idx in outlier_indices:
                child_counter += 1
                child_theory_id = parent.theory_ids[outlier_idx]
                
                child = TheoryCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    theory_ids=[child_theory_id],
                    centroid=parent_embeddings[outlier_idx],
                    parent_cluster_id=parent.cluster_id,
                    is_singleton=True
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
                self.stats['singleton_children'] += 1
                self.stats['outliers_preserved'] += 1
        
        self.children = all_children
        self.stats['num_children'] = len(all_children)
        self.stats['avg_coherence_child'] = np.mean([c.coherence_score for c in all_children if not c.is_singleton and c.coherence_score > 0])
        
        print(f"✓ Created {len(all_children)} child theories")
        print(f"  Regular clusters: {len(all_children) - self.stats['singleton_children']}")
        print(f"  Singleton clusters: {self.stats['singleton_children']}")
        if self.stats['avg_coherence_child'] > 0:
            print(f"  Avg coherence: {self.stats['avg_coherence_child']:.3f}")
        
        return all_children
    
    def _find_optimal_k(self, embeddings: np.ndarray, min_k: int = 10, max_k: int = 100) -> int:
        """Find optimal number of clusters using silhouette score."""
        best_k = self.target_families
        best_score = -1
        
        # Test range of k values
        k_range = range(min_k, min(max_k, len(embeddings) // 2), 5)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(embeddings)
            
            # Skip if too many singletons
            cluster_sizes = [np.sum(labels == i) for i in range(k)]
            if sum(1 for size in cluster_sizes if size < self.min_cluster_size) > k * 0.3:
                continue
            
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _split_large_cluster(self, theories: List[Dict], embeddings: np.ndarray,
                            features: List[Dict], indices: np.ndarray,
                            level: str, counter: int) -> List[TheoryCluster]:
        """Split a large cluster into smaller sub-clusters."""
        n_subclusters = (len(indices) // self.max_cluster_size) + 1
        
        sub_embeddings = embeddings[indices]
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
        sub_labels = kmeans.fit_predict(sub_embeddings)
        
        clusters = []
        for label in range(n_subclusters):
            sub_indices = indices[sub_labels == label]
            theory_ids = [theories[i]['theory_id'] for i in sub_indices]
            
            counter += 1
            centroid = embeddings[sub_indices].mean(axis=0)
            coherence = self._calculate_coherence(embeddings[sub_indices])
            
            cluster = TheoryCluster(
                cluster_id=f"F{counter:03d}" if level == 'family' else f"P{counter:04d}",
                level=level,
                theory_ids=theory_ids,
                centroid=centroid,
                coherence_score=coherence,
                is_singleton=False
            )
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within cluster."""
        if len(embeddings) < 2:
            return 1.0
        
        similarity = cosine_similarity(embeddings)
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarity), k=1).astype(bool)
        return similarity[mask].mean()
    
    def save_clusters(self, theories: List[Dict], output_path: str):
        """Save clustering results."""
        print(f"\n💾 Saving clusters to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage2_clustering_alternative',
                'strategy': {
                    'level1': 'K-Means with optimal K',
                    'level2': 'DBSCAN (density-based)',
                    'level3': 'Agglomerative with adaptive threshold'
                },
                'statistics': self.stats,
                'parameters': {
                    'target_families': self.target_families,
                    'min_cluster_size': self.min_cluster_size,
                    'max_cluster_size': self.max_cluster_size
                }
            },
            'theories': theories,
            'families': [f.to_dict() for f in self.families],
            'parents': [p.to_dict() for p in self.parents],
            'children': [c.to_dict() for c in self.children]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print clustering statistics."""
        print("\n" + "="*60)
        print("STAGE 2: ALTERNATIVE CLUSTERING STATISTICS")
        print("="*60)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"\nLevel 1 - Theory Families: {self.stats['num_families']}")
        print(f"  Regular clusters: {self.stats['num_families'] - self.stats['singleton_families']}")
        print(f"  Singleton clusters: {self.stats['singleton_families']}")
        print(f"  Avg coherence: {self.stats['avg_coherence_family']:.3f}")
        print(f"\nLevel 2 - Parent Theories: {self.stats['num_parents']}")
        print(f"  Regular clusters: {self.stats['num_parents'] - self.stats['singleton_parents']}")
        print(f"  Singleton clusters: {self.stats['singleton_parents']}")
        if self.stats['avg_coherence_parent'] > 0:
            print(f"  Avg coherence: {self.stats['avg_coherence_parent']:.3f}")
        print(f"\nLevel 3 - Child Theories: {self.stats['num_children']}")
        print(f"  Regular clusters: {self.stats['num_children'] - self.stats['singleton_children']}")
        print(f"  Singleton clusters: {self.stats['singleton_children']}")
        if self.stats['avg_coherence_child'] > 0:
            print(f"  Avg coherence: {self.stats['avg_coherence_child']:.3f}")
        print(f"\nOutliers preserved as singletons: {self.stats['outliers_preserved']}")
        print(f"Silhouette score (families): {self.stats['silhouette_score']:.3f}")
        print(f"Compression ratio: {self.stats['total_theories']/self.stats['num_children']:.1f}:1")
        print("="*60)


def main():
    """Run Stage 2 alternative hierarchical clustering."""
    print("🚀 Starting Stage 2: Alternative Hierarchical Clustering\n")
    
    # Initialize clusterer
    clusterer = AlternativeClusterer(
        target_families=50,      # Target number of families
        min_cluster_size=2,      # Minimum theories per cluster (was 3)
        max_cluster_size=50      # Maximum theories per cluster
    )
    
    # Load embeddings from Stage 1
    theories, detailed_emb, features_list = clusterer.load_embeddings(
        'output/stage1_embeddings.json'
    )
    
    # Extract features
    features = [emb.get('concept_features', {}) for emb in features_list]
    
    # Level 1: Theory families (K-Means)
    families = clusterer.cluster_level1_families(theories, detailed_emb, features)
    
    # Level 2: Parent theories (DBSCAN)
    parents = clusterer.cluster_level2_parents(theories, detailed_emb, features, families)
    
    # Level 3: Child theories (Agglomerative with adaptive threshold)
    children = clusterer.cluster_level3_children(theories, detailed_emb, features, parents)
    
    # Save results
    clusterer.save_clusters(theories, 'output/stage2_clusters_alternative.json')
    
    # Print statistics
    clusterer.print_statistics()
    
    print("\n✅ Stage 2 (Alternative) complete!")


if __name__ == '__main__':
    main()
