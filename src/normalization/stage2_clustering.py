"""
Stage 2: Three-Level Hierarchical Clustering
Creates theory families, parent theories, and child theories.
"""

import json
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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
    is_singleton: bool = False  # Flag for singleton clusters (outliers)
    
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


class HierarchicalClusterer:
    """Performs three-level hierarchical clustering on theories."""
    
    def __init__(self, 
                 family_threshold: float = 0.7,
                 parent_threshold: float = 0.5,
                 child_threshold: float = 0.4):
        """
        Initialize clusterer with distance thresholds.
        
        Args:
            family_threshold: Distance threshold for Level 1 (families)
            parent_threshold: Distance threshold for Level 2 (parents)
            child_threshold: Distance threshold for Level 3 (children)
        """
        self.family_threshold = family_threshold
        self.parent_threshold = parent_threshold
        self.child_threshold = child_threshold
        
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
            'outliers_preserved': 0
        }
    
    def load_embeddings(self, embeddings_path: str) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
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
        embeddings = data['embeddings']
        
        # Extract embedding matrices
        name_embeddings = []
        semantic_embeddings = []
        detailed_embeddings = []
        
        for emb in embeddings:
            name_embeddings.append(np.array(emb['name_embedding']))
            semantic_embeddings.append(np.array(emb['semantic_embedding']))
            detailed_embeddings.append(np.array(emb['detailed_embedding']))
        
        name_matrix = np.array(name_embeddings)
        semantic_matrix = np.array(semantic_embeddings)
        detailed_matrix = np.array(detailed_embeddings)
        
        print(f"✓ Loaded {len(theories)} theories with embeddings")
        print(f"  Name embeddings: {name_matrix.shape}")
        print(f"  Semantic embeddings: {semantic_matrix.shape}")
        print(f"  Detailed embeddings: {detailed_matrix.shape}")
        
        self.stats['total_theories'] = len(theories)
        
        return theories, name_matrix, semantic_matrix, detailed_matrix
    
    def cluster_level1_families(self, theories: List[Dict], 
                                name_embeddings: np.ndarray) -> List[TheoryCluster]:
        """
        Level 1: Cluster into theory families (30-50 clusters).
        Uses name-only embeddings for broad categorization.
        """
        print(f"\n🔄 Level 1: Clustering into theory families...")
        print(f"   Threshold: {self.family_threshold}")
        
        # Compute distance matrix
        similarity_matrix = cosine_similarity(name_embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.family_threshold,
            linkage='average',
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Create clusters
        # Separate regular clusters and outliers
        families = []
        outlier_indices = []
        unique_labels = set(labels)
        regular_label_count = 0
        
        # Create regular clusters
        for label in sorted(unique_labels):
            if label == -1:  # Collect outliers
                outlier_indices = np.where(labels == label)[0].tolist()
                continue
            
            theory_indices = np.where(labels == label)[0]
            theory_ids = [theories[i]['theory_id'] for i in theory_indices]
            centroid = name_embeddings[theory_indices].mean(axis=0)
            
            regular_label_count += 1
            family = TheoryCluster(
                cluster_id=f"F{regular_label_count:03d}",
                level='family',
                theory_ids=theory_ids,
                centroid=centroid,
                is_singleton=False
            )
            families.append(family)
        
        # Create singleton clusters for outliers
        singleton_count = 0
        for outlier_idx in outlier_indices:
            singleton_count += 1
            cluster_num = regular_label_count + singleton_count
            
            family = TheoryCluster(
                cluster_id=f"F{cluster_num:03d}",
                level='family',
                theory_ids=[theories[outlier_idx]['theory_id']],
                centroid=name_embeddings[outlier_idx],
                is_singleton=True,
                canonical_name=f"Singleton: {theories[outlier_idx]['name']}"
            )
            families.append(family)
        
        self.families = families
        self.stats['num_families'] = len(families)
        self.stats['singleton_families'] = singleton_count
        self.stats['outliers_preserved'] += singleton_count
        
        print(f"✓ Created {len(families)} theory families")
        print(f"  Regular clusters: {regular_label_count}")
        print(f"  Singleton clusters: {singleton_count}")
        if singleton_count > 0:
            print(f"  ⚠️  {singleton_count} unique theories (no close matches)")
        if regular_label_count > 0:
            avg_size = sum(len(f.theory_ids) for f in families if not f.is_singleton) / regular_label_count
            print(f"  Avg theories per regular family: {avg_size:.1f}")
        
        return families
    
    def cluster_level2_parents(self, theories: List[Dict], 
                              semantic_embeddings: np.ndarray,
                              families: List[TheoryCluster]) -> List[TheoryCluster]:
        """
        Level 2: Cluster into parent theories within each family.
        Uses semantic embeddings (name + concepts).
        """
        print(f"\n🔄 Level 2: Clustering into parent theories...")
        print(f"   Threshold: {self.parent_threshold}")
        
        all_parents = []
        parent_counter = 0
        
        for family in families:
            # Singletons and single-theory families stay as single parents
            if family.is_singleton or len(family.theory_ids) < 2:
                parent_counter += 1
                parent = TheoryCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    theory_ids=family.theory_ids,
                    parent_cluster_id=family.cluster_id,
                    is_singleton=family.is_singleton  # Propagate singleton flag
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
                if family.is_singleton:
                    self.stats['singleton_parents'] += 1
                continue
            
            # Get embeddings for theories in this family
            theory_id_to_idx = {t['theory_id']: i for i, t in enumerate(theories)}
            family_indices = [theory_id_to_idx[tid] for tid in family.theory_ids]
            family_embeddings = semantic_embeddings[family_indices]
            
            # Cluster within family
            if len(family_indices) < 3:
                # Too few theories - single parent
                parent_counter += 1
                parent = TheoryCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    theory_ids=family.theory_ids,
                    centroid=family_embeddings.mean(axis=0),
                    parent_cluster_id=family.cluster_id
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
            else:
                # Cluster into parents
                similarity_matrix = cosine_similarity(family_embeddings)
                distance_matrix = 1 - similarity_matrix
                
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.parent_threshold,
                    linkage='average',
                    metric='precomputed'
                )
                
                labels = clustering.fit_predict(distance_matrix)
                
                # Separate regular clusters and outliers
                outlier_indices_in_family = []
                for label in set(labels):
                    if label == -1:
                        outlier_indices_in_family = np.where(labels == label)[0].tolist()
                        continue
                    
                    parent_counter += 1
                    cluster_indices = np.where(labels == label)[0]
                    parent_theory_ids = [family.theory_ids[i] for i in cluster_indices]
                    
                    centroid = family_embeddings[cluster_indices].mean(axis=0)
                    
                    parent = TheoryCluster(
                        cluster_id=f"P{parent_counter:04d}",
                        level='parent',
                        theory_ids=parent_theory_ids,
                        centroid=centroid,
                        parent_cluster_id=family.cluster_id,
                        is_singleton=False
                    )
                    all_parents.append(parent)
                    family.child_cluster_ids.append(parent.cluster_id)
                
                # Create singleton parents for outliers within this family
                for outlier_idx in outlier_indices_in_family:
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
        
        print(f"✓ Created {len(all_parents)} parent theories")
        print(f"  Avg parents per family: {len(all_parents)/len(families):.1f}")
        
        return all_parents
    
    def cluster_level3_children(self, theories: List[Dict],
                               detailed_embeddings: np.ndarray,
                               parents: List[TheoryCluster],
                               embeddings_data: List[Dict]) -> List[TheoryCluster]:
        """
        Level 3: Cluster into child theories within each parent.
        Uses detailed embeddings + concept features for fine-grained distinction.
        """
        print(f"\n🔄 Level 3: Clustering into child theories...")
        print(f"   Threshold: {self.child_threshold}")
        
        all_children = []
        child_counter = 0
        
        theory_id_to_idx = {t['theory_id']: i for i, t in enumerate(theories)}
        
        for parent in parents:
            # Singletons and single-theory parents stay as single children
            if parent.is_singleton or len(parent.theory_ids) < 2:
                child_counter += 1
                child = TheoryCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    theory_ids=parent.theory_ids,
                    parent_cluster_id=parent.cluster_id,
                    is_singleton=parent.is_singleton  # Propagate singleton flag
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
                if parent.is_singleton:
                    self.stats['singleton_children'] += 1
                continue
            
            # Get embeddings and features for theories in this parent
            parent_indices = [theory_id_to_idx[tid] for tid in parent.theory_ids]
            parent_embeddings = detailed_embeddings[parent_indices]
            
            # Extract concept features
            parent_features = [embeddings_data[i]['concept_features'] for i in parent_indices]
            
            # Compute combined similarity (embedding + feature-based)
            similarity_matrix = self._compute_combined_similarity(
                parent_embeddings, 
                parent_features
            )
            distance_matrix = 1 - similarity_matrix
            
            # Cluster into children
            if len(parent_indices) < 3:
                # Too few theories - single child
                child_counter += 1
                child = TheoryCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    theory_ids=parent.theory_ids,
                    centroid=parent_embeddings.mean(axis=0),
                    parent_cluster_id=parent.cluster_id
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.child_threshold,
                    linkage='average',
                    metric='precomputed'
                )
                
                labels = clustering.fit_predict(distance_matrix)
                
                # Separate regular clusters and outliers
                outlier_indices_in_parent = []
                for label in set(labels):
                    if label == -1:
                        outlier_indices_in_parent = np.where(labels == label)[0].tolist()
                        continue
                    
                    child_counter += 1
                    cluster_indices = np.where(labels == label)[0]
                    child_theory_ids = [parent.theory_ids[i] for i in cluster_indices]
                    
                    centroid = parent_embeddings[cluster_indices].mean(axis=0)
                    
                    child = TheoryCluster(
                        cluster_id=f"C{child_counter:04d}",
                        level='child',
                        theory_ids=child_theory_ids,
                        centroid=centroid,
                        parent_cluster_id=parent.cluster_id,
                        is_singleton=False
                    )
                    all_children.append(child)
                    parent.child_cluster_ids.append(child.cluster_id)
                
                # Create singleton children for outliers within this parent
                for outlier_idx in outlier_indices_in_parent:
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
        
        print(f"✓ Created {len(all_children)} child theories")
        print(f"  Avg children per parent: {len(all_children)/len(parents):.1f}")
        
        return all_children
    
    def _compute_combined_similarity(self, embeddings: np.ndarray, 
                                    features: List[Dict]) -> np.ndarray:
        """
        Compute combined similarity using embeddings and concept features.
        This is key for preserving fine-grained distinctions.
        """
        n = len(embeddings)
        
        # Embedding similarity
        emb_similarity = cosine_similarity(embeddings)
        
        # Feature-based similarity
        feature_similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    feature_similarity[i, j] = 1.0
                else:
                    # Compute feature similarity
                    sim = self._compute_feature_similarity(features[i], features[j])
                    feature_similarity[i, j] = sim
                    feature_similarity[j, i] = sim
        
        # Combined similarity (weighted)
        # Higher weight on embeddings, but features can reduce similarity if mechanisms differ
        combined = 0.6 * emb_similarity + 0.4 * feature_similarity
        
        return combined
    
    def _compute_feature_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """
        Compute similarity between two feature dictionaries.
        Reduces similarity if mechanisms are different (key for distinction preservation).
        """
        score = 0.0
        
        # Mechanism overlap (different mechanisms → lower similarity)
        mech1 = set(feat1.get('mechanisms', []))
        mech2 = set(feat2.get('mechanisms', []))
        
        if mech1 and mech2:
            if mech1 & mech2:  # Some overlap
                score += 0.3
            elif mech1 != mech2:  # Different mechanisms
                score -= 0.2  # Penalty for different mechanisms
        
        # Pathway overlap
        path1 = set(feat1.get('pathways', []))
        path2 = set(feat2.get('pathways', []))
        
        if path1 and path2:
            overlap = len(path1 & path2) / len(path1 | path2)
            score += 0.3 * overlap
        
        # Process overlap
        proc1 = set(feat1.get('processes', []))
        proc2 = set(feat2.get('processes', []))
        
        if proc1 and proc2:
            overlap = len(proc1 & proc2) / len(proc1 | proc2)
            score += 0.3 * overlap
        
        # Biological level match
        if feat1.get('biological_level') == feat2.get('biological_level'):
            score += 0.1
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def save_clusters(self, theories: List[Dict], output_path: str):
        """Save clustering results."""
        print(f"\n💾 Saving clusters to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage2_clustering',
                'statistics': self.stats,
                'thresholds': {
                    'family': self.family_threshold,
                    'parent': self.parent_threshold,
                    'child': self.child_threshold
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
    
    def validate_no_data_loss(self, theories: List[Dict]) -> bool:
        """Validate that no theories were lost during clustering."""
        total_theories_before = len(theories)
        total_theories_after = sum(len(f.theory_ids) for f in self.families)
        
        if total_theories_before != total_theories_after:
            print(f"\n❌ DATA LOSS DETECTED!")
            print(f"   Before clustering: {total_theories_before} theories")
            print(f"   After clustering: {total_theories_after} theories")
            print(f"   Lost: {total_theories_before - total_theories_after} theories")
            return False
        else:
            print(f"\n✅ Data integrity verified: All {total_theories_before} theories preserved")
            return True
    
    def print_statistics(self):
        """Print clustering statistics."""
        print("\n" + "="*60)
        print("STAGE 2: CLUSTERING STATISTICS")
        print("="*60)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"\nLevel 1 - Theory Families: {self.stats['num_families']}")
        print(f"  Regular clusters: {self.stats['num_families'] - self.stats['singleton_families']}")
        print(f"  Singleton clusters: {self.stats['singleton_families']}")
        print(f"  Avg theories per family: {self.stats['total_theories']/self.stats['num_families']:.1f}")
        print(f"\nLevel 2 - Parent Theories: {self.stats['num_parents']}")
        print(f"  Regular clusters: {self.stats['num_parents'] - self.stats['singleton_parents']}")
        print(f"  Singleton clusters: {self.stats['singleton_parents']}")
        print(f"  Avg parents per family: {self.stats['num_parents']/self.stats['num_families']:.1f}")
        print(f"\nLevel 3 - Child Theories: {self.stats['num_children']}")
        print(f"  Regular clusters: {self.stats['num_children'] - self.stats['singleton_children']}")
        print(f"  Singleton clusters: {self.stats['singleton_children']}")
        print(f"  Avg children per parent: {self.stats['num_children']/self.stats['num_parents']:.1f}")
        print(f"\nOutliers preserved as singletons: {self.stats['outliers_preserved']}")
        print(f"Compression ratio: {self.stats['total_theories']/self.stats['num_children']:.1f}:1")
        print("="*60)


def main():
    """Run Stage 2 hierarchical clustering."""
    print("🚀 Starting Stage 2: Hierarchical Clustering\n")
    
    # Initialize clusterer
    clusterer = HierarchicalClusterer(
        family_threshold=0.7,
        parent_threshold=0.5,
        child_threshold=0.4
    )
    
    # Load embeddings from Stage 1
    theories, name_emb, semantic_emb, detailed_emb = clusterer.load_embeddings(
        'output/stage1_embeddings.json'
    )
    
    # Load full embedding data for features
    with open('output/stage1_embeddings.json', 'r') as f:
        emb_data = json.load(f)
    embeddings_list = emb_data['embeddings']
    
    # Level 1: Theory families
    families = clusterer.cluster_level1_families(theories, name_emb)
    
    # Validate no data loss
    clusterer.validate_no_data_loss(theories)
    
    # Level 2: Parent theories
    parents = clusterer.cluster_level2_parents(theories, semantic_emb, families)
    
    # Level 3: Child theories
    children = clusterer.cluster_level3_children(
        theories, detailed_emb, parents, embeddings_list
    )
    
    # Save results
    clusterer.save_clusters(theories, 'output/stage2_clusters.json')
    
    # Print statistics
    clusterer.print_statistics()
    
    print("\n✅ Stage 2 complete!")


if __name__ == '__main__':
    main()
