"""
Stage 4: Clustering Methods for Theory Normalization

Implements and compares multiple clustering approaches:
1. Graph-based (Community Detection) - RECOMMENDED
2. Hierarchical Agglomerative Clustering
3. DBSCAN (Density-based)
4. Hybrid Ensemble Approach
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import community.community_louvain as community_louvain  # python-louvain
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# METHOD 1: GRAPH-BASED CLUSTERING (RECOMMENDED)
# ============================================================================

class GraphBasedClustering:
    """
    Build similarity graph and detect communities
    
    PROS:
    - Handles transitive relationships naturally
    - No need to specify number of clusters
    - Works well with varied cluster sizes
    - Preserves hierarchical structure
    
    CONS:
    - Sensitive to threshold parameter
    - Can create very large clusters
    """
    
    def __init__(self, similarity_threshold: float = 0.75,
                 hierarchical_boost: float = 0.85):
        self.threshold = similarity_threshold
        self.hierarchical_boost = hierarchical_boost
    
    def build_graph(self, theories: List, similarity_matrix: np.ndarray,
                   similarity_details: List[List[Dict]] = None) -> nx.Graph:
        """
        Build weighted graph from similarity matrix
        
        Args:
            theories: List of Theory objects
            similarity_matrix: NxN similarity matrix
            similarity_details: Optional detailed similarity info
        """
        G = nx.Graph()
        n = len(theories)
        
        # Add nodes with theory metadata
        for i, theory in enumerate(theories):
            G.add_node(i, 
                      name=theory.original_name,
                      canonical=theory.canonical_form,
                      confidence=theory.confidence_score)
        
        # Add edges for similar theories
        edges_added = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]
                
                # Dynamic threshold based on relationship type
                threshold = self.threshold
                
                # If hierarchical relationship detected, lower threshold
                if similarity_details and similarity_details[i][j]:
                    hierarchy = similarity_details[i][j].get('hierarchy', {})
                    if hierarchy.get('relationship') == 'parent-child':
                        threshold = self.threshold * self.hierarchical_boost
                
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)
                    edges_added += 1
        
        logger.info(f"Built graph: {G.number_of_nodes()} nodes, {edges_added} edges")
        logger.info(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        
        return G
    
    def detect_communities_louvain(self, G: nx.Graph) -> Dict[int, int]:
        """
        Louvain community detection
        
        Optimizes modularity to find natural communities
        Best for: Most cases, balanced cluster sizes
        """
        communities = community_louvain.best_partition(G, weight='weight')
        
        n_communities = len(set(communities.values()))
        logger.info(f"Louvain detected {n_communities} communities")
        
        return communities
    
    def detect_communities_connected_components(self, G: nx.Graph) -> Dict[int, int]:
        """
        Simple connected components
        
        Each connected component = one cluster
        Best for: High threshold, conservative merging
        """
        communities = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                communities[node] = i
        
        n_communities = len(set(communities.values()))
        logger.info(f"Connected components: {n_communities} clusters")
        
        return communities
    
    def detect_communities_label_propagation(self, G: nx.Graph) -> Dict[int, int]:
        """
        Label propagation algorithm
        
        Fast, semi-random community detection
        Best for: Large graphs, quick iterations
        """
        communities_list = list(nx.algorithms.community.label_propagation_communities(G))
        
        communities = {}
        for i, community_set in enumerate(communities_list):
            for node in community_set:
                communities[node] = i
        
        n_communities = len(communities_list)
        logger.info(f"Label propagation: {n_communities} communities")
        
        return communities
    
    def cluster(self, theories: List, similarity_matrix: np.ndarray,
                method: str = "louvain") -> List[Set[int]]:
        """
        Main clustering method
        
        Args:
            theories: List of Theory objects
            similarity_matrix: NxN similarity matrix
            method: "louvain", "components", or "label_prop"
        
        Returns:
            List of clusters (each cluster is a set of theory indices)
        """
        G = self.build_graph(theories, similarity_matrix)
        
        # Choose community detection method
        if method == "louvain":
            community_dict = self.detect_communities_louvain(G)
        elif method == "components":
            community_dict = self.detect_communities_connected_components(G)
        elif method == "label_prop":
            community_dict = self.detect_communities_label_propagation(G)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert dict to list of sets
        clusters_dict = defaultdict(set)
        for node_id, cluster_id in community_dict.items():
            clusters_dict[cluster_id].add(node_id)
        
        clusters = list(clusters_dict.values())
        
        # Statistics
        cluster_sizes = [len(c) for c in clusters]
        logger.info(f"Cluster sizes: min={min(cluster_sizes)}, "
                   f"max={max(cluster_sizes)}, "
                   f"mean={np.mean(cluster_sizes):.1f}, "
                   f"median={np.median(cluster_sizes):.1f}")
        
        return clusters


# ============================================================================
# METHOD 2: HIERARCHICAL AGGLOMERATIVE CLUSTERING
# ============================================================================

class HierarchicalClustering:
    """
    Bottom-up hierarchical clustering
    
    PROS:
    - Creates natural hierarchy
    - Can cut at different levels
    - Deterministic
    
    CONS:
    - Need to specify number of clusters OR distance threshold
    - Can create unbalanced clusters
    - Computationally expensive (O(n²) or O(n³))
    """
    
    def __init__(self, linkage_method: str = "average"):
        """
        Args:
            linkage_method: "single", "complete", "average", "ward"
            - single: min distance (creates chains)
            - complete: max distance (creates tight clusters)
            - average: mean distance (RECOMMENDED)
            - ward: minimize variance (requires euclidean)
        """
        self.linkage_method = linkage_method
    
    def cluster_by_threshold(self, similarity_matrix: np.ndarray,
                            distance_threshold: float = 0.3) -> List[Set[int]]:
        """
        Cluster by cutting dendrogram at distance threshold
        
        Args:
            similarity_matrix: NxN similarity matrix
            distance_threshold: Cut threshold (lower = more clusters)
                Note: This is DISTANCE, so 1 - similarity
        
        Returns:
            List of clusters
        """
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.linkage_method,
            distance_threshold=distance_threshold
        )
        
        labels = clusterer.fit_predict(distance_matrix)
        
        # Convert to clusters
        clusters_dict = defaultdict(set)
        for i, label in enumerate(labels):
            clusters_dict[label].add(i)
        
        clusters = list(clusters_dict.values())
        logger.info(f"Hierarchical clustering: {len(clusters)} clusters at threshold {distance_threshold}")
        
        return clusters
    
    def cluster_by_count(self, similarity_matrix: np.ndarray,
                        n_clusters: int = 300) -> List[Set[int]]:
        """
        Cluster by specifying exact number of clusters
        
        Args:
            similarity_matrix: NxN similarity matrix
            n_clusters: Target number of clusters
        
        Returns:
            List of clusters
        """
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering with fixed cluster count
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=self.linkage_method
        )
        
        labels = clusterer.fit_predict(distance_matrix)
        
        # Convert to clusters
        clusters_dict = defaultdict(set)
        for i, label in enumerate(labels):
            clusters_dict[label].add(i)
        
        clusters = list(clusters_dict.values())
        logger.info(f"Hierarchical clustering: {len(clusters)} clusters (target: {n_clusters})")
        
        return clusters


# ============================================================================
# METHOD 3: DBSCAN (DENSITY-BASED)
# ============================================================================

class DensityBasedClustering:
    """
    DBSCAN: Density-Based Spatial Clustering
    
    PROS:
    - Handles noise/outliers (creates "-1" cluster)
    - No need to specify number of clusters
    - Can find arbitrary-shaped clusters
    
    CONS:
    - Sensitive to eps parameter
    - Struggles with varying density clusters
    - May create many singletons
    """
    
    def __init__(self, eps: float = 0.25, min_samples: int = 2):
        """
        Args:
            eps: Maximum distance for points to be neighbors
                (This is DISTANCE, so 1 - similarity)
            min_samples: Minimum cluster size
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def cluster(self, similarity_matrix: np.ndarray) -> List[Set[int]]:
        """
        Perform DBSCAN clustering
        
        Returns:
            List of clusters (noise points go to singleton clusters)
        """
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # DBSCAN
        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        
        labels = clusterer.fit_predict(distance_matrix)
        
        # Convert to clusters
        clusters_dict = defaultdict(set)
        for i, label in enumerate(labels):
            if label == -1:
                # Noise point - create singleton cluster
                clusters_dict[f"singleton_{i}"].add(i)
            else:
                clusters_dict[label].add(i)
        
        clusters = list(clusters_dict.values())
        n_noise = sum(1 for label in labels if label == -1)
        
        logger.info(f"DBSCAN: {len(clusters)} clusters, {n_noise} noise points")
        
        return clusters


# ============================================================================
# METHOD 4: HYBRID ENSEMBLE APPROACH (RECOMMENDED FOR PRODUCTION)
# ============================================================================

class EnsembleClustering:
    """
    Combine multiple clustering methods for robust results
    
    Strategy:
    1. Run multiple clustering algorithms
    2. Theories clustered together by MAJORITY vote are merged
    3. Provides confidence scores for each merge
    
    PROS:
    - More robust than single method
    - Reduces algorithm-specific biases
    - Provides confidence metrics
    
    CONS:
    - Computationally expensive
    - More complex to tune
    """
    
    def __init__(self):
        self.methods = {
            'graph_louvain': GraphBasedClustering(threshold=0.75),
            'graph_components': GraphBasedClustering(threshold=0.80),
            'hierarchical': HierarchicalClustering(linkage_method='average'),
            'dbscan': DensityBasedClustering(eps=0.25, min_samples=2)
        }
    
    def run_all_methods(self, theories: List, 
                       similarity_matrix: np.ndarray) -> Dict[str, List[Set[int]]]:
        """
        Run all clustering methods
        
        Returns:
            Dictionary mapping method name to cluster assignments
        """
        results = {}
        
        # Graph-based methods
        logger.info("Running graph-based clustering (Louvain)...")
        results['graph_louvain'] = self.methods['graph_louvain'].cluster(
            theories, similarity_matrix, method='louvain'
        )
        
        logger.info("Running graph-based clustering (Components)...")
        results['graph_components'] = self.methods['graph_components'].cluster(
            theories, similarity_matrix, method='components'
        )
        
        # Hierarchical clustering
        logger.info("Running hierarchical clustering...")
        results['hierarchical'] = self.methods['hierarchical'].cluster_by_threshold(
            similarity_matrix, distance_threshold=0.25
        )
        
        # DBSCAN
        logger.info("Running DBSCAN...")
        results['dbscan'] = self.methods['dbscan'].cluster(similarity_matrix)
        
        return results
    
    def compute_consensus(self, all_results: Dict[str, List[Set[int]]],
                         n_theories: int,
                         agreement_threshold: float = 0.5) -> List[Set[int]]:
        """
        Compute consensus clustering from multiple methods
        
        Args:
            all_results: Results from each method
            n_theories: Total number of theories
            agreement_threshold: Minimum agreement to merge (0.5 = majority vote)
        
        Returns:
            Consensus clusters
        """
        # Build co-occurrence matrix: how often are theories i,j clustered together?
        co_occurrence = np.zeros((n_theories, n_theories))
        
        for method_name, clusters in all_results.items():
            # For this method, which theories are together?
            for cluster in clusters:
                cluster_list = list(cluster)
                for i in range(len(cluster_list)):
                    for j in range(i, len(cluster_list)):
                        idx_i, idx_j = cluster_list[i], cluster_list[j]
                        co_occurrence[idx_i, idx_j] += 1
                        co_occurrence[idx_j, idx_i] += 1
        
        # Normalize by number of methods
        n_methods = len(all_results)
        co_occurrence /= n_methods
        
        logger.info(f"Co-occurrence matrix computed ({n_methods} methods)")
        logger.info(f"Mean co-occurrence: {co_occurrence.mean():.3f}")
        
        # Build consensus graph
        G = nx.Graph()
        for i in range(n_theories):
            G.add_node(i)
        
        edges_added = 0
        for i in range(n_theories):
            for j in range(i + 1, n_theories):
                if co_occurrence[i, j] >= agreement_threshold:
                    G.add_edge(i, j, weight=co_occurrence[i, j])
                    edges_added += 1
        
        logger.info(f"Consensus graph: {edges_added} edges (threshold={agreement_threshold})")
        
        # Extract connected components as final clusters
        consensus_clusters = [set(c) for c in nx.connected_components(G)]
        
        logger.info(f"Consensus: {len(consensus_clusters)} final clusters")
        
        return consensus_clusters
    
    def cluster_with_confidence(self, theories: List,
                               similarity_matrix: np.ndarray) -> Dict:
        """
        Main ensemble clustering with confidence scores
        
        Returns:
            {
                'clusters': List of final clusters,
                'method_results': Individual method results,
                'confidence': Per-cluster confidence scores
            }
        """
        # Run all methods
        method_results = self.run_all_methods(theories, similarity_matrix)
        
        # Compute consensus
        consensus_clusters = self.compute_consensus(
            method_results, 
            len(theories),
            agreement_threshold=0.5  # Majority vote
        )
        
        # Compute confidence for each cluster
        cluster_confidences = self._compute_cluster_confidence(
            consensus_clusters, method_results, len(theories)
        )
        
        return {
            'clusters': consensus_clusters,
            'method_results': method_results,
            'confidence': cluster_confidences,
            'statistics': self._compute_statistics(consensus_clusters, method_results)
        }
    
    def _compute_cluster_confidence(self, clusters: List[Set[int]],
                                   method_results: Dict,
                                   n_theories: int) -> Dict[int, float]:
        """
        Compute confidence score for each cluster
        
        Confidence = average agreement across methods that this cluster should exist
        """
        confidences = {}
        
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) == 1:
                # Singleton - low confidence (probably noise or unique)
                confidences[cluster_id] = 0.3
                continue
            
            # Count how many methods agree on this cluster
            agreements = []
            cluster_list = list(cluster)
            
            for method_name, method_clusters in method_results.items():
                # Check if this cluster exists in this method
                method_agreement = 0
                for method_cluster in method_clusters:
                    overlap = len(cluster & method_cluster)
                    if overlap > 0:
                        # Jaccard similarity
                        jaccard = overlap / len(cluster | method_cluster)
                        method_agreement = max(method_agreement, jaccard)
                
                agreements.append(method_agreement)
            
            confidences[cluster_id] = np.mean(agreements)
        
        return confidences
    
    def _compute_statistics(self, consensus_clusters: List[Set[int]],
                          method_results: Dict) -> Dict:
        """Compute clustering statistics"""
        stats = {
            'n_clusters': len(consensus_clusters),
            'cluster_sizes': {
                'min': min(len(c) for c in consensus_clusters),
                'max': max(len(c) for c in consensus_clusters),
                'mean': np.mean([len(c) for c in consensus_clusters]),
                'median': np.median([len(c) for c in consensus_clusters])
            },
            'singletons': sum(1 for c in consensus_clusters if len(c) == 1),
            'large_clusters': sum(1 for c in consensus_clusters if len(c) > 10),
            'method_cluster_counts': {
                name: len(clusters) for name, clusters in method_results.items()
            }
        }
        return stats


# ============================================================================
# RECOMMENDATION & COMPARISON
# ============================================================================

def compare_clustering_methods(theories: List, similarity_matrix: np.ndarray):
    """
    Run all methods and compare results
    
    Helps you decide which method works best for your data
    """
    print("\n" + "="*80)
    print("CLUSTERING METHODS COMPARISON")
    print("="*80)
    
    # Method 1: Graph-based (Louvain)
    print("\n1. GRAPH-BASED (Louvain) - Best for most cases")
    graph_cluster = GraphBasedClustering(threshold=0.75)
    graph_result = graph_cluster.cluster(theories, similarity_matrix, method='louvain')
    print(f"   Clusters: {len(graph_result)}")
    print(f"   Size range: {min(len(c) for c in graph_result)} - {max(len(c) for c in graph_result)}")
    
    # Method 2: Graph-based (Components)
    print("\n2. GRAPH-BASED (Connected Components) - More conservative")
    graph_cluster2 = GraphBasedClustering(threshold=0.80)
    graph_result2 = graph_cluster2.cluster(theories, similarity_matrix, method='components')
    print(f"   Clusters: {len(graph_result2)}")
    print(f"   Size range: {min(len(c) for c in graph_result2)} - {max(len(c) for c in graph_result2)}")
    
    # Method 3: Hierarchical
    print("\n3. HIERARCHICAL CLUSTERING - Deterministic, creates hierarchy")
    hier_cluster = HierarchicalClustering(linkage_method='average')
    hier_result = hier_cluster.cluster_by_threshold(similarity_matrix, distance_threshold=0.25)
    print(f"   Clusters: {len(hier_result)}")
    print(f"   Size range: {min(len(c) for c in hier_result)} - {max(len(c) for c in hier_result)}")
    
    # Method 4: DBSCAN
    print("\n4. DBSCAN - Handles outliers, may create many singletons")
    dbscan_cluster = DensityBasedClustering(eps=0.25, min_samples=2)
    dbscan_result = dbscan_cluster.cluster(similarity_matrix)
    print(f"   Clusters: {len(dbscan_result)}")
    print(f"   Singletons: {sum(1 for c in dbscan_result if len(c) == 1)}")
    
    # Method 5: Ensemble
    print("\n5. ENSEMBLE (RECOMMENDED) - Most robust")
    ensemble = EnsembleClustering()
    ensemble_result = ensemble.cluster_with_confidence(theories, similarity_matrix)
    print(f"   Clusters: {len(ensemble_result['clusters'])}")
    print(f"   Mean confidence: {np.mean(list(ensemble_result['confidence'].values())):.3f}")
    print(f"   Statistics: {ensemble_result['statistics']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("- Start with GRAPH-BASED (Louvain) for speed and good results")
    print("- Use ENSEMBLE for production (most accurate but slower)")
    print("- Tune threshold based on your domain knowledge")
    print("="*80 + "\n")
    
    return {
        'graph_louvain': graph_result,
        'graph_components': graph_result2,
        'hierarchical': hier_result,
        'dbscan': dbscan_result,
        'ensemble': ensemble_result
    }


# ============================================================================
# MY RECOMMENDATION FOR YOUR USE CASE
# ============================================================================

"""
RECOMMENDED APPROACH FOR 6,000 THEORIES → 300-350 CLUSTERS:

1. START: Graph-Based (Louvain) with threshold=0.75
   - Fast (~1 minute for 6K theories)
   - Good results
   - Handles varying cluster sizes
   
2. VALIDATE: Run Ensemble on uncertain clusters (size > 15 OR low internal similarity)
   - Adds robustness
   - ~5-10 minutes additional time
   
3. REFINE: Use LLM validation (next stage) on:
   - Large clusters (>20 theories)
   - Low-confidence clusters (<0.6)
   - ~10-15% of clusters need LLM review

PARAMETER TUNING:
- threshold=0.75: Good starting point
- threshold=0.80: More conservative (more clusters)
- threshold=0.70: More aggressive (fewer clusters)

Run a sweep: [0.70, 0.75, 0.80, 0.85] and see which gives ~300-350 clusters
"""


# ============================================================================
# COMPLETE PIPELINE: STAGES 1-4
# ============================================================================

def run_complete_clustering_pipeline(theories: List,
                                    similarity_matrix: np.ndarray,
                                    method: str = "graph_louvain",
                                    threshold: float = 0.75) -> Dict:
    """
    Complete clustering pipeline (Stages 1-4)
    
    Args:
        theories: List of Theory objects (from Stage 1-2)
        similarity_matrix: From Stage 3
        method: "graph_louvain", "graph_components", "hierarchical", 
                "dbscan", or "ensemble"
        threshold: Similarity threshold (for graph methods)
    
    Returns:
        Dictionary with clusters and metadata
    """
    logger.info(f"Starting clustering with method={method}, threshold={threshold}")
    
    if method == "graph_louvain":
        clusterer = GraphBasedClustering(threshold=threshold)
        clusters = clusterer.cluster(theories, similarity_matrix, method='louvain')
        result = {'clusters': clusters, 'method': method}
    
    elif method == "graph_components":
        clusterer = GraphBasedClustering(threshold=threshold)
        clusters = clusterer.cluster(theories, similarity_matrix, method='components')
        result = {'clusters': clusters, 'method': method}
    
    elif method == "hierarchical":
        clusterer = HierarchicalClustering(linkage_method='average')
        clusters = clusterer.cluster_by_threshold(similarity_matrix, 
                                                  distance_threshold=1-threshold)
        result = {'clusters': clusters, 'method': method}
    
    elif method == "dbscan":
        clusterer = DensityBasedClustering(eps=1-threshold, min_samples=2)
        clusters = clusterer.cluster(similarity_matrix)
        result = {'clusters': clusters, 'method': method}
    
    elif method == "ensemble":
        clusterer = EnsembleClustering()
        result = clusterer.cluster_with_confidence(theories, similarity_matrix)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Clustering complete: {len(result['clusters'])} clusters created")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Stage 4: Clustering Methods")
    print("Use run_complete_clustering_pipeline() with your data from Stages 1-3")
    print("\nRecommended: method='graph_louvain', threshold=0.75")