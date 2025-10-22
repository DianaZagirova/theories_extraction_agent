"""
Test script for Stage 5: Cluster-Based Theory Refinement

Tests the stage5_cluster_refinement.py script on a small subset of clusters.
"""

import json
from pathlib import Path
from src.normalization.stage5_cluster_refinement import Stage5ClusterRefiner


def test_stage5_on_subset():
    """Test Stage 5 on a small subset of clusters."""
    print("Testing Stage 5 on subset of clusters...")
    
    # Load full clusters
    clusters_path = Path('data/clusters_with_paper_counts.json')
    with open(clusters_path, 'r') as f:
        all_clusters = json.load(f)
    
    # Select first 3 clusters for testing
    test_clusters = dict(list(all_clusters.items())[:3])
    
    # Save test clusters to temporary file
    test_clusters_path = Path('data/test_clusters.json')
    with open(test_clusters_path, 'w') as f:
        json.dump(test_clusters, f, indent=2)
    
    print(f"\nSelected {len(test_clusters)} clusters for testing:")
    for cluster_id, cluster_data in test_clusters.items():
        print(f"  - Cluster {cluster_id}: {cluster_data['size']} theories")
    
    # Initialize refiner with test clusters
    refiner = Stage5ClusterRefiner(
        clusters_path=str(test_clusters_path),
        max_concurrent=3
    )
    
    # Process clusters
    test_output_path = Path('output/test_stage5_output.json')
    output = refiner.process_clusters(
        output_path=str(test_output_path),
        resume_from_checkpoint=False
    )
    
    # Save test results
    refiner.save_results(output, output_path=str(test_output_path))
    
    # Analyze results
    print("\n" + "="*80)
    print("TEST RESULTS ANALYSIS")
    print("="*80)
    
    for cluster_result in output['clusters']:
        cluster_id = cluster_result['cluster_id']
        normalizations = cluster_result.get('normalizations', [])
        error = cluster_result.get('error')
        
        print(f"\nCluster {cluster_id}:")
        if error:
            print(f"  ⚠️ Error: {error}")
        
        print(f"  Normalizations: {len(normalizations)}")
        for norm in normalizations:
            print(f"    - {norm['original_name']}")
            print(f"      Strategy: {norm.get('strategy', 'N/A')}")
            print(f"      Normalized: {norm.get('normalized_name', 'N/A')}")
            print(f"      Confidence: {norm.get('mapping_confidence', 'N/A')}")
            reasoning = norm.get('reasoning', 'N/A')
            print(f"      Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
    
    print("\n✅ Test completed successfully!")
    print(f"Full test results saved to: {test_output_path}")


if __name__ == '__main__':
    test_stage5_on_subset()
