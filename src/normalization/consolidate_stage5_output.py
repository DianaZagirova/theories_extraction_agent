#!/usr/bin/env python3
"""
Consolidate Stage 5 output with theory ID mapping.

This script maps filtered_paper_data.json through Stage 5 normalization:
1. Loads filtered_paper_data.json (theory IDs with final_name_normalized)
2. Loads clusters_with_paper_counts.json (paper counts per theory)
3. Loads stage5_cluster_refined_theories.json (Stage 5 normalization)
4. Creates consolidated mapping:
   - For theories with paper_count >= 6: final_name = original theory_name (reference)
   - For theories with paper_count < 6: final_name = normalized_name from Stage 5
   - Collects all theory_ids and dois for theories mapped to each final_name
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_stage0_index(stage0_path: str) -> Dict[str, str]:
    """
    Build index from stage0_filtered_theories.json.
    
    Returns:
        Dict mapping theory_id -> initial name
    """
    stage0_data = load_json(stage0_path)
    
    theory_id_to_name = {}
    
    for theory in stage0_data.get('theories', []):
        theory_id = theory.get('theory_id')
        name = theory.get('name')
        if theory_id and name:
            theory_id_to_name[theory_id] = name
    
    return theory_id_to_name


def build_filtered_data_index(filtered_data_path: str) -> Dict[str, List[dict]]:
    """
    Build index from filtered_paper_data.json.
    
    Returns:
        Dict mapping final_name_normalized -> list of {'theory_id': str, 'doi': str}
    """
    filtered_data = load_json(filtered_data_path)
    
    name_to_entries = defaultdict(list)
    
    for theory_id, entry in filtered_data.items():
        final_name_normalized = entry.get('final_name_normalized')
        doi = entry.get('doi')
        if final_name_normalized:
            name_to_entries[final_name_normalized].append({
                'theory_id': theory_id,
                'doi': doi
            })
    
    return dict(name_to_entries)


def consolidate_stage5_output(
    clusters_path: str,
    stage5_output_path: str,
    filtered_data_path: str,
    stage0_path: str,
    output_path: str
):
    """
    Consolidate Stage 5 output with theory ID mapping.
    """
    print("Loading input files...")
    clusters = load_json(clusters_path)
    stage5_output = load_json(stage5_output_path)
    
    print("Building stage0 index...")
    stage0_index = build_stage0_index(stage0_path)
    
    print("Building filtered data index...")
    filtered_data_index = build_filtered_data_index(filtered_data_path)
    
    # Build mapping from original_name (Stage 4 name) to normalized_name (Stage 5 name)
    print("Building Stage 5 normalization mapping...")
    original_to_normalized = {}
    
    for cluster in stage5_output['clusters']:
        normalizations = cluster.get('normalizations', [])
        
        for norm in normalizations:
            original_name = norm.get('original_name')  # This is final_name_normalized from Stage 4
            normalized_name = norm.get('normalized_name')  # This is the Stage 5 normalized name
            strategy = norm.get('strategy')
            
            if original_name and normalized_name:
                original_to_normalized[original_name] = {
                    'final_name': normalized_name,
                    'strategy': strategy
                }
    
    # Get paper counts from clusters
    print("Extracting paper counts...")
    theory_paper_counts = {}
    for cluster_id, cluster_data in clusters.items():
        for member in cluster_data.get('members', []):
            theory_name = member['theory_name']
            paper_count = member['paper_count']
            theory_paper_counts[theory_name] = paper_count
    
    # Build final mapping: for each theory in filtered_data, determine its final_name
    print("\nBuilding consolidated mapping...")
    final_name_groups = defaultdict(lambda: {
        'original_names': set(),
        'theory_ids': [],
        'dois': [],
        'initial_names': [],
        'strategies': set()
    })
    
    stats = {
        'total_theories_in_filtered_data': 0,
        'reference_theories': 0,
        'normalized_theories': 0,
        'missing_from_stage5': 0
    }
    
    for theory_name_normalized, entries in filtered_data_index.items():
        stats['total_theories_in_filtered_data'] += len(entries)
        
        # Determine final_name for this theory
        paper_count = theory_paper_counts.get(theory_name_normalized, 0)
        
        if paper_count >= 6:
            # Reference theory - keep original name
            final_name = theory_name_normalized
            strategy = 'reference'
            stats['reference_theories'] += len(entries)
        else:
            # Rare theory - use Stage 5 normalized name
            if theory_name_normalized in original_to_normalized:
                mapping = original_to_normalized[theory_name_normalized]
                final_name = mapping['final_name']
                strategy = mapping['strategy']
                stats['normalized_theories'] += len(entries)
            else:
                # Missing from Stage 5 - retain original
                final_name = theory_name_normalized
                strategy = 'retain'
                stats['missing_from_stage5'] += len(entries)
                print(f"  ⚠️  Theory missing from Stage 5: {theory_name_normalized}")
        
        # Collect theory_ids, dois, and initial_names for this final_name
        final_name_groups[final_name]['original_names'].add(theory_name_normalized)
        final_name_groups[final_name]['strategies'].add(strategy)
        
        for entry in entries:
            theory_id = entry['theory_id']
            doi = entry['doi']
            initial_name = stage0_index.get(theory_id, 'Unknown')
            
            final_name_groups[final_name]['theory_ids'].append(theory_id)
            final_name_groups[final_name]['dois'].append(doi)
            final_name_groups[final_name]['initial_names'].append(initial_name)
    
    # Create final summary
    print("\nCreating final summary...")
    final_name_summary = []
    
    for final_name, data in final_name_groups.items():
        original_names = sorted(list(data['original_names']))
        theory_ids = data['theory_ids']
        dois = data['dois']
        initial_names = data['initial_names']
        strategies = sorted(list(data['strategies']))
        
        final_name_summary.append({
            'final_name': final_name,
            'original_names_count': len(original_names),
            'original_names': original_names,
            'total_papers': len(theory_ids),  # Count of theory_ids = count of DOIs
            'theory_ids_count': len(theory_ids),
            'theory_ids': theory_ids,
            'dois': dois,
            'initial_names': initial_names,
            'strategies': strategies
        })
    
    # Sort by total papers descending
    final_name_summary.sort(key=lambda x: x['total_papers'], reverse=True)
    
    # Create output
    output = {
        'metadata': {
            'source_clusters': clusters_path,
            'source_stage5': stage5_output_path,
            'source_filtered_data': filtered_data_path,
            'total_theory_ids': stats['total_theories_in_filtered_data'],
            'reference_theory_ids': stats['reference_theories'],
            'normalized_theory_ids': stats['normalized_theories'],
            'missing_from_stage5': stats['missing_from_stage5'],
            'unique_final_names': len(final_name_groups),
            'stage5_metadata': stage5_output.get('metadata', {})
        },
        'final_name_summary': final_name_summary
    }
    
    # Save output
    print(f"\nSaving consolidated output to {output_path}...")
    save_json(output, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("CONSOLIDATION SUMMARY")
    print("="*80)
    print(f"Total theory IDs processed: {stats['total_theories_in_filtered_data']}")
    print(f"  - Reference theory IDs (≥6 papers): {stats['reference_theories']}")
    print(f"  - Normalized theory IDs (<6 papers): {stats['normalized_theories']}")
    print(f"  - Missing from Stage 5: {stats['missing_from_stage5']}")
    print(f"\nUnique final names: {len(final_name_groups)}")
    print(f"  - Reduction: {stats['total_theories_in_filtered_data']} theory IDs → {len(final_name_groups)} final names")
    print(f"  - Compression ratio: {len(final_name_groups) / stats['total_theories_in_filtered_data']:.2%}")
    
    print(f"\n✅ Consolidated output saved to: {output_path}")
    
    # Print top 10 final names by paper count
    print("\n" + "="*80)
    print("TOP 10 FINAL NAMES BY THEORY ID COUNT")
    print("="*80)
    for i, entry in enumerate(final_name_summary[:10], 1):
        print(f"{i}. {entry['final_name']}")
        print(f"   Theory IDs: {entry['theory_ids_count']} | Original names: {entry['original_names_count']}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate Stage 5 output with theory ID mapping'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        default='data/clustering_data/clusters_with_paper_counts.json',
        help='Path to clusters_with_paper_counts.json'
    )
    parser.add_argument(
        '--stage5-output',
        type=str,
        default='output/stage5_cluster_refined_theories.json',
        help='Path to Stage 5 output JSON'
    )
    parser.add_argument(
        '--filtered-data',
        type=str,
        default='data/clustering_data/filtered_paper_data.json',
        help='Path to filtered_paper_data.json (max paper_focus per DOI)'
    )
    parser.add_argument(
        '--stage0',
        type=str,
        default='output/stage0_filtered_theories.json',
        help='Path to stage0_filtered_theories.json (initial theory names)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/stage5_consolidated_final_theories.json',
        help='Path to output consolidated JSON'
    )
    
    args = parser.parse_args()
    
    consolidate_stage5_output(
        clusters_path=args.clusters,
        stage5_output_path=args.stage5_output,
        filtered_data_path=args.filtered_data,
        stage0_path=args.stage0,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
