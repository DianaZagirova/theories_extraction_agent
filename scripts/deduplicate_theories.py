#!/usr/bin/env python3
"""
Deduplicate theories from Stage 1.5 output.

Handles cases where multiple papers extract the same theory or
where the same paper extracts the same theory multiple times.

Usage:
    python scripts/deduplicate_theories.py
    python scripts/deduplicate_theories.py --input output/stage1_5_llm_mapped.json --output output/stage1_5_deduplicated.json
"""

import json
import argparse
from collections import defaultdict
from typing import List, Dict


def deduplicate_by_paper_and_theory(theories: List[Dict]) -> Dict:
    """
    Deduplicate theories: keep one instance per (canonical_name, paper).
    
    Args:
        theories: List of theory dictionaries
        
    Returns:
        Dictionary with deduplicated theories and statistics
    """
    # Group by (canonical_name, paper_id)
    grouped = defaultdict(list)
    
    for theory in theories:
        canonical = theory.get('canonical_name')
        paper_id = theory.get('doi') or theory.get('pmid') or theory.get('theory_id', '').split('_')[0]
        
        key = (canonical, paper_id)
        grouped[key].append(theory)
    
    # Keep first instance of each group
    deduplicated = []
    duplicate_count = 0
    
    for (canonical, paper_id), group in grouped.items():
        if len(group) > 1:
            duplicate_count += len(group) - 1
        
        # Keep the first one (or merge if needed)
        deduplicated.append(group[0])
    
    return {
        'theories': deduplicated,
        'original_count': len(theories),
        'deduplicated_count': len(deduplicated),
        'duplicates_removed': duplicate_count
    }


def aggregate_by_canonical(theories: List[Dict]) -> List[Dict]:
    """
    Aggregate theories by canonical name with statistics.
    
    Args:
        theories: List of theory dictionaries
        
    Returns:
        List of aggregated theory statistics
    """
    aggregated = defaultdict(lambda: {
        'canonical_name': None,
        'count': 0,
        'papers': set(),
        'variants': set(),
        'theory_ids': []
    })
    
    for theory in theories:
        canonical = theory.get('canonical_name')
        if not canonical:
            continue
            
        aggregated[canonical]['canonical_name'] = canonical
        aggregated[canonical]['count'] += 1
        
        paper_id = theory.get('doi') or theory.get('pmid')
        if paper_id:
            aggregated[canonical]['papers'].add(paper_id)
        
        aggregated[canonical]['variants'].add(theory.get('original_name', ''))
        aggregated[canonical]['theory_ids'].append(theory.get('theory_id'))
    
    # Convert to list and sort by count
    result = []
    for data in aggregated.values():
        result.append({
            'canonical_name': data['canonical_name'],
            'total_mentions': data['count'],
            'unique_papers': len(data['papers']),
            'name_variants': sorted(list(data['variants'])),
            'theory_ids': data['theory_ids'][:10]  # Sample of IDs
        })
    
    return sorted(result, key=lambda x: x['total_mentions'], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description='Deduplicate theories from Stage 1.5 output'
    )
    parser.add_argument(
        '--input',
        default='output/stage1_5_llm_mapped.json',
        help='Input JSON file (default: output/stage1_5_llm_mapped.json)'
    )
    parser.add_argument(
        '--output',
        default='output/stage1_5_deduplicated.json',
        help='Output JSON file (default: output/stage1_5_deduplicated.json)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics without creating output file'
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Create aggregated statistics by canonical theory'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Process each category
    categories = ['mapped_theories', 'novel_theories', 'still_unmatched', 'invalid_theories']
    
    print("\n" + "="*80)
    print("DEDUPLICATION ANALYSIS")
    print("="*80)
    
    total_original = 0
    total_deduplicated = 0
    total_removed = 0
    
    deduplicated_data = {
        'metadata': data.get('metadata', {}),
    }
    
    for category in categories:
        theories = data.get(category, [])
        if not theories:
            continue
        
        result = deduplicate_by_paper_and_theory(theories)
        
        print(f"\n{category}:")
        print(f"  Original: {result['original_count']}")
        print(f"  Deduplicated: {result['deduplicated_count']}")
        print(f"  Removed: {result['duplicates_removed']}")
        
        total_original += result['original_count']
        total_deduplicated += result['deduplicated_count']
        total_removed += result['duplicates_removed']
        
        deduplicated_data[category] = result['theories']
    
    print(f"\n{'='*80}")
    print(f"TOTAL:")
    print(f"  Original: {total_original}")
    print(f"  Deduplicated: {total_deduplicated}")
    print(f"  Removed: {total_removed} ({total_removed/total_original*100:.1f}%)")
    print("="*80)
    
    # Show aggregated statistics if requested
    if args.aggregate:
        print("\n" + "="*80)
        print("AGGREGATED STATISTICS BY CANONICAL THEORY")
        print("="*80)
        
        mapped = data.get('mapped_theories', [])
        aggregated = aggregate_by_canonical(mapped)
        
        print(f"\nTop 20 most mentioned theories:")
        for i, theory in enumerate(aggregated[:20], 1):
            print(f"\n{i}. {theory['canonical_name']}")
            print(f"   Total mentions: {theory['total_mentions']}")
            print(f"   Unique papers: {theory['unique_papers']}")
            print(f"   Name variants: {len(theory['name_variants'])}")
            if len(theory['name_variants']) <= 3:
                for variant in theory['name_variants']:
                    print(f"     - {variant}")
    
    # Save deduplicated data
    if not args.stats_only:
        print(f"\n💾 Saving deduplicated data to {args.output}...")
        
        # Update metadata
        deduplicated_data['metadata']['deduplication'] = {
            'original_count': total_original,
            'deduplicated_count': total_deduplicated,
            'duplicates_removed': total_removed,
            'deduplication_method': 'by_paper_and_canonical_theory'
        }
        
        with open(args.output, 'w') as f:
            json.dump(deduplicated_data, f, indent=2)
        
        print(f"✓ Saved deduplicated data")
    
    print("\n✅ Deduplication complete!")


if __name__ == '__main__':
    main()
