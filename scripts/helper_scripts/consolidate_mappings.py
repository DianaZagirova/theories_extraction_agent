#!/usr/bin/env python3
"""
Consolidate inconsistent LLM mappings.

Fixes the issue where the same theory name maps to different canonical theories
by using majority voting and applying the most common mapping to all instances.

Usage:
    python scripts/consolidate_mappings.py
    python scripts/consolidate_mappings.py --input output/stage1_5_llm_mapped.json --output output/stage1_5_consolidated.json
"""

import json
import argparse
from collections import defaultdict, Counter
from typing import List, Dict


def normalize_theory_name(name: str) -> str:
    """Normalize theory names for comparison."""
    name = name.strip().rstrip('.')
    name = name.replace('Signalling', 'Signaling')
    name = name.replace('signalling', 'signaling')
    name = name.replace('Ageing', 'Aging')
    name = name.replace('ageing', 'aging')
    name = ' '.join(name.split())
    return name


def consolidate_mappings(theories: List[Dict], category: str) -> List[Dict]:
    """
    Consolidate mappings so same name always maps to same canonical theory.
    
    Uses majority voting: most common mapping wins.
    """
    # Group by normalized name
    name_groups = defaultdict(list)
    for theory in theories:
        name = theory.get('original_name', '')
        normalized = normalize_theory_name(name)
        name_groups[normalized].append(theory)
    
    consolidated = []
    changes = 0
    
    for normalized_name, group in name_groups.items():
        if len(group) == 1:
            consolidated.extend(group)
            continue
        
        # Count canonical mappings (for mapped theories)
        if category == 'mapped':
            canonical_votes = Counter(
                t.get('canonical_name') for t in group 
                if t.get('canonical_name')
            )
            
            if canonical_votes:
                winner, count = canonical_votes.most_common(1)[0]
                confidence_sum = sum(
                    t.get('mapping_confidence', 0) 
                    for t in group 
                    if t.get('canonical_name') == winner
                )
                avg_confidence = confidence_sum / count
                
                # Apply winner to all
                for theory in group:
                    if theory.get('canonical_name') != winner:
                        changes += 1
                        theory['canonical_name'] = winner
                        theory['mapping_confidence'] = avg_confidence
                        
                        # Update match_result if it exists
                        if 'match_result' in theory:
                            theory['match_result']['canonical_name'] = winner
                            theory['match_result']['confidence'] = avg_confidence
                            theory['match_result']['score'] = avg_confidence
                        
                        # Update stage1_5_result if it exists
                        if 'stage1_5_result' in theory:
                            theory['stage1_5_result']['canonical_name'] = winner
                            theory['stage1_5_result']['mapping_confidence'] = avg_confidence
                        
                        # Add consolidation note
                        if 'consolidation_note' not in theory:
                            theory['consolidation_note'] = f'Consolidated to majority vote: {winner}'
        
        # Count proposed names (for novel theories)
        elif category == 'novel':
            proposed_votes = Counter(
                t.get('proposed_name') for t in group 
                if t.get('proposed_name')
            )
            
            if proposed_votes:
                winner, count = proposed_votes.most_common(1)[0]
                
                # Apply winner to all
                for theory in group:
                    if theory.get('proposed_name') != winner:
                        changes += 1
                        theory['proposed_name'] = winner
                        
                        # Update stage1_5_result if it exists
                        if 'stage1_5_result' in theory:
                            theory['stage1_5_result']['proposed_name'] = winner
                        
                        if 'consolidation_note' not in theory:
                            theory['consolidation_note'] = f'Consolidated to majority vote: {winner}'
        
        consolidated.extend(group)
    
    return consolidated, changes


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate inconsistent LLM mappings'
    )
    parser.add_argument(
        '--input',
        default='output/stage1_5_llm_mapped.json',
        help='Input JSON file'
    )
    parser.add_argument(
        '--output',
        default='output/stage1_5_consolidated.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify input file in place'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Analyze inconsistencies before consolidation
    print("\n" + "="*80)
    print("ANALYZING INCONSISTENCIES")
    print("="*80)
    
    mapped = data.get('mapped_theories', [])
    novel = data.get('novel_theories', [])
    
    # Check for inconsistencies
    name_to_canonical = defaultdict(set)
    for theory in mapped:
        name = normalize_theory_name(theory.get('original_name', ''))
        canonical = theory.get('canonical_name')
        if canonical:
            name_to_canonical[name].add(canonical)
    
    inconsistent_count = sum(1 for canonicals in name_to_canonical.values() if len(canonicals) > 1)
    
    print(f"\nMapped theories: {len(mapped)}")
    print(f"Inconsistent mappings: {inconsistent_count}")
    
    if inconsistent_count > 0:
        print(f"\nExamples of inconsistent mappings:")
        shown = 0
        for name, canonicals in name_to_canonical.items():
            if len(canonicals) > 1 and shown < 5:
                shown += 1
                print(f"\n{shown}. \"{name}\"")
                for canonical in canonicals:
                    count = sum(1 for t in mapped 
                              if normalize_theory_name(t.get('original_name', '')) == name 
                              and t.get('canonical_name') == canonical)
                    print(f"     {count:2d}x → {canonical}")
    
    # Consolidate
    print("\n" + "="*80)
    print("CONSOLIDATING MAPPINGS")
    print("="*80)
    
    consolidated_mapped, mapped_changes = consolidate_mappings(mapped, 'mapped')
    consolidated_novel, novel_changes = consolidate_mappings(novel, 'novel')
    
    print(f"\nMapped theories: {mapped_changes} changes")
    print(f"Novel theories: {novel_changes} changes")
    
    # Update data
    data['mapped_theories'] = consolidated_mapped
    data['novel_theories'] = consolidated_novel
    
    # Add consolidation metadata
    if 'metadata' not in data:
        data['metadata'] = {}
    
    data['metadata']['consolidation'] = {
        'applied': True,
        'mapped_changes': mapped_changes,
        'novel_changes': novel_changes,
        'total_changes': mapped_changes + novel_changes
    }
    
    # Verify consistency
    print("\n" + "="*80)
    print("VERIFYING CONSISTENCY")
    print("="*80)
    
    name_to_canonical_after = defaultdict(set)
    for theory in consolidated_mapped:
        name = normalize_theory_name(theory.get('original_name', ''))
        canonical = theory.get('canonical_name')
        if canonical:
            name_to_canonical_after[name].add(canonical)
    
    inconsistent_after = sum(1 for canonicals in name_to_canonical_after.values() if len(canonicals) > 1)
    
    if inconsistent_after == 0:
        print("✅ Perfect consistency achieved!")
        print("   All identical theory names now map to the same canonical theory.")
    else:
        print(f"⚠️  Still {inconsistent_after} inconsistencies remaining")
    
    # Save
    output_path = args.input if args.in_place else args.output
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved consolidated data")
    
    print("\n✅ Consolidation complete!")
    print(f"\nSummary:")
    print(f"  Total changes: {mapped_changes + novel_changes}")
    print(f"  Inconsistencies before: {inconsistent_count}")
    print(f"  Inconsistencies after: {inconsistent_after}")


if __name__ == '__main__':
    main()
