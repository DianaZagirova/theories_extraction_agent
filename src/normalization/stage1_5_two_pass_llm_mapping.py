#!/usr/bin/env python3
"""
Stage 1.5 (Revised): Two-Pass LLM Mapping for Consistent Results

This approach solves the inconsistency problem by:
1. Pass 1: Map UNIQUE theory names only (not full theories)
2. Pass 2: Apply mappings to all theories with those names

This guarantees that identical theory names always get the same canonical mapping.

Usage:
    python src/normalization/stage1_5_two_pass_llm_mapping.py
    python src/normalization/stage1_5_two_pass_llm_mapping.py --reset
"""

import json
import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.stage1_5_llm_mapping import (
    LLMMapper, MappingResult, CheckpointDB
)


def normalize_theory_name(name: str) -> str:
    """
    Normalize theory names for better deduplication.
    
    Args:
        name: Original theory name
        
    Returns:
        Normalized name
    """
    name = name.strip()
    # Remove trailing periods
    name = name.rstrip('.')
    # Normalize spelling variations
    name = name.replace('Signalling', 'Signaling')
    name = name.replace('signalling', 'signaling')
    name = name.replace('Ageing', 'Aging')
    name = name.replace('ageing', 'aging')
    # Normalize whitespace
    name = ' '.join(name.split())
    return name


class TwoPassLLMMapper(LLMMapper):
    """
    Two-pass LLM mapper for consistent results.
    
    Pass 1: Map unique theory names
    Pass 2: Apply mappings to all theories
    """
    
    def process_unmatched_theories_two_pass(self,
                                           stage1_output_path: str,
                                           output_path: str,
                                           batch_size: int = 25,
                                           max_theories: Optional[int] = None,
                                           max_workers: int = 4) -> Dict:
        """
        Process unmatched theories using two-pass approach.
        
        Args:
            stage1_output_path: Path to Stage 1 output
            output_path: Path to save results
            batch_size: Number of theories per batch
            max_theories: Maximum theories to process (for testing)
            max_workers: Number of parallel workers
        """
        print("🚀 Starting Stage 1.5: Two-Pass LLM Mapping\n")
        
        # Load Stage 1 output
        print(f"📂 Loading Stage 1 output from {stage1_output_path}...")
        with open(stage1_output_path, 'r') as f:
            stage1_data = json.load(f)
        
        unmatched_theories = stage1_data.get('unmatched_theories', [])
        
        if max_theories:
            unmatched_theories = unmatched_theories[:max_theories]
        
        print(f"✓ Found {len(unmatched_theories)} unmatched theories")
        
        # ============================================================
        # PASS 1: Extract and Map Unique Theory Names
        # ============================================================
        print("\n" + "="*80)
        print("PASS 1: MAPPING UNIQUE THEORY NAMES")
        print("="*80)
        
        # Extract unique theory names
        unique_names = {}  # normalized_name → original_name
        name_to_theories = defaultdict(list)  # normalized_name → [theory_ids]
        
        for theory in unmatched_theories:
            original_name = theory.get('name') or theory.get('original_name', '')
            normalized = normalize_theory_name(original_name)
            
            if normalized not in unique_names:
                unique_names[normalized] = original_name
            
            name_to_theories[normalized].append(theory['theory_id'])
        
        print(f"\n📊 Statistics:")
        print(f"   Total theories: {len(unmatched_theories)}")
        print(f"   Unique names (after normalization): {len(unique_names)}")
        print(f"   Deduplication ratio: {len(unmatched_theories)/len(unique_names):.1f}x")
        
        # Create pseudo-theories for unique names
        unique_theory_objects = []
        for idx, (normalized, original) in enumerate(unique_names.items()):
            # Get concept text from first theory with this name
            first_theory_id = name_to_theories[normalized][0]
            first_theory = next(t for t in unmatched_theories if t['theory_id'] == first_theory_id)
            
            unique_theory_objects.append({
                'theory_id': f'UNIQUE_{idx:05d}',
                'name': original,
                'original_name': original,
                'concept_text': first_theory.get('concept_text', ''),
                'confidence_is_theory': first_theory.get('confidence_is_theory', 'high'),
                '_normalized_name': normalized,
                '_instance_count': len(name_to_theories[normalized])
            })
        
        print(f"\n🔄 Processing {len(unique_theory_objects)} unique theory names...")
        
        # Process unique names through LLM (using parent class method)
        name_mapping_results = super().process_unmatched_theories(
            stage1_output_path=stage1_output_path,  # Not used, but required
            output_path=output_path.replace('.json', '_unique_names.json'),
            batch_size=batch_size,
            max_theories=None,
            max_workers=max_workers
        )
        
        # Build lookup: normalized_name → MappingResult
        name_to_mapping = {}
        for result in name_mapping_results.get('mapped_theories', []) + \
                      name_mapping_results.get('novel_theories', []) + \
                      name_mapping_results.get('still_unmatched', []) + \
                      name_mapping_results.get('invalid_theories', []):
            
            normalized = normalize_theory_name(result.get('original_name', ''))
            name_to_mapping[normalized] = result.get('stage1_5_result', {})
        
        # ============================================================
        # PASS 2: Apply Mappings to All Theories
        # ============================================================
        print("\n" + "="*80)
        print("PASS 2: APPLYING MAPPINGS TO ALL THEORIES")
        print("="*80)
        
        mapped_theories = []
        novel_theories = []
        still_unmatched = []
        invalid_theories = []
        
        for theory in tqdm(unmatched_theories, desc="Applying mappings"):
            original_name = theory.get('name') or theory.get('original_name', '')
            normalized = normalize_theory_name(original_name)
            
            # Get mapping from Pass 1
            mapping = name_to_mapping.get(normalized)
            
            if not mapping:
                print(f"⚠️  Warning: No mapping found for {normalized}")
                continue
            
            # Apply mapping to theory
            theory_with_result = theory.copy()
            theory_with_result['stage1_5_result'] = mapping
            
            # Categorize
            if not mapping['is_valid_theory']:
                invalid_theories.append(theory_with_result)
            elif mapping['is_mapped'] and mapping['canonical_name']:
                # Add match_result for consistency
                theory_with_result['match_result'] = {
                    'matched': True,
                    'canonical_name': mapping['canonical_name'],
                    'match_type': 'llm_mapping_two_pass',
                    'confidence': mapping['mapping_confidence'],
                    'score': mapping['mapping_confidence']
                }
                mapped_theories.append(theory_with_result)
            elif mapping['is_novel']:
                novel_theories.append(theory_with_result)
            else:
                still_unmatched.append(theory_with_result)
        
        # Save results
        print(f"\n💾 Saving results to {output_path}...")
        
        output_data = {
            'metadata': {
                'stage': 'stage1_5_two_pass_llm_mapping',
                'method': 'two_pass_unique_names',
                'unique_names_processed': len(unique_names),
                'total_theories_processed': len(unmatched_theories),
                'deduplication_ratio': len(unmatched_theories) / len(unique_names),
                'statistics': {
                    'total_processed': len(unmatched_theories),
                    'valid_theories': len(mapped_theories) + len(novel_theories) + len(still_unmatched),
                    'mapped_to_canonical': len(mapped_theories),
                    'novel_theories': len(novel_theories),
                    'still_unmatched': len(still_unmatched),
                    'invalid_theories': len(invalid_theories)
                }
            },
            'mapped_theories': mapped_theories,
            'novel_theories': novel_theories,
            'still_unmatched': still_unmatched,
            'invalid_theories': invalid_theories
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
        
        # Print statistics
        print("\n" + "="*80)
        print("TWO-PASS MAPPING STATISTICS")
        print("="*80)
        print(f"Total theories processed: {len(unmatched_theories)}")
        print(f"Unique names: {len(unique_names)}")
        print(f"Deduplication: {len(unmatched_theories)/len(unique_names):.1f}x reduction")
        print(f"\nResults:")
        print(f"  Mapped to canonical: {len(mapped_theories)}")
        print(f"  Novel theories: {len(novel_theories)}")
        print(f"  Still unmatched: {len(still_unmatched)}")
        print(f"  Invalid theories: {len(invalid_theories)}")
        print("="*80)
        
        # Verify consistency
        self._verify_consistency(mapped_theories + novel_theories)
        
        return output_data
    
    def _verify_consistency(self, theories: List[Dict]):
        """Verify that same names always map to same canonical theories."""
        print("\n🔍 Verifying mapping consistency...")
        
        name_to_canonical = defaultdict(set)
        for theory in theories:
            name = theory.get('name') or theory.get('original_name', '')
            result = theory.get('stage1_5_result', {})
            canonical = result.get('canonical_name')
            if canonical:
                name_to_canonical[name].add(canonical)
        
        inconsistent = sum(1 for canonicals in name_to_canonical.values() if len(canonicals) > 1)
        
        if inconsistent == 0:
            print("✅ Perfect consistency: All identical names map to the same canonical theory!")
        else:
            print(f"❌ Found {inconsistent} inconsistent mappings!")
            for name, canonicals in name_to_canonical.items():
                if len(canonicals) > 1:
                    print(f"   {name} → {canonicals}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 1.5: Two-Pass LLM mapping')
    parser.add_argument('--input', default='output/stage1_fuzzy_matched.json')
    parser.add_argument('--output', default='output/stage1_5_llm_mapped.json')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--max-theories', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--reset', action='store_true')
    
    args = parser.parse_args()
    
    # Handle checkpoint reset
    checkpoint_path = args.output.replace('.json', '_checkpoint.db')
    if args.reset and os.path.exists(checkpoint_path):
        print(f"🗑️  Removing checkpoint: {checkpoint_path}")
        os.remove(checkpoint_path)
    
    # Run two-pass mapping
    mapper = TwoPassLLMMapper()
    mapper.process_unmatched_theories_two_pass(
        stage1_output_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        max_theories=args.max_theories,
        max_workers=args.max_workers
    )
    
    print("\n✅ Two-pass mapping complete!")


if __name__ == '__main__':
    main()
