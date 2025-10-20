#!/usr/bin/env python3
"""
Export Stage 1.5 checkpoint database to JSON format.

This script loads results from the checkpoint database and exports them
to JSON format for use in subsequent pipeline stages.

Usage:
    python scripts/export_checkpoint_to_json.py
    python scripts/export_checkpoint_to_json.py --checkpoint output/stage1_5_llm_mapped_checkpoint.db --output output/stage1_5_exported.json
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def load_results_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load all mapping results from checkpoint database.
    
    Args:
        checkpoint_path: Path to checkpoint database
        
    Returns:
        Dictionary with results and metadata
    """
    print(f"📂 Loading from checkpoint: {checkpoint_path}")
    
    conn = sqlite3.connect(checkpoint_path)
    
    # Load mapping results
    cursor = conn.execute("""
        SELECT 
            theory_id, original_name, is_valid_theory, validation_reasoning,
            is_mapped, novelty_reasoning, canonical_name, mapping_confidence,
            is_novel, proposed_name, batch_number, timestamp
        FROM mapping_results
        ORDER BY batch_number, theory_id
    """)
    
    results = []
    for row in cursor.fetchall():
        result = {
            'theory_id': row[0],
            'original_name': row[1],
            'is_valid_theory': bool(row[2]),
            'validation_reasoning': row[3],
            'is_mapped': bool(row[4]),
            'novelty_reasoning': row[5],
            'canonical_name': row[6],
            'mapping_confidence': row[7],
            'is_novel': bool(row[8]),
            'proposed_name': row[9],
            'batch_number': row[10],
            'timestamp': row[11]
        }
        results.append(result)
    
    print(f"✓ Loaded {len(results)} theory results")
    
    # Load batch metadata
    cursor = conn.execute("""
        SELECT 
            batch_number, theory_count, input_tokens, output_tokens, 
            cost, timestamp
        FROM batch_metadata
        ORDER BY batch_number
    """)
    
    batches = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for row in cursor.fetchall():
        batch = {
            'batch_number': row[0],
            'theory_count': row[1],
            'input_tokens': row[2],
            'output_tokens': row[3],
            'cost': row[4],
            'timestamp': row[5]
        }
        batches.append(batch)
        total_input_tokens += row[2] or 0
        total_output_tokens += row[3] or 0
        total_cost += row[4] or 0.0
    
    print(f"✓ Loaded {len(batches)} batch records")
    
    # Load run metadata
    cursor = conn.execute("SELECT key, value FROM run_metadata")
    metadata = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Calculate statistics
    valid_count = sum(1 for r in results if r['is_valid_theory'])
    mapped_count = sum(1 for r in results if r['is_mapped'])
    novel_count = sum(1 for r in results if r['is_novel'])
    invalid_count = sum(1 for r in results if not r['is_valid_theory'])
    
    stats = {
        'total_processed': len(results),
        'valid_theories': valid_count,
        'invalid_theories': invalid_count,
        'mapped_to_canonical': mapped_count,
        'novel_theories': novel_count,
        'still_unmatched': len(results) - mapped_count - novel_count - invalid_count,
        'batch_count': len(batches),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_cost': total_cost
    }
    
    return {
        'results': results,
        'batches': batches,
        'metadata': metadata,
        'statistics': stats
    }


def categorize_results(results: List[Dict]) -> Dict:
    """
    Categorize results into mapped, novel, unmatched, and invalid.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with categorized results
    """
    mapped_theories = []
    novel_theories = []
    still_unmatched = []
    invalid_theories = []
    
    for result in results:
        if not result['is_valid_theory']:
            invalid_theories.append(result)
        elif result['is_mapped'] and result['canonical_name']:
            mapped_theories.append(result)
        elif result['is_novel']:
            novel_theories.append(result)
        else:
            still_unmatched.append(result)
    
    return {
        'mapped_theories': mapped_theories,
        'novel_theories': novel_theories,
        'still_unmatched': still_unmatched,
        'invalid_theories': invalid_theories
    }


def export_to_json(checkpoint_path: str, output_path: str, include_batches: bool = False):
    """
    Export checkpoint database to JSON format.
    
    Args:
        checkpoint_path: Path to checkpoint database
        output_path: Path to output JSON file
        include_batches: Whether to include batch metadata in output
    """
    # Load data from checkpoint
    data = load_results_from_checkpoint(checkpoint_path)
    
    # Categorize results
    print("\n📊 Categorizing results...")
    categorized = categorize_results(data['results'])
    
    print(f"   Mapped to canonical: {len(categorized['mapped_theories'])}")
    print(f"   Novel theories: {len(categorized['novel_theories'])}")
    print(f"   Still unmatched: {len(categorized['still_unmatched'])}")
    print(f"   Invalid theories: {len(categorized['invalid_theories'])}")
    
    # Build output structure
    output_data = {
        'metadata': {
            'stage': 'stage1_5_llm_mapping',
            'export_timestamp': datetime.now().isoformat(),
            'source': checkpoint_path,
            'statistics': data['statistics'],
            'run_metadata': data['metadata']
        },
        'mapped_theories': categorized['mapped_theories'],
        'novel_theories': categorized['novel_theories'],
        'still_unmatched': categorized['still_unmatched'],
        'invalid_theories': categorized['invalid_theories']
    }
    
    # Optionally include batch metadata
    if include_batches:
        output_data['metadata']['batches'] = data['batches']
    
    # Save to JSON
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Exported {len(data['results'])} results to JSON")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    print(f"Total theories: {data['statistics']['total_processed']}")
    print(f"Valid theories: {data['statistics']['valid_theories']}")
    print(f"Mapped to canonical: {data['statistics']['mapped_to_canonical']}")
    print(f"Novel theories: {data['statistics']['novel_theories']}")
    print(f"Invalid theories: {data['statistics']['invalid_theories']}")
    print(f"\nProcessing cost: ${data['statistics']['total_cost']:.4f}")
    print(f"Total tokens: {data['statistics']['total_input_tokens'] + data['statistics']['total_output_tokens']:,}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Export Stage 1.5 checkpoint database to JSON format'
    )
    parser.add_argument(
        '--checkpoint',
        default='output/stage1_5_llm_mapped_checkpoint.db',
        help='Path to checkpoint database (default: output/stage1_5_llm_mapped_checkpoint.db)'
    )
    parser.add_argument(
        '--output',
        default='output/stage1_5_llm_mapped.json',
        help='Path to output JSON file (default: output/stage1_5_llm_mapped.json)'
    )
    parser.add_argument(
        '--include-batches',
        action='store_true',
        help='Include batch metadata in output'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint database not found: {args.checkpoint}")
        print("   Make sure you've run stage1_5_llm_mapping.py first")
        return 1
    
    # Export to JSON
    try:
        export_to_json(args.checkpoint, args.output, args.include_batches)
        print(f"\n✅ Export complete! Output saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
