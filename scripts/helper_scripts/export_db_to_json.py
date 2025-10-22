"""
Export theories.db to theories_per_paper.json format.
Useful for creating output file while extraction script is still running.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import argparse


def export_db_to_json(db_path: str, output_path: str):
    """Export theories database to JSON format matching extract_theories_per_paper.py output."""
    
    print(f"📖 Reading from: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all papers
    cur.execute("""
        SELECT doi, pmid, title, validation_result, confidence_score, timestamp
        FROM paper_results
        ORDER BY timestamp
    """)
    
    paper_rows = cur.fetchall()
    
    if not paper_rows:
        print("❌ No results found in database")
        conn.close()
        return
    
    print(f"✓ Found {len(paper_rows)} papers in database")
    
    # Get all theories
    cur.execute("""
        SELECT doi, name, mode, evidence, confidence_is_theory, 
               criteria_reasoning, paper_focus, key_concepts
        FROM theories
        ORDER BY doi, paper_focus DESC
    """)
    
    theory_rows = cur.fetchall()
    conn.close()
    
    print(f"✓ Found {len(theory_rows)} theories in database")
    
    # Group theories by DOI
    theories_by_doi = {}
    for theory_row in theory_rows:
        doi, name, mode, evidence, confidence, criteria, focus, key_concepts_json = theory_row
        
        if doi not in theories_by_doi:
            theories_by_doi[doi] = []
        
        # Parse key_concepts JSON
        try:
            key_concepts = json.loads(key_concepts_json) if key_concepts_json else []
        except json.JSONDecodeError:
            key_concepts = []
        
        theory = {
            'name': name,
            'key_concepts': key_concepts,
            'confidence_is_theory': confidence,
            'mode': mode,
            'evidence': evidence,
            'criteria_reasoning': criteria,
            'paper_focus': focus
        }
        
        theories_by_doi[doi].append(theory)
    
    # Convert to output format
    results = []
    stats = {
        'total_papers': len(paper_rows),
        'papers_with_theories': 0,
        'papers_without_theories': 0,
        'total_theories_extracted': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'estimated_cost_usd': 0.0
    }
    
    for paper_row in paper_rows:
        doi, pmid, title, validation_result, confidence_score, timestamp = paper_row
        
        # Get theories for this paper
        theories = theories_by_doi.get(doi, [])
        contains_theory = len(theories) > 0
        
        # Build result object
        result = {
            'doi': doi,
            'pmid': pmid,
            'title': title,
            'validation_result': validation_result,
            'confidence_score': confidence_score,
            'contains_theory': contains_theory,
            'theories': theories,
            'timestamp': timestamp
        }
        
        results.append(result)
        
        # Update stats
        if contains_theory:
            stats['papers_with_theories'] += 1
            stats['total_theories_extracted'] += len(theories)
        else:
            stats['papers_without_theories'] += 1
    
    # Create output data
    output_data = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'export_source': 'theories.db',
            'statistics': stats
        },
        'results': results
    }
    
    # Write to JSON
    print(f"💾 Writing to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   Papers exported: {stats['total_papers']}")
    print(f"   Papers with theories: {stats['papers_with_theories']}")
    print(f"   Papers without theories: {stats['papers_without_theories']}")
    print(f"   Total theories extracted: {stats['total_theories_extracted']}")
    
    if stats['papers_with_theories'] > 0:
        avg_theories = stats['total_theories_extracted'] / stats['papers_with_theories']
        print(f"   Avg theories per paper: {avg_theories:.1f}")
    
    print(f"\n✓ Exported to: {output_path}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export theories.db to JSON format"
    )
    
    parser.add_argument(
        '--db',
        type=str,
        default='theories.db',
        help='Path to theories database (default: theories.db)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='theories_per_paper.json',
        help='Output JSON file (default: theories_per_paper.json)'
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.db).exists():
        print(f"❌ Database not found: {args.db}")
        return
    
    export_db_to_json(args.db, args.output)


if __name__ == "__main__":
    main()
