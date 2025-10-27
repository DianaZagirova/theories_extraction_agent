#!/usr/bin/env python3
"""
Export 50 random DOIs from theories.db and theories_abstract.db to JSON samples.
"""

import sqlite3
import json
import random
from pathlib import Path


def export_sample_from_db(db_path: str, output_path: str, sample_size: int = 50):
    """Export sample_size random DOIs with all their theories from a database."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all unique DOIs
    cursor.execute("SELECT DISTINCT doi FROM theories")
    all_dois = [row['doi'] for row in cursor.fetchall()]
    
    # Sample random DOIs
    sample_dois = random.sample(all_dois, min(sample_size, len(all_dois)))
    
    # Build the output structure
    output_data = []
    
    for doi in sample_dois:
        # Get paper metadata
        cursor.execute("""
            SELECT doi, pmid, title, validation_result, confidence_score, timestamp
            FROM paper_results
            WHERE doi = ?
        """, (doi,))
        
        paper_row = cursor.fetchone()
        
        if paper_row:
            paper_data = {
                "doi": paper_row['doi'],
                "pmid": paper_row['pmid'],
                "title": paper_row['title'],
                "validation_result": paper_row['validation_result'],
                "confidence_score": paper_row['confidence_score'],
                "timestamp": paper_row['timestamp'],
                "theories": []
            }
        else:
            # If no paper_results entry, create minimal structure
            paper_data = {
                "doi": doi,
                "pmid": None,
                "title": None,
                "validation_result": None,
                "confidence_score": None,
                "timestamp": None,
                "theories": []
            }
        
        # Get all theories for this DOI
        cursor.execute("""
            SELECT name, mode, evidence, confidence_is_theory, 
                   criteria_reasoning, paper_focus, key_concepts
            FROM theories
            WHERE doi = ?
        """, (doi,))
        
        for theory_row in cursor.fetchall():
            theory_data = {
                "name": theory_row['name'],
                "mode": theory_row['mode'],
                "evidence": theory_row['evidence'],
                "confidence_is_theory": theory_row['confidence_is_theory'],
                "criteria_reasoning": theory_row['criteria_reasoning'],
                "paper_focus": theory_row['paper_focus'],
                "key_concepts": json.loads(theory_row['key_concepts']) if theory_row['key_concepts'] else []
            }
            paper_data["theories"].append(theory_data)
        
        output_data.append(paper_data)
    
    conn.close()
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(output_data)} papers from {db_path} to {output_path}")
    return len(output_data)


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    base_dir = Path(__file__).parent.parent
    
    # Export from theories.db (full-text)
    theories_db = base_dir / "theories.db"
    theories_sample = base_dir / "data" / "sample_theories_fulltext.json"
    theories_sample.parent.mkdir(parents=True, exist_ok=True)
    
    count1 = export_sample_from_db(str(theories_db), str(theories_sample), sample_size=50)
    
    # Export from theories_abstract.db (abstract-only)
    theories_abstract_db = base_dir / "theories_abstract.db"
    theories_abstract_sample = base_dir / "data" / "sample_theories_abstract.json"
    
    count2 = export_sample_from_db(str(theories_abstract_db), str(theories_abstract_sample), sample_size=50)
    
    print(f"\n✓ Total papers exported: {count1 + count2}")
    print(f"  - Full-text: {count1} papers")
    print(f"  - Abstract-only: {count2} papers")


if __name__ == "__main__":
    main()
