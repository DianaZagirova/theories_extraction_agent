#!/usr/bin/env python3
"""
Export 50 random DOIs with their QA results from qa_results.db to JSON sample.
"""

import sqlite3
import json
import random
from pathlib import Path


def export_sample_qa_results(db_path: str, output_path: str, sample_size: int = 50):
    """Export sample_size random DOIs with all their QA data from the database."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all unique DOIs
    cursor.execute("SELECT DISTINCT doi FROM paper_metadata")
    all_dois = [row['doi'] for row in cursor.fetchall()]
    
    # Sample random DOIs
    sample_dois = random.sample(all_dois, min(sample_size, len(all_dois)))
    
    # Build the output structure
    output_data = []
    
    for doi in sample_dois:
        # Get paper metadata
        cursor.execute("""
            SELECT doi, pmid, title, validation_result, confidence_score,
                   processed_text_length, used_full_text, timestamp
            FROM paper_metadata
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
                "processed_text_length": paper_row['processed_text_length'],
                "used_full_text": bool(paper_row['used_full_text']),
                "timestamp": paper_row['timestamp'],
                "answers": []
            }
            
            # Get all answers for this DOI
            cursor.execute("""
                SELECT question_name, question_text, answer, confidence_score,
                       reasoning, original_answer
                FROM paper_answers
                WHERE doi = ?
            """, (doi,))
            
            for answer_row in cursor.fetchall():
                answer_data = {
                    "question_name": answer_row['question_name'],
                    "question_text": answer_row['question_text'],
                    "answer": answer_row['answer'],
                    "confidence_score": answer_row['confidence_score'],
                    "reasoning": answer_row['reasoning'],
                    "original_answer": answer_row['original_answer']
                }
                paper_data["answers"].append(answer_data)
            
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
    
    # Export from qa_results.db
    qa_db = base_dir / "qa_results" / "qa_results.db"
    qa_sample = base_dir / "data" / "sample_qa_results.json"
    qa_sample.parent.mkdir(parents=True, exist_ok=True)
    
    count = export_sample_qa_results(str(qa_db), str(qa_sample), sample_size=50)
    
    print(f"\n✓ Total papers exported: {count}")
    print(f"  - Sample file: data/sample_qa_results.json")


if __name__ == "__main__":
    main()
