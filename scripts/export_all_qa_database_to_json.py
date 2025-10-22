
import sys
import os
from pathlib import Path
import sqlite3
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_all_results(results_db: str, output_file: str, validation_data: dict = None):
    """Export QA results for all papers in the database."""
    
    # Build validation lookup by DOI (if validation_data given)
    validation_lookup = {}
    # if validation_data:
    #     for entry in validation_data:
    #         doi = entry.get('doi')
    #         if doi:
    #             validation_lookup[doi] = {}
    #             for q_text, answer in entry.items():
    #                 if q_text != 'doi' and q_text in QUESTION_MAPPING:
    #                     q_name = QUESTION_MAPPING[q_text]
    #                     validation_lookup[doi][q_name] = answer
    
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    
    # Fetch all DOIs from the paper_metadata table
    cur.execute("SELECT doi FROM paper_metadata")
    all_dois = [row[0] for row in cur.fetchall()]
    
    results = []  
    for doi in all_dois:
        # Get paper metadata
        cur.execute(
            """
            SELECT pmid, title, used_full_text
            FROM paper_metadata
            WHERE doi = ?
            """,
            (doi,)
        )
        metadata = cur.fetchone()
        if not metadata:
            print(f"⚠️  Paper metadata not found for DOI: {doi}")
            continue
        
        pmid, title, used_full_text = metadata
        
        # Get answers
        cur.execute(
            """
            SELECT question_name, question_text, answer, confidence_score, reasoning
            FROM paper_answers
            WHERE doi = ?
            ORDER BY question_name
            """,
            (doi,)
        )
        answers = cur.fetchall()
        if not answers:
            print(f"⚠️  No answers found for DOI: {doi}")
            continue
        
        # Format result
        result = {
            'doi': doi,
            'pmid': pmid,
            'title': title,
            'used_full_text': bool(used_full_text),
            'answers': {}
        }
        
        for q_name, q_text, answer, confidence, reasoning in answers:
            answer_data = {
                'question': q_text,
                'answer': answer,
                'confidence': float(confidence) if confidence else 0.0,
                'reasoning': reasoning
            }
            
            # Add correct answer if available from validation set
            if doi in validation_lookup and q_name in validation_lookup[doi]:
                answer_data['correct_answer'] = validation_lookup[doi][q_name]
            
            result['answers'][q_name] = answer_data
        
        results.append(result)
        # print(f"✓ Exported results for: {doi}")
    
    conn.close()
    
    # Save all results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Exported {len(results)} papers to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Export QA results for papers')
    parser.add_argument('--results-db', default='./qa_results/qa_results.db',
                        help='Path to results database')
    parser.add_argument('--dois-file', default=None, help='File containing DOIs (one per line)')
    parser.add_argument('--validation-file', default='data/validation_qa/qa_validation_set_extended.json',
                        help='Validation file to extract DOIs from')
    parser.add_argument('--output-file', default='./qa_results/qa_evaluation_results.json',
                        help='Output JSON file')
    parser.add_argument('--all', action='store_true',
                        help='Export results for ALL papers in the database')
    
    args = parser.parse_args()

    validation_data = None
    if args.validation_file and Path(args.validation_file).exists():
        print(f"📄 Loading validation set: {args.validation_file}")
        with open(args.validation_file, 'r') as f:
            validation_data = json.load(f)
        print(f"✓ Loaded {len(validation_data)} validation entries")
    
    # if args.all:
    print("📄 Exporting results for ALL papers in the database.")
    export_all_results(args.results_db, args.output_file, validation_data)
if __name__ == "__main__":
    main()