"""
Export QA results for specific DOIs to JSON format.

Extracts answers from the database and formats them for comparison.
"""
import sys
import os
from pathlib import Path
import sqlite3
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


# Question mapping from validation set to our question names
QUESTION_MAPPING = {
    "Q1: Does the paper suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state  associated with mortality or age-related conditions)?": "aging_biomarker",
    "Q2: Does the paper suggest a molecular mechanism of aging?": "molecular_mechanism_of_aging",
    "Q3: Does the paper suggest a longevity intervention to test?": "longevity_intervention_to_test",
    "Q4: Does the paper claim that aging cannot be reversed? (Yes / No)": "aging_cannot_be_reversed",
    "Q5: Does the paper suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)": "cross_species_longevity_biomarker",
    "Q6: Does the paper explain why the naked mole rat can live 40+ years despite its small size?": "naked_mole_rat_lifespan_explanation",
    "Q7: Does the paper explain why birds live much longer than mammals on average?": "birds_lifespan_explanation",
    "Q8: Does the paper explain why large animals live longer than small ones?": "large_animals_lifespan_explanation",
    "Q9: Does the paper explain why calorie restriction increases the lifespan of vertebrates?": "calorie_restriction_lifespan_explanation"
}


def export_results(results_db: str, dois: list, output_file: str, validation_data: dict = None):
    """Export QA results for given DOIs."""
    
    # Build validation lookup by DOI
    validation_lookup = {}
    if validation_data:
        for entry in validation_data:
            doi = entry.get('doi')
            if doi:
                validation_lookup[doi] = {}
                for q_text, answer in entry.items():
                    if q_text != 'doi' and q_text in QUESTION_MAPPING:
                        q_name = QUESTION_MAPPING[q_text]
                        validation_lookup[doi][q_name] = answer
    
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    
    results = []
    
    for doi in dois:
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
            print(f"⚠️  Paper not found in database: {doi}")
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
            print(f"⚠️  No answers found for: {doi}")
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
        print(f"✓ Exported results for: {doi}")
    
    conn.close()
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Exported {len(results)} papers to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Export QA results for specific DOIs')
    parser.add_argument('--results-db', default='./qa_results/qa_results.db',
                        help='Path to results database')
    parser.add_argument('--dois-file', default='data/validation_qa/sample_from_ext.txt', help='File containing DOIs (one per line)')
    parser.add_argument('--validation-file', default='data/validation_qa/qa_validation_set_extended.json',
                        help='Validation file to extract DOIs from')
    parser.add_argument('--output-file', default='./qa_results/qa_evaluation_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load validation data if available
    validation_data = None
    if args.validation_file and Path(args.validation_file).exists():
        print(f"📄 Loading validation set: {args.validation_file}")
        with open(args.validation_file, 'r') as f:
            validation_data = json.load(f)
        print(f"✓ Loaded {len(validation_data)} validation entries")
    
    # Get DOIs from file or validation set
    if args.dois_file:
        print(f"📄 Loading DOIs from: {args.dois_file}")
        with open(args.dois_file, 'r') as f:
            dois = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    else:
        print(f"📄 Using DOIs from validation set")
        dois = [entry['doi'] for entry in validation_data] if validation_data else []
    
    print(f"✓ Found {len(dois)} DOIs to export\n")
    
    # Export results
    export_results(args.results_db, dois, args.output_file, validation_data)


if __name__ == "__main__":
    main()
