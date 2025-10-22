"""
Evaluate question-answering results against validation set.

Compares LLM answers from the database with ground truth validation data
and computes accuracy metrics per question and overall.
"""
import sys
import os
from pathlib import Path
import sqlite3
import json
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mapping from validation set question keys to our question names
QUESTION_MAPPING = {
    # Extended validation set format (without /theory)
    "Q1: Does the paper suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state  associated with mortality or age-related conditions)?": "aging_biomarker",
    "Q2: Does the paper suggest a molecular mechanism of aging?": "molecular_mechanism_of_aging",
    "Q3: Does the paper suggest a longevity intervention to test?": "longevity_intervention_to_test",
    "Q4: Does the paper claim that aging cannot be reversed? (Yes / No)": "aging_cannot_be_reversed",
    "Q5: Does the paper suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)": "cross_species_longevity_biomarker",
    "Q6: Does the paper explain why the naked mole rat can live 40+ years despite its small size?": "naked_mole_rat_lifespan_explanation",
    "Q7: Does the paper explain why birds live much longer than mammals on average?": "birds_lifespan_explanation",
    "Q8: Does the paper explain why large animals live longer than small ones?": "large_animals_lifespan_explanation",
    "Q9: Does the paper explain why calorie restriction increases the lifespan of vertebrates?": "calorie_restriction_lifespan_explanation",
    # Original validation set format (with /theory)
    "Q1: Does the paper/theory suggest an aging biomarker (i.e. measurable entity reflecting the pace of aging or organism health state, associated with mortality or age-related conditions)?": "aging_biomarker",
    "Q2: Does the paper/theory suggest a molecular mechanism of aging?": "molecular_mechanism_of_aging",
    "Q3: Does the paper/theory suggest a longevity intervention to test?": "longevity_intervention_to_test",
    "Q4: Does the paper/theory claim that aging cannot be reversed? (Yes / No)": "aging_cannot_be_reversed",
    "Q5: Does the paper/theory suggest a biomarker that explains differences in maximum lifespan between species? (Yes / No)": "cross_species_longevity_biomarker",
    "Q6: Does the paper/theory explain why the naked mole rat can live 40+ years despite its small size?": "naked_mole_rat_lifespan_explanation",
    "Q7: Does the paper/theory explain why birds live much longer than mammals on average?": "birds_lifespan_explanation",
    "Q8: Does the paper/theory explain why large animals live longer than small ones?": "large_animals_lifespan_explanation",
    "Q9: Does the paper/theory explain why calorie restriction increases the lifespan of vertebrates?": "calorie_restriction_lifespan_explanation"
}


def normalize_answer(answer: str) -> str:
    """Normalize answer to Yes/No for comparison."""
    answer = answer.strip()
    
    # Handle our multi-option answers
    if "Yes, quantitatively shown" in answer or "Yes, but not shown" in answer:
        return "Yes"
    
    # Handle "Not available" as incorrect (should have been Yes or No)
    if answer == "Not available":
        return "Not available"
    
    return answer


def load_validation_set(validation_file: str) -> Dict[str, Dict[str, str]]:
    """Load validation set from JSON file."""
    with open(validation_file, 'r') as f:
        data = json.load(f)
    
    # Convert to dict keyed by DOI
    validation_dict = {}
    for entry in data:
        doi = entry['doi']
        validation_dict[doi] = {}
        for q_text, answer in entry.items():
            if q_text == 'doi':
                continue
            if q_text in QUESTION_MAPPING:
                q_name = QUESTION_MAPPING[q_text]
                validation_dict[doi][q_name] = answer
    
    return validation_dict


def load_llm_answers(results_db: str, dois: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Load LLM answers from database for given DOIs."""
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    
    llm_answers = {}
    for doi in dois:
        cur.execute(
            """
            SELECT question_name, answer, confidence_score, reasoning
            FROM paper_answers
            WHERE doi = ?
            """,
            (doi,)
        )
        rows = cur.fetchall()
        if rows:
            llm_answers[doi] = {}
            for q_name, answer, confidence, reasoning in rows:
                llm_answers[doi][q_name] = {
                    'answer': answer,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
    
    conn.close()
    return llm_answers


def compute_metrics(validation_set: Dict, llm_answers: Dict) -> Dict:
    """Compute accuracy metrics."""
    
    # Per-question metrics
    question_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'details': []})
    
    # Overall metrics
    total_correct = 0
    total_questions = 0
    missing_papers = []
    missing_answers = []
    
    # Per-paper results
    paper_results = []
    
    for doi, validation_answers in validation_set.items():
        if doi not in llm_answers:
            missing_papers.append(doi)
            continue
        
        paper_correct = 0
        paper_total = 0
        paper_details = []
        
        for q_name, true_answer in validation_answers.items():
            if q_name not in llm_answers[doi]:
                missing_answers.append((doi, q_name))
                continue
            
            llm_data = llm_answers[doi][q_name]
            llm_answer = llm_data['answer']
            confidence = llm_data['confidence']
            
            # Normalize answers
            true_norm = normalize_answer(true_answer)
            llm_norm = normalize_answer(llm_answer)
            
            is_correct = (true_norm == llm_norm)
            
            # Update metrics
            question_metrics[q_name]['total'] += 1
            total_questions += 1
            paper_total += 1
            
            if is_correct:
                question_metrics[q_name]['correct'] += 1
                total_correct += 1
                paper_correct += 1
            
            # Store details
            detail = {
                'question': q_name,
                'true_answer': true_answer,
                'llm_answer': llm_answer,
                'correct': is_correct,
                'confidence': confidence
            }
            question_metrics[q_name]['details'].append(detail)
            paper_details.append(detail)
        
        paper_results.append({
            'doi': doi,
            'correct': paper_correct,
            'total': paper_total,
            'accuracy': paper_correct / paper_total if paper_total > 0 else 0,
            'details': paper_details
        })
    
    # Compute per-question accuracy
    for q_name in question_metrics:
        metrics = question_metrics[q_name]
        metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
    
    return {
        'overall_accuracy': total_correct / total_questions if total_questions > 0 else 0,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'question_metrics': dict(question_metrics),
        'paper_results': paper_results,
        'missing_papers': missing_papers,
        'missing_answers': missing_answers
    }


def print_results(metrics: Dict, output_file: str = None):
    """Print evaluation results."""
    
    print("\n" + "="*70)
    print("QUESTION ANSWERING EVALUATION RESULTS")
    print("="*70)
    
    # Overall metrics
    print(f"\n📊 OVERALL METRICS")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['total_correct']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    
    # Per-question metrics
    print(f"\n📋 PER-QUESTION ACCURACY")
    print("-" * 70)
    question_metrics = metrics['question_metrics']
    for q_name, q_metrics in sorted(question_metrics.items()):
        accuracy = q_metrics['accuracy']
        correct = q_metrics['correct']
        total = q_metrics['total']
        print(f"{q_name:45s} {accuracy:6.1%} ({correct}/{total})")
    
    # Per-paper results
    print(f"\n📄 PER-PAPER RESULTS")
    print("-" * 70)
    for paper in sorted(metrics['paper_results'], key=lambda x: x['accuracy'], reverse=True):
        doi = paper['doi']
        accuracy = paper['accuracy']
        correct = paper['correct']
        total = paper['total']
        print(f"{doi:40s} {accuracy:6.1%} ({correct}/{total})")
    
    # Detailed mismatches
    print(f"\n❌ INCORRECT ANSWERS")
    print("-" * 70)
    for paper in metrics['paper_results']:
        doi = paper['doi']
        for detail in paper['details']:
            if not detail['correct']:
                print(f"\nDOI: {doi}")
                print(f"Question: {detail['question']}")
                print(f"  True answer: {detail['true_answer']}")
                print(f"  LLM answer:  {detail['llm_answer']} (confidence: {detail['confidence']:.2f})")
    
    # Missing data
    if metrics['missing_papers']:
        print(f"\n⚠️  MISSING PAPERS (not in LLM results)")
        for doi in metrics['missing_papers']:
            print(f"  - {doi}")
    
    if metrics['missing_answers']:
        print(f"\n⚠️  MISSING ANSWERS")
        for doi, q_name in metrics['missing_answers']:
            print(f"  - {doi}: {q_name}")
    
    print("\n" + "="*70)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate QA results against validation set')
    parser.add_argument('--validation-file', default='data/qa_validation_set_extended.json',
                        help='Path to validation set JSON file')
    parser.add_argument('--results-db', default='./qa_results/qa_results.db',
                        help='Path to results database')
    parser.add_argument('--output-file', default='qa_evaluation_results_extended.json',
                        help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Load validation set
    print(f"📖 Loading validation set from: {args.validation_file}")
    validation_set = load_validation_set(args.validation_file)
    print(f"✓ Loaded {len(validation_set)} papers with validation answers")
    
    # Get DOIs from validation set
    dois = list(validation_set.keys())
    
    # Load LLM answers
    print(f"\n🔍 Loading LLM answers from: {args.results_db}")
    llm_answers = load_llm_answers(args.results_db, dois)
    print(f"✓ Loaded answers for {len(llm_answers)} papers")
    
    # Compute metrics
    print(f"\n⚙️  Computing metrics...")
    metrics = compute_metrics(validation_set, llm_answers)
    
    # Print results
    print_results(metrics, args.output_file)


if __name__ == "__main__":
    main()
