#!/usr/bin/env python3
"""
Example script to run the question answering pipeline.

Usage:
    python run_question_answering.py --limit 10
"""
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from scripts.answer_questions_per_paper import QuestionAnsweringPipeline


def main():
    # Configuration
    EVALUATIONS_DB = "path/to/evaluations.db"  # Update this path
    PAPERS_DB = "path/to/papers.db"  # Update this path
    QUESTIONS_FILE = "/home/diana.z/hack/rag_agent/data/questions_part2.json"
    OUTPUT_FILE = "paper_answers.json"
    RESULTS_DB = "paper_answers.db"
    
    # Check if databases exist
    if not Path(EVALUATIONS_DB).exists():
        print(f"❌ Evaluations database not found: {EVALUATIONS_DB}")
        print("Please update EVALUATIONS_DB path in this script")
        return
    
    if not Path(PAPERS_DB).exists():
        print(f"❌ Papers database not found: {PAPERS_DB}")
        print("Please update PAPERS_DB path in this script")
        return
    
    if not Path(QUESTIONS_FILE).exists():
        print(f"❌ Questions file not found: {QUESTIONS_FILE}")
        return
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("INITIALIZING QUESTION ANSWERING PIPELINE")
    print("="*70)
    
    pipeline = QuestionAnsweringPipeline(QUESTIONS_FILE)
    
    # Run pipeline
    # For testing, use limit=10 to process only 10 papers
    # Remove limit or set to None to process all papers
    pipeline.run_pipeline(
        evaluations_db=EVALUATIONS_DB,
        papers_db=PAPERS_DB,
        output_file=OUTPUT_FILE,
        results_db=RESULTS_DB,
        limit=10,  # Process only 10 papers for testing
        resume_from_db=True,  # Skip already processed papers
        store_processed_text=True  # Store processed text in DB
    )
    
    print("\n✓ Pipeline completed successfully!")
    print(f"Results saved to:")
    print(f"  - JSON: {OUTPUT_FILE}")
    print(f"  - Database: {RESULTS_DB}")


if __name__ == "__main__":
    main()
