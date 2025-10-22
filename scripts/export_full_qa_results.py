"""Export the entire QA results database to JSON.

Usage:
    python scripts/export_full_qa_results.py \
        --results-db qa_results/qa_results.db \
        --output-file qa_results_export.json [--limit 1000]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def fetch_all_results(conn: sqlite3.Connection, limit: int | None) -> List[Dict[str, Any]]:
    cur = conn.cursor()

    # Fetch ordered list of DOIs with metadata
    query = """
        SELECT doi, pmid, title, validation_result, confidence_score,
               processed_text_length, used_full_text, timestamp
        FROM paper_metadata
        ORDER BY timestamp
    """
    if limit:
        query += " LIMIT ?"
        cur.execute(query, (limit,))
    else:
        cur.execute(query)

    papers = []
    for row in cur.fetchall():
        (
            doi,
            pmid,
            title,
            validation_result,
            confidence_score,
            processed_text_length,
            used_full_text,
            timestamp,
        ) = row

        # Fetch all answers for the DOI
        cur.execute(
            """
                SELECT question_name, question_text, answer, original_answer,
                       confidence_score, reasoning
                FROM paper_answers
                WHERE doi = ?
                ORDER BY question_name
            """,
            (doi,),
        )

        answers_rows = cur.fetchall()
        answers: Dict[str, Dict[str, Any]] = {}
        for (
            question_name,
            question_text,
            answer,
            original_answer,
            confidence,
            reasoning,
        ) in answers_rows:
            answer_entry: Dict[str, Any] = {
                "question": question_text,
                "answer": answer,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "reasoning": reasoning,
            }
            if original_answer:
                answer_entry["original_answer"] = original_answer
            answers[question_name] = answer_entry

        papers.append(
            {
                "doi": doi,
                "pmid": pmid,
                "title": title,
                "validation_result": validation_result,
                "confidence_score": confidence_score,
                "processed_text_length": processed_text_length,
                "used_full_text": bool(used_full_text),
                "timestamp": timestamp,
                "answers": answers,
            }
        )

    return papers


def export_results(results_db: Path, output_file: Path, limit: int | None) -> Dict[str, Any]:
    if not results_db.exists():
        raise FileNotFoundError(f"Results database not found: {results_db}")

    print(f"📖 Reading QA results from: {results_db}")
    conn = sqlite3.connect(results_db)
    try:
        papers = fetch_all_results(conn, limit)
    finally:
        conn.close()

    if not papers:
        print("⚠️  No papers found in database")
        return {"total_papers": 0, "results": []}

    # Serialize to JSON
    payload = {"total_papers": len(papers), "results": papers}
    with output_file.open("w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"✓ Exported {len(papers)} papers to {output_file}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full QA results database to JSON")
    parser.add_argument(
        "--results-db",
        default="./qa_results/qa_results.db",
        help="Path to qa_results SQLite database",
    )
    parser.add_argument(
        "--output-file",
        default="./qa_results/qa_results_export.json",
        help="Destination JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of papers to export (ordered by timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_db = Path(args.results_db)
    output_file = Path(args.output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    export_results(results_db, output_file, args.limit)


if __name__ == "__main__":
    main()
