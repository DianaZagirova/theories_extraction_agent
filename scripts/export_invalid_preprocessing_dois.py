#!/usr/bin/env python3
"""
Export DOIs of papers that had full text but became invalid after preprocessing.
These papers should be retried with different text extraction methods.

Usage:
    python export_invalid_preprocessing_dois.py --results-db theories.db --output invalid_dois.txt
    python export_invalid_preprocessing_dois.py --results-db theories.db --format json --output invalid_papers.json
"""
import sqlite3
import argparse
import json
from pathlib import Path
from datetime import datetime


def export_invalid_dois(results_db: str, output_file: str, format: str = 'txt'):
    """
    Export DOIs from invalid_after_preprocessing table.
    
    Args:
        results_db: Path to theories.db
        output_file: Output file path
        format: 'txt' for plain DOI list, 'json' for detailed info, 'csv' for spreadsheet
    """
    if not Path(results_db).exists():
        print(f"❌ Database not found: {results_db}")
        return
    
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    
    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='invalid_after_preprocessing'")
    if not cur.fetchone():
        print(f"❌ Table 'invalid_after_preprocessing' not found in {results_db}")
        print("   This table is created when running extract_theories_per_paper.py")
        conn.close()
        return
    
    # Fetch all invalid papers
    cur.execute("""
        SELECT doi, pmid, title, had_full_text, had_sections, preprocessing_issue, timestamp
        FROM invalid_after_preprocessing
        ORDER BY timestamp DESC
    """)
    
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        print("✓ No invalid papers found in database")
        return
    
    print(f"📊 Found {len(rows)} papers with invalid preprocessing")
    
    # Export based on format
    if format == 'txt':
        # Plain text: one DOI per line
        with open(output_file, 'w') as f:            
            for row in rows:
                f.write(f"{row[0]}\n")
        print(f"✓ Exported {len(rows)} DOIs to: {output_file}")
    
    elif format == 'json':
        # JSON: detailed information
        papers = []
        for row in rows:
            papers.append({
                'doi': row[0],
                'pmid': row[1],
                'title': row[2],
                'had_full_text': bool(row[3]),
                'had_sections': bool(row[4]),
                'preprocessing_issue': row[5],
                'timestamp': row[6]
            })
        
        output_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_papers': len(papers),
                'description': 'Papers with full text that became invalid after preprocessing'
            },
            'papers': papers
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Exported {len(rows)} papers to: {output_file}")
    
    elif format == 'csv':
        # CSV: spreadsheet format
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['DOI', 'PMID', 'Title', 'Had_Full_Text', 'Had_Sections', 'Issue', 'Timestamp'])
            for row in rows:
                writer.writerow(row)
        print(f"✓ Exported {len(rows)} papers to: {output_file}")
    
    # Print summary statistics
    had_full_text_count = sum(1 for row in rows if row[3])
    had_sections_count = sum(1 for row in rows if row[4])
    had_both = sum(1 for row in rows if row[3] and row[4])
    
    print(f"\n📈 Statistics:")
    print(f"   Papers with full_text field: {had_full_text_count}")
    print(f"   Papers with full_text_sections: {had_sections_count}")
    print(f"   Papers with both: {had_both}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review these DOIs to understand why preprocessing failed")
    print(f"   2. Try alternative text extraction methods for these papers")
    print(f"   3. Update papers.db with better full text data")


def main():
    parser = argparse.ArgumentParser(
        description="Export DOIs of papers with invalid preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-db',
        type=str,
        default='theories.db',
        help='Path to results database (default: theories.db)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='invalid_preprocessing_dois.txt',
        help='Output file path'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'json', 'csv'],
        default='txt',
        help='Output format: txt (DOI list), json (detailed), csv (spreadsheet)'
    )
    
    args = parser.parse_args()
    export_invalid_dois(args.results_db, args.output, args.format)


if __name__ == "__main__":
    main()
