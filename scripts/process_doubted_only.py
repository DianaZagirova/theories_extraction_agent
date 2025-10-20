"""
Extract doubted papers only and add them to theories.db.
"""
import sqlite3
import sys
from pathlib import Path

EVAL_DB = '/home/diana.z/hack/llm_judge/data/evaluations.db'
OUTPUT_FILE = 'doubted_dois.txt'

# Get all doubted DOIs
conn = sqlite3.connect(EVAL_DB)
cursor = conn.cursor()

cursor.execute("""
    SELECT DISTINCT doi
    FROM paper_evaluations
    WHERE result = 'doubted'
      AND doi IS NOT NULL 
      AND doi != ''
    ORDER BY doi
""")

dois = [row[0] for row in cursor.fetchall()]
conn.close()

# Write to file
with open(OUTPUT_FILE, 'w') as f:
    for doi in dois:
        f.write(f"{doi}\n")

print(f"✓ Wrote {len(dois)} doubted DOIs to {OUTPUT_FILE}")
print(f"\nTo process them, run:")
print(f"python scripts/extract_theories_per_paper.py \\")
print(f"  --evaluations-db {EVAL_DB} \\")
print(f"  --papers-db /home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db \\")
print(f"  --results-db theories.db \\")
print(f"  --max-workers 4")
print(f"\nOr to test first:")
print(f"python scripts/extract_theories_per_paper.py \\")
print(f"  --test \\")
print(f"  --test-dois-file {OUTPUT_FILE} \\")
print(f"  --limit 10")
