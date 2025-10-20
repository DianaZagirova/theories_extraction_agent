import sqlite3
import json
import sys

def export_last_100_entries(db_path, table_name, output_json):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # To get dict-like rows
    cursor = conn.cursor()

    query = f"""
    SELECT *
    FROM {table_name}
    LIMIT 100
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert rows to list of dicts
    data = [dict(row) for row in rows]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(data)} entries to {output_json}")

    conn.close()

if __name__ == "__main__":
    db_path = '/home/diana.z/hack/theories_extraction_agent/theories.db'
    table_name = 'theories'
    output_json = 'theories_201025.json'

    export_last_100_entries(db_path, table_name, output_json)