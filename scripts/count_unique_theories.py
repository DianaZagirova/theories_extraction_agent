#!/usr/bin/env python3
import json
from pathlib import Path

def count_unique_theories(json_path: str) -> int:
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    # Extract all unique theory names
    unique_theories = set()
    for cluster in clusters.values():
        for member in cluster['members']:
            unique_theories.add(member['theory_name'])
    
    return len(unique_theories)

if __name__ == "__main__":
    json_path = Path(__file__).parent.parent / 'data' / 'stage7' / 'clusters_from_stage6_names.json'
    count = count_unique_theories(str(json_path))
    print(f"Number of unique theories: {count}")
