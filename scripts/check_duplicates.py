import json

with open('output/stage1_fuzzy_matched.json', 'r') as f:
    stage1 = json.load(f)

with open('output/stage1_5_llm_mapped.json', 'r') as f:
    stage1_5 = json.load(f)

unmatched = {t['theory_id']: t.get('name') or t.get('original_name') 
             for t in stage1.get('unmatched_theories', [])}

mapped = stage1_5.get('mapped_theories', [])

mismatches = 0
for theory in mapped[:20]:
    theory_id = theory['theory_id']
    original_input = unmatched.get(theory_id, 'NOT FOUND')
    llm_output = theory['original_name']
    
    if original_input != llm_output:
        mismatches += 1
        print(f'{theory_id}:')
        print(f'  Input:  {original_input}')
        print(f'  Output: {llm_output}')
        print()

print(f'Total mismatches in first 20: {mismatches}/20')
