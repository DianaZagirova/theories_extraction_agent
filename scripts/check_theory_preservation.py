#!/usr/bin/env python3
"""Check if theories are preserved in stage6 consolidated output."""

import json

# Load all data
with open('output/stage6_consolidated_final_theories.json', 'r') as f:
    stage6_data = json.load(f)

with open('output/stage5_consolidated_final_theories.json', 'r') as f:
    stage5_data = json.load(f)

# Get all theory IDs from stage5
stage5_all_ids = set()
for cluster in stage5_data['final_name_summary']:
    stage5_all_ids.update(cluster['theory_ids'])

# Get all theory IDs from stage6 consolidated
stage6_all_ids = set()
for cluster in stage6_data['final_name_summary']:
    stage6_all_ids.update(cluster['theory_ids'])

print('='*80)
print('THEORY PRESERVATION CHECK')
print('='*80)
print(f'Stage 5 total theories: {len(stage5_all_ids)}')
print(f'Stage 6 consolidated total theories: {len(stage6_all_ids)}')
print(f'Missing in stage6: {len(stage5_all_ids - stage6_all_ids)}')
print(f'Extra in stage6: {len(stage6_all_ids - stage5_all_ids)}')
print()

if stage5_all_ids == stage6_all_ids:
    print('✅ ALL THEORIES PRESERVED!')
else:
    print('❌ THEORIES MISSING OR EXTRA')
    
    missing = stage5_all_ids - stage6_all_ids
    extra = stage6_all_ids - stage5_all_ids
    
    if missing:
        print(f'\n⚠️  {len(missing)} theories missing from stage6:')
        for tid in sorted(list(missing)[:10]):
            print(f'  - {tid}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')
    
    if extra:
        print(f'\n⚠️  {len(extra)} extra theories in stage6:')
        for tid in sorted(list(extra)[:10]):
            print(f'  - {tid}')
        if len(extra) > 10:
            print(f'  ... and {len(extra) - 10} more')

print()
print('='*80)
print('CELLULAR SENESCENCE THEORY DETAILED CHECK')
print('='*80)

# From stage5
stage5_cellular = None
for cluster in stage5_data['final_name_summary']:
    if cluster['final_name'] == 'Cellular Senescence Theory':
        stage5_cellular = cluster
        break

# From stage6 - check all clusters with "Cellular Senescence" in name
stage6_cellular_clusters = []
for cluster in stage6_data['final_name_summary']:
    if 'Cellular Senescence' in cluster['final_name']:
        stage6_cellular_clusters.append(cluster)

if stage5_cellular:
    stage5_ids = set(stage5_cellular['theory_ids'])
    print(f'\nStage 5 "Cellular Senescence Theory": {len(stage5_ids)} theories')
    
    # Check all stage6 clusters with this name
    print(f'\nStage 6 clusters with "Cellular Senescence":')
    total_stage6_ids = set()
    for cluster in stage6_cellular_clusters:
        cluster_ids = set(cluster['theory_ids'])
        total_stage6_ids.update(cluster_ids)
        print(f'  - {cluster["final_name"]}: {len(cluster_ids)} theories')
    
    print(f'\nTotal in stage6 Cellular Senescence clusters: {len(total_stage6_ids)}')
    print(f'Missing: {len(stage5_ids - total_stage6_ids)}')
    print(f'Extra: {len(total_stage6_ids - stage5_ids)}')
    
    # Check if missing theories went to OTHER clusters
    missing_from_cellular = stage5_ids - total_stage6_ids
    if missing_from_cellular:
        print(f'\n🔍 Checking if {len(missing_from_cellular)} missing theories are in other stage6 clusters...')
        
        found_elsewhere = {}
        for cluster in stage6_data['final_name_summary']:
            if 'Cellular Senescence' not in cluster['final_name']:
                cluster_ids = set(cluster['theory_ids'])
                found = missing_from_cellular & cluster_ids
                if found:
                    found_elsewhere[cluster['final_name']] = found
        
        if found_elsewhere:
            print(f'\n✅ Found {sum(len(v) for v in found_elsewhere.values())} theories in other clusters:')
            for cluster_name, theory_ids in sorted(found_elsewhere.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                print(f'  - {cluster_name}: {len(theory_ids)} theories')
        
        truly_missing = missing_from_cellular - set().union(*found_elsewhere.values()) if found_elsewhere else missing_from_cellular
        if truly_missing:
            print(f'\n❌ {len(truly_missing)} theories TRULY MISSING (not in any stage6 cluster):')
            for tid in sorted(list(truly_missing)[:10]):
                print(f'  - {tid}')
            if len(truly_missing) > 10:
                print(f'  ... and {len(truly_missing) - 10} more')
        else:
            print(f'\n✅ All theories accounted for (just redistributed to other clusters)')
