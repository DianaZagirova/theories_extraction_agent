#!/usr/bin/env python3
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Any:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        sys.exit(1)

def get_dois_with_low_focus(stage0_data: Dict, max_focus_threshold: int = 6) -> Set[str]:
    """
    Identify DOIs with maximum focus score below threshold.
    
    Args:
        stage0_data: Loaded stage0 filtered theories data
        max_focus_threshold: Maximum focus score threshold for exclusion
        
    Returns:
        Set of DOIs to exclude
    """
    # Calculate max focus for each DOI
    doi_max_focus = defaultdict(float)
    for theory in stage0_data.get('theories', []):
        doi = theory.get('doi')
        focus = float(theory.get('paper_focus', 0))
        if doi:
            doi_max_focus[doi] = max(doi_max_focus[doi], focus)
    
    # Get DOIs with max focus <= threshold
    
    print(f"Number of DOIs with max focus <= {max_focus_threshold}: {len(doi_max_focus)}")
    return {doi for doi, focus in doi_max_focus.items() if focus <= max_focus_threshold}

    
def process_theory_data(
    stage7_consolidated: Dict,
    stage0_filtered: Dict,
    excluded_dois: Set[str] = None
) -> Dict[str, List[str]]:
    """
    Process theory data and map final names to DOIs.
    
    Args:
        stage7_consolidated: Stage 7 consolidated theories data
        stage0_filtered: Stage 0 filtered theories data
        excluded_dois: Set of DOIs to exclude from results
        
    Returns:
        Dictionary mapping final theory names to lists of DOIs
    """
    if excluded_dois is None:
        excluded_dois = set()
    
    # Build theory_id to DOI mapping
    theory_id_to_doi = {}
    for theory in stage0_filtered.get('theories', []):
        theory_id_to_doi[theory['theory_id']] = {
            "doi": theory['doi'],
            "paper_focus": theory.get('paper_focus', '')
        }
    
    # Process final theories
    final_name_to_dois: Dict[str, List[str]] = {}
    
    for item in stage7_consolidated.get('final_name_summary', []):
        final_name = item.get('final_name', '')
        if not final_name:
            continue
            
        theory_ids = item.get('theory_ids', [])
        doi_set: Set[str] = set()
        
        # Add DOIs from theory IDs (excluding low-focus DOIs)
        for tid in theory_ids:
            if tid in theory_id_to_doi:
                doi = theory_id_to_doi[tid]['doi']
                if doi and doi not in excluded_dois:
                    doi_set.add(doi)
        
        if doi_set:
            final_name_to_dois[final_name] = sorted(list(doi_set))
    
    return final_name_to_dois

def save_output(data: Dict, output_path: Path) -> None:
    """Save output to JSON file with pretty printing."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved output to {output_path}")
    except IOError as e:
        logger.error(f"Error saving output to {output_path}: {str(e)}")
        sys.exit(1)


def deduplicate_dois_across_groups(data: Dict) -> Dict:
    """
    Ensure each DOI appears in only one theory group, preferring the smallest group.
    
    Args:
        data: Dictionary containing 'theories' list with 'name' and 'dois' keys
        
    Returns:
        Updated data dictionary with deduplicated DOIs
    """
    # Make a deep copy to avoid modifying the input
    data = json.loads(json.dumps(data))
    
    # 1. Build mapping from DOI to all group names
    doi_to_groups = defaultdict(list)
    for group in data["theories"]:
        group_name = group["name"]
        for doi in group["dois"]:
            doi_to_groups[doi].append(group_name)
    
    # 2. Compute size of each initial group
    group_to_size = {group["name"]: len(group["dois"]) for group in data["theories"]}
    
    # 3. Assign each DOI to the smallest group (by current size)
    doi_to_preferred_group = {}
    for doi, groups in doi_to_groups.items():
        if len(groups) == 1:
            doi_to_preferred_group[doi] = groups[0]
        else:
            # Smallest cluster wins; ties go to first encountered
            smallest = min(groups, key=lambda g: group_to_size[g])
            doi_to_preferred_group[doi] = smallest
    
    # 4. Filter out DOIs not belonging to their "owner" group
    for group in data["theories"]:
        group_name = group["name"]
        group["dois"] = [doi for doi in group["dois"] 
                         if doi_to_preferred_group[doi] == group_name]
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Process final theory outputs and map to DOIs')
    parser.add_argument('--stage7-input', type=Path, default='output/stage7_consolidated_final_theories.json',
                       help='Path to stage7 consolidated theories JSON')
    parser.add_argument('--stage0-input', type=Path, default='output/stage0_filtered_theories.json',
                       help='Path to stage0 filtered theories JSON')
    parser.add_argument('--output', type=Path, default='output/final_output/final_theory_to_dois_mapping.json',
                       help='Output JSON file path')
    parser.add_argument('--focus-threshold', type=float, default=6.0,
                       help='Exclude DOIs with max focus score <= this value')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting processing of final theory outputs...")
    
    # Load input files
    logger.info("Loading input files...")
    stage7_data = load_json_file(args.stage7_input)
    stage0_data = load_json_file(args.stage0_input)
    
    # Identify DOIs to exclude based on focus threshold
    logger.info(f"Identifying DOIs with max focus <= {args.focus_threshold}...")
    excluded_dois = get_dois_with_low_focus(stage0_data, args.focus_threshold)
    logger.info(f"Excluding {len(excluded_dois)} DOIs with low focus")
    
    # Process data
    logger.info("Processing theory data...")
    final_mapping = process_theory_data(
        stage7_consolidated=stage7_data,
        stage0_filtered=stage0_data,
        excluded_dois=excluded_dois
    )
    
    # Prepare initial output
    output_data = {
        'metadata': {
            'source_stage7': str(args.stage7_input),
            'source_stage0': str(args.stage0_input),
            'focus_threshold': args.focus_threshold,
            'excluded_dois_count': len(excluded_dois),
            'generated_at': str(datetime.datetime.utcnow().isoformat())
        },
        'theories': [{'name': name, 'dois': dois} 
                    for name, dois in final_mapping.items()]
    }
    
    # Deduplicate DOIs across groups
    logger.info("Deduplicating DOIs across theory groups...")
    output_data = deduplicate_dois_across_groups(output_data)
    
    # Update metadata with post-deduplication counts
    output_data['metadata'].update({
        'total_theories': len(output_data['theories']),
        'total_dois': sum(len(group['dois']) for group in output_data['theories']),
        'doi_deduplication_applied': True
    })
    
    # Save output
    save_output(output_data, args.output)
    logger.info("Processing completed successfully!")

if __name__ == '__main__':
    import datetime
    main()