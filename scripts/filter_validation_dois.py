#!/usr/bin/env python3
"""
Filter out specific DOIs from the validation set.

Usage:
    python filter_validation_dois.py --input-dir data/dois_processing_check/validation_set_qa --output-dir data/dois_processing_check/validation_set_qa_filtered
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Set, List

# DOIs to remove (lowercase for case-insensitive comparison)
DOIS_TO_REMOVE = {
    '10.1038/287141a0',
    '10.1038/361724a0',
    '10.1093/hmg/ddu541',
    '10.1016/j.pnpbp.2019.109854',
    'http://revistamedica.imss.gob.mx/editorial/index.php/revista_medica/article/view/825/3002',
    '46/869',
    '48/5/677',
    '76/1/111',
    'BSP/CAS/E-Pub/000013',
    'clica13015158',
    'file/5703/FB2013A0030.pdf',
    'https://www.soa.org/Library/Monographs/Life/Living-To-100/2017/table-of-contents.aspx'
}

def normalize_doi(doi: str) -> str:
    """Normalize DOI for comparison (lowercase and strip)."""
    return doi.lower().strip()

def get_doi_from_filename(filename: str) -> str:
    """Extract DOI from filename (without .json extension)."""
    return filename.rsplit('.', 1)[0]

def filter_dois(input_dir: str, output_dir: str) -> None:
    """
    Filter out specified DOIs from the validation set.
    
    Args:
        input_dir: Directory containing the validation set JSON files
        output_dir: Directory to save the filtered validation set
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Normalize DOIs to remove
    dois_to_remove = {normalize_doi(doi) for doi in DOIS_TO_REMOVE}
    
    # Process each file in the input directory
    json_files = list(input_path.glob('*.json'))
    removed_count = 0
    
    for json_file in json_files:
        file_doi = get_doi_from_filename(json_file.name)
        normalized_doi = normalize_doi(file_doi)
        
        # Skip files with DOIs in the removal list
        if normalized_doi in dois_to_remove:
            print(f"Removing: {file_doi}")
            removed_count += 1
            continue
            
        # Copy file to output directory
        shutil.copy2(json_file, output_path / json_file.name)
    
    print(f"\nRemoved {removed_count} DOIs from the validation set")
    print(f"Kept {len(json_files) - removed_count} files in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Filter out specific DOIs from the validation set')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing validation set JSON files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for filtered validation set')
    
    args = parser.parse_args()
    filter_dois(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
