"""
Theory Tracker: Comprehensive tracking of theory IDs across all normalization stages.

Tracks each theory from Stage 0 through Stage 3, collecting metadata at each step:
- Stage 0: Quality filtering
- Stage 1: Fuzzy matching to ontology
- Stage 1.5: LLM-assisted mapping
- Stage 2: Group normalization
- Stage 3: Iterative refinement

Output: Complete lineage and status for each theory ID
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import pandas as pd
import re

class TheoryTracker:
    """Track theory IDs across all normalization stages."""
    
    def __init__(self, ontology_path: str = 'ontology/groups_ontology_alliases.json'):
        """Initialize tracker."""
        self.theory_lineage = {}  # theory_id -> stage metadata
        self.stage_stats = {}
        self.ontology_path = ontology_path
        self.ontology_theories = self._load_ontology()
        
    def _load_json(self, path: str) -> dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_ontology(self) -> Set[str]:
        """Load theory names from ontology (main names only, not aliases)."""
        print(f"📂 Loading ontology from {self.ontology_path}...")
        
        try:
            with open(self.ontology_path, 'r') as f:
                data = json.load(f)
            
            theories = set()
            
            def extract_theories(obj):
                """Recursively extract theory names from nested structure."""
                if isinstance(obj, dict):
                    # Check if this is a theory entry (has 'name' field)
                    if 'name' in obj:
                        theories.add(obj['name'])
                        # NOTE: Not adding aliases - only main names count as "completed"
                    # Recurse into nested dicts
                    for value in obj.values():
                        extract_theories(value)
                elif isinstance(obj, list):
                    # Recurse into lists
                    for item in obj:
                        extract_theories(item)
            
            extract_theories(data)
            
            print(f"  ✓ Loaded {len(theories)} ontology theories (main names only)")
            return theories
        except FileNotFoundError:
            print(f"  ⚠️  Ontology file not found, continuing without ontology tracking")
            return set()
    
    def track_stage0(self, stage0_path: str = 'output/stage0_filtered_theories.json'):
        """
        Track Stage 0: Quality filtering.
        
        Collects:
        - theory_id
        - theory_name
        - confidence_level
        - paper_id
        - source (abstract/fulltext)
        """
        print("=" * 80)
        print("TRACKING STAGE 0: Quality Filtering")
        print("=" * 80)
        
        data = self._load_json(stage0_path)
        theories = data.get('theories', [])
        
        print(f"📊 Found {len(theories)} theories in Stage 0")
        
        for theory in theories:
            theory_id = theory.get('theory_id')
            
            theory_name = theory.get('theory_name', '')
            
            self.theory_lineage[theory_id] = {
                'theory_id': theory_id,
                'original_name': theory_name,
                'paper_id': theory.get('paper_id', ''),
                'doi': theory.get('doi', ''),
                'paper_focus': theory.get('paper_focus', ''),
                "mode": theory.get('mode', ''),
      
                'source': theory.get('source', ''),
                'confidence_level': theory.get('confidence_level', ''),
                
                # Stage 0
                'stage0_status': 'passed_filter',
                'stage0_name': theory_name,
                'stage0_completed': False,
                
                # Stage 1
                'stage1_status': None,
                'stage1_name': None,
                'stage1_completed': None,
                
                # Stage 1.5
                'stage1_5_status': None,
                'stage1_5_name': None,
                'stage1_5_completed': None,
                
                # Stage 2
                'stage2_status': None,
                'stage2_name': None,
                'stage2_completed': None,
                
                # Stage 3
                'stage3_status': None,
                'stage3_name': None,
                'stage3_completed': None,
                
                # Stage 4
                'stage4_status': None,
                'stage4_name': None,
                'stage4_completed': None,
                'is_valid_by_stage4': None,
                
                # Legacy fields (for backward compatibility)
                'stage1_matched_name': None,
                'stage1_5_mapped_name': None,
                'stage2_normalized_name': None,
                'stage3_final_name': None,
                'stage3_ontology_match': None,
                'stage1_is_ontology': None,
                'stage1_5_is_ontology': None,
                'stage2_is_ontology': None,
                'stage3_is_ontology': None
            }
        
        self.stage_stats['stage0'] = {
            'total_theories': len(theories),
            'passed_filter': len(theories)
        }
        
        print(f"✓ Tracked {len(self.theory_lineage)} theories from Stage 0")
    
    def normalize_name(self, name: str) -> str:
        normalized = name.strip()
        
        # Standardize "ageing" to "aging"
        normalized = re.sub(r'\bageing\b', 'aging', normalized, flags=re.IGNORECASE)
        
        if "hallmark" not in name.lower():
            # Remove "of Aging" / "of Ageing" suffix
            normalized = re.sub(r'\s+of\s+aging\s*$', '', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\s+of\s+ageing\s*$', '', normalized, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Convert to title case for consistency
        # normalized = normalized.title()
        if name == 'SINE Retrotransposon Activation Theory':
            normalized = "Retrotransposon Activation Theory"
        
        return normalized
        
    def track_stage1(self, stage1_path: str = 'output/stage1_fuzzy_matched.json'):
        """
        Track Stage 1: Fuzzy matching.
        
        Collects:
        - Matched to ontology or not
        - Matched ontology name (if matched)
        - Match score
        """
        print("\n" + "=" * 80)
        print("TRACKING STAGE 1: Fuzzy Matching")
        print("=" * 80)
        
        data = self._load_json(stage1_path)
        
        # Track matched theories
        matched = data.get('matched_theories', [])
        unmatched = data.get('unmatched_theories', [])
        
        print(f"📊 Matched: {len(matched)}, Unmatched: {len(unmatched)}")
        
        matched_count = 0
        for theory in matched:
            theory_id = theory.get('theory_id')
            if theory_id in self.theory_lineage:
                # Check match_result field
                match_result = theory.get('match_result', {})
                is_matched = match_result.get('matched', False)
                canonical_name = match_result.get('canonical_name')
                
                if is_matched and canonical_name:
                    # Status: mapped
                    self.theory_lineage[theory_id]['stage1_status'] = 'mapped'
                    # Name: canonical_name from match_result
                    self.theory_lineage[theory_id]['stage1_name'] = canonical_name
                    # Completed: true if canonical_name is in ontology
                    is_in_ontology = canonical_name in self.ontology_theories
                    self.theory_lineage[theory_id]['stage1_completed'] = is_in_ontology
                    
                    # Legacy fields
                    self.theory_lineage[theory_id]['stage1_matched_name'] = canonical_name
                    self.theory_lineage[theory_id]['stage1_is_ontology'] = is_in_ontology
                    matched_count += 1
                else:
                    # Status: not_mapped
                    self.theory_lineage[theory_id]['stage1_status'] = 'not_mapped'
                    # Name: original_name
                    self.theory_lineage[theory_id]['stage1_name'] = theory.get('original_name', self.theory_lineage[theory_id]['original_name'])
                    # Completed: false
                    self.theory_lineage[theory_id]['stage1_completed'] = False
                
                self.theory_lineage[theory_id]['stage1_match_score'] = theory.get('match_score')
        
        unmatched_count = 0
        for theory in unmatched:
            theory_id = theory.get('theory_id')
            if theory_id in self.theory_lineage:
                # Status: not_mapped
                self.theory_lineage[theory_id]['stage1_status'] = 'not_mapped'
                # Name: original_name
                self.theory_lineage[theory_id]['stage1_name'] = theory.get('original_name', self.theory_lineage[theory_id]['original_name'])
                # Completed: false
                self.theory_lineage[theory_id]['stage1_completed'] = False
                unmatched_count += 1
        
        # Count theories per matched name and completed theories
        matched_names_count = defaultdict(int)
        completed_count = 0
        unique_names_to_process = set()
        
        for theory in self.theory_lineage.values():
            if theory.get('stage1_status') == 'mapped':
                matched_name = theory.get('stage1_matched_name')
                if matched_name:
                    matched_names_count[matched_name] += 1
                if theory.get('stage1_completed'):
                    completed_count += 1
            elif theory.get('stage1_status') == 'not_mapped':
                # These will be processed in Stage 1.5
                name = theory.get('stage1_name') or theory.get('original_name')
                if name:
                    unique_names_to_process.add(name)
        
        self.stage_stats['stage1'] = {
            'matched': matched_count,
            'unmatched': unmatched_count,
            'completed': completed_count,
            'to_process_next_stage': unmatched_count,
            'unique_names_to_process': len(unique_names_to_process),
            'matched_names_distribution': dict(matched_names_count)
        }
        
        print(f"✓ Tracked Stage 1: {matched_count} matched, {unmatched_count} unmatched")
        print(f"  → Completed (matched to ontology): {completed_count}")
        print(f"  → Mapped to {len(matched_names_count)} unique names")
        print(f"  → To process in Stage 1.5: {unmatched_count} theories ({len(unique_names_to_process)} unique names)")
    
    def track_stage1_5(self, stage1_5_path: str = 'output/stage1_5_llm_mapped.json'):
        """
        Track Stage 1.5: LLM-assisted mapping.
        
        Collects:
        - LLM mapping status
        - Mapped name (if mapped)
        """
        print("\n" + "=" * 80)
        print("TRACKING STAGE 1.5: LLM-Assisted Mapping")
        print("=" * 80)
        
        data = self._load_json(stage1_5_path)
        
        # Track mapped theories
        mapped = data.get('mapped_theories', [])
        unmapped = data.get('unmapped_theories', [])
        
        print(f"📊 Mapped: {len(mapped)}, Unmapped: {len(unmapped)}")
        
        # Create sets of theory IDs in Stage 1.5
        stage1_5_theory_ids = set()
        for theory in mapped:
            stage1_5_theory_ids.add(theory.get('theory_id'))
        for theory in unmapped:
            stage1_5_theory_ids.add(theory.get('theory_id'))
        
        mapped_count = 0
        skipped_count = 0
        
        for theory in mapped:
            theory_id = theory.get('theory_id')
            if theory_id in self.theory_lineage:
                mapped_name = theory.get('mapped_name')
                # Status: mapped
                self.theory_lineage[theory_id]['stage1_5_status'] = 'mapped'
                # Name: mapped_name
                self.theory_lineage[theory_id]['stage1_5_name'] = mapped_name
                # Completed: true if mapped_name is in ontology
                is_in_ontology = mapped_name in self.ontology_theories if mapped_name else False
                self.theory_lineage[theory_id]['stage1_5_completed'] = is_in_ontology
                
                # Legacy fields
                self.theory_lineage[theory_id]['stage1_5_mapped_name'] = mapped_name
                self.theory_lineage[theory_id]['stage1_5_is_ontology'] = is_in_ontology
                mapped_count += 1
        
        unmapped_count = 0
        for theory in unmapped:
            theory_id = theory.get('theory_id')
            if theory_id in self.theory_lineage:
                # Status: not_mapped
                self.theory_lineage[theory_id]['stage1_5_status'] = 'not_mapped'
                # Name: original_name
                self.theory_lineage[theory_id]['stage1_5_name'] = theory.get('original_name', self.theory_lineage[theory_id]['original_name'])
                # Completed: false
                self.theory_lineage[theory_id]['stage1_5_completed'] = False
                unmapped_count += 1
        
        # Mark theories NOT in Stage 1.5 as skipped (they were completed in Stage 1)
        for theory_id, theory_data in self.theory_lineage.items():
            if theory_id not in stage1_5_theory_ids:
                # Check if completed at Stage 1
                if theory_data.get('stage1_completed'):
                    # Status: skipped (completed at previous stage)
                    self.theory_lineage[theory_id]['stage1_5_status'] = 'skipped'
                    # Name: keep stage1_name
                    self.theory_lineage[theory_id]['stage1_5_name'] = theory_data['stage1_name']
                    # Completed: null
                    self.theory_lineage[theory_id]['stage1_5_completed'] = None
                    skipped_count += 1
        
        # Count theories per mapped name and completed theories
        mapped_names_count = defaultdict(int)
        completed_count = 0
        unique_names_to_process = set()
        
        for theory in self.theory_lineage.values():
            if theory.get('stage1_5_status') == 'mapped':
                mapped_name = theory.get('stage1_5_mapped_name')
                if mapped_name:
                    mapped_names_count[mapped_name] += 1
                if theory.get('stage1_5_completed'):
                    completed_count += 1
                else:
                    # Mapped but not completed - will be processed in Stage 2
                    if mapped_name:
                        unique_names_to_process.add(mapped_name)
            elif theory.get('stage1_5_status') == 'not_mapped':
                # These will be processed in Stage 2
                name = theory.get('stage1_5_name') or theory.get('original_name')
                if name:
                    unique_names_to_process.add(name)
        
        # Total completed before Stage 2 (Stage 1 + Stage 1.5)
        total_completed = skipped_count + completed_count
        # To process in Stage 2: unmapped + mapped but not completed
        mapped_not_completed = mapped_count - completed_count
        to_process_stage2 = unmapped_count + mapped_not_completed
        
        self.stage_stats['stage1_5'] = {
            'mapped': mapped_count,
            'unmapped': unmapped_count,
            'skipped': skipped_count,
            'completed': completed_count,
            'total_completed_so_far': total_completed,
            'to_process_next_stage': to_process_stage2,
            'unique_names_to_process': len(unique_names_to_process),
            'mapped_names_distribution': dict(mapped_names_count)
        }
        
        print(f"✓ Tracked Stage 1.5: {mapped_count} mapped, {unmapped_count} unmapped, {skipped_count} skipped")
        print(f"  → Completed (mapped to ontology): {completed_count}")
        print(f"  → Total completed so far: {total_completed} (Stage 1: {skipped_count} + Stage 1.5: {completed_count})")
        print(f"  → Mapped to {len(mapped_names_count)} unique names")
        print(f"  → To process in Stage 2: {to_process_stage2} theories ({len(unique_names_to_process)} unique names)")
    
    def track_stage2(self, stage2_path: str = 'output/stage2_grouped_theories.json'):
        """
        Track Stage 2: Group normalization.
        
        Collects:
        - Normalized name for each theory
        """
        print("\n" + "=" * 80)
        print("TRACKING STAGE 2: Group Normalization")
        print("=" * 80)
        
        data = self._load_json(stage2_path)
        mappings = data.get('mappings', {})
        
        print(f"📊 Found {len(mappings)} mappings in Stage 2")
        
        # Create reverse lookup: theory_name -> normalized_name
        # Need to match theory names to theory_ids
        
        normalized_count = 0
        skipped_count = 0
        warning_count = 0
        
        for theory_id, theory_data in self.theory_lineage.items():
            # Check if completed at previous stages
            if theory_data.get('stage1_5_completed') or theory_data.get('stage1_completed'):
                # Status: skipped (completed at previous stage)
                self.theory_lineage[theory_id]['stage2_status'] = 'skipped'
                # Name: keep previous stage name
                prev_name = theory_data.get('stage1_5_name') or theory_data.get('stage1_name')
                self.theory_lineage[theory_id]['stage2_name'] = prev_name
                # Completed: null
                self.theory_lineage[theory_id]['stage2_completed'] = None
                skipped_count += 1
                continue
            
            # Get the name that went into Stage 2
            # Priority: stage1_5_name > stage1_name > original_name
            input_name = theory_data.get('stage1_5_name') or theory_data.get('stage1_name') or theory_data.get('original_name')
            
            if input_name in mappings:
                normalized_name = mappings[input_name]
                # Status: mapped
                self.theory_lineage[theory_id]['stage2_status'] = 'mapped'
                # Name: normalized_name from mappings
                self.theory_lineage[theory_id]['stage2_name'] = normalized_name
                # Completed: true if normalized_name is in ontology
                is_in_ontology = normalized_name in self.ontology_theories if normalized_name else False
                self.theory_lineage[theory_id]['stage2_completed'] = is_in_ontology
                
                # Legacy fields
                self.theory_lineage[theory_id]['stage2_normalized_name'] = normalized_name
                self.theory_lineage[theory_id]['stage2_is_ontology'] = is_in_ontology
                normalized_count += 1
            else:
                # Try to find by original name
                original_name = theory_data.get('original_name')
                if original_name in mappings:
                    normalized_name = mappings[original_name]
                    # Status: mapped
                    self.theory_lineage[theory_id]['stage2_status'] = 'mapped'
                    # Name: normalized_name from mappings
                    self.theory_lineage[theory_id]['stage2_name'] = normalized_name
                    # Completed: true if normalized_name is in ontology
                    is_in_ontology = normalized_name in self.ontology_theories if normalized_name else False
                    self.theory_lineage[theory_id]['stage2_completed'] = is_in_ontology
                    
                    # Legacy fields
                    self.theory_lineage[theory_id]['stage2_normalized_name'] = normalized_name
                    self.theory_lineage[theory_id]['stage2_is_ontology'] = is_in_ontology
                    normalized_count += 1
                else:
                    # Status: warning (not completed and not in json)
                    self.theory_lineage[theory_id]['stage2_status'] = 'warning'
                    # Name: keep input_name
                    self.theory_lineage[theory_id]['stage2_name'] = input_name
                    # Completed: false
                    self.theory_lineage[theory_id]['stage2_completed'] = False
                    warning_count += 1
        
        # Count theories per normalized name and completed theories
        normalized_names_count = defaultdict(int)
        completed_count = 0
        unique_names_to_process = set()
        
        for theory in self.theory_lineage.values():
            if theory.get('stage2_status') == 'mapped':
                normalized_name = theory.get('stage2_normalized_name')
                if normalized_name:
                    normalized_names_count[normalized_name] += 1
                if theory.get('stage2_completed'):
                    completed_count += 1
                else:
                    # Not completed, will be processed in Stage 3
                    if normalized_name:
                        unique_names_to_process.add(normalized_name)
        
        # Total completed before Stage 3 (Stage 1 + Stage 1.5 + Stage 2)
        total_completed = skipped_count + completed_count
        to_process_stage3 = normalized_count - completed_count
        
        self.stage_stats['stage2'] = {
            'mapped': normalized_count,
            'skipped': skipped_count,
            'warnings': warning_count,
            'completed': completed_count,
            'total_completed_so_far': total_completed,
            'to_process_next_stage': to_process_stage3,
            'unique_names_to_process': len(unique_names_to_process),
            'normalized_names_distribution': dict(normalized_names_count)
        }
        
        print(f"✓ Tracked Stage 2: {normalized_count} mapped, {skipped_count} skipped, {warning_count} warnings")
        print(f"  → Completed (mapped to ontology): {completed_count}")
        print(f"  → Total completed so far: {total_completed}")
        print(f"  → Normalized to {len(normalized_names_count)} unique names")
        print(f"  → To process in Stage 3: {to_process_stage3} theories ({len(unique_names_to_process)} unique names)")
    
    def track_stage3(self, stage3_path: str = 'output/stage3_refined_theories.json'):
        """
        Track Stage 3: Iterative refinement.
        
        Collects:
        - Final name
        - Whether matched to ontology
        """
        print("\n" + "=" * 80)
        print("TRACKING STAGE 3: Iterative Refinement")
        print("=" * 80)
        
        try:
            data = self._load_json(stage3_path)
        except FileNotFoundError:
            print("⚠️  Stage 3 output not found. Skipping.")
            return
        
        # Stage 3 has 'mappings' dict: input_name -> refined_name
        mappings = data.get('mappings', {})
        ontology_theories_stage3 = set(data.get('ontology_theories', []))
        
        print(f"📊 Found {len(mappings)} mappings in Stage 3")
        
        matched_count = 0
        unmatched_count = 0
        skipped_count = 0
        warning_count = 0
        
        for theory_id, theory_data in self.theory_lineage.items():
            # Check if completed at previous stages
            if (theory_data.get('stage2_completed') or 
                theory_data.get('stage1_5_completed') or 
                theory_data.get('stage1_completed')):
                # Status: skipped (completed at previous stage)
                self.theory_lineage[theory_id]['stage3_status'] = 'skipped'
                # Name: keep previous stage name
                prev_name = (theory_data.get('stage2_name') or 
                           theory_data.get('stage1_5_name') or 
                           theory_data.get('stage1_name'))
                self.theory_lineage[theory_id]['stage3_name'] = prev_name
                # Completed: null
                self.theory_lineage[theory_id]['stage3_completed'] = None
                skipped_count += 1
                continue
            
            # Get the name that went into Stage 3 (from Stage 2)
            input_name = theory_data.get('stage2_name') or theory_data.get('stage2_normalized_name')
            
            if not input_name:
                # Status: warning (no input name)
                self.theory_lineage[theory_id]['stage3_status'] = 'warning'
                self.theory_lineage[theory_id]['stage3_name'] = theory_data.get('original_name')
                self.theory_lineage[theory_id]['stage3_completed'] = False
                warning_count += 1
                continue
            
            # Check if in Stage 3 mappings
            if input_name in mappings:
                refined_name = mappings[input_name]
                
                # Check if refined name is in ontology
                is_in_ontology = refined_name in self.ontology_theories
                
                if is_in_ontology:
                    # Status: matched_to_ontology
                    self.theory_lineage[theory_id]['stage3_status'] = 'matched_to_ontology'
                    self.theory_lineage[theory_id]['stage3_completed'] = True
                    matched_count += 1
                else:
                    # Status: refined_not_matched
                    self.theory_lineage[theory_id]['stage3_status'] = 'refined_not_matched'
                    self.theory_lineage[theory_id]['stage3_completed'] = False
                    unmatched_count += 1
                
                # Set name
                self.theory_lineage[theory_id]['stage3_name'] = refined_name
                
                # Legacy fields
                self.theory_lineage[theory_id]['stage3_final_name'] = refined_name
                self.theory_lineage[theory_id]['stage3_ontology_match'] = is_in_ontology
                self.theory_lineage[theory_id]['stage3_is_ontology'] = is_in_ontology
            else:
                # Status: warning (not completed and not in json)
                self.theory_lineage[theory_id]['stage3_status'] = 'warning'
                # Name: keep input_name
                self.theory_lineage[theory_id]['stage3_name'] = input_name
                # Completed: false
                self.theory_lineage[theory_id]['stage3_completed'] = False
                warning_count += 1
        
        # Count theories per final name and completed theories
        final_names_count = defaultdict(int)
        completed_count = 0
        not_matched_names = set()
        
        for theory in self.theory_lineage.values():
            final_name = theory.get('stage3_final_name')
            if final_name:
                final_names_count[final_name] += 1
            if theory.get('stage3_completed'):
                completed_count += 1
            elif theory.get('stage3_status') == 'refined_not_matched':
                # Still not matched to ontology
                if final_name:
                    not_matched_names.add(final_name)
        
        # Total completed (all stages)
        total_completed = skipped_count + completed_count
        
        self.stage_stats['stage3'] = {
            'matched': matched_count,
            'refined_not_matched': unmatched_count,
            'skipped': skipped_count,
            'warnings': warning_count,
            'completed': completed_count,
            'total_completed': total_completed,
            'unique_not_matched_names': len(not_matched_names),
            'final_names_distribution': dict(final_names_count)
        }
        
        print(f"✓ Tracked Stage 3: {matched_count} matched, {unmatched_count} refined, {skipped_count} skipped, {warning_count} warnings")
        print(f"  → Completed (matched to ontology): {completed_count}")
        print(f"  → Total completed (all stages): {total_completed}")
        print(f"  → Final {len(final_names_count)} unique names")
        print(f"  → Still not matched to ontology: {unmatched_count} theories ({len(not_matched_names)} unique names)")
    
    def track_stage4(self, stage4_path: str = 'output/stage4_validated_theories.json'):
        """
        Track Stage 4: Theory validation.
        
        Collects:
        - Validation status (is_valid_theory)
        - Final name (introduced_name or listed_name)
        - Whether matched to ontology (is_listed)
        """
        print("\n" + "=" * 80)
        print("TRACKING STAGE 4: Theory Validation")
        print("=" * 80)
        
        try:
            data = self._load_json(stage4_path)
        except FileNotFoundError:
            print("⚠️  Stage 4 output not found. Skipping.")
            return
        
        # Stage 4 has 'validations' list
        validations = data.get('validations', [])
        
        print(f"📊 Found {len(validations)} validations in Stage 4")
        
        # Create mapping: original_name (from Stage 3) -> validation
        validation_map = {}
        for validation in validations:
            original_name = validation.get('original_name')
            if original_name:
                validation_map[original_name] = validation
        
        validated_count = 0
        invalid_count = 0
        doubted_count = 0
        skipped_count = 0
        warning_count = 0
        completed_count = 0
        
        for theory_id, theory_data in self.theory_lineage.items():
            # Check if completed at previous stages
            if (theory_data.get('stage3_completed') or
                theory_data.get('stage2_completed') or 
                theory_data.get('stage1_5_completed') or 
                theory_data.get('stage1_completed')):
                # Status: skipped (completed at previous stage)
                self.theory_lineage[theory_id]['stage4_status'] = 'skipped'
                # Name: keep previous stage name
                prev_name = (theory_data.get('stage3_name') or
                           theory_data.get('stage2_name') or 
                           theory_data.get('stage1_5_name') or 
                           theory_data.get('stage1_name'))
                self.theory_lineage[theory_id]['stage4_name'] = prev_name
                # Completed: null
                self.theory_lineage[theory_id]['stage4_completed'] = None
                self.theory_lineage[theory_id]['is_valid_by_stage4'] = None
                skipped_count += 1
                continue
            
            # Get the name that went into Stage 4 (from Stage 3)
            input_name = theory_data.get('stage3_name') or theory_data.get('stage3_final_name')
            
            if not input_name:
                # Status: warning (no input name)
                self.theory_lineage[theory_id]['stage4_status'] = 'warning'
                self.theory_lineage[theory_id]['stage4_name'] = theory_data.get('original_name')
                self.theory_lineage[theory_id]['stage4_completed'] = False
                self.theory_lineage[theory_id]['is_valid_by_stage4'] = None
                warning_count += 1
                continue
            
            # Check if in Stage 4 validations
            if input_name in validation_map:
                validation = validation_map[input_name]
                
                is_valid_theory = validation.get('is_valid_theory')
                is_listed = validation.get('is_listed')
                listed_name = validation.get('listed_name')
                introduced_name = validation.get('introduced_name')
                
                # Store validation status
                self.theory_lineage[theory_id]['is_valid_by_stage4'] = is_valid_theory
                
                # Determine final name
                if is_listed:
                    # Should use listed_name
                    if listed_name and listed_name.strip():
                        # Check if listed_name is in ontology
                        if listed_name in self.ontology_theories:
                            final_name = listed_name
                            self.theory_lineage[theory_id]['stage4_status'] = 'mapped_to_ontology'
                            self.theory_lineage[theory_id]['stage4_completed'] = True
                            completed_count += 1
                        else:
                            # Listed but not in ontology - warning
                            final_name = listed_name if listed_name else introduced_name
                            final_name = final_name if final_name else input_name
                            self.theory_lineage[theory_id]['stage4_status'] = 'warning'
                            self.theory_lineage[theory_id]['stage4_completed'] = False
                            warning_count += 1
                            print(f"  ⚠️  Listed name not in ontology: {listed_name}")
                    else:
                        # is_listed=true but listed_name is empty - warning, use introduced_name
                        final_name = introduced_name if introduced_name else input_name
                        self.theory_lineage[theory_id]['stage4_status'] = 'warning'
                        self.theory_lineage[theory_id]['stage4_completed'] = False
                        warning_count += 1
                        print(f"  ⚠️  is_listed=true but listed_name empty for: {input_name}")
                else:
                    # Not listed, use introduced_name
                    final_name = introduced_name if introduced_name else input_name
                    
                    if is_valid_theory == True:
                        self.theory_lineage[theory_id]['stage4_status'] = 'validated'
                        self.theory_lineage[theory_id]['stage4_completed'] = False
                        validated_count += 1
                    elif is_valid_theory == False:
                        self.theory_lineage[theory_id]['stage4_status'] = 'invalid'
                        self.theory_lineage[theory_id]['stage4_completed'] = False
                        invalid_count += 1
                    else:  # 'doubted'
                        self.theory_lineage[theory_id]['stage4_status'] = is_valid_theory
                        self.theory_lineage[theory_id]['stage4_completed'] = False
                        doubted_count += 1
                
                self.theory_lineage[theory_id]['stage4_name'] = final_name
                
            else:
                # Status: warning (not in Stage 4 validations)
                # self.theory_lineage[theory_id]['stage4_status'] = 'warning'
                # self.theory_lineage[theory_id]['stage4_name'] = input_name
                # self.theory_lineage[theory_id]['stage4_completed'] = False
                # self.theory_lineage[theory_id]['is_valid_by_stage4'] = None
                warning_count += 1
        
        # Count theories per final name
        final_names_count = defaultdict(int)
        for theory in self.theory_lineage.values():
            final_name = theory.get('stage4_name')
            if final_name:
                final_names_count[final_name] += 1
        
        # Total completed (all stages)
        total_completed = skipped_count + completed_count
        
        self.stage_stats['stage4'] = {
            'validated': validated_count,
            'invalid': invalid_count,
            'doubted': doubted_count,
            'mapped_to_ontology': completed_count,
            'skipped': skipped_count,
            'warnings': warning_count,
            'total_completed': total_completed,
            'final_names_distribution': dict(final_names_count)
        }
        
        print(f"✓ Tracked Stage 4: {validated_count} validated, {invalid_count} invalid, {doubted_count} doubted")
        print(f"  → Mapped to ontology: {completed_count}")
        print(f"  → Skipped (from previous stages): {skipped_count}")
        print(f"  → Warnings: {warning_count}")
        print(f"  → Total completed (all stages): {total_completed}")
        print(f"  → Final {len(final_names_count)} unique names")
    
    def _compute_final_fields(self):
        """Compute final_name and final_status for each theory."""
        for theory_id, theory_data in self.theory_lineage.items():
            # Compute final_name
            final_name = None
            final_status = None
            name_warning = False
            status_warning = False
            
            # Check stages in order: 1, 1.5, 2, 3, 4
            if theory_data.get('stage1_completed'):
                final_name = theory_data.get('stage1_name')
            elif theory_data.get('stage1_5_completed'):
                final_name = theory_data.get('stage1_5_name')
            elif theory_data.get('stage2_completed'):
                final_name = theory_data.get('stage2_name')
            elif theory_data.get('stage3_completed'):
                final_name = theory_data.get('stage3_name')
            elif theory_data.get('stage4_completed'):
                final_name = theory_data.get('stage4_name')
            else:
                # Not completed at any stage - check if we have stage4_name
                stage4_name = theory_data.get('stage4_name')
                if stage4_name:
                    # Has Stage 4 name (validated but not completed)
                    final_name = stage4_name
                    name_warning = False
                else:
                    # No Stage 4 name either - use last available name
                    final_name = (theory_data.get('stage3_name') or 
                                theory_data.get('stage2_name') or 
                                theory_data.get('stage1_5_name') or 
                                theory_data.get('stage1_name') or 
                                theory_data.get('original_name'))
                    name_warning = True  # No normalized name from ontology AND no Stage 4 name
            
            # Compute final_status
            # 1. If completed at any stage before stage 4: "valid"
            if (theory_data.get('stage1_completed') or 
                theory_data.get('stage1_5_completed') or 
                theory_data.get('stage2_completed') or 
                theory_data.get('stage3_completed')):
                final_status = 'valid'
            # 2. If is_valid_by_stage4 is True: "valid"
            elif theory_data.get('is_valid_by_stage4') == True:
                final_status = 'valid'
            # 3. If is_valid_by_stage4 is False or "doubted": "not valid"
            elif theory_data.get('is_valid_by_stage4') == False or theory_data.get('is_valid_by_stage4') == 'doubted':
                final_status = 'not valid'
            # 4. If not completed and is_valid_by_stage4 is None: status_warning = True
            else:
                final_status = 'unknown'
                status_warning = True  # Validation status unknown
            
            # Store final fields
            self.theory_lineage[theory_id]['final_name'] = final_name
            self.theory_lineage[theory_id]['final_name_normalized'] = self.normalize_name(final_name)
            self.theory_lineage[theory_id]['final_status'] = final_status
            self.theory_lineage[theory_id]['name_warning'] = name_warning
            self.theory_lineage[theory_id]['status_warning'] = status_warning
    
    def generate_report(self, output_path: str = 'output/theory_tracking_report.json'):
        """Generate comprehensive tracking report."""
        print("\n" + "=" * 80)
        print("GENERATING TRACKING REPORT")
        print("=" * 80)
        
        # Compute final fields
        self._compute_final_fields()
        
        # Count final statuses and warnings
        final_status_counts = defaultdict(int)
        name_warning_count = 0
        status_warning_count = 0
        for theory_data in self.theory_lineage.values():
            final_status_counts[theory_data['final_status']] += 1
            if theory_data.get('name_warning'):
                name_warning_count += 1
            if theory_data.get('status_warning'):
                status_warning_count += 1
        
        report = {
            'metadata': {
                'total_theories': len(self.theory_lineage),
                'stages_tracked': list(self.stage_stats.keys()),
                'final_status_distribution': dict(final_status_counts),
                'name_warnings': name_warning_count,
                'status_warnings': status_warning_count
            },
            'stage_statistics': self.stage_stats,
            'theory_lineage': self.theory_lineage
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved JSON report to {output_path}")
        
        # Generate CSV for easier analysis
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame.from_dict(self.theory_lineage, orient='index')
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved CSV report to {csv_path}")
        
        # Generate summary statistics
        self._print_summary()
        
        return report
    
    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("TRACKING SUMMARY")
        print("=" * 80)
        
        total = len(self.theory_lineage)
        
        print(f"\n📊 Total theories tracked: {total}")
        
        # Stage 0
        if 'stage0' in self.stage_stats:
            print(f"\n🔹 Stage 0 (Quality Filtering):")
            print(f"   Passed filter: {self.stage_stats['stage0']['passed_filter']}")
        
        # Stage 1
        if 'stage1' in self.stage_stats:
            stats = self.stage_stats['stage1']
            print(f"\n🔹 Stage 1 (Fuzzy Matching):")
            print(f"   Matched: {stats['matched']} ({stats['matched']/total*100:.1f}%)")
            print(f"   Completed (to ontology): {stats['completed']} ({stats['completed']/total*100:.1f}%)")
            print(f"   Unmatched: {stats['unmatched']} ({stats['unmatched']/total*100:.1f}%)")
            print(f"   To process in Stage 1.5: {stats['to_process_next_stage']} theories ({stats['unique_names_to_process']} unique names)")
        
        # Stage 1.5
        if 'stage1_5' in self.stage_stats:
            stats = self.stage_stats['stage1_5']
            print(f"\n🔹 Stage 1.5 (LLM-Assisted Mapping):")
            print(f"   Mapped: {stats['mapped']} ({stats['mapped']/total*100:.1f}%)")
            print(f"   Completed (to ontology): {stats['completed']} ({stats['completed']/total*100:.1f}%)")
            print(f"   Skipped (from Stage 1): {stats['skipped']} ({stats['skipped']/total*100:.1f}%)")
            print(f"   Unmapped: {stats['unmapped']} ({stats['unmapped']/total*100:.1f}%)")
            print(f"   Total completed so far: {stats['total_completed_so_far']} ({stats['total_completed_so_far']/total*100:.1f}%)")
            print(f"   To process in Stage 2: {stats['to_process_next_stage']} theories ({stats['unique_names_to_process']} unique names)")
        
        # Stage 2
        if 'stage2' in self.stage_stats:
            stats = self.stage_stats['stage2']
            print(f"\n🔹 Stage 2 (Group Normalization):")
            print(f"   Mapped: {stats['mapped']} ({stats['mapped']/total*100:.1f}%)")
            print(f"   Completed (to ontology): {stats['completed']} ({stats['completed']/total*100:.1f}%)")
            print(f"   Skipped (from previous stages): {stats['skipped']} ({stats['skipped']/total*100:.1f}%)")
            print(f"   Warnings: {stats['warnings']} ({stats['warnings']/total*100:.1f}%)")
            print(f"   Total completed so far: {stats['total_completed_so_far']} ({stats['total_completed_so_far']/total*100:.1f}%)")
            print(f"   To process in Stage 3: {stats['to_process_next_stage']} theories ({stats['unique_names_to_process']} unique names)")
        
        # Stage 3
        if 'stage3' in self.stage_stats:
            stats = self.stage_stats['stage3']
            print(f"\n🔹 Stage 3 (Iterative Refinement):")
            print(f"   Matched: {stats['matched']} ({stats['matched']/total*100:.1f}%)")
            print(f"   Completed (to ontology): {stats['completed']} ({stats['completed']/total*100:.1f}%)")
            print(f"   Refined (not matched): {stats['refined_not_matched']} ({stats['refined_not_matched']/total*100:.1f}%)")
            print(f"   Skipped (from previous stages): {stats['skipped']} ({stats['skipped']/total*100:.1f}%)")
            print(f"   Warnings: {stats['warnings']} ({stats['warnings']/total*100:.1f}%)")
            print(f"   Total completed (all stages): {stats['total_completed']} ({stats['total_completed']/total*100:.1f}%)")
            print(f"   Still not matched: {stats['refined_not_matched']} theories ({stats['unique_not_matched_names']} unique names)")
        
        # Stage 4
        if 'stage4' in self.stage_stats:
            stats = self.stage_stats['stage4']
            print(f"\n🔹 Stage 4 (Theory Validation):")
            print(f"   Validated: {stats['validated']} ({stats['validated']/total*100:.1f}%)")
            print(f"   Invalid: {stats['invalid']} ({stats['invalid']/total*100:.1f}%)")
            print(f"   Doubted: {stats['doubted']} ({stats['doubted']/total*100:.1f}%)")
            print(f"   Mapped to ontology: {stats['mapped_to_ontology']} ({stats['mapped_to_ontology']/total*100:.1f}%)")
            print(f"   Skipped (from previous stages): {stats['skipped']} ({stats['skipped']/total*100:.1f}%)")
            print(f"   Warnings: {stats['warnings']} ({stats['warnings']/total*100:.1f}%)")
            print(f"   Total completed (all stages): {stats['total_completed']} ({stats['total_completed']/total*100:.1f}%)")
        
        # Final Status Summary
        print(f"\n🔹 Final Status Summary:")
        final_status_counts = defaultdict(int)
        name_warning_count = 0
        status_warning_count = 0
        for theory_data in self.theory_lineage.values():
            final_status_counts[theory_data.get('final_status', 'unknown')] += 1
            if theory_data.get('name_warning'):
                name_warning_count += 1
            if theory_data.get('status_warning'):
                status_warning_count += 1
        
        for status, count in sorted(final_status_counts.items()):
            print(f"   {status}: {count} ({count/total*100:.1f}%)")
        print(f"   Name warnings (no normalized name): {name_warning_count} ({name_warning_count/total*100:.1f}%)")
        print(f"   Status warnings (validation unknown): {status_warning_count} ({status_warning_count/total*100:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def query_theory(self, theory_id: str) -> dict:
        """Query lineage for a specific theory ID."""
        return self.theory_lineage.get(theory_id, None)
    
    def get_theories_by_status(self, stage: str, status: str) -> List[dict]:
        """Get all theories with a specific status at a stage."""
        status_field = f'{stage}_status'
        return [
            theory for theory in self.theory_lineage.values()
            if theory.get(status_field) == status
        ]
    
    def get_final_mapping_summary(self) -> Dict[str, int]:
        """Get summary of final mappings."""
        summary = {
            'stage1_matched': 0,
            'stage1_5_mapped': 0,
            'stage2_normalized': 0,
            'stage3_ontology_matched': 0,
            'stage3_refined': 0,
            'unmapped': 0
        }
        
        for theory in self.theory_lineage.values():
            if theory.get('stage3_ontology_match'):
                summary['stage3_ontology_matched'] += 1
            elif theory.get('stage3_final_name'):
                summary['stage3_refined'] += 1
            elif theory.get('stage2_normalized_name'):
                summary['stage2_normalized'] += 1
            elif theory.get('stage1_5_mapped_name'):
                summary['stage1_5_mapped'] += 1
            elif theory.get('stage1_matched_name'):
                summary['stage1_matched'] += 1
            else:
                summary['unmapped'] += 1
        
        return summary


def main():
    """Main entry point for theory tracking."""
    print("=" * 80)
    print("THEORY TRACKER: Comprehensive Stage Tracking")
    print("=" * 80)
    
    tracker = TheoryTracker()
    
    # Track all stages
    tracker.track_stage0('output/stage0_filtered_theories.json')
    tracker.track_stage1('output/stage1_fuzzy_matched.json')
    tracker.track_stage1_5('output/stage1_5_llm_mapped.json')
    tracker.track_stage2('output/stage2_grouped_theories.json')
    tracker.track_stage3('output/stage3_refined_theories.json')
    tracker.track_stage4('output/stage4_validated_theories.json')
    
    # Generate report
    report = tracker.generate_report('output/theory_tracking_report.json')
    
    # Print final mapping summary
    print("\n" + "=" * 80)
    print("FINAL MAPPING SUMMARY")
    print("=" * 80)
    
    summary = tracker.get_final_mapping_summary()
    total = len(tracker.theory_lineage)
    
    print(f"\n📊 Final Status Distribution:")
    print(f"   Stage 3 - Matched to ontology: {summary['stage3_ontology_matched']} ({summary['stage3_ontology_matched']/total*100:.1f}%)")
    print(f"   Stage 3 - Refined (not matched): {summary['stage3_refined']} ({summary['stage3_refined']/total*100:.1f}%)")
    print(f"   Stage 2 - Normalized only: {summary['stage2_normalized']} ({summary['stage2_normalized']/total*100:.1f}%)")
    print(f"   Stage 1.5 - Mapped only: {summary['stage1_5_mapped']} ({summary['stage1_5_mapped']/total*100:.1f}%)")
    print(f"   Stage 1 - Matched only: {summary['stage1_matched']} ({summary['stage1_matched']/total*100:.1f}%)")
    print(f"   Unmapped: {summary['unmapped']} ({summary['unmapped']/total*100:.1f}%)")
    
    print("\n✅ Tracking complete!")
    print(f"\n📁 Output files:")
    print(f"   - JSON: output/theory_tracking_report.json")
    print(f"   - CSV: output/theory_tracking_report.csv")


if __name__ == '__main__':
    main()
