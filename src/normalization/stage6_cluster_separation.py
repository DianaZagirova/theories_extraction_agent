"""
Stage 6: Cluster Separation for Overly General Theory Names

This stage separates large clusters (>40 papers) into more specific subclusters:
1. Identify clusters with >40 papers from stage5_consolidated_final_theories.json
2. For each large cluster, extract all theory_ids
3. For each theory_id, get original name, key_concepts, and paper info from stage0
4. Use LLM to assign more specific subcluster names based on key_concepts
5. Batch intelligently: max 30 theories per prompt, smart splitting for larger clusters
6. Validate: ensure no subcluster has <4 theories, retry if needed

Input:
- output/stage5_consolidated_final_theories.json
- output/stage0_filtered_theories.json

Output: output/stage6_separated_clusters.json
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import time
import asyncio
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient


class Stage6ClusterSeparator:
    """
    Stage 6: Separate overly general theory clusters into specific subclusters.
    """
    
    def __init__(self,
                 stage5_path: str = 'output/stage5_consolidated_final_theories.json',
                 stage0_path: str = 'output/stage0_filtered_theories.json',
                 output_path: str = 'output/stage6_separated_clusters.json',
                 paper_threshold: int = 40,
                 max_theories_per_batch: int = 35,
                 min_subcluster_size: int = 2,
                 max_retries: int = 2):
        """Initialize Stage 6 separator."""
        
        self.stage5_path = Path(stage5_path)
        self.stage0_path = Path(stage0_path)
        self.output_path = Path(output_path)
        self.paper_threshold = paper_threshold
        self.max_theories_per_batch = max_theories_per_batch
        self.min_subcluster_size = min_subcluster_size
        self.max_retries = max_retries
        
        # Statistics
        self.stats = {
            'total_clusters_analyzed': 0,
            'clusters_to_separate': 0,
            'total_theories_in_large_clusters': 0,
            'total_batches_processed': 0,
            'total_subclusters_created': 0,
            'total_retries': 0,
            'successful_separations': 0,
            'failed_separations': 0,
            'batches_with_singleton_warning': 0,
            'theories_with_singleton_warning': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
        
        # Load data
        print("📂 Loading data...")
        self.stage5_data = self._load_stage5()
        self.stage0_theories = self._load_stage0()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE6', 'openai')
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY_STAGE6_EX'))
        else:
            self.llm = AzureOpenAIClient()
        self.model = 'gpt-4.1-mini'
        
        self._lock = threading.Lock()
    
    def _load_stage5(self) -> Dict:
        """Load stage5 consolidated theories."""
        print(f"  Loading {self.stage5_path}...")
        with open(self.stage5_path, 'r') as f:
            data = json.load(f)
        
        final_name_summary = data.get('final_name_summary', [])
        print(f"  ✓ Loaded {len(final_name_summary)} unique theory names")
        return data
    
    def _load_stage0(self) -> Dict:
        """Load stage0 theories with key concepts."""
        print(f"  Loading {self.stage0_path}...")
        with open(self.stage0_path, 'r') as f:
            data = json.load(f)
        
        theories = {}
        for theory in data.get('theories', []):
            theory_id = theory.get('theory_id')
            if theory_id:
                theories[theory_id] = {
                    'name': theory.get('name', ''),
                    'key_concepts': theory.get('key_concepts', []),
                    'paper_title': theory.get('paper_title', ''),
                    'doi': theory.get('doi', ''),
                    'pmid': theory.get('pmid', '')
                }
        
        print(f"  ✓ Loaded {len(theories)} theories with key concepts")
        return theories
    
    def _identify_large_clusters(self) -> List[Dict]:
        """Identify clusters with more than threshold papers."""
        large_clusters = []
        
        for cluster_info in self.stage5_data.get('final_name_summary', []):
            final_name = cluster_info['final_name']
            total_papers = cluster_info['total_papers']
            theory_ids = cluster_info['theory_ids']
            
            if total_papers > self.paper_threshold:
                large_clusters.append({
                    'final_name': final_name,
                    'total_papers': total_papers,
                    'theory_ids': theory_ids,
                    'theory_count': len(theory_ids)
                })
        
        # Sort by paper count descending
        large_clusters.sort(key=lambda x: x['total_papers'], reverse=True)
        
        self.stats['clusters_to_separate'] = len(large_clusters)
        self.stats['total_theories_in_large_clusters'] = sum(c['theory_count'] for c in large_clusters)
        
        print(f"\n📊 Identified {len(large_clusters)} clusters with >{self.paper_threshold} papers")
        print(f"   Total theories to process: {self.stats['total_theories_in_large_clusters']}")
        
        return large_clusters
    
    def _create_smart_batches(self, theory_ids: List[str]) -> List[List[str]]:
        """
        Create smart batches ensuring no batch is too small.
        
        Rules:
        - Max 30 theories per batch
        - If remainder would be small (<15), split more evenly
        - Example: 35 theories -> [18, 17] not [30, 5]
        """
        total = len(theory_ids)
        max_batch = self.max_theories_per_batch
        
        if total <= max_batch:
            return [theory_ids]
        
        # Calculate initial number of batches
        num_batches = total // max_batch
        remainder = total % max_batch
        
        # If remainder is small but non-zero, redistribute
        if remainder > 0 and remainder < max_batch // 2:
            # Add one more batch to distribute more evenly
            num_batches += 1
        elif remainder >= max_batch // 2:
            # Remainder is large enough to be its own batch
            num_batches += 1
        
        # Calculate batch size for even distribution
        batch_size = total // num_batches
        extra = total % num_batches
        
        batches = []
        start = 0
        for i in range(num_batches):
            # Add one extra to first 'extra' batches
            size = batch_size + (1 if i < extra else 0)
            batches.append(theory_ids[start:start + size])
            start += size
        
        return batches
    
    def _create_separation_prompt(self, cluster_name: str, theories_data: List[Dict], 
                                   batch_info: str = "") -> str:
        """
        Create prompt for LLM to separate cluster into subclusters.
        
        Args:
            cluster_name: Original cluster name
            theories_data: List of theory data with name, key_concepts, paper_title
            batch_info: Optional batch information string
        """
        # Build theories section
        theories_section = []
        theory_id_only = []
        for idx, theory in enumerate(theories_data, 1):
            theory_id = theory['theory_id']
            theory_id_only.append(theory_id)
            original_name = theory['name']
            paper_title = theory['paper_title']
            key_concepts = theory['key_concepts']
            
            theories_section.append(f"\n# {theory_id})")
            theories_section.append(f"Initial name (might be too specific): {original_name}")
            theories_section.append(f"Paper: {paper_title}")
            theories_section.append("Key Concepts:")
            
            if key_concepts:
                for concept in key_concepts[:4]:  # Limit to 4 concepts
                    concept_name = concept.get('concept', '')
                    concept_desc = concept.get('description', '')
                    theories_section.append(f"  • {concept_name}: {concept_desc}")
            else:
                theories_section.append("  (No key concepts available)")
        
        theories_text = "\n".join(theories_section)
        min_subcluster_size_appender  = 5
        prompt = f"""# TASK
You are analyzing aging theories grouped under the broad name "{cluster_name}".
Your goal is to SEPARATE these theories into MORE SPECIFIC subclusters (with at least {self.min_subcluster_size+min_subcluster_size_appender} members) based on their mechanisms/initial names.

# CONTEXT
The current cluster contains theories that are too broadly grouped. These theories likely represent distinct mechanistic subcategories that should be separated.

# GOAL
Create 2-4 WELL-DEFINED subclusters that:
1. Are MORE SPECIFIC than "{cluster_name}" (add mechanistic details or refer to initial names)
2. Each have AT LEAST {self.min_subcluster_size+min_subcluster_size_appender} theories (preferably 8-12)
3. Capture distinct mechanistic themes

# INSTRUCTIONS
1. Analyze ALL theories below and identify 2-4 MAJOR mechanistic themes (not 10+ tiny themes)
2. Create specific subcluster names (more specific than "{cluster_name}"). Try to reflect the common mechanisms. IMPORTANT: Ensure each subcluster has AT LEAST {self.min_subcluster_size+min_subcluster_size_appender} theories.
3. Assign EACH theory id to exactly ONE subcluster (NO DUPLICATES)
4. If a theme has fewer than {self.min_subcluster_size+min_subcluster_size_appender} theories, MERGE it with the most similar theme
5. Better to have 3 large subclusters than 10 tiny ones
6. Subcluster names should be theory names (not just mechanisms)

# NAMING RULES
- Make names MORE SPECIFIC than "{cluster_name}" by adding mechanistic details
- Examples of good specificity:
  * "{cluster_name}" → "Mitochondrial ROS-Induced {cluster_name}"
  * "{cluster_name}" → "Nuclear DNA Damage-Driven {cluster_name}"
  * "{cluster_name}" → "Telomere-Associated {cluster_name}"
- Use "aging" not "ageing"
- Keep names clear and mechanism-focused
- Avoid overly long names (max 6-7 words)

# THEORIES TO SEPARATE
{theories_text}

# VALIDATION CHECKLIST (Check before responding)
✓ Every subcluster has >= {self.min_subcluster_size+min_subcluster_size_appender} theories?
✓ Created 2-4 subclusters (not 10+)?
✓ All {len(theories_data)} theories assigned?
✓ No duplicate theory_ids?
✓ No subclusters with only 2-3 theories?

If any subcluster has < {self.min_subcluster_size+min_subcluster_size_appender} theories, STOP and merge it with another subcluster before responding.

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):
Thus, map these theories to subclusters: {(", ").join(theory_id_only)}
{{
  "theory_assignments": [
    {{
      "theory_id": "T00235001",
      "subcluster_name": "More Specific Name 1",
      "reasoning": "Brief explanation why this theory belongs to this subcluster"
    }},
    {{
      "theory_id": "T0011002",
      "subcluster_name": "More Specific Name 1",
      "reasoning": "Brief explanation why this theory belongs to this subcluster"
    }},
    {{
      "theory_id": "T000010",
      "subcluster_name": "More Specific Name 2",
      "reasoning": "..."
    }},
    {{
      "theory_id": "T001230",
      "subcluster_name": "More Specific Name 2",
      "reasoning": "...."
    }},
    ...
  ]
}}

# IMPORTANT REMINDERS
1. Target: 2-4 subclusters with 8-12 theories each (NOT 10+ subclusters with 2-3 theories each)
2. Each subcluster MUST have >= {self.min_subcluster_size+min_subcluster_size_appender} theories
3. All {len(theories_data)} theory IDs must be assigned
4. Subcluster names are MORE SPECIFIC than "{cluster_name}"
5. CRITICAL: NO DUPLICATES - each theory_id appears EXACTLY ONCE (if you assign T001133 to subcluster A, do NOT assign it to subcluster B)
6. When in doubt, create BROADER subclusters (merge small themes together)
7. Double-check your output: count how many times each theory_id appears (should be exactly 1)
"""
        
        return prompt
    
    def _validate_separation(self, separation_result: Dict, input_theory_ids: List[str], cluster_name: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate separation result (works with converted format: subclusters with theory_ids).
        
        Returns:
            (is_valid, error_message, modified_result_with_singletons)
            - If some theories are missing, adds them as singletons and returns modified result
            - If some subclusters are too small, merges them into singleton warning
            - If validation passes, returns (True, "", None)
            - If critical error (no subclusters, duplicates, extra IDs), returns (False, error_msg, None)
        """
        subclusters = separation_result.get('subclusters', [])
        
        if not subclusters:
            return False, "No subclusters created", None
        
        # Separate valid and too-small subclusters
        valid_subclusters = []
        small_subclusters = []
        
        for subcluster in subclusters:
            theory_count = len(subcluster.get('theory_ids', []))
            if theory_count < self.min_subcluster_size:
                small_subclusters.append(subcluster)
            else:
                valid_subclusters.append(subcluster)
        
        # Check all theories are assigned
        assigned_ids = set()
        for subcluster in subclusters:
            assigned_ids.update(subcluster.get('theory_ids', []))
        
        input_ids_set = set(input_theory_ids)
        missing_ids = input_ids_set - assigned_ids
        extra_ids = assigned_ids - input_ids_set
        
        if extra_ids:
            return False, f"Found {len(extra_ids)} unexpected theory IDs in output", None
        
        # Check for duplicates
        all_ids = []
        for subcluster in subclusters:
            all_ids.extend(subcluster.get('theory_ids', []))
        
        if len(all_ids) != len(set(all_ids)):
            duplicates = [tid for tid in all_ids if all_ids.count(tid) > 1]
            return False, f"Duplicate theory IDs found: {duplicates[:5]}", None
        
        # If no valid subclusters at all, fail completely
        if not valid_subclusters:
            return False, f"All {len(subclusters)} subclusters are too small (min: {self.min_subcluster_size})", None
        
        # If we have small subclusters or missing theories, merge them into singleton warning
        singleton_ids = list(missing_ids)
        for small_sc in small_subclusters:
            singleton_ids.extend(small_sc.get('theory_ids', []))
        
        if singleton_ids:
            warnings = []
            if small_subclusters:
                warnings.append(f"{len(small_subclusters)} subclusters too small")
            if missing_ids:
                warnings.append(f"{len(missing_ids)} theories not assigned")
            
            warning_msg = ", ".join(warnings)
            print(f"    ⚠️  {warning_msg} - adding {len(singleton_ids)} theories as singletons with original name")
            
            modified_result = separation_result.copy()
            modified_subclusters = list(valid_subclusters)  # Only keep valid ones
            
            # Add singleton subcluster for all problematic theories
            modified_subclusters.append({
                'subcluster_name': cluster_name,
                'theory_ids': singleton_ids,
                'theory_count': len(singleton_ids),
                'mechanism_focus': '',
                'status': 'singleton_warning',
                'warning_reason': warning_msg
            })
            
            modified_result['subclusters'] = modified_subclusters
            return True, f"Saved {len(valid_subclusters)} valid subclusters, {len(singleton_ids)} singletons", modified_result
        
        return True, "", None
    
    async def _separate_batch_async(self, cluster_name: str, batch_theories: List[Dict],
                                     batch_num: int, total_batches: int) -> Optional[Dict]:
        """
        Separate a batch of theories into subclusters.
        
        CRITICAL GUARANTEES:
        1. All theories in batch_theories are from the SAME cluster (cluster_name)
        2. The LLM prompt will ONLY contain data from these theories
        3. If validation fails, the ENTIRE BATCH is retried, not individual theories
        
        This ensures the LLM sees coherent, related theories for separation.
        
        Returns:
            Dictionary with subclusters or None if failed after max_retries
        """
        batch_info = f"# BATCH INFO\nThis is batch {batch_num} of {total_batches} for cluster '{cluster_name}'."
        
        theory_ids = [t['theory_id'] for t in batch_theories]
        
        # Retry the ENTIRE BATCH if validation fails
        for attempt in range(self.max_retries):
            try:
                # Create prompt
                prompt = self._create_separation_prompt(cluster_name, batch_theories, batch_info)
                
                # Call LLM
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert in aging biology and theory classification. You excel at identifying distinct mechanistic themes."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.0,
                        max_tokens=4000
                    )
                )
                
                # Track tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
                
                with self._lock:
                    self.stats['total_input_tokens'] += input_tokens
                    self.stats['total_output_tokens'] += output_tokens
                    self.stats['total_cost'] += cost
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                
                # Clean JSON markers
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                # Parse JSON
                data = json.loads(response_text)
                
                # Convert to old format for compatibility with rest of code
                # New format: theory_assignments (individual assignments)
                # Old format: subclusters with theory_ids (grouped)
                if 'theory_assignments' in data:
                    # Convert new format to old format
                    theory_assignments = data['theory_assignments']
                    
                    # CRITICAL: Remove duplicates from LLM response
                    # Keep only first occurrence of each theory_id
                    seen_theory_ids = set()
                    deduplicated_assignments = []
                    duplicates_removed = []
                    
                    for assignment in theory_assignments:
                        theory_id = assignment['theory_id']
                        if theory_id in seen_theory_ids:
                            duplicates_removed.append(theory_id)
                            continue
                        seen_theory_ids.add(theory_id)
                        deduplicated_assignments.append(assignment)
                    
                    if duplicates_removed:
                        print(f"    ⚠️  LLM created duplicates - removed {len(duplicates_removed)} duplicate assignments: {duplicates_removed[:5]}")
                    
                    theory_assignments = deduplicated_assignments
                    
                    # Group theories by subcluster
                    subcluster_dict = {}
                    for assignment in theory_assignments:
                        subcluster_name = assignment['subcluster_name']
                        theory_id = assignment['theory_id']
                        
                        if subcluster_name not in subcluster_dict:
                            subcluster_dict[subcluster_name] = {
                                'subcluster_name': subcluster_name,
                                'theory_ids': [],
                                'mechanism_focus': ''
                            }
                        subcluster_dict[subcluster_name]['theory_ids'].append(theory_id)
                    
                    # Add theory_count
                    for name in subcluster_dict:
                        subcluster_dict[name]['theory_count'] = len(subcluster_dict[name]['theory_ids'])
                    
                    # Convert to list
                    data = {
                        'subclusters': list(subcluster_dict.values()),
                        'separation_rationale': 'Separated based on mechanistic themes',
                        'original_format': {
                            'theory_assignments': theory_assignments
                        }
                    }
                
                # Validate
                is_valid, error_msg, modified_data = self._validate_separation(data, theory_ids, cluster_name)
                
                if is_valid:
                    # Use modified data if singletons were added
                    result_data = modified_data if modified_data else data
                    if modified_data:
                        # Count theories in singleton warning subclusters
                        singleton_theories = sum(
                            len(s.get('theory_ids', [])) 
                            for s in modified_data['subclusters'] 
                            if s.get('status') == 'singleton_warning'
                        )
                        with self._lock:
                            self.stats['theories_with_singleton_warning'] += singleton_theories
                    return result_data
                else:
                    print(f"  ⚠️  Validation failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}")
                    if attempt < self.max_retries - 1:
                        print(f"      → Retrying ENTIRE BATCH {batch_num} with all {len(theory_ids)} theories...")
                        with self._lock:
                            self.stats['total_retries'] += 1
                        await asyncio.sleep(1)
                    else:
                        print(f"  ❌ Max retries reached for batch {batch_num} - all {len(theory_ids)} theories failed")
                        return None
                        
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"      → Retrying ENTIRE BATCH {batch_num}...")
                    with self._lock:
                        self.stats['total_retries'] += 1
                    await asyncio.sleep(1)
                else:
                    print(f"  ❌ Batch {batch_num} failed after {self.max_retries} attempts")
                    return None
                    
            except Exception as e:
                print(f"  ❌ Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"      → Retrying ENTIRE BATCH {batch_num}...")
                    with self._lock:
                        self.stats['total_retries'] += 1
                    await asyncio.sleep(1)
                else:
                    print(f"  ❌ Batch {batch_num} failed after {self.max_retries} attempts")
                    return None
        
        return None
    
    def _merge_batch_results(self, cluster_name: str, batch_results: List[Dict]) -> Dict:
        """
        Merge results from multiple batches.
        
        If multiple batches, we need to potentially merge similar subclusters across batches.
        For now, we'll keep them separate but add batch identifier to names.
        Preserves 'status' field for singleton warnings.
        """
        if len(batch_results) == 1:
            return batch_results[0]
        
        # Merge subclusters from all batches
        all_subclusters = []
        has_warnings = False
        
        for batch_idx, batch_result in enumerate(batch_results, 1):
            for subcluster in batch_result.get('subclusters', []):
                # Check if this is a singleton warning
                status = subcluster.get('status', 'normal')
                if status == 'singleton_warning':
                    has_warnings = True
                
                # Add batch identifier if multiple batches (but not for singleton warnings)
                subcluster_name = subcluster['subcluster_name']
                if len(batch_results) > 1 and status != 'singleton_warning':
                    subcluster_name = f"{subcluster_name} (Batch {batch_idx})"
                
                subcluster_entry = {
                    'subcluster_name': subcluster_name,
                    'mechanism_focus': subcluster.get('mechanism_focus', ''),
                    'theory_ids': subcluster.get('theory_ids', []),
                    'theory_count': len(subcluster.get('theory_ids', []))
                }
                
                # Preserve status and warning_reason if present
                if status == 'singleton_warning':
                    subcluster_entry['status'] = status
                    subcluster_entry['warning_reason'] = subcluster.get('warning_reason', '')
                
                all_subclusters.append(subcluster_entry)
        
        rationale = f"Merged from {len(batch_results)} batches"
        if has_warnings:
            rationale += " (includes singleton warnings for failed batches)"
        
        return {
            'subclusters': all_subclusters,
            'separation_rationale': rationale
        }
    
    def _calculate_concept_similarity(self, theory_ids_1: List[str], theory_ids_2: List[str]) -> float:
        """
        Calculate similarity between two sets of theories based on key concept overlap.
        
        Uses overlap coefficient (less strict than Jaccard) to find related subclusters.
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get all key concepts from both theory sets
        concepts_1 = set()
        concepts_2 = set()
        
        for tid in theory_ids_1:
            if tid in self.stage0_theories:
                for concept in self.stage0_theories[tid].get('key_concepts', []):
                    concept_name = concept.get('concept', '').lower().strip()
                    if concept_name:
                        concepts_1.add(concept_name)
        
        for tid in theory_ids_2:
            if tid in self.stage0_theories:
                for concept in self.stage0_theories[tid].get('key_concepts', []):
                    concept_name = concept.get('concept', '').lower().strip()
                    if concept_name:
                        concepts_2.add(concept_name)
        
        if not concepts_1 or not concepts_2:
            return 0.0
        
        # Calculate overlap coefficient (intersection / smaller set)
        # This is less strict than Jaccard and works better for sets of different sizes
        intersection = len(concepts_1 & concepts_2)
        smaller_set = min(len(concepts_1), len(concepts_2))
        
        return intersection / smaller_set if smaller_set > 0 else 0.0
    
    async def _recluster_with_previous_batches(self, cluster_name: str, failed_ids: List[str], 
                                                batch_results: List[Dict]) -> bool:
        """
        Recluster failed theories by finding most similar previous subclusters.
        
        Strategy:
        1. Calculate concept similarity between failed theories and each subcluster
        2. Select top N most similar subclusters
        3. Combine and recluster
        
        Returns:
            True if reclustering succeeded, False otherwise
        """
        if not batch_results:
            return False
        
        # Collect all valid subclusters from previous batches (exclude singleton warnings)
        all_subclusters = []
        for batch_result in batch_results:
            for sc in batch_result.get('subclusters', []):
                if sc.get('status') != 'singleton_warning':
                    all_subclusters.append(sc)
        
        if len(all_subclusters) < 2:
            print(f"      ⚠️  Not enough previous subclusters ({len(all_subclusters)}) for reclustering")
            return False
        
        # Calculate similarity scores for each subcluster
        print(f"      🔍 Analyzing similarity between {len(failed_ids)} failed theories and {len(all_subclusters)} subclusters...")
        
        MIN_SIMILARITY = 0.05  # At least 5% concept overlap required (lowered from 0.10)
        
        subcluster_scores = []
        for sc in all_subclusters:
            similarity = self._calculate_concept_similarity(failed_ids, sc.get('theory_ids', []))
            subcluster_scores.append({
                'subcluster': sc,
                'similarity': similarity,
                'size': sc['theory_count']
            })
        
        # Sort by similarity (descending)
        subcluster_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Check if we have any subclusters with sufficient similarity
        high_similarity = [x for x in subcluster_scores if x['similarity'] >= MIN_SIMILARITY]
        if len(high_similarity) < 2:
            print(f"      ⚠️  No subclusters with sufficient similarity (min: {MIN_SIMILARITY:.2f}, best: {subcluster_scores[0]['similarity']:.3f})")
            return False
        
        # Smart selection based on failed theory count and similarity
        num_failed = len(failed_ids)
        
        if num_failed <= 10:
            # Few failed theories - select 2-3 most similar smaller subclusters
            num_to_select = min(3, len(subcluster_scores))
            # Filter for smaller subclusters (bottom 60% by size) from top similar ones
            size_threshold = sorted([x['size'] for x in subcluster_scores])[int(len(subcluster_scores) * 0.6)]
            candidates = [x for x in subcluster_scores if x['size'] <= size_threshold and x['similarity'] > 0.0]
            
            if len(candidates) < num_to_select:
                # Not enough small similar ones, take top similar regardless of size
                candidates = [x for x in subcluster_scores if x['similarity'] > 0.0]
            
            selected = candidates[:num_to_select] if candidates else subcluster_scores[:num_to_select]
        else:
            # Many failed theories - select 3-4 most similar medium-sized subclusters
            num_to_select = min(4, len(subcluster_scores))
            # Filter for medium-sized subclusters (25th to 75th percentile)
            sizes = sorted([x['size'] for x in subcluster_scores])
            size_25 = sizes[len(sizes) // 4] if sizes else 0
            size_75 = sizes[3 * len(sizes) // 4] if sizes else float('inf')
            
            candidates = [x for x in subcluster_scores 
                         if size_25 <= x['size'] <= size_75 and x['similarity'] > 0.0]
            
            if len(candidates) < num_to_select:
                # Not enough medium similar ones, take top similar regardless of size
                candidates = [x for x in subcluster_scores if x['similarity'] > 0.0]
            
            selected = candidates[:num_to_select] if candidates else subcluster_scores[:num_to_select]
        
        selected_subclusters = [x['subcluster'] for x in selected]
        
        print(f"      📊 Selected {len(selected_subclusters)} most similar subclusters for reclustering:")
        for i, sel in enumerate(selected):
            sc = sel['subcluster']
            sim = sel['similarity']
            print(f"         {i+1}. {sc['subcluster_name']}: {sc['theory_count']} theories (similarity: {sim:.3f})")
        
        # Combine theories from selected subclusters + failed theories
        recluster_theory_ids = failed_ids.copy()
        for sc in selected_subclusters:
            recluster_theory_ids.extend(sc.get('theory_ids', []))
        
        print(f"      🔄 Reclustering {len(recluster_theory_ids)} theories total")
        
        # Get theory data
        recluster_theories = []
        for theory_id in recluster_theory_ids:
            if theory_id in self.stage0_theories:
                theory_data = self.stage0_theories[theory_id].copy()
                theory_data['theory_id'] = theory_id
                recluster_theories.append(theory_data)
        
        # Deduplicate recluster_theory_ids before processing
        original_count = len(recluster_theory_ids)
        recluster_theory_ids = list(dict.fromkeys(recluster_theory_ids))  # Preserve order
        if len(recluster_theory_ids) < original_count:
            print(f"      ⚠️  Removed {original_count - len(recluster_theory_ids)} duplicate theory IDs before reclustering")
        
        # Try to recluster (with 2 retries)
        for attempt in range(3):
            result = await self._separate_batch_async(
                cluster_name, recluster_theories, 
                batch_num=999,  # Special batch number for reclustering
                total_batches=999
            )

            if result:
                # Check if result is valid (not all singleton warnings)
                valid_subclusters = [sc for sc in result.get('subclusters', []) 
                                    if sc.get('status') != 'singleton_warning']
                
                if valid_subclusters:
                    # Remove the selected subclusters from batch_results
                    for batch_result in batch_results:
                        batch_result['subclusters'] = [
                            sc for sc in batch_result.get('subclusters', [])
                            if sc not in selected_subclusters
                        ]
                    
                    # Add reclustered result
                    batch_results.append(result)
                    print(f"      ✓ Created {len(valid_subclusters)} new subclusters from reclustering")
                    return True
                else:
                    print(f"      ⚠️  Reclustering produced only singleton warnings")
                    if attempt < 2:
                        print(f"      🔄 Trying with different subclusters (attempt {attempt + 2}/3)...")
                        # Try with next best subclusters
                        if len(subcluster_scores) > len(selected) + 2:
                            selected = subcluster_scores[len(selected):len(selected) + num_to_select]
                            selected_subclusters = [x['subcluster'] for x in selected]
                            recluster_theory_ids = failed_ids.copy()
                            for sc in selected_subclusters:
                                recluster_theory_ids.extend(sc.get('theory_ids', []))
                            recluster_theories = []
                            for theory_id in recluster_theory_ids:
                                if theory_id in self.stage0_theories:
                                    theory_data = self.stage0_theories[theory_id].copy()
                                    theory_data['theory_id'] = theory_id
                                    recluster_theories.append(theory_data)
                        await asyncio.sleep(1)
            else:
                if attempt < 2:
                    print(f"      ⚠️  Reclustering attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(1)
        
        return False
    
    def _save_batch_result(self, cluster_name: str, batch_num: int, batch_result: Dict):
        """Save individual batch result to file."""
        # Create batches directory
        batches_dir = self.output_path.parent / 'stage6_batches'
        batches_dir.mkdir(exist_ok=True)
        
        # Clean cluster name for filename
        clean_name = cluster_name.replace("/", "_").replace(" ", "_").replace(":", "")
        batch_path = batches_dir / f'{clean_name}_batch_{batch_num:03d}.json'
        
        batch_data = {
            'cluster_name': cluster_name,
            'batch_num': batch_num,
            'batch_result': batch_result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
    
    def _save_checkpoint(self, cluster_name: str, batch_results: List[Dict]):
        """Save checkpoint with all batch results so far."""
        checkpoint_path = self.output_path.parent / f'stage6_checkpoint_{cluster_name.replace("/", "_").replace(" ", "_").replace(":", "")}.json'
        checkpoint_data = {
            'cluster_name': cluster_name,
            'batches_completed': len(batch_results),
            'batch_results': batch_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    async def _process_cluster_async(self, cluster_info: Dict) -> Dict:
        """
        Process a single large cluster.
        
        CRITICAL GUARANTEE: Each LLM prompt will ONLY contain theories (T IDs) from this
        single cluster_name. Theories from different clusters are NEVER mixed in one prompt.
        This ensures the LLM sees coherent, related theories for separation.
        """
        cluster_name = cluster_info['final_name']
        theory_ids = cluster_info['theory_ids']
        total_papers = cluster_info['total_papers']
        
        print(f"\n{'='*80}")
        print(f"Processing: {cluster_name}")
        print(f"  Papers: {total_papers} | Theories: {len(theory_ids)}")
        
        # Create smart batches - ALL from this single cluster only
        batches = self._create_smart_batches(theory_ids)
        print(f"  Batches: {len(batches)} (sizes: {[len(b) for b in batches]})")
        print(f"  ✓ All batches contain ONLY theories from '{cluster_name}'")
        
        # Get theory data for each batch
        batch_results = []
        failed_theory_ids = []  # Track theories that failed in current batch
        MAX_CARRY_FORWARD = 10  # Max theories to carry forward to avoid bloating batches
        
        for batch_num, batch_ids in enumerate(batches, 1):
            # Convert to list and ensure no duplicates in original batch
            batch_ids = list(set(batch_ids))
            
            # Smart carry-forward logic
            if failed_theory_ids:
                # Remove any failed theories that are already in this batch
                failed_theory_ids = [tid for tid in failed_theory_ids if tid not in batch_ids]
                
                if not failed_theory_ids:
                    print(f"\n  📦 Processing batch {batch_num}/{len(batches)} ({len(batch_ids)} theories)...")
                elif len(failed_theory_ids) <= MAX_CARRY_FORWARD:
                    # Small number - carry forward normally
                    print(f"\n  📦 Processing batch {batch_num}/{len(batches)} ({len(batch_ids)} theories + {len(failed_theory_ids)} carried forward)...")
                    batch_ids = batch_ids + failed_theory_ids
                    failed_theory_ids = []  # Reset for this batch
                else:
                    # Too many failed theories - trigger early reclustering
                    print(f"\n  ⚠️  {len(failed_theory_ids)} theories failed (too many to carry forward)")
                    print(f"  🔄 Triggering early reclustering with previous batches...")
                    
                    reclustered = await self._recluster_with_previous_batches(
                        cluster_name, failed_theory_ids, batch_results
                    )
                    
                    if reclustered:
                        print(f"  ✓ Early reclustering successful")
                        failed_theory_ids = []  # Clear after successful reclustering
                    else:
                        # Reclustering failed - split failed theories across next few batches
                        print(f"  ⚠️  Reclustering failed - splitting {len(failed_theory_ids)} theories across next batches")
                        carry_now = failed_theory_ids[:MAX_CARRY_FORWARD]
                        failed_theory_ids = failed_theory_ids[MAX_CARRY_FORWARD:]  # Keep rest for later
                        print(f"  📦 Processing batch {batch_num}/{len(batches)} ({len(batch_ids)} theories + {len(carry_now)} carried forward)...")
                        batch_ids = batch_ids + carry_now
            else:
                print(f"\n  📦 Processing batch {batch_num}/{len(batches)} ({len(batch_ids)} theories)...")
            
            # Final deduplication check
            original_count = len(batch_ids)
            batch_ids = list(dict.fromkeys(batch_ids))  # Preserve order while deduplicating
            if len(batch_ids) < original_count:
                print(f"    ⚠️  Removed {original_count - len(batch_ids)} duplicate theory IDs from batch")
            
            # Extract theory data - validate all from same cluster
            batch_theories = []
            seen_ids = set()
            for theory_id in batch_ids:
                if theory_id in seen_ids:
                    print(f"    ⚠️  Skipping duplicate theory ID: {theory_id}")
                    continue
                seen_ids.add(theory_id)
                
                if theory_id in self.stage0_theories:
                    theory_data = self.stage0_theories[theory_id].copy()
                    theory_data['theory_id'] = theory_id
                    batch_theories.append(theory_data)
                else:
                    print(f"    ⚠️  Warning: Theory {theory_id} not found in stage0")
            
            # Update batch_ids to match batch_theories (remove any skipped duplicates)
            batch_ids = [t['theory_id'] for t in batch_theories]
            
            # Validation: Ensure all theories in this batch are from the same cluster
            # (This should always be true by design, but we validate for safety)
            assert all(tid in theory_ids for tid in batch_ids), \
                f"CRITICAL ERROR: Batch contains theories not from cluster '{cluster_name}'"
            
            if not batch_theories:
                print(f"    ⚠️  No valid theories in batch {batch_num}, skipping")
                continue
            
            # Separate batch
            result = await self._separate_batch_async(
                cluster_name, batch_theories, batch_num, len(batches)
            )
            
            if result:
                batch_results.append(result)
                subclusters = result.get('subclusters', [])
                print(f"    ✓ Created {len(subclusters)} subclusters")
                
                # Check for singleton warnings and extract failed theory IDs
                for sc in subclusters:
                    status_marker = " [⚠️ singleton warning]" if sc.get('status') == 'singleton_warning' else ""
                    print(f"      - {sc['subcluster_name']}: {sc['theory_count']} theories{status_marker}")
                    
                    # If this is a singleton warning and not the last batch, carry forward
                    if sc.get('status') == 'singleton_warning' and batch_num < len(batches):
                        singleton_ids = sc.get('theory_ids', [])
                        failed_theory_ids.extend(singleton_ids)
                        
                        # Smart messaging based on count
                        if len(singleton_ids) <= MAX_CARRY_FORWARD:
                            print(f"        → Carrying {len(singleton_ids)} theories forward to next batch")
                        else:
                            print(f"        ⚠️  {len(singleton_ids)} theories to handle (will trigger reclustering or split)")
                
                # If we have failed theories to carry forward, remove singleton warning from result
                if failed_theory_ids and batch_num < len(batches):
                    # Remove singleton warning subclusters from this batch result (only if not last batch)
                    result['subclusters'] = [sc for sc in subclusters if sc.get('status') != 'singleton_warning']
                elif failed_theory_ids and batch_num == len(batches):
                    # Last batch - keep singleton warnings in result
                    pass
                
                # Save individual batch result
                self._save_batch_result(cluster_name, batch_num, result)
                
                # Save checkpoint with all batches so far
                self._save_checkpoint(cluster_name, batch_results)
                
                with self._lock:
                    self.stats['total_batches_processed'] += 1
                    self.stats['total_subclusters_created'] += len(result['subclusters'])
            else:
                print(f"    ❌ Failed to separate batch {batch_num}")
                print(f"    ⚠️  Assigning {len(batch_ids)} theories to original name with 'singleton warning' status")
                
                # If not last batch, carry forward all theories (deduplicate first)
                if batch_num < len(batches):
                    # Remove any that are already in failed_theory_ids
                    new_failed = [tid for tid in batch_ids if tid not in failed_theory_ids]
                    print(f"    → Carrying {len(new_failed)} theories forward to next batch")
                    failed_theory_ids.extend(new_failed)
                else:
                    # Last batch - try reclustering with previous batches
                    print(f"    🔄 Last batch failed - attempting reclustering with previous batches")
                    reclustered = await self._recluster_with_previous_batches(
                        cluster_name, batch_ids, batch_results
                    )
                    
                    if reclustered:
                        print(f"    ✓ Reclustering successful")
                        # Reclustered results already added to batch_results
                    else:
                        # Reclustering failed - create fallback
                        print(f"    ❌ Reclustering failed - assigning to original name")
                        fallback_result = {
                            'subclusters': [{
                                'subcluster_name': cluster_name,
                                'theory_ids': batch_ids,
                                'theory_count': len(batch_ids),
                                'mechanism_focus': '',
                                'status': 'singleton_warning',
                                'warning_reason': f'Failed to separate after {self.max_retries} retries and reclustering'
                            }],
                            'separation_rationale': 'Fallback: assigned to original name due to separation failure'
                        }
                        batch_results.append(fallback_result)
                        self._save_batch_result(cluster_name, batch_num, fallback_result)
                        self._save_checkpoint(cluster_name, batch_results)
                        
                        # Track singleton warnings
                        with self._lock:
                            self.stats['batches_with_singleton_warning'] += 1
                            self.stats['theories_with_singleton_warning'] += len(batch_ids)
        
        # Handle any remaining failed theories that were never processed
        if failed_theory_ids:
            print(f"\n  ⚠️  {len(failed_theory_ids)} theories still pending after all batches")
            print(f"  → Adding them to final results with singleton warning")
            
            fallback_result = {
                'subclusters': [{
                    'subcluster_name': cluster_name,
                    'theory_ids': failed_theory_ids,
                    'theory_count': len(failed_theory_ids),
                    'mechanism_focus': '',
                    'status': 'singleton_warning',
                    'warning_reason': f'Carried forward through batches but never successfully processed'
                }],
                'separation_rationale': 'Fallback: theories carried forward but not processed'
            }
            batch_results.append(fallback_result)
            
            with self._lock:
                self.stats['batches_with_singleton_warning'] += 1
                self.stats['theories_with_singleton_warning'] += len(failed_theory_ids)
        
        # Merge batch results
        if batch_results:
            merged_result = self._merge_batch_results(cluster_name, batch_results)
            
            with self._lock:
                self.stats['successful_separations'] += 1
            
            return {
                'original_cluster_name': cluster_name,
                'original_total_papers': total_papers,
                'original_theory_count': len(theory_ids),
                'separation_successful': True,
                'subclusters': merged_result['subclusters'],
                'separation_rationale': merged_result.get('separation_rationale', '')
            }
        else:
            with self._lock:
                self.stats['failed_separations'] += 1
            
            return {
                'original_cluster_name': cluster_name,
                'original_total_papers': total_papers,
                'original_theory_count': len(theory_ids),
                'separation_successful': False,
                'error': 'All batches failed'
            }
    
    async def process_all_clusters_async(self) -> Dict:
        """
        Process all large clusters.
        
        CRITICAL GUARANTEE: Each cluster is processed independently and sequentially.
        The LLM will NEVER see theories from different clusters in the same prompt.
        """
        large_clusters = self._identify_large_clusters()
        
        if not large_clusters:
            print("\n✓ No clusters need separation")
            return {'separated_clusters': [], 'statistics': self.stats}
        
        print(f"\n🚀 Starting separation of {len(large_clusters)} clusters...")
        print(f"   Each cluster processed independently - no mixing of theories across clusters")
        
        results = []
        # Don't use tqdm wrapper - it hides the batch progress
        for idx, cluster_info in enumerate(large_clusters, 1):
            print(f"\n[Cluster {idx}/{len(large_clusters)}]")
            # Process ONE cluster at a time - ensures no mixing
            result = await self._process_cluster_async(cluster_info)
            results.append(result)
            
            # Small delay between clusters
            await asyncio.sleep(0.5)
        
        return {
            'metadata': {
                'stage': 'stage6_cluster_separation',
                'source_stage5': str(self.stage5_path),
                'source_stage0': str(self.stage0_path),
                'paper_threshold': self.paper_threshold,
                'min_subcluster_size': self.min_subcluster_size,
                'max_theories_per_batch': self.max_theories_per_batch,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'separated_clusters': results,
            'statistics': self.stats
        }
    
    def _clean_batch_number_from_name(self, name: str) -> str:
        """
        Remove batch number suffix from subcluster names.
        
        Examples:
            "Mitochondrial ROS-Induced Damage Theory (Batch 33)" -> "Mitochondrial ROS-Induced Damage Theory"
            "Some Theory (Batch 1)" -> "Some Theory"
            "Normal Theory" -> "Normal Theory"
        """
        import re
        # Pattern to match " (Batch N)" at the end of the string
        cleaned = re.sub(r'\s*\(Batch\s+\d+\)\s*$', '', name, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def _create_consolidated_output(self, separation_results: Dict) -> Dict:
        """
        Create consolidated output combining stage5 and stage6 results.
        
        This maintains full tracking:
        - Theories in separated clusters get new stage6 names
        - Theories in non-separated clusters keep stage5 names
        - All theory_id mappings are preserved
        
        Returns:
            Dictionary with consolidated final_name_summary
        """
        print("\n📊 Creating consolidated output (stage5 + stage6)...")
        
        # Build mapping: theory_id -> new subcluster name (for separated clusters)
        theory_id_to_new_name = {}
        separated_cluster_names = set()
        
        for cluster_result in separation_results['separated_clusters']:
            if cluster_result.get('separation_successful', False):
                original_name = cluster_result['original_cluster_name']
                separated_cluster_names.add(original_name)
                
                for subcluster in cluster_result.get('subclusters', []):
                    subcluster_name = subcluster['subcluster_name']
                    # Clean batch numbers from subcluster names
                    cleaned_name = self._clean_batch_number_from_name(subcluster_name)
                    for theory_id in subcluster.get('theory_ids', []):
                        theory_id_to_new_name[theory_id] = cleaned_name
        
        print(f"  Separated {len(separated_cluster_names)} clusters into {len(set(theory_id_to_new_name.values()))} subclusters")
        print(f"  Mapped {len(theory_id_to_new_name)} theory IDs to new names")
        
        # Build new final_name_summary
        new_summary = []
        
        # Group theories by their final name (stage6 if separated, stage5 otherwise)
        name_to_theories = defaultdict(lambda: {
            'theory_ids': [],
            'original_names': set(),
            'stage5_parent': None,
            'was_separated': False
        })
        
        for cluster_info in self.stage5_data.get('final_name_summary', []):
            stage5_name = cluster_info['final_name']
            theory_ids = cluster_info['theory_ids']
            original_names = cluster_info.get('original_names', [])
            
            if stage5_name in separated_cluster_names:
                # This cluster was separated - distribute theories to subclusters
                for theory_id in theory_ids:
                    new_name = theory_id_to_new_name.get(theory_id)
                    if new_name:
                        name_to_theories[new_name]['theory_ids'].append(theory_id)
                        name_to_theories[new_name]['original_names'].update(original_names)
                        name_to_theories[new_name]['stage5_parent'] = stage5_name
                        name_to_theories[new_name]['was_separated'] = True
                    else:
                        # Theory not found in separation - keep with original stage5 name
                        # This can happen if theory was in a carried-forward batch that's still pending
                        print(f"  ⚠️  Warning: Theory {theory_id} from '{stage5_name}' not found in separation results - keeping with stage5 name")
                        name_to_theories[stage5_name]['theory_ids'].append(theory_id)
                        name_to_theories[stage5_name]['original_names'].update(original_names)
                        name_to_theories[stage5_name]['stage5_parent'] = stage5_name
                        name_to_theories[stage5_name]['was_separated'] = False
            else:
                # This cluster was not separated - keep as is
                name_to_theories[stage5_name]['theory_ids'].extend(theory_ids)
                name_to_theories[stage5_name]['original_names'].update(original_names)
                name_to_theories[stage5_name]['stage5_parent'] = stage5_name
                name_to_theories[stage5_name]['was_separated'] = False
        
        # Convert to final summary format
        for final_name, data in name_to_theories.items():
            theory_ids = data['theory_ids']
            original_names = sorted(list(data['original_names']))
            
            summary_entry = {
                'final_name': final_name,
                'original_names_count': len(original_names),
                'original_names': original_names,
                'total_papers': len(theory_ids),
                'theory_ids_count': len(theory_ids),
                'theory_ids': theory_ids,
                'stage5_parent': data['stage5_parent'],
                'was_separated_in_stage6': data['was_separated']
            }
            
            new_summary.append(summary_entry)
        
        # Sort by paper count descending
        new_summary.sort(key=lambda x: x['total_papers'], reverse=True)
        
        print(f"  ✓ Created {len(new_summary)} final theory names")
        print(f"    - {sum(1 for s in new_summary if s['was_separated_in_stage6'])} from stage6 separation")
        print(f"    - {sum(1 for s in new_summary if not s['was_separated_in_stage6'])} from stage5 (unchanged)")
        
        return {
            'metadata': {
                'source_stage5': str(self.stage5_path),
                'source_stage6': str(self.output_path),
                'total_theory_ids': sum(len(s['theory_ids']) for s in new_summary),
                'unique_final_names': len(new_summary),
                'separated_in_stage6': sum(1 for s in new_summary if s['was_separated_in_stage6']),
                'unchanged_from_stage5': sum(1 for s in new_summary if not s['was_separated_in_stage6']),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'final_name_summary': new_summary,
            'stage6_separation_details': separation_results
        }
    
    def run(self):
        """Run the stage6 separation process."""
        print("\n" + "="*80)
        print("STAGE 6: CLUSTER SEPARATION")
        print("="*80)
        
        # Process clusters
        loop = asyncio.get_event_loop()
        output_data = loop.run_until_complete(self.process_all_clusters_async())
        
        # Save separation results
        print(f"\n💾 Saving separation results to {self.output_path}...")
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  ✓ Saved")
        
        # Create and save consolidated output
        consolidated_path = self.output_path.parent / 'stage6_consolidated_final_theories.json'
        print(f"\n💾 Creating consolidated output (stage5 + stage6)...")
        consolidated_data = self._create_consolidated_output(output_data)
        
        print(f"   Saving to {consolidated_path}...")
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        print(f"  ✓ Saved")
        
        # Print statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        stats = self.stats
        print(f"Clusters analyzed: {stats['total_clusters_analyzed']}")
        print(f"Clusters to separate: {stats['clusters_to_separate']}")
        print(f"Theories in large clusters: {stats['total_theories_in_large_clusters']}")
        print(f"Batches processed: {stats['total_batches_processed']}")
        print(f"Subclusters created: {stats['total_subclusters_created']}")
        print(f"Successful separations: {stats['successful_separations']}")
        print(f"Failed separations: {stats['failed_separations']}")
        print(f"Total retries: {stats['total_retries']}")
        print(f"Batches with singleton warning: {stats['batches_with_singleton_warning']}")
        print(f"Theories with singleton warning: {stats['theories_with_singleton_warning']}")
        print(f"\nToken usage:")
        print(f"  Input tokens: {stats['total_input_tokens']:,}")
        print(f"  Output tokens: {stats['total_output_tokens']:,}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print("="*80)
        
        print(f"\n✅ Stage 6 complete!")
        print(f"   Separation details: {self.output_path}")
        print(f"   Consolidated output: {consolidated_path}")
        print(f"\n💡 Use the consolidated output for downstream processing:")
        print(f"   {consolidated_path}")


def main():
    """Main entry point."""
    separator = Stage6ClusterSeparator(
        paper_threshold=40,
        max_theories_per_batch=26,
        min_subcluster_size=2,
        max_retries=3
    )
    separator.run()


if __name__ == '__main__':
    main()
