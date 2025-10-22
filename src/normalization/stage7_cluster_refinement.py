#!/usr/bin/env python3
"""
Stage 7: Cluster-Based Theory Name Gathering

This stage gathers small theory names (from Stage 6) under broader names within embedding clusters:
1. Load clusters from data/stage7/clusters_from_stage6_names.json
2. For each cluster, separate theories into REFERENCE_LIST (paper_count >= reference_min_size) and RARE_LIST (paper_count <= rare_max_size)
3. For theories, sample key_concepts from stage0
4. Use LLM to normalize RARE_LIST theories:
   - Preferably map to REFERENCE_LIST names
   - Create 1-2 generalized names if needed
   - Retain original if neither works

Input:
- data/stage7/clusters_from_stage6_names.json (predefined clusters)
- output/stage6_consolidated_final_theories.json (theory name to ID mapping)
- output/stage0_filtered_theories.json (key concepts)

Output: 
- output/stage7_name_normalizations.json (per-cluster name mapping decisions)
- output/stage7_consolidated_final_theories.json (updated consolidated summary)
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import time
from tqdm import tqdm
import asyncio
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient


class RateLimiter:
    """Rate limiter for API calls respecting TPM and RPM limits."""
    
    def __init__(self, max_tokens_per_minute: int, max_requests_per_minute: int):
        self.max_tpm = max_tokens_per_minute
        self.max_rpm = max_requests_per_minute
        
        self.token_timestamps = deque()
        self.request_timestamps = deque()
        self._lock = threading.Lock()
    
    async def acquire(self, estimated_tokens: int):
        """Wait until we can make a request within rate limits."""
        while True:
            with self._lock:
                now = time.time()
                
                # Remove old timestamps (older than 1 minute)
                cutoff = now - 60
                while self.token_timestamps and self.token_timestamps[0][0] < cutoff:
                    self.token_timestamps.popleft()
                while self.request_timestamps and self.request_timestamps[0] < cutoff:
                    self.request_timestamps.popleft()
                
                # Calculate current usage
                current_tokens = sum(t[1] for t in self.token_timestamps)
                current_requests = len(self.request_timestamps)
                
                # Check if we can proceed (leave 10% buffer)
                if (current_tokens + estimated_tokens) <= self.max_tpm * 0.9 and \
                   (current_requests + 1) <= self.max_rpm * 0.9:
                    # Record this request
                    self.token_timestamps.append((now, estimated_tokens))
                    self.request_timestamps.append(now)
                    return
            
            # Wait before retrying
            await asyncio.sleep(0.5)


class Stage7ClusterRefiner:
    """
    Stage 7: Gather small theory names within predefined embedding-based clusters.
    
    Process:
    1. Load clusters from data/stage7/clusters_from_stage6_names.json
    2. For each cluster, separate into REFERENCE_LIST (>=reference_min_size papers) and RARE_LIST (<=rare_max_size papers)
    3. Extract key_concepts for theories (sample from stage0)
    4. Use LLM to normalize RARE_LIST theories
    """
    
    def __init__(self,
                 clusters_path: str = 'data/stage7/clusters_from_stage6_names.json',
                 stage6_path: str = 'output/stage6_consolidated_final_theories.json',
                 stage0_path: str = 'output/stage0_filtered_theories.json',
                 rare_max_size: int = 1,
                 reference_min_size: int = 3,
                 max_concurrent: int = 8,
                 retry_poor_grouping: bool = False):
        """Initialize Stage 7 refiner."""
        # Statistics
        self.stats = {
            'total_clusters': 0,
            'total_theories': 0,
            'reference_theories': 0,
            'rare_theories': 0,
            'grouped_theories': 0,
            'mapped_to_reference': 0,
            'retained_original': 0,
            'clusters_with_missing': 0,
            'total_retries': 0,
            'theories_recovered_by_retry': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
        
        self.clusters_path = Path(clusters_path)
        self.stage6_path = Path(stage6_path)
        self.stage0_path = Path(stage0_path)
        self.rare_max_size = rare_max_size
        self.reference_min_size = reference_min_size
        self.retry_poor_grouping = retry_poor_grouping
        
        # Load data
        self.clusters = self._load_clusters()
        self.stage6 = self._load_stage6()
        self.stage0 = self._load_stage0()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE7', os.getenv('USE_MODULE_FILTERING_LLM_STAGE5', 'azure'))
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            self.llm = AzureOpenAIClient()
        self.model = 'gpt-4.1-mini'
        
        # Concurrency control
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(
            max_tokens_per_minute=180000,
            max_requests_per_minute=450
        )
        self._lock = threading.Lock()

    def _load_clusters(self) -> Dict:
        """Load predefined clusters from JSON."""
        print(f"📂 Loading clusters from {self.clusters_path}...")
        
        with open(self.clusters_path, 'r') as f:
            clusters = json.load(f)
        
        self.stats['total_clusters'] = len(clusters)
        total_theories = sum(cluster['size'] for cluster in clusters.values())
        self.stats['total_theories'] = total_theories
        
        print(f"  ✓ Loaded {len(clusters)} clusters with {total_theories} total theories")
        return clusters

    def _load_stage6(self) -> Dict:
        """Load stage6 consolidated theories to map theory names to IDs."""
        print(f"📂 Loading stage6 data from {self.stage6_path}...")
        
        with open(self.stage6_path, 'r') as f:
            data = json.load(f)
        
        name_to_ids = {}
        name_to_count = {}
        entries = data.get('final_name_summary', [])
        
        for e in entries:
            name = e.get('final_name')
            ids = e.get('theory_ids', [])
            if name:
                name_to_ids[name] = ids
                name_to_count[name] = len(ids)
        
        print(f"  ✓ Loaded {len(name_to_ids)} unique theory names")
        return {
            'name_to_ids': name_to_ids,
            'name_to_count': name_to_count,
            'entries': entries,
            'metadata': data.get('metadata', {})
        }

    def _load_stage0(self) -> Dict:
        """Load theories with key_concepts from stage0."""
        print(f"📂 Loading theories from {self.stage0_path}...")
        
        with open(self.stage0_path, 'r') as f:
            data = json.load(f)
        
        theories = {}
        theory_list = data.get('theories', [])
        
        for theory in theory_list:
            theory_id = theory.get('theory_id', '')
            if theory_id:
                theories[theory_id] = {
                    'name': theory.get('name', ''),
                    'key_concepts': theory.get('key_concepts', []),
                    'evidence': theory.get('evidence', ''),
                    'doi': theory.get('doi', ''),
                    'pmid': theory.get('pmid', '')
                }
        
        print(f"  ✓ Loaded {len(theories)} theories with key concepts")
        return theories

    def _get_concepts_for_name(self, theory_name: str, sample_size: int = 2) -> List[Dict]:
        """
        Get key_concepts for a theory name by:
        1. Looking up theory_ids from stage6
        2. Randomly sampling up to sample_size theory_ids
        3. Extracting key_concepts from stage0
        
        Args:
            theory_name: The theory name to look up
            sample_size: Number of random samples to extract
            
        Returns:
            List of dictionaries with theory_id and key_concepts
        """
        theory_ids = self.stage6['name_to_ids'].get(theory_name, [])
        
        if not theory_ids:
            return []
        
        # Sample up to sample_size theory_ids
        sample_count = min(len(theory_ids), sample_size)
        sampled_ids = random.sample(theory_ids, sample_count)
        
        # Extract key_concepts for sampled IDs
        concepts_samples = []
        for theory_id in sampled_ids:
            if theory_id in self.stage0:
                concepts_samples.append({
                    'theory_id': theory_id,
                    'key_concepts': self.stage0[theory_id].get('key_concepts', [])
                })
        
        return concepts_samples

    def _create_prompt(self, cluster_id: str, reference_list: List[Dict], rare_list: List[Dict]) -> str:
        """
        Create prompt for LLM-based gathering of rare theories.
        
        Args:
            cluster_id: Cluster identifier
            reference_list: List of reference theories (paper_count >= reference_min_size) with concepts
            rare_list: List of rare theories (paper_count <= rare_max_size) with concepts
            
        Returns:
            Prompt string
        """
        # Build REFERENCE_LIST section
        ref_sections = []
        for ref in reference_list:
            ref_sections.append(f"\n# {ref['theory_name']}\n(papers: {ref['paper_count']})")
            for idx, sample in enumerate(ref.get('concepts_samples', []), 1):
                ref_sections.append(f"  - sample {idx}:")
                for concept in sample.get('key_concepts', [])[:3]:
                    ref_sections.append(f"    * {concept.get('concept','')}: {concept.get('description','')}")
        ref_text = "\n".join(ref_sections) if ref_sections else "(no reference names in this cluster)"

        # Build INPUT_LIST section
        rare_sections = []
        rare_names = []
        for rare in rare_list:
            rare_names.append(rare['theory_name'])
            rare_sections.append(f"\n# {rare['theory_name']}\n(papers: {rare['paper_count']})")
            for idx, sample in enumerate(rare.get('concepts_samples', []), 1):
                rare_sections.append(f"  - sample {idx}:")
                for concept in sample.get('key_concepts', [])[:3]:
                    rare_sections.append(f"    * {concept.get('concept','')}: {concept.get('description','')}")
        rare_text = "\n".join(rare_sections)

        prompt = f"""# TASK
Gather small theory names (INPUT_LIST) under broader names within this cluster when mechanisms are even moderately related. Prefer mapping to existing REFERENCE_LIST names. Create at most 1-2 generalized names only if necessary.

# CONTEXT
- REFERENCE_LIST: Well-established names in this cluster (do not change these names)
- INPUT_LIST: Small names (size <= {self.rare_max_size}) to be gathered

# GOAL
- Reduce the number of singleton/small names by assigning them to broader names
- Prefer mapping to REFERENCE_LIST when reasonable
- If mechanisms differ, allow 1-2 new generalized names to gather several small names together

# REFERENCE_LIST
{ref_text}

# INPUT_LIST (TO GATHER)
{rare_text}
Thus, names to gather are: {'; '.join(rare_names)}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown):
{{
  "normalizations": [
    {{
      "original_name": "Name from INPUT_LIST",
      "strategy": "map" | "assign_common" | "retain",
      "normalized_name": "Reference name or generalized name or original name",
      "decision_confidence": 0.0,
      "reasoning": "short"
    }}
  ]
}}

# RULES
- Prefer mapping to REFERENCE_LIST names over creating new names
- Avoid creating unique names for each INPUT item
- If creating a new generalized name, ensure multiple INPUT items share it
- Use "aging" not "ageing"
- Return all INPUT names in the output
"""
        return prompt

    async def _retry_cluster(self, cluster_id: str, reference_list: List[Dict],
                            rare_list: List[Dict], retry_reason: str) -> Optional[Dict]:
        """
        Retry processing for entire cluster.
        
        Args:
            cluster_id: Cluster identifier
            reference_list: Reference theories
            rare_list: Rare theories list
            retry_reason: Reason for retry (for logging)
            
        Returns:
            Dictionary with normalizations or None if retry fails
        """
        print(f"  🔄 Retrying cluster {cluster_id} due to: {retry_reason}")
        
        # Create prompt with modified instructions for retry
        prompt = self._create_prompt(cluster_id, reference_list, rare_list)
        
        # Modify prompt for retry with stronger instructions
        if "poor grouping" in retry_reason.lower():
            prompt = prompt.replace(
                "# GOAL",
                "# CRITICAL GOAL (RETRY - PREVIOUS ATTEMPT FAILED TO GROUP)"
            )
            prompt = prompt.replace(
                "when reasonable",
                "REQUIRED"
            )
        
        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 3000
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Call LLM
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in aging biology and theory classification."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=3000
                )
            )
            
            # Track token usage
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
            try:
                data = json.loads(response_text)
                normalizations = data.get('normalizations', [])
                return {'normalizations': normalizations}
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error in retry for cluster {cluster_id}: {e}")
                return None
                
        except Exception as e:
            print(f"  ❌ LLM error in retry for cluster {cluster_id}: {e}")
            return None
    
    async def _process_cluster_async(self, cluster_id: str, cluster_data: Dict) -> Dict:
        """
        Process a single cluster asynchronously.
        
        Args:
            cluster_id: Cluster identifier
            cluster_data: Cluster data with members
            
        Returns:
            Dictionary with normalization results
        """
        members = cluster_data.get('members', [])
        
        # Separate into REFERENCE_LIST and RARE_LIST
        reference_list = []
        rare_list = []
        
        for member in members:
            theory_name = member['theory_name']
            paper_count = member.get('paper_count', self.stage6['name_to_count'].get(theory_name, 0))
            
            # Get concepts for this theory
            concepts_samples = self._get_concepts_for_name(theory_name, sample_size=2)
            item = {
                'theory_name': theory_name,
                'paper_count': paper_count,
                'concepts_samples': concepts_samples
            }
            
            if paper_count >= self.reference_min_size:
                reference_list.append(item)
            elif paper_count <= self.rare_max_size:
                rare_list.append(item)
        
        # Update statistics
        with self._lock:
            self.stats['reference_theories'] += len(reference_list)
            self.stats['rare_theories'] += len(rare_list)
        
        # If no rare theories, skip LLM call
        if not rare_list:
            return {
                'cluster_id': cluster_id,
                'reference_list': reference_list,
                'rare_list': [],
                'normalizations': []
            }
        
        # Create prompt
        prompt = self._create_prompt(cluster_id, reference_list, rare_list)
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt) // 4 + 3000
        
        # Wait for rate limiter
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Call LLM
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in aging biology and theory classification."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=3000
                )
            )
            
            # Track token usage
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
            try:
                data = json.loads(response_text)
                normalizations = data.get('normalizations', [])
                
                # Validate: check all input theories are present in output
                input_names = {theory['theory_name'] for theory in rare_list}
                output_names = {norm.get('original_name') for norm in normalizations}
                missing_names = input_names - output_names
                
                # Determine if cluster needs retry
                needs_retry = False
                retry_reason = ""
                
                if missing_names:
                    needs_retry = True
                    retry_reason = f"missing {len(missing_names)} theories"
                    print(f"  ⚠️  WARNING: {len(missing_names)} theories missing from output for cluster {cluster_id}")
                    for name in missing_names:
                        print(f"    - Missing: {name}")
                
                # Validate grouping: check if assign_common actually groups theories
                if not needs_retry:  # Only check grouping if no missing theories
                    assign_common_norms = [n for n in normalizations if n.get('strategy') == 'assign_common']
                    if assign_common_norms:
                        normalized_names = [n.get('normalized_name') for n in assign_common_norms]
                        unique_names = set(normalized_names)
                        # Check if grouping is poor (all unique names)
                        if len(unique_names) == len(normalized_names) and len(normalized_names) > 3:
                            if self.retry_poor_grouping:
                                needs_retry = True
                                retry_reason = "poor grouping"
                            print(f"  ⚠️  WARNING: LLM used 'assign_common' but created unique names for cluster {cluster_id}")
                            print(f"    Expected: Multiple theories sharing 2-3 common names")
                            print(f"    Got: {len(unique_names)} unique names for {len(normalized_names)} theories")
                
                # Retry entire cluster if needed
                if needs_retry:
                    with self._lock:
                        self.stats['clusters_with_missing'] += 1
                        self.stats['total_retries'] += 1
                    
                    retry_result = await self._retry_cluster(cluster_id, reference_list, rare_list, retry_reason)
                    
                    if retry_result and retry_result.get('normalizations'):
                        retry_normalizations = retry_result['normalizations']
                        
                        # Validate retry result
                        retry_output_names = {norm.get('original_name') for norm in retry_normalizations}
                        retry_missing = input_names - retry_output_names
                        
                        if not retry_missing:
                            print(f"  ✓ Retry successful: all theories present")
                            normalizations = retry_normalizations
                            
                            # Track recovered theories
                            with self._lock:
                                self.stats['theories_recovered_by_retry'] += len(input_names)
                        else:
                            print(f"  ⚠️  Retry still missing {len(retry_missing)} theories, keeping original and filling gaps")
                            # Add missing theories with retain strategy
                            for name in missing_names:
                                normalizations.append({
                                    'original_name': name,
                                    'strategy': 'retain',
                                    'normalized_name': name,
                                    'decision_confidence': 0.0,
                                    'reasoning': 'Missing from LLM output after cluster retry - retained original name'
                                })
                    else:
                        print(f"  ⚠️  Retry failed, keeping original and filling gaps")
                        # Add missing theories with retain strategy
                        for name in missing_names:
                            normalizations.append({
                                'original_name': name,
                                'strategy': 'retain',
                                'normalized_name': name,
                                'decision_confidence': 0.0,
                                'reasoning': 'Missing from LLM output after cluster retry - retained original name'
                            })
                
                # Update strategy statistics
                for norm in normalizations:
                    strategy = norm.get('strategy', 'retain')
                    with self._lock:
                        if strategy == 'assign_common':
                            self.stats['grouped_theories'] += 1
                        elif strategy == 'map':
                            self.stats['mapped_to_reference'] += 1
                        elif strategy == 'retain':
                            self.stats['retained_original'] += 1
                
                return {
                    'cluster_id': cluster_id,
                    'reference_list': reference_list,
                    'rare_list': rare_list,
                    'normalizations': normalizations,
                    'had_missing': len(missing_names) > 0
                }
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error for cluster {cluster_id}: {e}")
                
                # Retry entire cluster on JSON parsing error
                with self._lock:
                    self.stats['clusters_with_missing'] += 1
                    self.stats['total_retries'] += 1
                
                retry_result = await self._retry_cluster(cluster_id, reference_list, rare_list, "JSON parsing error")
                
                if retry_result and retry_result.get('normalizations'):
                    print(f"  ✓ Retry successful after JSON error")
                    normalizations = retry_result['normalizations']
                    
                    # Validate retry result
                    input_names = {theory['theory_name'] for theory in rare_list}
                    retry_output_names = {norm.get('original_name') for norm in normalizations}
                    retry_missing = input_names - retry_output_names
                    
                    if retry_missing:
                        # Fill gaps with retain strategy
                        for name in retry_missing:
                            normalizations.append({
                                'original_name': name,
                                'strategy': 'retain',
                                'normalized_name': name,
                                'decision_confidence': 0.0,
                                'reasoning': 'Missing from LLM output after JSON error retry - retained original name'
                            })
                    
                    # Update strategy statistics
                    for norm in normalizations:
                        strategy = norm.get('strategy', 'retain')
                        with self._lock:
                            if strategy == 'assign_common':
                                self.stats['grouped_theories'] += 1
                            elif strategy == 'map':
                                self.stats['mapped_to_reference'] += 1
                            elif strategy == 'retain':
                                self.stats['retained_original'] += 1
                    
                    return {
                        'cluster_id': cluster_id,
                        'reference_list': reference_list,
                        'rare_list': rare_list,
                        'normalizations': normalizations,
                        'had_missing': len(retry_missing) > 0
                    }
                else:
                    # Retry failed, return empty normalizations with error
                    return {
                        'cluster_id': cluster_id,
                        'reference_list': reference_list,
                        'rare_list': rare_list,
                        'normalizations': [],
                        'error': f'JSON parsing error: {str(e)} - retry also failed'
                    }
                
        except Exception as e:
            print(f"  ❌ LLM error for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'reference_list': reference_list,
                'rare_list': rare_list,
                'normalizations': [],
                'error': f'LLM error: {str(e)}'
            }

    async def _process_all_clusters_async(self, start_cluster: int, output_path: str, all_results: List[Dict]) -> List[Dict]:
        """Process all clusters asynchronously with controlled concurrency and checkpointing."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_cluster_with_semaphore(cluster_id, cluster_data, cluster_num, total_clusters):
            async with semaphore:
                return await self._process_cluster_async(cluster_id, cluster_data)
        
        # Convert clusters dict to list for indexing
        cluster_items = list(self.clusters.items())
        total_clusters = len(cluster_items)
        
        # Process clusters in groups
        group_size = 50  # Process 50 clusters at a time
        
        with tqdm(total=total_clusters, initial=start_cluster, desc="Processing clusters", unit="cluster") as pbar:
            for group_start in range(start_cluster, total_clusters, group_size):
                group_end = min(group_start + group_size, total_clusters)
                
                # Create tasks for this group
                tasks = []
                for i in range(group_start, group_end):
                    cluster_id, cluster_data = cluster_items[i]
                    cluster_num = i + 1
                    tasks.append(process_cluster_with_semaphore(cluster_id, cluster_data, cluster_num, total_clusters))
                
                # Process group concurrently
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results and count successes
                successful_clusters = 0
                for result in group_results:
                    if isinstance(result, Exception):
                        print(f"\n⚠️  Cluster failed with exception: {result}")
                    elif result:
                        all_results.append(result)
                        successful_clusters += 1
                        
                        # Save after each batch completes
                        current_cluster = len(all_results) + start_cluster
                        is_final = (current_cluster == total_clusters)
                        self._save_checkpoint(
                            all_results,
                            output_path,
                            clusters_completed=current_cluster,
                            is_final=is_final
                        )
                
                # Update progress bar once for the entire group
                clusters_processed_in_group = group_end - group_start
                pbar.update(clusters_processed_in_group)
                pbar.set_postfix({'processed': len(all_results), 'failed': len(group_results) - successful_clusters})
        
        return all_results
    
    def _save_checkpoint(self, all_results: List[Dict], output_path: str,
                        clusters_completed: int = 0, is_final: bool = True):
        """Save checkpoint results (intermediate or final).
        
        Args:
            all_results: List of cluster results
            output_path: Path to save output
            clusters_completed: Number of clusters completed (for checkpoints)
            is_final: Whether this is the final save
        """
        # Extract only normalizations to save space
        compact_results = []
        for result in all_results:
            compact_results.append({
                'cluster_id': result['cluster_id'],
                'normalizations': result.get('normalizations', []),
                'error': result.get('error')  # Keep error if present
            })
        
        # Compile output
        output = {
            'metadata': {
                'stage': 'stage7_cluster_refinement',
                'status': 'complete' if is_final else 'in_progress',
                'clusters_completed': clusters_completed,
                'total_clusters': self.stats['total_clusters'],
                'total_theories': self.stats['total_theories'],
                'reference_theories': self.stats['reference_theories'],
                'rare_theories': self.stats['rare_theories'],
                'grouped_theories': self.stats['grouped_theories'],
                'mapped_to_reference': self.stats['mapped_to_reference'],
                'retained_original': self.stats['retained_original'],
                'clusters_with_missing': self.stats['clusters_with_missing'],
                'total_retries': self.stats['total_retries'],
                'theories_recovered_by_retry': self.stats['theories_recovered_by_retry'],
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_cost': round(self.stats['total_cost'], 4),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clusters': compact_results
        }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if not is_final:
            # Only print every 5 saves to avoid spam
            if len(all_results) % 5 == 0:
                print(f"    💾 Checkpoint saved: {len(all_results)} clusters processed")
    
    def _load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load checkpoint from previous run.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Tuple of (all_results, clusters_completed, stats) or (None, 0, None) if no valid checkpoint
        """
        if not Path(checkpoint_path).exists():
            return None, 0, None
        
        print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        
        # Check if it's complete
        if metadata.get('status') == 'complete':
            print("  ⚠️  Checkpoint is already complete!")
            user_input = input("  Do you want to re-run from scratch? (y/n): ")
            if user_input.lower() != 'y':
                return None, 0, None
            else:
                return None, 0, None
        
        all_results = data.get('clusters', [])
        clusters_completed = metadata.get('clusters_completed', 0)
        
        # Restore stats
        stats = {
            'total_clusters': metadata.get('total_clusters', 0),
            'total_theories': metadata.get('total_theories', 0),
            'reference_theories': metadata.get('reference_theories', 0),
            'rare_theories': metadata.get('rare_theories', 0),
            'grouped_theories': metadata.get('grouped_theories', 0),
            'mapped_to_reference': metadata.get('mapped_to_reference', 0),
            'retained_original': metadata.get('retained_original', 0),
            'clusters_with_missing': metadata.get('clusters_with_missing', 0),
            'total_retries': metadata.get('total_retries', 0),
            'theories_recovered_by_retry': metadata.get('theories_recovered_by_retry', 0),
            'total_input_tokens': metadata.get('total_input_tokens', 0),
            'total_output_tokens': metadata.get('total_output_tokens', 0),
            'total_cost': metadata.get('total_cost', 0.0)
        }
        
        print(f"  ✓ Loaded checkpoint: {clusters_completed} clusters completed")
        print(f"  ✓ Resuming from cluster {clusters_completed + 1}")
        
        return all_results, clusters_completed, stats
    
    def process_clusters(self, output_path: str = 'output/stage7_name_normalizations.json',
                        resume_from_checkpoint: bool = False):
        """Main processing function.
        
        Args:
            output_path: Path to save output
            resume_from_checkpoint: If True, resume from checkpoint
        """
        print("\n" + "="*80)
        print("STAGE 7: CLUSTER-BASED THEORY NAME GATHERING")
        print("="*80)
        
        # Check for checkpoint
        all_results = []
        start_cluster = 0
        
        if resume_from_checkpoint:
            checkpoint_results, clusters_completed, checkpoint_stats = self._load_checkpoint(output_path)
            if checkpoint_results is not None:
                all_results = checkpoint_results
                start_cluster = clusters_completed
                # Restore stats
                for key, value in checkpoint_stats.items():
                    self.stats[key] = value
        
        # Process clusters
        print(f"\n🚀 Starting processing from cluster {start_cluster + 1}/{self.stats['total_clusters']}...")
        print(f"   Max concurrent: {self.max_concurrent}")
        print(f"   Rare max size: {self.rare_max_size}")
        print(f"   Reference min size: {self.reference_min_size}\n")
        
        # Run async processing
        all_results = asyncio.run(self._process_all_clusters_async(start_cluster, output_path, all_results))
        
        # Print final statistics
        print("\n" + "="*80)
        print("🎉 PROCESSING COMPLETE")
        print("="*80)
        print(f"Total clusters: {self.stats['total_clusters']}")
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"  - Reference theories: {self.stats['reference_theories']}")
        print(f"  - Rare theories: {self.stats['rare_theories']}")
        print(f"\nNormalization strategies:")
        print(f"  - Grouped: {self.stats['grouped_theories']}")
        print(f"  - Mapped to reference: {self.stats['mapped_to_reference']}")
        print(f"  - Retained original: {self.stats['retained_original']}")
        print(f"\nRetries:")
        print(f"  - Clusters with missing: {self.stats['clusters_with_missing']}")
        print(f"  - Total retries: {self.stats['total_retries']}")
        print(f"  - Theories recovered: {self.stats['theories_recovered_by_retry']}")
        print(f"\nToken usage:")
        print(f"  - Input tokens: {self.stats['total_input_tokens']:,}")
        print(f"  - Output tokens: {self.stats['total_output_tokens']:,}")
        print(f"  - Total cost: ${self.stats['total_cost']:.4f}")
        print(f"\n💾 Results saved to: {output_path}")
        print("="*80 + "\n")
        
        return all_results
    
    def save_results(self, output: Dict, output_path: str = 'output/stage7_name_normalizations.json'):
        """Save final results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved results to {output_path}")
    
    def generate_consolidated_output(self, normalizations_path: str, output_consolidated: str):
        """Generate consolidated output from normalizations."""
        print(f"\n🔄 Generating consolidated output...")
        
        # Load normalizations
        with open(normalizations_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('clusters', [])
        
        # Build mapping from normalizations
        mapping: Dict[str, str] = {}
        for res in results:
            for n in res.get('normalizations', []):
                orig = n.get('original_name')
                norm = n.get('normalized_name') or orig
                if orig:
                    mapping[orig] = norm
        
        # Apply mapping to stage6 entries
        stage6_entries = self.stage6['entries']
        grouped: Dict[str, Dict] = {}
        
        for e in stage6_entries:
            old_name = e.get('final_name')
            new_name = mapping.get(old_name, old_name)
            ids = e.get('theory_ids', [])
            
            if new_name not in grouped:
                grouped[new_name] = {
                    'final_name': new_name,
                    'original_names_count': 0,
                    'original_names': set(),
                    'total_papers': 0,
                    'theory_ids': [],
                }
            
            grouped[new_name]['original_names'].add(old_name)
            grouped[new_name]['total_papers'] += e.get('theory_ids_count', len(ids))
            grouped[new_name]['theory_ids'].extend(ids)
        
        # Build final list
        final_list: List[Dict] = []
        for name, g in grouped.items():
            ids = list(dict.fromkeys(g['theory_ids']))
            final_list.append({
                'final_name': name,
                'original_names_count': len(g['original_names']),
                'original_names': sorted(list(g['original_names'])),
                'total_papers': len(ids),
                'theory_ids_count': len(ids),
                'theory_ids': ids,
            })
        
        final_list.sort(key=lambda x: x['total_papers'], reverse=True)
        
        # Create consolidated output
        consolidated = {
            'metadata': {
                'source_stage6': str(self.stage6_path),
                'clusters_source': str(self.clusters_path),
                'rare_max_size': self.rare_max_size,
                'reference_min_size': self.reference_min_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'final_name_summary': final_list,
        }
        
        # Save consolidated output
        output_path = Path(output_consolidated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        print(f"✓ Consolidated output saved to: {output_consolidated}")
        print(f"✓ Total final theories: {len(final_list)}")


def main():
    """Main entry point."""
    import argparse
    
    p = argparse.ArgumentParser(description='Stage 7: Gather small names after Stage 6')
    p.add_argument('--clusters', default='data/stage7/clusters_from_stage6_names.json',
                   help='Path to clusters JSON file')
    p.add_argument('--stage6', default='output/stage6_consolidated_final_theories.json',
                   help='Path to stage6 consolidated theories')
    p.add_argument('--stage0', default='output/stage0_filtered_theories.json',
                   help='Path to stage0 filtered theories')
    p.add_argument('--rare-max-size', type=int, default=1,
                   help='Maximum paper count for rare theories')
    p.add_argument('--reference-min-size', type=int, default=3,
                   help='Minimum paper count for reference theories')
    p.add_argument('--max-concurrent', type=int, default=8,
                   help='Maximum concurrent LLM requests')
    p.add_argument('--retry-poor-grouping', action='store_true',
                   help='Retry clusters with poor grouping')
    p.add_argument('--resume', action='store_true',
                   help='Resume from checkpoint if available')
    p.add_argument('--out-norms', default='output/stage7_name_normalizations.json',
                   help='Output path for normalizations')
    p.add_argument('--out-consolidated', default='output/stage7_consolidated_final_theories.json',
                   help='Output path for consolidated theories')
    
    args = p.parse_args()
    
    # Initialize refiner
    refiner = Stage7ClusterRefiner(
        clusters_path=args.clusters,
        stage6_path=args.stage6,
        stage0_path=args.stage0,
        rare_max_size=args.rare_max_size,
        reference_min_size=args.reference_min_size,
        max_concurrent=args.max_concurrent,
        retry_poor_grouping=args.retry_poor_grouping
    )
    
    # Process clusters
    refiner.process_clusters(
        output_path=args.out_norms,
        resume_from_checkpoint=args.resume
    )
    
    # Generate consolidated output
    refiner.generate_consolidated_output(
        normalizations_path=args.out_norms,
        output_consolidated=args.out_consolidated
    )


if __name__ == '__main__':
    main()
