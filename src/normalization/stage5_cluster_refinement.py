"""
Stage 5: Cluster-Based Theory Refinement

This stage refines theory names within predefined embedding-based clusters:
1. Load predefined clusters from data/clusters_with_paper_counts.json
2. For each cluster, separate theories into REFERENCE_LIST (paper_count >= 6) and RARE_LIST (paper_count < 6)
3. For REFERENCE_LIST theories, sample 3 random entries and extract their key_concepts
4. Use LLM to normalize RARE_LIST theories:
   - Preferably group into 2-3 distinct names different from REFERENCE_LIST
   - If not possible, map to REFERENCE_LIST
   - If neither works, retain original name

Input:
- data/clusters_with_paper_counts.json (predefined clusters)
- output/theory_tracking_report.json (theory name to ID mapping)
- output/stage0_filtered_theories.json (key concepts)

Output: output/stage5_cluster_refined_theories.json
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


class Stage5ClusterRefiner:
    """
    Stage 5: Refine theory names within predefined embedding-based clusters.
    
    Process:
    1. Load clusters from data/clusters_with_paper_counts.json
    2. For each cluster, separate into REFERENCE_LIST (>=6 papers) and RARE_LIST (<6 papers)
    3. Extract key_concepts for REFERENCE_LIST theories (3 random samples each)
    4. Use LLM to normalize RARE_LIST theories
    """
    
    def __init__(self,
                 clusters_path: str = 'data/clustering_data/clusters_with_paper_counts.json',
                 tracker_path: str = 'output/theory_tracking_report.json',
                 stage0_path: str = 'output/stage0_filtered_theories.json',
                 max_concurrent: int = 10,
                 retry_poor_grouping: bool = False):
        """Initialize Stage 5 refiner."""
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
        self.tracker_path = Path(tracker_path)
        self.stage0_path = Path(stage0_path)
        self.retry_poor_grouping = retry_poor_grouping
        
        # Load data
        self.clusters = self._load_clusters()
        self.theory_tracker = self._load_theory_tracker()
        self.stage0_theories = self._load_stage0_theories()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE5', 'azure')
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
    
    def _load_theory_tracker(self) -> Dict:
        """Load theory tracker to map theory names to IDs."""
        print(f"📂 Loading theory tracker from {self.tracker_path}...")
        
        with open(self.tracker_path, 'r') as f:
            tracker = json.load(f)
        
        theory_lineage = tracker.get('theory_lineage', {})
        
        # Create mapping: final_name_normalized -> list of theory_ids
        name_to_ids = defaultdict(list)
        for theory_id, lineage in theory_lineage.items():
            final_name = lineage.get('final_name_normalized')
            if final_name:
                name_to_ids[final_name].append(theory_id)
        
        print(f"  ✓ Loaded {len(name_to_ids)} unique theory names")
        return dict(name_to_ids)
    
    def _load_stage0_theories(self) -> Dict:
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
    
    def _get_concepts_for_theory_name(self, theory_name: str, sample_size: int = 3) -> List[Dict]:
        """
        Get key_concepts for a theory name by:
        1. Looking up theory_ids from theory_tracker
        2. Randomly sampling up to sample_size theory_ids
        3. Extracting key_concepts from stage0_theories
        
        Args:
            theory_name: The theory name to look up
            sample_size: Number of random samples to extract
            
        Returns:
            List of dictionaries with theory_id and key_concepts
        """
        theory_ids = self.theory_tracker.get(theory_name, [])
        
        if not theory_ids:
            return []
        
        # Sample up to sample_size random theory_ids
        sample_count = min(len(theory_ids), sample_size)
        sampled_ids = random.sample(theory_ids, sample_count)
        
        # Extract key_concepts
        concepts_list = []
        for theory_id in sampled_ids:
            if theory_id in self.stage0_theories:
                theory_data = self.stage0_theories[theory_id]
                concepts_list.append({
                    'theory_id': theory_id,
                    'key_concepts': theory_data.get('key_concepts', [])
                })
        
        return concepts_list
    
    def _create_refinement_prompt(self, cluster_id: str, reference_list: List[Dict], 
                                  rare_list: List[Dict]) -> str:
        """
        Create prompt for LLM-based refinement of rare theories.
        
        Args:
            cluster_id: Cluster identifier
            reference_list: List of reference theories (paper_count >= 6) with concepts
            rare_list: List of rare theories (paper_count < 6) with concepts
            
        Returns:
            Prompt string
        """
        # Build REFERENCE_LIST section
        reference_section = []
        for ref in reference_list:
            theory_name = ref['theory_name']
            paper_count = ref['paper_count']
            concepts_samples = ref.get('concepts_samples', [])
            
            reference_section.append(f"\n# {theory_name}\n (papers: {paper_count})")
            
            if concepts_samples:
                for idx, sample in enumerate(concepts_samples, 1):
                    reference_section.append(f"\n### V. {idx}:")
                    key_concepts = sample.get('key_concepts', [])
                    if key_concepts:
                        for concept in key_concepts[:3]:
                            concept_name = concept.get('concept', '')
                            concept_desc = concept.get('description', '')
                            reference_section.append(f"  - {concept_name}: {concept_desc}")
                    else:
                        reference_section.append("  (No key concepts available)")
            else:
                reference_section.append("  (No concept samples available)")
        
        reference_text = "\n".join(reference_section) if reference_section else "(No reference theories in this cluster)"
        
        # Build RARE_LIST section
        rare_section = []
        theories_names_only = []
        for rare in rare_list:
            theory_name = rare['theory_name']
            theories_names_only.append(theory_name)
            paper_count = rare['paper_count']
            concepts_samples = rare.get('concepts_samples', [])
            
            rare_section.append(f"\n# {theory_name}\n(papers: {paper_count})")
            
            if concepts_samples:
                for idx, sample in enumerate(concepts_samples, 1):
                    rare_section.append(f"\n# V. {idx}:")
                    key_concepts = sample.get('key_concepts', [])
                    if key_concepts:
                        for concept in key_concepts[:3]:
                            concept_name = concept.get('concept', '')
                            concept_desc = concept.get('description', '')
                            rare_section.append(f"  - {concept_name}: {concept_desc}")
                    else:
                        rare_section.append("  (No key concepts available)")
            else:
                rare_section.append("  (No concept samples available)")
        
        rare_text = "\n".join(rare_section)
        
        prompt = f"""# TASK
Generalize and group aging theories from INPUT_LIST under 2-3 SHARED common theories names, when possible. prioritize broader groupings to ensure multiple theories share the same normalized_name where possible, even if involves meaning generalization

# CONTEXT
- REFERENCE_LIST: Well-established theories (for reference only)
- INPUT_LIST: Rare theories that need to be grouped under common theories names

# GOAL
Your goal is to GROUP multiple theories from INPUT_LIST under 2-3 SHARED common theories names, if it is even a bit possible. This means MULTIPLE theories should have the SAME normalized_name.
DO NOT create unique names for each theory - that defeats the purpose of grouping!
But assume that clustering might be not percise and some theories might have completely different mechanisms.
Assigning a more general name, than original, in order to group multiple theories is much more preferable than creating unique names for each theory. You may generalize names if the meaning is not changed. Generalize theory names as needed, as long as the core mechanism is preserved.
ON the other hand, you can make a original names nore specific, if it helps to group multiple theories.

# INSTRUCTIONS
1. Analyze ALL theories in INPUT_LIST together 
2. Identify if there are common mechanistic themes across these theories (depends on the cluster, but usually 2-3)
3. Create generalized theories names that capture these themes
4. Assign MULTIPLE theories to each of these common names, if it is possible.

# STRATEGIES

**HIGHLY PREFERRED: "assign_common"**
- Assign MULTIPLE theories from INPUT_LIST to the SAME common name
- Create only 2-3 distinct common names for the entire INPUT_LIST
- Each common name should be:
  - DISTINCT from REFERENCE_LIST names
  - Clear, generalizable, and mechanism-based
  - Shared by MULTIPLE theories (not unique per theory)
- Example: If you have 10 theories, you might assign 4 to "Common Theory Name A", 4 to "Common NTheory ame B", and 2 to "Common Theory Name C"

**FALLBACK: "map"**
- If mechanisms are very similar to a reference theory, map to a name from REFERENCE_LIST
- Provide the exact reference_name

**NOT RECOMMENDED: "retain"**
- Retain the original name
- Only if mechanisms are too dissimilar from both rare theories and reference theories
- Extremely not recommended for theories with paper_count <3

# NAMING RULES
- It should be an aging theory name
- Avoid excessively specific names with many details
- Generalize based on mechanisms, not specific diseases/organs/pathways
- If name ends with "Theory", don't add "of Aging", just Theory.
- Use "aging" not "ageing"
- Never create composite names that combine multiple theories.

# REFERENCE_LIST
{reference_text}

# RARE_LIST  - TO BE NORMALIZED
{rare_text}
Thus, names to be normalized are: {("; ").join(theories_names_only)}

# EXAMPLE OF CORRECT GROUPING
If INPUT_LIST has theories about mitochondrial damage mechanisms:
- "Mitochondrial Protein Cross-Linking Theory" → normalized_name: "Mitochondrial Damage Mechanisms"
- "Aberrant Mitochondrial RNA Theory" → normalized_name: "Mitochondrial Damage Mechanisms"  
- "Mitochondrial Mutagenesis Theory" → normalized_name: "Mitochondrial Damage Mechanisms"
Notice: All three theories share the SAME normalized_name (as they share similar mechanisms)!

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):

{{  
  "normalizations": [
    {{
      "original_name": "Theory Name",
      "strategy": "assign_common" | "map" | "retain",
      "normalized_name": "New standardized name/ mapped reference name/ or original name",
      "decision_confidence": 0.0-1.0,
      "reasoning": "Brief explanation of the decision"
    }},
    ...
  ]
}}

# CRITICAL REQUIREMENTS
- You must include ALL {len(rare_list)} rare theories in the output
- Prefer grouping over mapping
"""
        
        return prompt
    
    async def _retry_cluster(self, cluster_id: str, reference_list: List[Dict],
                            rare_list: List[Dict], retry_reason: str) -> Dict:
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
        print(f"  🔄 Retrying ENTIRE cluster {cluster_id} (reason: {retry_reason})...")
        
        # Create prompt for entire cluster
        prompt = self._create_refinement_prompt(cluster_id, reference_list, rare_list)
        
        # Modify prompt for retry with stronger instructions
        if "poor grouping" in retry_reason.lower():
            prompt = prompt.replace(
                "# GOAL",
                "# CRITICAL GOAL (RETRY - PREVIOUS ATTEMPT FAILED TO GROUP)"
            )
            prompt = prompt.replace(
                "when possible",
                "REQUIRED"
            )
        
        # Estimate tokens
        estimated_tokens = len(prompt) // 4 + 4000
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
                    max_tokens=4000
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
            paper_count = member['paper_count']
            
            # Get concepts for this theory
            if paper_count >= 6:
                # REFERENCE_LIST: sample 3 random entries
                concepts_samples = self._get_concepts_for_theory_name(theory_name, sample_size=2)
                reference_list.append({
                    'theory_name': theory_name,
                    'paper_count': paper_count,
                    'concepts_samples': concepts_samples
                })
            else:
                # RARE_LIST: get all available concepts (up to 3)
                concepts_samples = self._get_concepts_for_theory_name(theory_name, sample_size=2)
                rare_list.append({
                    'theory_name': theory_name,
                    'paper_count': paper_count,
                    'concepts_samples': concepts_samples
                })
        
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
        prompt = self._create_refinement_prompt(cluster_id, reference_list, rare_list)
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt) // 4 + 4000
        
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
                    max_tokens=4000
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
        # Note: Checkpoint is saved after EACH cluster completes (not every 5)
        
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
                        
                        # Save after each batch (cluster) completes
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
                'stage': 'stage5_cluster_refinement',
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
        
        print(f"  ✓ Loaded checkpoint:")
        print(f"    - Clusters completed: {clusters_completed}/{stats['total_clusters']}")
        print(f"    - Results: {len(all_results)}")
        print(f"    - Cost so far: ${stats['total_cost']:.4f}")
        
        return all_results, clusters_completed, stats
    
    def process_clusters(self, output_path: str = 'output/stage5_cluster_refined_theories.json',
                        resume_from_checkpoint: bool = False) -> Dict:
        """Main processing function.
        
        Args:
            output_path: Path to save output
            resume_from_checkpoint: If True, resume from checkpoint
        """
        print("\n" + "="*80)
        print("STAGE 5: CLUSTER-BASED THEORY REFINEMENT")
        print("="*80)
        
        # Try to load checkpoint if requested
        start_cluster = 0
        all_results = []
        
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint(output_path)
            if checkpoint_data[0] is None and checkpoint_data[1] == 0 and checkpoint_data[2] is None:
                # User declined to re-run complete checkpoint
                print("\n❌ Exiting without changes.")
                return None
            if checkpoint_data[0] is not None:
                all_results, start_cluster, loaded_stats = checkpoint_data
                # Restore stats
                self.stats.update(loaded_stats)
                print(f"\n🔄 Resuming from cluster {start_cluster + 1}")
        
        # Process all clusters
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            all_results = loop.run_until_complete(
                self._process_all_clusters_async(start_cluster, output_path, all_results)
            )
        finally:
            loop.close()
        
        # Compile final output (compact version with only normalizations)
        compact_results = []
        for result in all_results:
            compact_results.append({
                'cluster_id': result['cluster_id'],
                'normalizations': result.get('normalizations', []),
                'error': result.get('error')  # Keep error if present
            })
        
        output = {
            'metadata': {
                'stage': 'stage5_cluster_refinement',
                'status': 'complete',
                'clusters_completed': len(all_results),
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
        
        return output
    
    def save_results(self, output: Dict, output_path: str = 'output/stage5_cluster_refined_theories.json'):
        """Save final results to JSON file."""
        if output is None:
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        metadata = output['metadata']
        print(f"Total clusters processed: {metadata['clusters_completed']}/{metadata['total_clusters']}")
        print(f"Total theories: {metadata['total_theories']}")
        print(f"  - Reference theories (≥6 papers): {metadata['reference_theories']}")
        print(f"  - Rare theories (<6 papers): {metadata['rare_theories']}")
        print(f"\nNormalization strategies:")
        print(f"  - Grouped: {metadata['grouped_theories']}")
        print(f"  - Mapped to reference: {metadata['mapped_to_reference']}")
        print(f"  - Retained original: {metadata['retained_original']}")
        print(f"\nValidation & Retries:")
        print(f"  - Clusters with missing theories: {metadata['clusters_with_missing']}")
        print(f"  - Total retry attempts: {metadata['total_retries']}")
        print(f"  - Theories recovered by retry: {metadata['theories_recovered_by_retry']}")
        print(f"\nToken usage:")
        print(f"  - Input tokens: {metadata['total_input_tokens']:,}")
        print(f"  - Output tokens: {metadata['total_output_tokens']:,}")
        print(f"  - Total cost: ${metadata['total_cost']:.4f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 5: Cluster-Based Theory Refinement')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--output', type=str, default='output/stage5_cluster_refined_theories.json',
                       help='Output path (default: output/stage5_cluster_refined_theories.json)')
    parser.add_argument('--retry-poor-grouping', action='store_true',
                       help='Retry clusters where assign_common creates unique names (costs more)')
    args = parser.parse_args()
    
    refiner = Stage5ClusterRefiner(retry_poor_grouping=args.retry_poor_grouping)
    output = refiner.process_clusters(
        output_path=args.output,
        resume_from_checkpoint=args.resume
    )
    refiner.save_results(output, output_path=args.output)
    
    print("\n✅ Stage 5 complete!")


if __name__ == '__main__':
    main()
