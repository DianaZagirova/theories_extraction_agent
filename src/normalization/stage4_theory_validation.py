"""
Stage 4: Theory Validation and Mapping

This stage validates unique theory names and maps them to canonical theories:
1. Load unique names from Stage 3 output
2. For each unique name, sample up to 4 theories with that name
3. Extract key_concepts from stage0_filtered_theories.json
4. Batch process 5 unique names at a time
5. Send to LLM with canonical theories list for validation and mapping
6. Handle "doubted" cases with re-runs using different random samples
7. Output validated theories with mapping information

Input: 
- output/stage3_refined_theories.json (unique names)
- output/stage0_filtered_theories.json (key concepts)
- ontology/group_ontology_mechanisms.json (canonical theories)

Output: output/stage4_validated_theories.json
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


class Stage4TheoryValidator:
    """
    Stage 4: Validate unique theory names and map to canonical theories.
    
    Process:
    1. Extract unique names from Stage 3
    2. For each unique name, sample theories and their key_concepts
    3. Batch process with LLM validation
    4. Handle doubted cases with re-runs
    """
    
    def __init__(self,
                 stage3_path: str = 'output/stage3_refined_theories.json',
                 stage0_path: str = 'output/stage0_filtered_theories.json',
                 tracker_path: str = 'output/theory_tracking_report.json',
                 ontology_path: str = 'ontology/group_ontology_mechanisms.json',
                 max_concurrent: int = 10):
        """Initialize Stage 4 validator."""
        # Statistics
        self.stats = {
            'total_unique_names': 0,
            'total_batches': 0,
            'valid_theories': 0,
            'invalid_theories': 0,
            'doubted_theories': 0,
            'mapped_theories': 0,
            'novel_theories': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
        self.stage3_path = Path(stage3_path)
        self.stage0_path = Path(stage0_path)
        self.tracker_path = Path(tracker_path)
        self.ontology_path = Path(ontology_path)
        
        # Load data
        self.canonical_theories = self._load_canonical_theories()
        self.stage0_theories = self._load_stage0_theories()
        self.unique_name_to_theory_ids = self._load_stage3_unique_names()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE4', 'azure')
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY2'))
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
        
        
    
    def _load_canonical_theories(self) -> Dict:
        """Load canonical theories from ontology."""
        print(f"📂 Loading canonical theories from {self.ontology_path}...")
        
        with open(self.ontology_path, 'r') as f:
            data = json.load(f)
        
        canonical = {}
        for theory in data:
            name = theory.get('theory_name', '')
            if name:
                canonical[name] = {
                    'mechanisms': theory.get('mechanisms', []),
                    'pathways': theory.get('pathways', []),
                    'key_players': theory.get('key_players', []),
                    'level': theory.get('level_of_explanation', ''),
                    'type': theory.get('type_of_cause', ''),
                    'temporal': theory.get('temporal_focus', '')
                }
        
        print(f"  ✓ Loaded {len(canonical)} canonical theories")
        return canonical
    
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
    
    def _load_stage3_unique_names(self) -> Dict[str, List[str]]:
        """
        Load Stage 3 unique names from theory tracker and create mapping: stage3_final_name -> list of theory_ids.
        
        Uses theory_tracking_report.json to get correct mapping from theory_id to stage3_final_name.
        
        Returns:
            Dictionary mapping stage3_final_name -> list of theory_ids
        """
        print(f"\n📂 Loading Stage 3 mappings from theory tracker: {self.tracker_path}...")
        
        with open(self.tracker_path, 'r') as f:
            tracker_data = json.load(f)
        
        theory_lineage = tracker_data.get('theory_lineage', {})
        
        # Create reverse index: stage3_final_name -> list of theory_ids
        unique_to_theory_ids = defaultdict(list)
        
        for theory_id, lineage in theory_lineage.items():
            stage3_final_name = lineage.get('stage3_final_name')
            
            # Only include theories that have a stage3_final_name
            if stage3_final_name:
                unique_to_theory_ids[stage3_final_name].append(theory_id)
        
        # Convert to regular dict
        unique_to_theory_ids = dict(unique_to_theory_ids)
        
        print(f"  ✓ Found {len(unique_to_theory_ids)} unique stage3 names")
        print(f"  ✓ Total theory instances: {sum(len(ids) for ids in unique_to_theory_ids.values())}")
        
        # Show some statistics
        name_counts = sorted([(name, len(ids)) for name, ids in unique_to_theory_ids.items()], 
                            key=lambda x: x[1], reverse=True)
        print(f"  ✓ Top 5 most common names:")
        for name, count in name_counts[:5]:
            print(f"    - {name}: {count} theories")
        
        self.stats['total_unique_names'] = len(unique_to_theory_ids)
        
        return unique_to_theory_ids
    
    def _sample_theories_for_unique_name(self, unique_name: str, 
                                         max_samples: int = 4,
                                         exclude_ids: Set[str] = None) -> List[Dict]:
        """
        Sample up to max_samples theories for a given unique name.
        
        Args:
            unique_name: The unique theory name
            max_samples: Maximum number of theories to sample (default 4)
            exclude_ids: Set of theory IDs to exclude from sampling
            
        Returns:
            List of theory dictionaries with key_concepts
        """
        theory_ids = self.unique_name_to_theory_ids.get(unique_name, [])
        
        # Filter out excluded IDs
        if exclude_ids:
            theory_ids = [tid for tid in theory_ids if tid not in exclude_ids]
        
        if not theory_ids:
            return []
        
        # Sample up to max_samples
        sample_size = min(len(theory_ids), max_samples)
        sampled_ids = random.sample(theory_ids, sample_size)
        
        # Get theory data
        sampled_theories = []
        for theory_id in sampled_ids:
            if theory_id in self.stage0_theories:
                theory_data = self.stage0_theories[theory_id].copy()
                theory_data['theory_id'] = theory_id
                sampled_theories.append(theory_data)
        
        return sampled_theories
    
    def _create_validation_prompt(self, batch_data: List[Dict], include_evidence: bool = False) -> str:
        """
        Create prompt for LLM validation and mapping.
        
        Args:
            batch_data: List of dictionaries, each containing:
                - unique_name: str
                - theories: List of theory dicts with key_concepts
            include_evidence: If True, include evidence field (for second+ rounds)
        
        Returns:
            Prompt string
        """
        # Build canonical theories reference
        canonical_ref = []
        for name, data in self.canonical_theories.items():
            mechanisms_str = "; ".join(data['mechanisms'][:4]) if data['mechanisms'] else "N/A"
            if len(data['mechanisms']) > 4:
                mechanisms_str += f" ... ({len(data['mechanisms'])} total)"
            
            canonical_ref.append(f"""
# {name}
  Level: {data['level']} | Type: {data['type']}
  Pathways: {data['pathways']}
  Key mechanisms: {mechanisms_str}
""")
        
        canonical_theories_text = "\n".join(canonical_ref)
        
        # Build batch input
        batch_input = []
        for idx, item in enumerate(batch_data, 1):
            unique_name = item['unique_name']
            theories = item['theories']
            
            batch_input.append(f"\n## UNIQUE NAME {idx}: {unique_name}")
            # batch_input.append(f"Number of theory instances: {len(theories)}")
            
            for version_num, theory in enumerate(theories, 1):
                batch_input.append(f"\n### V.{version_num}:")
                # batch_input.append(f"Source: DOI {theory.get('doi', 'N/A')}, PMID {theory.get('pmid', 'N/A')}")
                
                key_concepts = theory.get('key_concepts', [])
                if key_concepts:
                    batch_input.append("Key concepts:")
                    for concept in key_concepts:
                        concept_name = concept.get('concept', '')
                        concept_desc = concept.get('description', '')
                        batch_input.append(f"  - {concept_name}: {concept_desc}")
                else:
                    batch_input.append("  (No key concepts available)")
                
                # Include evidence for second+ rounds
                if include_evidence:
                    evidence = theory.get('evidence', '')
                    if evidence:
                        # Truncate evidence if too long
                        evidence_truncated = evidence[:500] + '...' if len(evidence) > 500 else evidence
                        batch_input.append(f"Evidence: {evidence_truncated}")
        
        batch_input_text = "\n".join(batch_input)
        
        prompt = f"""# TASK
Validate whether each unique name represents a valid aging theory and name it (map it to listed theory or create a new name).

# INSTRUCTIONS
For each UNIQUE NAME:

1. **Validate if it's a valid aging theory**:
   - an "aging theory" is a proposal, model, hypothesis, or mechanism that tries to explain WHY or HOW biological or psychosocial aging occurs at a general, organism-level scale.
   - **Generalizability**: The theory must attempt to explain aging as a fundamental process, not just describe a narrow phenomenon. Addresses aging broadly, not in the context of a specific disease/ organ/ pathway/ etc.
   - **Causal explanation**: Must propose mechanisms or reasons for aging, not just describe patterns or correlations Could be mathematical/computational model/epigenetic clocks but has to discuss causal mechanisms.

   - TRUE: valid
   - FALSE: does not meet criteria (too vague, organ-specific, disease-specific, not generalizable to human or at all, not causal, methodology, other reasons. based on mechanisms)
   - DOUBTED

2. **Check if it maps to any listed theory**:
   - Compare key concepts/mechanisms (not name only) with listed theories
   - Set is_listed=true if it matches a listed theory
   - Try not to match theories that too different.
   - Provide the exact listed_name ONLY if mapped
   - Set mapping_confidence (0.0-1.0). Try to map only if very sure.

3. **If it is unlisted**:   - 
   - Provide reasoning for novel name creation, why not mapped to any listed theory.
   - Suggest a clear introduced_name (can retain the same or improve it):
   RULES FOR NAMES INTRODUCTION:
   - Avoid excessively specific names with a lot of details.
   - Try to reasonably generilize it based on the mechanisms.
   - If the name ends with "Theory", try not to add "of Aging". Use aging, not ageing.
   - If the name contains several aging theories: 1 - select one that is valid, 2 - if all are valid select one that fits the described mechanisms more than others. Never retain composite names with >=1 theory. 
   - Important! If the name contains the specific disease/organ/pathway/phenomenon/model/etc, analyze the corresponding mechanisms. 1 - If mechanisms are generalizable to human aging, generilize a name. 2 - If not, consider as not valid. 

# LISTED THEORIES REFERENCE
{canonical_theories_text}

# THEORIES TO VALIDATE
{batch_input_text}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):

{{
  "mappings": [
    {{
      "original_name": "Unique Name",
      "is_valid_theory": true/false/"doubted",
      "validation_reasoning": "Brief explanation why valid/invalid/doubted",
      "is_listed": true/false,
      "listed_name": "Exact name as in the list",
      "mapping_confidence": 0.0-1.0,
      "introduced_name_reasoning": "Brief explanation why not match with listed theories",
      "introduced_name": "Clear standardized name or null"
    }},
    ...
  ]
}}

# CRITICAL REQUIREMENTS
- You must include ALL {len(batch_data)} unique names in the output
- Be conservative with mapping (only map if very confident by the mechanism)
"""
        
        return prompt
    
    def _validate_llm_output(self, mappings: List[Dict], batch_data: List[Dict]) -> List[Dict]:
        """
        Validate LLM output and mark incomplete/invalid entries as doubted.
        
        Checks:
        1. All input names are present in output
        2. If is_listed=false and is_valid_theory=true, introduced_name must not be empty
        3. If is_listed=true, listed_name must not be empty
        
        Args:
            mappings: LLM output mappings
            batch_data: Original batch data
            
        Returns:
            Validated mappings with incomplete entries marked as doubted
        """
        validated_mappings = []
        
        # Check 1: Validate all input names are present
        output_names = {m['original_name'] for m in mappings}
        input_names = {item['unique_name'] for item in batch_data}
        missing_names = input_names - output_names
        
        if missing_names:
            print(f"  ⚠️  WARNING: {len(missing_names)} names missing from output!")
            for name in missing_names:
                mappings.append({
                    'original_name': name,
                    'is_valid_theory': 'doubted',
                    'validation_reasoning': 'Missing from LLM output',
                    'is_listed': False,
                    'listed_name': None,
                    'mapping_confidence': 0.0,
                    'introduced_name_reasoning': 'Not evaluated',
                    'introduced_name': name
                })
        
        # Check 2 & 3: Validate completeness of each mapping
        incomplete_count = 0
        for mapping in mappings:
            is_valid = mapping.get('is_valid_theory')
            is_listed = mapping.get('is_listed')
            listed_name = mapping.get('listed_name')
            introduced_name = mapping.get('introduced_name')
            
            # Check 3: If is_listed=false and is_valid_theory=true, introduced_name must not be empty
            if is_listed == False and is_valid == True:
                if not introduced_name or introduced_name.strip() == '' or introduced_name is None:
                    print(f"  ⚠️  WARNING: Valid unlisted theory missing introduced_name: {mapping['original_name']}")
                    mapping['is_valid_theory'] = 'doubted'
                    mapping['validation_reasoning'] = (mapping.get('validation_reasoning', '') + 
                                                      ' | Missing introduced_name for valid unlisted theory')
                    incomplete_count += 1
            
            # Check 4: If is_listed=true, listed_name must not be empty
            if is_listed == True:
                if not listed_name or listed_name.strip() == '' or listed_name is None:
                    print(f"  ⚠️  WARNING: Listed theory missing listed_name: {mapping['original_name']}")
                    mapping['is_valid_theory'] = 'doubted'
                    mapping['validation_reasoning'] = (mapping.get('validation_reasoning', '') + 
                                                      ' | Missing listed_name for mapped theory')
                    incomplete_count += 1
            
            validated_mappings.append(mapping)
        
        if incomplete_count > 0:
            print(f"  ⚠️  Marked {incomplete_count} incomplete entries as doubted")
        
        return validated_mappings
    
    async def _process_batch_async(self, batch_data: List[Dict], 
                                   batch_num: int, total_batches: int) -> List[Dict]:
        """
        Process a batch of unique names using LLM asynchronously.
        
        Args:
            batch_data: List of dictionaries with unique_name and theories
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            List of validation results
        """
        # Create prompt (without evidence for first round)
        prompt = self._create_validation_prompt(batch_data, include_evidence=False)
        
        # Estimate tokens for rate limiting (rough estimate)
        estimated_tokens = len(prompt) // 4 + 6000
        
        # Wait for rate limiter
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Call LLM in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in aging biology and theory validation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=6000
                )
            )
            
            # Track token usage (thread-safe)
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
                mappings = data.get('mappings', [])
                
                print(f"  ✓ Processed {len(mappings)} validations")
                
                # Validate LLM output
                validated_mappings = self._validate_llm_output(mappings, batch_data)
                
                return validated_mappings
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error: {e}")
                print(f"  Response length: {len(response_text)} chars")
                # Return doubted results on error
                return [{
                    'original_name': item['unique_name'],
                    'is_valid_theory': 'doubted',
                    'validation_reasoning': 'JSON parsing error',
                    'is_listed': False,
                    'listed_name': None,
                    'mapping_confidence': 0.0,
                    'introduced_name_reasoning': 'Not evaluated',
                    'introduced_name': item['unique_name']
                } for item in batch_data]
                
        except Exception as e:
            import traceback
            return [{
                'original_name': item['unique_name'],
                'is_valid_theory': 'doubted',
                'validation_reasoning': f'LLM error: {str(e)}',
                'is_listed': False,
                'listed_name': None,
                'mapping_confidence': 0.0,
                'introduced_name_reasoning': 'Not evaluated',
                'introduced_name': item['unique_name']
            } for item in batch_data]
    
    def _process_batch(self, batch_data: List[Dict], 
                      batch_num: int, total_batches: int) -> List[Dict]:
        """Synchronous wrapper for async batch processing."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._process_batch_async(batch_data, batch_num, total_batches)
        )
    
    async def _process_batch_with_evidence_async(self, batch_data: List[Dict], retry_num: int) -> List[Dict]:
        """
        Process a batch of unique names using LLM with evidence included (for doubted theories).
        
        Args:
            batch_data: List of dictionaries with unique_name and theories
            retry_num: Retry attempt number
            
        Returns:
            List of validation results
        """
        # Create prompt WITH evidence for second+ rounds
        prompt = self._create_validation_prompt(batch_data, include_evidence=True)
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt) // 4 + 16000
        
        # Wait for rate limiter
        await self.rate_limiter.acquire(estimated_tokens)
        
        # Call LLM in executor
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in aging biology."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=16000
                )
            )
            
            # Track token usage (thread-safe)
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
                mappings = data.get('mappings', [])
                
                print(f"  ✓ Processed {len(mappings)} validations")
                
                # Validate LLM output
                validated_mappings = self._validate_llm_output(mappings, batch_data)
                
                return validated_mappings
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error: {e}")
                print(f"  Response length: {len(response_text)} chars")
                # Return doubted results on error
                return [{
                    'original_name': item['unique_name'],
                    'is_valid_theory': 'doubted',
                    'validation_reasoning': 'JSON parsing error',
                    'is_listed': False,
                    'listed_name': None,
                    'mapping_confidence': 0.0,
                    'introduced_name_reasoning': 'Not evaluated',
                    'introduced_name': item['unique_name']
                } for item in batch_data]
                
        except Exception as e:
            import traceback
            return [{
                'original_name': item['unique_name'],
                'is_valid_theory': 'doubted',
                'validation_reasoning': f'LLM error: {str(e)}',
                'is_listed': False,
                'listed_name': None,
                'mapping_confidence': 0.0,
                'introduced_name_reasoning': 'Not evaluated',
                'introduced_name': item['unique_name']
            } for item in batch_data]
    
    def _process_batch_with_evidence(self, batch_data: List[Dict], retry_num: int) -> List[Dict]:
        """Synchronous wrapper for async batch processing with evidence."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._process_batch_with_evidence_async(batch_data, retry_num)
        )
    
    def _handle_doubted_theories(self, doubted_results: List[Dict], 
                                 max_retries: int = 3) -> List[Dict]:
        """
        Re-process doubted theories with different samples.
        
        Args:
            doubted_results: List of doubted validation results
            max_retries: Maximum number of retries per theory
            
        Returns:
            List of final validation results
        """
        print(f"\n🔄 Handling {len(doubted_results)} doubted theories...")
        
        final_results = []
        
        for result in tqdm(doubted_results, desc="Re-validating doubted theories"):
            unique_name = result['original_name']
            retry_count = 0
            used_theory_ids = set()
            
            while retry_count < max_retries:
                # Sample different theories
                theories = self._sample_theories_for_unique_name(
                    unique_name, 
                    max_samples=4,
                    exclude_ids=used_theory_ids
                )
                
                if not theories:
                    # No more theories to sample
                    result['is_valid_theory'] = False
                    result['validation_reasoning'] += ' | No more samples available'
                    break
                
                # Track used IDs
                for theory in theories:
                    used_theory_ids.add(theory['theory_id'])
                
                # Process single theory with evidence (second+ round)
                batch_data = [{'unique_name': unique_name, 'theories': theories}]
                # For doubted theories, use special processing with evidence
                new_results = self._process_batch_with_evidence(batch_data, retry_count + 1)
                
                if new_results:
                    new_result = new_results[0]
                    
                    if new_result['is_valid_theory'] == True:
                        final_results.append(new_result)
                        break
                    elif new_result['is_valid_theory'] == False:
                        final_results.append(new_result)
                        break
                    else:  # Still doubted
                        retry_count += 1
                        if retry_count >= max_retries:
                            # After max retries, mark as invalid
                            new_result['is_valid_theory'] = False
                            new_result['validation_reasoning'] += f' | Doubted after {max_retries} retries'
                            final_results.append(new_result)
                
                time.sleep(0.5)  # Small delay between retries
        
        return final_results
    
    def _save_results(self, all_results: List[Dict], output_path: str,
                     batches_completed: int = 0, is_final: bool = True):
        """Save validation results (intermediate or final).
        
        Args:
            all_results: List of validation results
            output_path: Path to save output
            batches_completed: Number of batches completed (for checkpoints)
            is_final: Whether this is the final save
        """
        # Count statistics
        valid_count = sum(1 for r in all_results if r['is_valid_theory'] == True)
        invalid_count = sum(1 for r in all_results if r['is_valid_theory'] == False)
        doubted_count = sum(1 for r in all_results if r['is_valid_theory'] == 'doubted')
        mapped_count = sum(1 for r in all_results if r['is_listed'] == True)
        
        if is_final:
            self.stats['valid_theories'] = valid_count
            self.stats['invalid_theories'] = invalid_count
            self.stats['doubted_theories'] = doubted_count
            self.stats['mapped_theories'] = mapped_count
        
        # Create output structure
        output_data = {
            'metadata': {
                'stage': 'stage4_theory_validation',
                'status': 'complete' if is_final else 'in_progress',
                'batches_completed': batches_completed,
                'total_unique_names': self.stats['total_unique_names'],
                'valid_theories': valid_count,
                'invalid_theories': invalid_count,
                'doubted_theories': doubted_count,
                'mapped_theories': mapped_count,
                'total_batches': self.stats['total_batches'],
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_cost': self.stats['total_cost'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'validations': all_results,
            'canonical_theories': list(self.canonical_theories.keys())
        }
        
        # Save to file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if is_final:
            print(f"\n✅ Results saved to {output_path}")
            print(f"\n📊 Final Statistics:")
            print(f"  Total unique names: {self.stats['total_unique_names']}")
            print(f"  Valid theories: {valid_count} ({valid_count/len(all_results)*100:.1f}%)")
            print(f"  Invalid theories: {invalid_count} ({invalid_count/len(all_results)*100:.1f}%)")
            print(f"  Doubted theories: {doubted_count} ({doubted_count/len(all_results)*100:.1f}%)")
            print(f"  Mapped to canonical: {mapped_count} ({mapped_count/len(all_results)*100:.1f}%)")
            print(f"  Total batches: {self.stats['total_batches']}")
            print(f"  Total tokens: {self.stats['total_input_tokens']:,} in / {self.stats['total_output_tokens']:,} out")
            print(f"  Total cost: ${self.stats['total_cost']:.4f}")
        else:
            print(f"    💾 Checkpoint saved: {len(all_results)} validations, {valid_count} valid, {invalid_count} invalid, {doubted_count} doubted")
    
    def _load_checkpoint(self, checkpoint_path: str, incremental: bool = False) -> tuple:
        """
        Load checkpoint from previous run.
        
        Args:
            checkpoint_path: Path to checkpoint file
            incremental: If True, allow loading complete checkpoint for incremental processing
        
        Returns:
            Tuple of (all_results, batches_completed, stats) or (None, 0, None) if no valid checkpoint
        """
        if not os.path.exists(checkpoint_path):
            return None, 0, None
        
        print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        
        # Check if it's complete
        if metadata.get('status') == 'complete':
            if incremental:
                print("  ✓ Checkpoint is complete - will process only new entries")
            else:
                print("  ⚠️  Checkpoint is already complete!")
                user_input = input("  Do you want to re-run from scratch? (y/n): ")
                if user_input.lower() != 'y':
                    return None, 0, None
                else:
                    return None, 0, None
        
        all_results = data.get('validations', [])
        batches_completed = metadata.get('batches_completed', 0)
        
        # Restore stats
        stats = {
            'total_unique_names': metadata.get('total_unique_names', 0),
            'total_batches': metadata.get('total_batches', 0),
            'valid_theories': metadata.get('valid_theories', 0),
            'invalid_theories': metadata.get('invalid_theories', 0),
            'doubted_theories': metadata.get('doubted_theories', 0),
            'mapped_theories': metadata.get('mapped_theories', 0),
            'total_input_tokens': metadata.get('total_input_tokens', 0),
            'total_output_tokens': metadata.get('total_output_tokens', 0),
            'total_cost': metadata.get('total_cost', 0.0)
        }
        
        print(f"  ✓ Loaded checkpoint:")
        print(f"    - Batches completed: {batches_completed}/{stats['total_batches']}")
        print(f"    - Validations: {len(all_results)}")
        print(f"    - Valid: {stats['valid_theories']}, Invalid: {stats['invalid_theories']}, Doubted: {stats['doubted_theories']}")
        print(f"    - Cost so far: ${stats['total_cost']:.4f}")
        
        return all_results, batches_completed, stats
    
    def run(self, output_path: str = 'output/stage4_validated_theories.json',
            batch_size: int = 5,
            handle_doubted: bool = True,
            resume_from_checkpoint: bool = False,
            incremental: bool = False):
        """
        Run Stage 4 validation.
        
        Args:
            output_path: Path to save output
            batch_size: Number of unique names per batch (default 5)
            handle_doubted: Whether to re-process doubted theories
            resume_from_checkpoint: If True, resume from checkpoint
            incremental: If True, process only new entries not in existing checkpoint
        """
        print("=" * 80)
        print("STAGE 4: THEORY VALIDATION AND MAPPING")
        print("=" * 80)
        
        # Try to load checkpoint if requested
        start_batch = 0
        all_results = []
        
        if resume_from_checkpoint or incremental:
            checkpoint_data = self._load_checkpoint(output_path, incremental=incremental)
            if checkpoint_data[0] is None and checkpoint_data[1] == 0 and checkpoint_data[2] is None:
                # User declined to re-run complete checkpoint
                print("\n❌ Exiting without changes.")
                return
            if checkpoint_data[0] is not None:
                all_results, start_batch, loaded_stats = checkpoint_data
                # Restore stats
                self.stats.update(loaded_stats)
                if not incremental:
                    print(f"\n🔄 Resuming from batch {start_batch + 1}")
        
        # Get all unique names
        all_unique_names = list(self.unique_name_to_theory_ids.keys())
        
        # Filter for incremental processing
        if incremental and all_results:
            processed_names = {r['original_name'] for r in all_results}
            unique_names = [name for name in all_unique_names if name not in processed_names]
            print(f"\n📊 Incremental mode:")
            print(f"  Total unique names in stage3: {len(all_unique_names)}")
            print(f"  Already processed: {len(processed_names)}")
            print(f"  New names to validate: {len(unique_names)}")
            
            if not unique_names:
                print("\n✅ No new entries to process!")
                return
        else:
            unique_names = all_unique_names
            print(f"\n📊 Total unique names to validate: {len(unique_names)}")
        
        # Create batches
        batches = []
        for i in range(0, len(unique_names), batch_size):
            batch_names = unique_names[i:i + batch_size]
            
            # Sample theories for each unique name
            batch_data = []
            for unique_name in batch_names:
                theories = self._sample_theories_for_unique_name(unique_name, max_samples=4)
                if theories:  # Only include if we have theories
                    batch_data.append({
                        'unique_name': unique_name,
                        'theories': theories
                    })
            
            if batch_data:
                batches.append(batch_data)
        
        if incremental:
            # For incremental mode, we start from batch 0 for new entries
            start_batch = 0
            # Update total batches to include both old and new
            self.stats['total_batches'] = len(batches)
        elif not resume_from_checkpoint or start_batch == 0:
            self.stats['total_batches'] = len(batches)
        
        print(f"  Created {len(batches)} batches for processing")
        
        # Process batches with async concurrency
        if start_batch == 0:
            print(f"\n🚀 Processing {len(batches)} batches with {self.max_concurrent} concurrent requests...")
        else:
            print(f"\n🚀 Processing remaining {len(batches) - start_batch} batches with {self.max_concurrent} concurrent requests...")
        
        # Run async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            all_results = loop.run_until_complete(
                self._process_all_batches_async(batches, start_batch, output_path, all_results)
            )
            
            # Handle doubted theories (while loop is still open)
            if handle_doubted:
                doubted_results = [r for r in all_results if r['is_valid_theory'] == 'doubted']
                
                if doubted_results:
                    print(f"\n🔄 Found {len(doubted_results)} doubted theories")
                    final_doubted = self._handle_doubted_theories(doubted_results, max_retries=3)
                    
                    # Replace doubted results with final results
                    doubted_names = {r['original_name'] for r in doubted_results}
                    all_results = [r for r in all_results if r['original_name'] not in doubted_names]
                    all_results.extend(final_doubted)
                    
                    # Save final results after handling doubted theories
                    self._save_results(all_results, output_path, batches_completed=len(batches), is_final=True)
        finally:
            loop.close()
        
        print("\n✅ Stage 4 complete!")
    
    async def _process_all_batches_async(self, batches: List[List[Dict]], 
                                         start_batch: int, output_path: str,
                                         all_results: List[Dict]) -> List[Dict]:
        """Process all batches asynchronously with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch_with_semaphore(batch_data, batch_num, total_batches):
            async with semaphore:
                return await self._process_batch_async(batch_data, batch_num, total_batches)
        
        # Process batches in groups
        batch_size = 50  # Process 50 batches at a time
        checkpoint_interval = 5  # Save checkpoint every 5 batches
        batches_since_checkpoint = 0
        
        with tqdm(total=len(batches), initial=start_batch, desc="Processing batches", unit="batch") as pbar:
            for group_start in range(start_batch, len(batches), batch_size):
                group_end = min(group_start + batch_size, len(batches))
                
                # Create tasks for this group
                tasks = []
                for i in range(group_start, group_end):
                    batch_data = batches[i]
                    batch_num = i + 1
                    tasks.append(process_batch_with_semaphore(batch_data, batch_num, len(batches)))
                
                # Process group concurrently
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results and count successes
                successful_batches = 0
                for result in group_results:
                    if isinstance(result, Exception):
                        print(f"\n⚠️  Batch failed with exception: {result}")
                    elif result:
                        all_results.extend(result)
                        successful_batches += 1
                
                # Update progress bar once for the entire group
                batches_processed_in_group = group_end - group_start
                pbar.update(batches_processed_in_group)
                pbar.set_postfix({'validated': len(all_results), 'failed': len(group_results) - successful_batches})
                
                # Track batches processed since last checkpoint
                batches_since_checkpoint += batches_processed_in_group
                current_batch = group_end
                
                # Save checkpoint every checkpoint_interval batches or at the end
                if batches_since_checkpoint >= checkpoint_interval or current_batch == len(batches):
                    self._save_results(
                        all_results,
                        output_path,
                        batches_completed=current_batch,
                        is_final=(current_batch == len(batches))
                    )
                    batches_since_checkpoint = 0  # Reset counter
                
                # Small delay between groups
                if group_end < len(batches):
                    await asyncio.sleep(2)
        
        return all_results


def main(resume: bool = False, incremental: bool = False, max_concurrent: int = 10):
    """
    Main entry point.
    
    Args:
        resume: If True, resume from checkpoint
        incremental: If True, process only new entries not in existing checkpoint
        max_concurrent: Maximum concurrent API calls
    """
    validator = Stage4TheoryValidator(
        stage3_path='output/stage3_refined_theories.json',
        stage0_path='output/stage0_filtered_theories.json',
        tracker_path='output/theory_tracking_report.json',
        ontology_path='ontology/group_ontology_mechanisms.json',
        max_concurrent=max_concurrent
    )
    
    validator.run(
        output_path='output/stage4_validated_theories.json',
        batch_size=10,
        handle_doubted=True,
        resume_from_checkpoint=resume,
        incremental=incremental
    )


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 4: Theory Validation and Mapping')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--incremental', action='store_true', 
                       help='Process only new entries not in existing checkpoint')
    parser.add_argument('--max-concurrent', type=int, default=10, 
                       help='Maximum concurrent API calls (default: 10)')
    
    args = parser.parse_args()
    
    print(f"\n🚀 Parallel processing enabled: {args.max_concurrent} concurrent requests")
    print(f"⚡ Rate limits: 180K tokens/min, 450 requests/min")
    
    main(resume=args.resume, incremental=args.incremental, max_concurrent=args.max_concurrent)
