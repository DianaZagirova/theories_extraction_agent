"""
Stage 1.5: LLM Assistant Mapping to Canonical Theories

Maps unmatched theories from Stage 1 (fuzzy matching) to canonical theories using LLM.
This stage uses a simplified approach:
1. Normalize theory names (remove abbreviations in parentheses, standardize terms)
2. Group theories by unique normalized names
3. For each unique name, ask LLM to map to ontology (only if 100% confident)
4. Assign mapped name to all entries in the group

Strategy:
- Conservative LLM mapping: only map when 100% sure
- Batch processing by unique names for efficiency
- Preserve all original theory data
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
from collections import defaultdict
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient

SAVE_PROMPT_FOR_DEBUG=True
@dataclass
class TheoryEntry:
    """Theory entry with normalization information."""
    theory_id: str
    original_name: str
    normalized_name: str
    mapped_name: Optional[str] = None
    mapping_confidence_stage1_5: float = 0.0
    mapping_reasoning: str = ""
    
    # Original theory data
    key_concepts: List[Dict] = field(default_factory=list)
    description: str = ""
    evidence: str = ""
    confidence_is_theory: str = "unknown"
    mode: str = ""
    criteria_reasoning: str = ""
    paper_focus: int = 0
    doi: str = ""
    pmid: str = ""
    paper_title: str = ""
    timestamp: str = ""
    enriched_text: Optional[str] = None
    concept_text: Optional[str] = None
    is_validated: bool = False
    validation_reason: str = ""
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'theory_id': self.theory_id,
            'original_name': self.original_name,
            'normalized_name': self.normalized_name,
            'mapped_name': self.mapped_name,
            'mapping_confidence_stage1_5': self.mapping_confidence_stage1_5,
            'mapping_reasoning': self.mapping_reasoning,
            'key_concepts': self.key_concepts,
            'description': self.description,
            'evidence': self.evidence,
            'confidence_is_theory': self.confidence_is_theory,
            'mode': self.mode,
            'criteria_reasoning': self.criteria_reasoning,
            'paper_focus': self.paper_focus,
            'doi': self.doi,
            'pmid': self.pmid,
            'paper_title': self.paper_title,
            'timestamp': self.timestamp,
            'enriched_text': self.enriched_text,
            'concept_text': self.concept_text,
            'is_validated': self.is_validated,
            'validation_reason': self.validation_reason
        }


class LLMAssistantMapper:
    """
    LLM-assisted mapper for theory names.
    
    Strategy:
    1. Normalize names (remove abbreviations in parentheses, standardize terms)
    2. Group by unique normalized names
    3. Use LLM to map unique names to ontology (conservative approach)
    4. Propagate mappings to all theories in each group
    """
    
    def __init__(self, ontology_path: str = 'ontology/groups_ontology_alliases.json'):
        """Initialize LLM assistant mapper."""
        self.ontology_path = Path(ontology_path)
        self.canonical_theories = self._load_ontology()
        
        # Track original ontology theories
        self.original_ontology_theories = set(self.canonical_theories.keys())
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM', 'openai')
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY2'))
        else:
            self.llm = AzureOpenAIClient()
        self.model = 'gpt-4.1-mini'
        
        self.stats = {
            'total_input': 0,
            'unique_names': 0,
            'mapped_by_llm': 0,
            'unmapped': 0,
            'total_output': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
    
    def _load_ontology(self) -> Dict[str, Dict]:
        """Load canonical theories from ontology."""
        print(f"📂 Loading ontology from {self.ontology_path}...")
        
        with open(self.ontology_path, 'r') as f:
            data = json.load(f)
        
        canonical = {}
        for category, subcats in data['TheoriesOfAging'].items():
            for subcat, theories in subcats.items():
                for theory in theories:
                    name = theory['name']
                    canonical[name] = {
                        'name': name,
                        'aliases': theory.get('aliases', []),
                        'abbreviations': theory.get('abbreviations', []),
                        'category': category,
                        'subcategory': subcat
                    }
        
        print(f"✓ Loaded {len(canonical)} canonical theories from ontology")
        return canonical
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize theory name for grouping.
        
        Rules:
        1. Remove abbreviations in parentheses (e.g., "(ABBR)")
        2. Remove "Theory of Aging" -> "Theory"
        3. Standardize "ageing" -> "aging"
        4. Remove extra whitespace
        5. Convert to title case for consistency
        """
        normalized = name.strip()
        
        # Remove abbreviations in parentheses/brackets
        # Pattern: (ABBR) or [ABBR] where ABBR is 2-6 uppercase letters
        normalized = re.sub(r'\s*[\(\[]([A-Z]{2,6}(?:\d*)?)[\)\]]\s*', ' ', normalized)
        
        # Standardize "ageing" to "aging"
        normalized = re.sub(r'\bageing\b', 'aging', normalized, flags=re.IGNORECASE)
        
        # Remove "of Aging" / "of Ageing" suffix
        normalized = re.sub(r'\s+of\s+aging\s*$', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+of\s+ageing\s*$', '', normalized, flags=re.IGNORECASE)
        
        # # Remove "Theory" suffix if followed by nothing
        # # But keep it if it's part of a compound name like "Free Radical Theory"
        # # Only remove if it's at the end
        # normalized = re.sub(r'\s+theory\s*$', '', normalized, flags=re.IGNORECASE)
        # normalized = re.sub(r'\s+hypothesis\s*$', '', normalized, flags=re.IGNORECASE)
        # normalized = re.sub(r'\s+model\s*$', '', normalized, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Convert to title case for consistency
        normalized = normalized.title()
        
        return normalized
    
    def _group_by_normalized_name(self, theories: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group theories by normalized name.
        
        Returns:
            Dictionary mapping normalized_name -> list of theory dicts
        """
        groups = defaultdict(list)
        
        for theory in theories:
            original_name = theory.get('original_name', '')
            normalized = self._normalize_name(original_name)
            groups[normalized].append(theory)
        
        return dict(groups)
    
    def _create_mapping_prompt(self, unique_names: List[str]) -> str:
        """
        Create prompt for LLM to map unique names to ontology.
        
        Args:
            unique_names: List of unique normalized theory names
            
        Returns:
            Prompt string
        """
        # Build canonical theories reference (primary names only)
        mapped_names = sorted(self.canonical_theories.keys())
        canonical_list = "\n".join([f"- {name}" for name in mapped_names])
        
        # Build list of names to map
        names_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(unique_names)])
        
        prompt = f"""Map names to aging theories names to names from an ontology.

# INSTRUCTIONS
For each theory name:
A. Does it seems to be a name of a valid aging theory - a proposal, model, hypothesis, mechanism, etc, that tries to explain WHY or HOW biological or psychosocial aging  occurs at a general, organism-level scale (even if shown on specific examples, should be still generalizable; not specific disease/condition) in humans (but not plants, etc)?  
B. If it seems to be valid:
1. Does it match ANY listed theory from the list above?
2. If it matches, map if you are 100% confident, it's the same theory. Consider semantic equivalence (e.g., "Mitochondrial Decline" = "Mitochondrial Decline Theory"). Note: Neural Dedifferentiation is not the Dysdifferentiation Theory

#NOTES:
1. If names contains several names of valid theories, select the main one or any one if main is not clear and output it as a mapped_name (but only one)
2. If several similiar names are mapped to a new theory, try to create the same name for the consistency. Exmaple: If for one theory created "Ampk Signaling In Aging" then for similiar use this name (and not Ampk Signaling In Senescence or other versions).
3. Try to create a new groups that are not too specific, but do not change completely the original meaning. 

# LISTED AGING THEORIES
{canonical_list}

# THEORY NAMES TO MAP
{names_list}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):
"input_name" - name from the list
"mapped_name" - name for a theory if not mapped from the list (not too specific) - try to strandartize it across similiar theories. Do not change the original name too match. Do not change the meaning. Just a name, not other notes, prefixes, etc.
"status" - valid, not_valid. Note not_valid for strongly not valid aging theories.
"confidence" - 10 - confidence level (10 is very sure that is theory, 1 is very unsure)

#Example
{{
  "mappings": [
    {{
      "input_name": "Theory Name",
      "mapped_name": "Mapped name from list",
      "status": "valid",
      "confidence": 1.0
    }},
    {{
      "input_name": "Theory Name not in the list",
      "mapped_name": "standard name for new theory",
      "status": "valid",
      "confidence": 0.9
    }},
    {{
      "input_name": "Aging of Plant roots",
      "mapped_name": null,
      "status": "not_valid",
      "confidence": 0.9
    }}
    ...
  ]
}}
# IMPORTANT
- Include ALL {len(unique_names)} theory names in output
"""
        
        if SAVE_PROMPT_FOR_DEBUG:
            with open('prompt_debug.txt', 'w') as f:
                f.write(prompt)
        return prompt
    
    def _map_batch(self, batch_names: List[str], batch_num: int, total_batches: int) -> Dict[str, Tuple[Optional[str], float]]:
        """
        Map a single batch of names using LLM.
        
        Args:
            batch_names: List of names in this batch
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            Dictionary mapping name -> (mapped_name, confidence)
        """
        print(f"\n  Batch {batch_num}/{total_batches}: Processing {len(batch_names)} names...")
        
        # Create prompt
        prompt = self._create_mapping_prompt(batch_names)
        
        # Call LLM
        try:
            response = self.llm.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in aging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            
            # Calculate cost (for gpt-4.1-mini)
            cost = (input_tokens / 1_000_000) * 0.04 + (output_tokens / 1_000_000) * 1.60
            
            self.stats['total_input_tokens'] += input_tokens
            self.stats['total_output_tokens'] += output_tokens
            self.stats['total_cost'] += cost
            
            print(f"    Tokens: {input_tokens:,} in / {output_tokens:,} out | Cost: ${cost:.4f}")
            
            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'length':
                print(f"    ⚠️  WARNING: Response truncated due to max_tokens limit!")
                print(f"    This batch will be retried with missing names.")
                return {}
            
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
                mappings_list = data.get('mappings', [])
            except json.JSONDecodeError as e:
                print(f"    ❌ JSON parsing error: {e}")
                print(f"    Response length: {len(response_text)} chars")
                print(f"    Last 200 chars: ...{response_text[-200:]}")
                # Try to salvage partial JSON by finding the last complete entry
                print(f"    Attempting to salvage partial response...")
                return {}
            
            # Convert to dictionary and track new theories
            mappings = {}
            new_theories_set = set()  # Track unique newly added theories
            not_valid_count = 0
            
            for mapping in mappings_list:
                input_name = mapping.get('input_name', '')
                mapped = mapping.get('mapped_name')
                status = mapping.get('status', 'valid')
                confidence = mapping.get('confidence', 0.0)
                
                # Only add if input_name is valid
                if input_name:
                    # Check status
                    if status == 'not_valid':
                        # Mark as not valid aging theory
                        mappings[input_name] = {
                            'mapped_name': mapped,
                            'confidence': confidence,
                            'status': 'not_valid'
                        }
                        not_valid_count += 1
                    elif mapped:
                        # Check if it's a new theory or existing one
                        is_from_ontology = mapped in self.original_ontology_theories
                        is_new_theory = mapped not in self.canonical_theories and mapped not in new_theories_set
                        
                        if is_new_theory:
                            # New theory discovered (first time in this batch)
                            new_theories_set.add(mapped)
                            print(f"    ✨ New theory: '{mapped}' (from '{input_name}')")
                        
                        mappings[input_name] = {
                            'mapped_name': mapped,
                            'confidence': confidence,
                            'status': status,
                            'is_from_ontology': is_from_ontology,
                            'is_new_theory': is_new_theory or (mapped not in self.original_ontology_theories and mapped in self.canonical_theories)
                        }
                    else:
                        # No mapping provided
                        mappings[input_name] = {
                            'mapped_name': None,
                            'confidence': confidence,
                            'status': status
                        }
            
            # Convert set to list for return
            new_theories = list(new_theories_set)
            
            # Report new theories
            if new_theories:
                print(f"    ✨ {len(new_theories)} unique new theories added to canonical list")
            if not_valid_count:
                print(f"    ❌ {not_valid_count} theories marked as not valid")
            
            # Check for missing names
            missing = set(batch_names) - set(mappings.keys())
            if missing:
                print(f"    ⚠️  Warning: {len(missing)} names missing from response")
            
            # Count mapped
            mapped_count = sum(1 for data in mappings.values() if data.get('mapped_name') is not None)
            print(f"    ✓ Mapped {mapped_count}/{len(batch_names)} names")
            
            return mappings, new_theories
            
        except Exception as e:
            print(f"    ❌ Error calling LLM: {e}")
            import traceback
            print(f"    Traceback: {traceback.format_exc()[:500]}")
            return {}, []
    
    def _save_intermediate_results(self, all_mappings: Dict[str, Dict], 
                                   batch_num: int, output_path: str, canonical_theories_count: int = 0):
        """Save intermediate mapping results to JSON."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        intermediate_data = {
            'metadata': {
                'stage': 'stage1_5_llm_assistant_mapping',
                'status': 'in_progress',
                'batches_completed': batch_num,
                'total_mappings': len(all_mappings),
                'mapped_count': sum(1 for _, data in all_mappings.items() if data.get('mapped_name') is not None),
                'not_valid_count': sum(1 for _, data in all_mappings.items() if data.get('status') == 'not_valid'),
                'canonical_theories_count': canonical_theories_count,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'mappings': all_mappings
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(intermediate_data, f, indent=2)
            print(f"    💾 Saved checkpoint: {len(all_mappings)} mappings")
        except Exception as e:
            print(f"    ⚠️  Failed to save checkpoint: {e}")
    
    def _save_canonical_theories(self, output_path: str):
        """Save final list of all canonical theories (original + newly discovered)."""
        # Separate original and new theories
        original_theories = []
        new_theories = []
        
        for theory_name in sorted(self.canonical_theories.keys()):
            theory_info = {
                'name': theory_name,
                'metadata': self.canonical_theories[theory_name]
            }
            
            if theory_name in self.original_ontology_theories:
                original_theories.append(theory_info)
            else:
                new_theories.append(theory_info)
        
        canonical_data = {
            'metadata': {
                'total_theories': len(self.canonical_theories),
                'original_ontology_theories': len(original_theories),
                'newly_discovered_theories': len(new_theories),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'original_theories': original_theories,
            'newly_discovered_theories': new_theories,
            'all_theories_list': sorted(self.canonical_theories.keys())
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(canonical_data, f, indent=2)
            print(f"  ✓ Canonical theories list saved: {output_path}")
            print(f"    - {len(original_theories)} from original ontology")
            print(f"    - {len(new_theories)} newly discovered")
        except Exception as e:
            print(f"  ⚠️  Failed to save canonical theories: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Dict]:
        """Load existing mappings from checkpoint file."""
        if not os.path.exists(checkpoint_path):
            return {}
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            mappings_dict = data.get('mappings', {})
            
            # Keep dict format (no conversion needed)
            loaded_mappings = mappings_dict
            
            print(f"  ✓ Loaded {len(loaded_mappings)} existing mappings from checkpoint")
            mapped_count = sum(1 for _, data in loaded_mappings.items() if data.get('mapped_name') is not None)
            not_valid_count = sum(1 for _, data in loaded_mappings.items() if data.get('status') == 'not_valid')
            print(f"  ✓ {mapped_count} names already mapped to canonical theories")
            print(f"  ✓ {not_valid_count} names marked as not valid")
            
            return loaded_mappings
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            return {}
    
    def _create_similarity_batches(self, unique_names: List[str], batch_size: int) -> List[List[str]]:
        """Create batches based on similarity using TF-IDF (optimized for large datasets)."""
        if len(unique_names) <= batch_size:
            return [unique_names]
        
        # For very large datasets (>5000), skip similarity and use simple batching
        if len(unique_names) > 5000:
            print(f"\n📊 Large dataset ({len(unique_names)} names), using simple alphabetical batching...")
            # Sort alphabetically for some grouping benefit
            sorted_names = sorted(unique_names)
            batches = [sorted_names[i:i + batch_size] for i in range(0, len(sorted_names), batch_size)]
            print(f"  ✓ Created {len(batches)} batches")
            return batches
        
        print(f"\n📊 Creating similarity-based batches...")
        
        try:
            # Create TF-IDF vectors with reduced features for speed
            vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(unique_names)
            
            # Fast greedy clustering
            batches = []
            used = set()
            
            for i in range(len(unique_names)):
                if i in used:
                    continue
                
                # Start new batch
                batch_indices = [i]
                used.add(i)
                
                # Only check next 500 items for similarity (not all remaining)
                check_limit = min(i + 500, len(unique_names))
                for j in range(i + 1, check_limit):
                    if j in used or len(batch_indices) >= batch_size:
                        break
                    
                    # Quick similarity check
                    sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
                    
                    if sim > 0.15:  # Higher threshold for speed
                        batch_indices.append(j)
                        used.add(j)
                
                batch = [unique_names[idx] for idx in batch_indices]
                batches.append(batch)
            
            print(f"  ✓ Created {len(batches)} similarity-based batches")
            return batches
            
        except Exception as e:
            print(f"  ⚠️  Error creating similarity batches: {e}, using simple batches")
            return [unique_names[i:i + batch_size] for i in range(0, len(unique_names), batch_size)]
    
    def _map_unique_names(self, unique_names: List[str], batch_size: int = 90, max_retries: int = 3,
                         intermediate_output_path: str = 'output/stage1_5_intermediate_UPD.json',
                         max_workers: int = 5,
                         resume_from_checkpoint: bool = True) -> Dict[str, Dict]:
        """
        Use LLM to map unique names to canonical names in batches (parallel processing).
        
        Args:
            unique_names: List of unique normalized names
            batch_size: Number of names per batch (default 130)
            max_retries: Maximum retry attempts for missing names (default 3)
            intermediate_output_path: Path to save intermediate results
            max_workers: Number of parallel workers (default 5)
            resume_from_checkpoint: Resume from checkpoint if exists (default True)
            
        Returns:
            Dictionary mapping normalized_name -> (mapped_name, confidence)
        """
        print(f"\n🤖 Mapping {len(unique_names)} unique names in batches of {batch_size}...")
        print(f"  Intermediate results will be saved to: {intermediate_output_path}")
        print(f"  Using {max_workers} parallel workers for faster processing")
        
        # Keep track of original list for validation
        original_unique_names = unique_names.copy()
        
        # Load checkpoint if exists and resume is enabled
        all_mappings = {}
        if resume_from_checkpoint and os.path.exists(intermediate_output_path):
            print(f"\n💾 Loading checkpoint from {intermediate_output_path}...")
            all_mappings = self._load_checkpoint(intermediate_output_path)
            
            if all_mappings:
                # Filter out already processed names
                remaining_names = [name for name in unique_names if name not in all_mappings]
                print(f"  ✓ {len(remaining_names)} names remaining to process (out of {len(unique_names)} total)")
                
                if not remaining_names:
                    print(f"  ✓ All names already processed! Skipping to validation...")
                    unique_names = []  # Empty list to skip processing
                else:
                    unique_names = remaining_names
            else:
                print(f"  ⚠️  Checkpoint file exists but is empty or invalid, starting fresh")
        
        # Create similarity-based batches (only remaining names)
        if unique_names:
            # Adjust batch size if canonical list is large
            current_batch_size = batch_size
            if len(self.canonical_theories) > 3000:
                current_batch_size = max(30, batch_size // 2)
                print(f"  ⚠️  Canonical list has {len(self.canonical_theories)} theories, reducing batch size to {current_batch_size}")
            
            batches = self._create_similarity_batches(unique_names, current_batch_size)
            total_batches = len(batches)
            print(f"  Total batches to process: {total_batches}")
        else:
            batches = []
            total_batches = 0
        
        # Process batches SEQUENTIALLY to update canonical list after each batch
        if batches:
            print(f"\\n🔄 Processing batches sequentially (to update canonical list dynamically)...")
            
            with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
                for batch_num, batch_names in enumerate(batches, 1):
                    try:
                        # Process batch
                        batch_mappings, new_theories = self._map_batch(batch_names, batch_num, total_batches)
                        
                        # Update all mappings
                        all_mappings.update(batch_mappings)
                        
                        # Add new theories to canonical list
                        if new_theories:
                            for theory in new_theories:
                                if theory not in self.canonical_theories:
                                    self.canonical_theories[theory] = {}  # Add with empty metadata
                            print(f"    📚 Canonical list now has {len(self.canonical_theories)} theories")
                            
                            # Adjust batch size if needed
                            if len(self.canonical_theories) > 3000 and current_batch_size > 30:
                                current_batch_size = max(30, current_batch_size // 2)
                                print(f"    ⚠️  Reducing batch size to {current_batch_size} due to large canonical list")
                        
                        # Save intermediate results every 5 batches (more frequent)
                        if batch_num % 5 == 0 or batch_num == total_batches:
                            self._save_intermediate_results(all_mappings, batch_num, intermediate_output_path, 
                                                          len(self.canonical_theories))
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'mapped': sum(1 for _, data in all_mappings.items() if data.get('mapped_name') is not None),
                            'canonical': len(self.canonical_theories)
                        })
                    except Exception as e:
                        print(f"\\n    ❌ Batch {batch_num} failed: {e}")
                        # Save checkpoint even on error
                        self._save_intermediate_results(all_mappings, batch_num, intermediate_output_path,
                                                      len(self.canonical_theories))
                        pbar.update(1)
        
        # Check for missing names and retry (use original list)
        missing_names = set(original_unique_names) - set(all_mappings.keys())
        
        if missing_names:
            print(f"\n🔄 Retrying {len(missing_names)} missing names...")
            
            for retry_attempt in range(1, max_retries + 1):
                if not missing_names:
                    break
                
                print(f"\n  Retry attempt {retry_attempt}/{max_retries} for {len(missing_names)} names")
                
                # Split missing names into batches
                retry_batches = [list(missing_names)[i:i + batch_size] for i in range(0, len(missing_names), batch_size)]
                
                with tqdm(total=len(retry_batches), desc=f"Retry {retry_attempt}", unit="batch") as pbar:
                    for batch_num, batch_names in enumerate(retry_batches, 1):
                        batch_mappings, new_theories = self._map_batch(batch_names, batch_num, len(retry_batches))
                        all_mappings.update(batch_mappings)
                        
                        # Add new theories to canonical list
                        if new_theories:
                            for theory in new_theories:
                                if theory not in self.canonical_theories:
                                    self.canonical_theories[theory] = {}
                        
                        # Save intermediate results
                        if batch_num % 10 == 0 or batch_num == len(retry_batches):
                            self._save_intermediate_results(all_mappings, batch_num, intermediate_output_path,
                                                          len(self.canonical_theories))
                        
                        pbar.update(1)
                
                # Update missing names
                missing_names = set(original_unique_names) - set(all_mappings.keys())
                
                if missing_names:
                    print(f"  Still missing {len(missing_names)} names after retry {retry_attempt}")
        
        # Add any remaining missing names as unmapped
        final_missing = set(original_unique_names) - set(all_mappings.keys())
        if final_missing:
            print(f"\n⚠️  {len(final_missing)} names could not be mapped after {max_retries} retries")
            for name in final_missing:
                all_mappings[name] = {
                    'mapped_name': None,
                    'confidence': 0.0,
                    'status': 'unmapped'
                }
        
        # Final verification
        if set(all_mappings.keys()) != set(original_unique_names):
            print(f"\n❌ ERROR: Mapping mismatch detected!")
            print(f"  Expected: {len(original_unique_names)} names")
            print(f"  Got: {len(all_mappings)} names")
            # Add any truly missing names
            for name in original_unique_names:
                if name not in all_mappings:
                    all_mappings[name] = {
                        'mapped_name': None,
                        'confidence': 0.0,
                        'status': 'unmapped'
                    }
        
        # Calculate statistics about ontology vs new theories
        mapped_to_ontology = sum(1 for _, data in all_mappings.items() 
                                if data.get('mapped_name') and data.get('is_from_ontology'))
        mapped_to_new = sum(1 for _, data in all_mappings.items() 
                           if data.get('mapped_name') and data.get('is_new_theory'))
        
        # Final save with completed status
        print(f"\n💾 Saving final results...")
        final_data = {
            'metadata': {
                'stage': 'stage1_5_llm_assistant_mapping',
                'status': 'completed',
                'total_mappings': len(all_mappings),
                'mapped_count': sum(1 for _, data in all_mappings.items() if data.get('mapped_name') is not None),
                'mapped_to_ontology': mapped_to_ontology,
                'mapped_to_new_theories': mapped_to_new,
                'not_valid_count': sum(1 for _, data in all_mappings.items() if data.get('status') == 'not_valid'),
                'original_ontology_count': len(self.original_ontology_theories),
                'canonical_theories_count': len(self.canonical_theories),
                'new_theories_discovered': len(self.canonical_theories) - len(self.original_ontology_theories),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'mappings': all_mappings
        }
        
        try:
            with open(intermediate_output_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            print(f"  ✓ Final checkpoint saved: {intermediate_output_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save final checkpoint: {e}")
        
        # Save final canonical theories list
        canonical_output_path = intermediate_output_path.replace('.json', '_canonical_theories.json')
        self._save_canonical_theories(canonical_output_path)
        
        # Count final statistics
        mapped_count = sum(1 for _, data in all_mappings.items() if data.get('mapped_name') is not None)
        original_theories = len(self.canonical_theories) - sum(1 for _, data in all_mappings.items() if data.get('mapped_name') and data.get('mapped_name') not in self.canonical_theories)
        new_theories_count = len(self.canonical_theories) - original_theories
        
        print(f"\n✓ Final results:")
        print(f"  - {mapped_count}/{len(original_unique_names)} names mapped")
        print(f"  - {len(self.canonical_theories)} total theories in canonical list")
        print(f"  - {new_theories_count} new theories discovered during mapping")
        
        return all_mappings
    
    def process_theories(self, input_path: str) -> List[TheoryEntry]:
        """
        Process theories from Stage 1 output.
        
        Args:
            input_path: Path to Stage 1 fuzzy matching output
            
        Returns:
            List of TheoryEntry objects
        """
        print(f"\n📂 Loading theories from {input_path}...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Get unmatched theories from Stage 1
        unmatched_theories = data.get('unmatched_theories', [])
        self.stats['total_input'] = len(unmatched_theories)
        
        print(f"✓ Loaded {len(unmatched_theories)} unmatched theories from Stage 1")
        
        # Step 1: Normalize names and group
        print(f"\n🔄 Step 1: Normalizing names and grouping...")
        groups = self._group_by_normalized_name(unmatched_theories)
        self.stats['unique_names'] = len(groups)
        
        print(f"✓ Found {len(groups)} unique normalized names")
        print(f"  Average group size: {len(unmatched_theories) / len(groups):.1f}")
        
        # Show sample groups
        print(f"\n📋 Sample groups:")
        for i, (norm_name, theories) in enumerate(list(groups.items())[:5]):
            print(f"  {i+1}. '{norm_name}' ({len(theories)} theories)")
            for theory in theories[:2]:
                print(f"     - {theory.get('original_name', '')}")
        
        # Step 2: Map unique names using LLM
        unique_names = list(groups.keys())
        intermediate_path = 'output/stage1_5_intermediate_UPD.json'
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
        mappings = self._map_unique_names(unique_names, intermediate_output_path=intermediate_path)
        
        # Count mapped
        self.stats['mapped_by_llm'] = sum(1 for _, data in mappings.items() if data.get('mapped_name') is not None)
        self.stats['not_valid'] = sum(1 for _, data in mappings.items() if data.get('status') == 'not_valid')
        self.stats['unmapped'] = len(unique_names) - self.stats['mapped_by_llm'] - self.stats['not_valid']
        
        # Step 3: Create TheoryEntry objects with mappings
        print(f"\n🔄 Step 3: Creating theory entries with mappings...")
        results = []
        
        for norm_name, theories in groups.items():
            # Get mapping for this normalized name
            mapping_data = mappings.get(norm_name, {'mapped_name': None, 'confidence': 0.0, 'status': 'unmapped'})
            mapped_name = mapping_data.get('mapped_name')
            confidence = mapping_data.get('confidence', 0.0)
            status = mapping_data.get('status', 'unmapped')
            is_from_ontology = mapping_data.get('is_from_ontology', False)
            is_new_theory = mapping_data.get('is_new_theory', False)
            
            # Create reasoning based on status
            if status == 'not_valid':
                reasoning = "Not a valid aging theory"
            elif mapped_name:
                if is_from_ontology:
                    reasoning = "Mapped by LLM to original ontology"
                elif is_new_theory:
                    reasoning = "Mapped by LLM to newly discovered theory"
                else:
                    reasoning = "Mapped by LLM"
            else:
                reasoning = "Not mapped by LLM"
            
            # Create TheoryEntry for each theory in group
            for theory in theories:
                entry = TheoryEntry(
                    theory_id=theory.get('theory_id', ''),
                    original_name=theory.get('original_name', ''),
                    normalized_name=norm_name,
                    mapped_name=mapped_name,
                    mapping_confidence_stage1_5=confidence,
                    mapping_reasoning=reasoning,
                    key_concepts=theory.get('key_concepts', []),
                    description=theory.get('description', ''),
                    evidence=theory.get('evidence', ''),
                    confidence_is_theory=theory.get('confidence_is_theory', 'unknown'),
                    mode=theory.get('mode', ''),
                    criteria_reasoning=theory.get('criteria_reasoning', ''),
                    paper_focus=theory.get('paper_focus', 0),
                    doi=theory.get('doi', ''),
                    pmid=theory.get('pmid', ''),
                    paper_title=theory.get('paper_title', ''),
                    timestamp=theory.get('timestamp', ''),
                    enriched_text=theory.get('enriched_text'),
                    concept_text=theory.get('concept_text'),
                    is_validated=theory.get('is_validated', False),
                    validation_reason=theory.get('validation_reason', '')
                )
                results.append(entry)
        
        self.stats['total_output'] = len(results)
        
        print(f"✓ Created {len(results)} theory entries")
        
        return results
    
    def save_results(self, results: List[TheoryEntry], output_path: str):
        """Save mapping results to JSON."""
        print(f"\n💾 Saving {len(results)} theories to {output_path}...")
        
        # Separate mapped and unmapped
        mapped = [r for r in results if r.mapped_name is not None]
        unmapped = [r for r in results if r.mapped_name is None]
        
        output_data = {
            'metadata': {
                'stage': 'stage1_5_llm_assistant_mapping',
                'statistics': self.stats,
                'mapped_count': len(mapped),
                'unmapped_count': len(unmapped),
                'mapping_rate': len(mapped) / len(results) * 100 if results else 0
            },
            'mapped_theories': [t.to_dict() for t in mapped],
            'unmapped_theories': [t.to_dict() for t in unmapped]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print mapping statistics."""
        print("\n" + "="*60)
        print("STAGE 1.5: LLM ASSISTANT MAPPING STATISTICS")
        print("="*60)
        print(f"Input theories (from Stage 1): {self.stats['total_input']}")
        print(f"Unique normalized names: {self.stats['unique_names']}")
        print(f"\nMapping results:")
        print(f"  Mapped by LLM: {self.stats['mapped_by_llm']} ({self.stats['mapped_by_llm']/self.stats['unique_names']*100:.1f}% of unique names)")
        print(f"  Unmapped: {self.stats['unmapped']} ({self.stats['unmapped']/self.stats['unique_names']*100:.1f}% of unique names)")
        
        # Calculate how many total theories were mapped
        total_mapped = sum(1 for _ in range(self.stats['total_output']) if _ < len([]))  # Will be calculated from results
        print(f"\nTotal theories mapped: {self.stats['mapped_by_llm']} unique names")
        print(f"Total theories unmapped: {self.stats['unmapped']} unique names")
        
        print(f"\nLLM Usage:")
        print(f"  Input tokens: {self.stats['total_input_tokens']:,}")
        print(f"  Output tokens: {self.stats['total_output_tokens']:,}")
        print(f"  Total cost: ${self.stats['total_cost']:.4f}")
        print("="*60)
    
    def print_sample_mappings(self, results: List[TheoryEntry], n: int = 10):
        """Print sample mappings for inspection."""
        print(f"\n📋 Sample Mappings (first {n}):")
        print("-" * 80)
        
        mapped = [r for r in results if r.mapped_name is not None][:n]
        
        for i, theory in enumerate(mapped, 1):
            print(f"\n{i}. Original: {theory.original_name}")
            print(f"   Normalized: {theory.normalized_name}")
            print(f"   Canonical: {theory.mapped_name}")
            print(f"   Confidence: {theory.mapping_confidence_stage1_5:.2f}")
            print(f"   Reasoning: {theory.mapping_reasoning}")


def main():
    """Run Stage 1.5 LLM assistant mapping."""
    print("🚀 Starting Stage 1.5: LLM Assistant Mapping\n")
    
    # Paths
    ontology_path = 'ontology/groups_ontology_alliases.json'
    input_path = 'output/stage1_fuzzy_matched.json'
    output_path = 'output/stage1_5_llm_mapped_UPD.json'
    
    # Initialize mapper
    mapper = LLMAssistantMapper(ontology_path=ontology_path)
    
    # Process theories
    results = mapper.process_theories(input_path)
    
    # Save results
    mapper.save_results(results, output_path)
    
    # Print statistics
    mapper.print_statistics()
    
    # Print sample mappings
    mapper.print_sample_mappings(results, n=10)
    
    print("\n✅ Stage 1.5 complete!")


if __name__ == '__main__':
    main()
