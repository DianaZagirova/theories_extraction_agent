"""
Stage 3: Refine Mapped Names to Align with Ontology

This stage takes the output from Stage 2 and further refines mapped names:
1. Load Stage 2 mappings
2. Extract unique mapped names (VALUES from Stage 2) that are NOT in initial ontology
3. Group these names alphabetically and create batches (like Stage 2)
4. Use LLM to further group and align with ontology
5. Save refined mappings

Input: Stage 2 output (mappings)
Output: Refined mappings with better ontology alignment
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient


class Stage3IterativeRefinement:
    """
    Further refine theory name mappings from Stage 2 by:
    1. Extracting unique mapped names (values) not in ontology
    2. Grouping alphabetically and creating batches
    3. Using LLM to align with ontology
    """
    
    def __init__(self,
                 stage2_path: str = 'output/stage2_grouped_theories.json',
                 ontology_path: str = 'ontology/groups_ontology_alliases.json'):
        """Initialize Stage 3 refinement."""
        self.stage2_path = Path(stage2_path)
        self.ontology_path = Path(ontology_path)
        
        # Load ontology
        self.ontology_theories = self._load_ontology()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE3', 'azure')
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY2'))
        else:
            self.llm = AzureOpenAIClient()
        self.model = 'gpt-4.1-mini'
        
        # Statistics
        self.stats = {
            'total_stage2_mappings': 0,
            'unique_mapped_names_from_stage2': 0,
            'names_not_in_ontology': 0,
            'total_batches': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
    
    def _load_ontology(self) -> Set[str]:
        """Load theory names from ontology (same logic as Stage 2)."""
        print(f"📂 Loading ontology from {self.ontology_path}...")
        
        with open(self.ontology_path, 'r') as f:
            data = json.load(f)
        
        theories = set()
        for category, subcats in data['TheoriesOfAging'].items():
            for subcat, theory_list in subcats.items():
                for theory in theory_list:
                    theories.add(theory['name'])
        
        print(f"  ✓ Loaded {len(theories)} ontology theories")
        return theories
    
    def _load_stage2_output(self) -> Tuple[Dict[str, str], Set[str]]:
        """Load Stage 2 mappings and extract unique mapped names."""
        print(f"\n📂 Loading Stage 2 output from {self.stage2_path}...")
        
        with open(self.stage2_path, 'r') as f:
            data = json.load(f)
        
        mappings = data.get('mappings', {})
        metadata = data.get('metadata', {})
        
        print(f"  ✓ Loaded {len(mappings)} mappings")
        print(f"  Status: {metadata.get('status', 'unknown')}")
        
        # Extract unique mapped names (values)
        unique_mapped_names = set(mappings.values())
        print(f"  ✓ Unique mapped names (values): {len(unique_mapped_names)}")
        
        self.stats['total_stage2_mappings'] = len(mappings)
        self.stats['unique_mapped_names_from_stage2'] = len(unique_mapped_names)
        
        return mappings, unique_mapped_names
    
    def _filter_names_not_in_ontology(self, unique_mapped_names: Set[str]) -> List[str]:
        """
        Filter out names that are already in the ontology.
        Only exact case-insensitive matches are filtered.
        
        Returns:
            List of unique mapped names NOT in ontology
        """
        print("\n🔍 Filtering names not in ontology...")
        
        # Create lowercase lookup for case-insensitive matching
        ontology_lower = {name.lower(): name for name in self.ontology_theories}
        
        names_not_in_ontology = []
        names_in_ontology = []
        
        for mapped_name in unique_mapped_names:
            mapped_lower = mapped_name.lower()
            
            # Check exact match (case-insensitive) only - no fuzzy matching
            if mapped_lower in ontology_lower:
                names_in_ontology.append(mapped_name)
            else:
                names_not_in_ontology.append(mapped_name)
        
        # Sort for consistency
        names_not_in_ontology = sorted(names_not_in_ontology)
        
        print(f"  ✓ Names in ontology (exact match): {len(names_in_ontology)}")
        print(f"  ⚠️  Names NOT in ontology (to process): {len(names_not_in_ontology)}")
        
        self.stats['names_not_in_ontology'] = len(names_not_in_ontology)
        
        return names_not_in_ontology
    
    def _group_by_first_char(self, names: List[str]) -> Dict[str, List[str]]:
        """
        Group names by first character (letter or symbol).
        
        Returns:
            Dictionary mapping first_char -> list of names
        """
        groups = defaultdict(list)
        
        for name in names:
            if not name:
                continue
            first_char = name[0].upper()
            groups[first_char].append(name)
        
        # Sort groups by key
        sorted_groups = dict(sorted(groups.items()))
        
        print(f"\n📊 Grouped into {len(sorted_groups)} alphabetical groups:")
        for char, names_list in sorted_groups.items():
            print(f"  {char}: {len(names_list)} names")
        
        return sorted_groups
    
    def _create_batches(self, alphabetical_groups: Dict[str, List[str]], 
                       max_batch_size: int = 100) -> List[Tuple[str, List[str]]]:
        """
        Create batches from alphabetical groups (same logic as Stage 2).
        
        Returns:
            List of (batch_id, names_list) tuples
        """
        batches = []
        current_batch = []
        current_batch_chars = []
        
        for char, names_list in alphabetical_groups.items():
            # If this group alone exceeds max_batch_size, split it
            if len(names_list) > max_batch_size:
                # First, try to fill current batch if it has space
                if current_batch and len(current_batch) < max_batch_size:
                    space_left = max_batch_size - len(current_batch)
                    current_batch.extend(names_list[:space_left])
                    current_batch_chars.append(char)
                    
                    # Flush filled batch
                    batch_id = f"batch_{'_'.join(current_batch_chars)}"
                    batches.append((batch_id, current_batch))
                    current_batch = []
                    current_batch_chars = []
                    
                    # Update names_list to remaining items
                    names_list = names_list[space_left:]
                elif current_batch:
                    # Current batch is full, flush it
                    batch_id = f"batch_{'_'.join(current_batch_chars)}"
                    batches.append((batch_id, current_batch))
                    current_batch = []
                    current_batch_chars = []
                
                # Split remaining large group into multiple batches
                for i in range(0, len(names_list), max_batch_size):
                    chunk = names_list[i:i + max_batch_size]
                    batch_id = f"batch_{char}_{i//max_batch_size + 1}"
                    batches.append((batch_id, chunk))
            
            # If current batch is empty or small, try to add more groups
            elif len(current_batch) == 0 or len(current_batch) + len(names_list) <= max_batch_size:
                # Add to current batch
                current_batch.extend(names_list)
                current_batch_chars.append(char)
            
            # If adding this group would exceed max_batch_size, flush current batch
            else:
                # Try to fill current batch to max_batch_size first
                space_left = max_batch_size - len(current_batch)
                if space_left > 0:
                    current_batch.extend(names_list[:space_left])
                    current_batch_chars.append(char)
                    
                    # Flush filled batch
                    batch_id = f"batch_{'_'.join(current_batch_chars)}"
                    batches.append((batch_id, current_batch))
                    
                    # Start new batch with remaining items
                    current_batch = names_list[space_left:].copy()
                    current_batch_chars = [char]
                else:
                    # Flush current batch
                    batch_id = f"batch_{'_'.join(current_batch_chars)}"
                    batches.append((batch_id, current_batch))
                    
                    # Start new batch with this group
                    current_batch = names_list.copy()
                    current_batch_chars = [char]
        
        # Flush remaining batch
        if current_batch:
            batch_id = f"batch_{'_'.join(current_batch_chars)}"
            batches.append((batch_id, current_batch))
        
        print(f"\n📊 Created {len(batches)} batches (smart batching to reach ~{max_batch_size} names):")
        for batch_id, names_list in batches[:10]:  # Show first 10
            print(f"  {batch_id}: {len(names_list)} names")
        if len(batches) > 10:
            print(f"  ... and {len(batches) - 10} more batches")
        
        return batches
    
    def _process_batch(self, names: List[str], batch_id: str,
                      batch_num: int, total_batches: int) -> Dict[str, str]:
        """
        Process a batch of names using LLM to align with ontology.
        
        Returns:
            Dictionary mapping name -> refined_name
        """
        print(f"\n🔄 Processing batch {batch_num}/{total_batches}: {batch_id}")
        print(f"  Input: {len(names)} names")
        
        # Create prompt
        prompt = self._create_refinement_prompt(names, batch_id)
        
        # Call LLM
        try:
            response = self.llm.client.chat.completions.create(
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
                max_tokens=31000
            )
            
            # Track token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
            
            self.stats['total_input_tokens'] += input_tokens
            self.stats['total_output_tokens'] += output_tokens
            self.stats['total_cost'] += cost
            
            print(f"  💰 Tokens: {input_tokens:,} in / {output_tokens:,} out | Cost: ${cost:.4f}")
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
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
                mappings = data.get('mappings', {})
                
                print(f"  ✓ Created {len(set(mappings.values()))} unique names from {len(mappings)} mappings")
                
                # Validate all input names are present
                output_names = set(mappings.keys())
                input_names_set = set(names)
                missing_names = input_names_set - output_names
                
                if missing_names:
                    print(f"  ⚠️  WARNING: {len(missing_names)} names missing from output!")
                    for name in missing_names:
                        mappings[name] = name
                        print(f"     Added missing name: {name}")
                
                return mappings
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error: {e}")
                print(f"  Response length: {len(response_text)} chars")
                # Return identity mappings on error
                return {name: name for name in names}
                
        except Exception as e:
            print(f"  ❌ Error calling LLM: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()[:500]}")
            return {name: name for name in names}
    
    def _create_refinement_prompt(self, names: List[str], batch_id: str) -> str:
        """Create prompt for refining names to align with ontology."""
        
        # Build ontology reference
        ontology_text = "\n# ONTOLOGY REFERENCE NAMES (prefer these when applicable)\n"
        ontology_text += "Try to match these names when the theory is the same:\n"
        for name in sorted(list(self.ontology_theories)):
            ontology_text += f"- {name}\n"
        
        names_text = "\n# THEORY NAMES TO STANDARDIZE\n"
        for i, name in enumerate(names, 1):
            names_text += f"{i}. {name}\n"
        
        prompt = f"""# TASK
You are given a list of potential aging theory names that are not normalized (use different wording for the same theory).
1. Identify all names that refer to the same THEORY and assign to them the same name.
2. If you assign the same name to several not stardart names, try to make it not too generic (preserve specific meaning) but in the same time not too specific.
- If you identified several names with that are too specific (for example, with several notions of the mechanisms, specific authors, etc) that refer to one theory - give one more generic name for all of them. Do not retain excessively specific names. 
- Do not combine names of theories that refer to different organs/tissues/diseases.
3. If a name doesn't match any other name, keep it as is or create a more clear standardized version. Single theories are accepted

{names_text}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):

{{
  "mappings": {{
    "Theory Name 1": "Standardized Name 1",
    "Theory Name 2": "Standardized Name 1",
    ...
  }}
}}

# CRITICAL REQUIREMENTS
- You must include ALL {len(names)} initial names in the output
"""
        
        return prompt
    
    def _load_checkpoint(self, checkpoint_path: str) -> tuple:
        """
        Load checkpoint from previous run.
        
        Returns:
            Tuple of (all_mappings, batches_completed, stats)
        """
        if not os.path.exists(checkpoint_path):
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
                # User wants to re-run - return empty to start fresh
                return {}, 0, None
        
        all_mappings = data.get('mappings', {})
        batches_completed = metadata.get('batches_completed', 0)
        
        # Restore stats
        stats = {
            'total_stage2_mappings': metadata.get('total_stage2_mappings', 0),
            'unique_mapped_names_from_stage2': metadata.get('unique_mapped_names_from_stage2', 0),
            'names_not_in_ontology': metadata.get('names_not_in_ontology', 0),
            'total_batches': metadata.get('total_batches', 0),
            'total_input_tokens': metadata.get('total_input_tokens', 0),
            'total_output_tokens': metadata.get('total_output_tokens', 0),
            'total_cost': metadata.get('total_cost', 0.0)
        }
        
        print(f"  ✓ Loaded checkpoint:")
        print(f"    - Batches completed: {batches_completed}/{stats['total_batches']}")
        print(f"    - Mappings: {len(all_mappings)}")
        print(f"    - Unique names: {len(set(all_mappings.values()))}")
        print(f"    - Cost so far: ${stats['total_cost']:.4f}")
        
        return all_mappings, batches_completed, stats
    
    def _save_intermediate_results(self, all_mappings: Dict[str, str],
                                   output_path: str,
                                   batches_completed: int,
                                   is_final: bool = False):
        """Save intermediate or final results."""
        
        # Count unique mapped names
        unique_mapped_names = len(set(all_mappings.values()))
        
        output_data = {
            'metadata': {
                'stage': 'stage3_iterative_refinement',
                'status': 'complete' if is_final else 'in_progress',
                'batches_completed': batches_completed,
                'total_batches': self.stats['total_batches'],
                'total_stage2_mappings': self.stats['total_stage2_mappings'],
                'unique_mapped_names_from_stage2': self.stats['unique_mapped_names_from_stage2'],
                'names_not_in_ontology': self.stats['names_not_in_ontology'],
                'total_mappings': len(all_mappings),
                'unique_mapped_names': unique_mapped_names,
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_cost': self.stats['total_cost'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'mappings': all_mappings,
            'ontology_theories': sorted(list(self.ontology_theories))
        }
        
        # Save to file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if not is_final:
            print(f"    💾 Checkpoint saved: {len(all_mappings)} mappings, {unique_mapped_names} unique names")
        else:
            print(f"\n💾 Final results saved to {output_path}")
    
    
    def run(self, output_path: str = 'output/stage3_refined_theories.json',
            batch_size: int = 200,
            resume_from_checkpoint: bool = False):
        """
        Run Stage 3 refinement.
        
        Process:
        1. Load Stage 2 mappings
        2. Extract unique mapped names (values) not in ontology
        3. Group by alphabet and create batches
        4. Process batches with LLM to align with ontology
        
        Args:
            output_path: Path to save final output
            batch_size: Maximum names per batch
            resume_from_checkpoint: If True, resume from checkpoint
        """
        print("=" * 80)
        print("STAGE 3: REFINE MAPPED NAMES TO ALIGN WITH ONTOLOGY")
        print("=" * 80)
        
        # Try to load checkpoint if requested
        start_batch = 0
        all_mappings = {}
        
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint(output_path)
            if checkpoint_data[0] is None and checkpoint_data[1] == 0 and checkpoint_data[2] is None:
                # User declined to re-run complete checkpoint
                print("\n❌ Exiting without changes.")
                return
            if checkpoint_data[0] is not None:
                all_mappings, start_batch, loaded_stats = checkpoint_data
                # Restore stats
                self.stats.update(loaded_stats)
                print(f"\n🔄 Resuming from batch {start_batch + 1}")
        
        # Load Stage 2 output
        stage2_mappings, unique_mapped_names = self._load_stage2_output()
        
        # Filter names not in ontology
        names_not_in_ontology = self._filter_names_not_in_ontology(unique_mapped_names)
        
        if len(names_not_in_ontology) == 0:
            print("\n✅ All names are already in ontology! Nothing to process.")
            return
        
        # Group by first character
        alphabetical_groups = self._group_by_first_char(names_not_in_ontology)
        
        # Create batches
        batches = self._create_batches(alphabetical_groups, batch_size)
        
        if not resume_from_checkpoint or start_batch == 0:
            self.stats['total_batches'] = len(batches)
        
        # Process batches
        if start_batch == 0:
            print(f"\n🚀 Processing {len(batches)} batches...")
        else:
            print(f"\n🚀 Processing remaining {len(batches) - start_batch} batches...")
        
        # Use tqdm for progress bar
        with tqdm(total=len(batches), initial=start_batch, desc="Processing batches", unit="batch") as pbar:
            for i, (batch_id, batch_names) in enumerate(batches, 1):
                # Skip already processed batches
                if i <= start_batch:
                    continue
                
                batch_mappings = self._process_batch(batch_names, batch_id, i, len(batches))
                
                # Merge mappings
                all_mappings.update(batch_mappings)
                
                # Save intermediate results: after batch 1 and then every 2 batches
                if i == 1 or i % 2 == 0 or i == len(batches):
                    self._save_intermediate_results(
                        all_mappings,
                        output_path,
                        batches_completed=i,
                        is_final=(i == len(batches))
                    )
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'mappings': len(all_mappings),
                    'unique': len(set(all_mappings.values()))
                })
                
                # Small delay to avoid rate limiting
                if i < len(batches):
                    time.sleep(1)
        
        # Save final results
        print(f"\n{'=' * 80}")
        print("FINAL RESULTS")
        print(f"{'=' * 80}")
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"\n📊 Final Statistics:")
        print(f"  Total Stage 2 mappings: {self.stats['total_stage2_mappings']}")
        print(f"  Unique mapped names from Stage 2: {self.stats['unique_mapped_names_from_stage2']}")
        print(f"  Names NOT in ontology (processed): {self.stats['names_not_in_ontology']}")
        print(f"  Total refined mappings: {len(all_mappings)}")
        print(f"  Unique refined names: {len(set(all_mappings.values()))}")
        print(f"  Total batches processed: {self.stats['total_batches']}")
        print(f"  Total tokens: {self.stats['total_input_tokens']:,} in / {self.stats['total_output_tokens']:,} out")
        print(f"  Total cost: ${self.stats['total_cost']:.4f}")
        
        print("\n✅ Stage 3 complete!")


def main(resume: bool = False):
    """Main entry point.
    
    Args:
        resume: If True, resume from checkpoint
    """
    refiner = Stage3IterativeRefinement(
        stage2_path='output/stage2_grouped_theories.json',
        ontology_path='ontology/groups_ontology_alliases.json'
    )
    
    refiner.run(
        output_path='output/stage3_refined_theories.json',
        batch_size=500, #500
        resume_from_checkpoint=resume
    )


if __name__ == '__main__':
    import sys
    # Check if --resume flag is passed
    resume = '--resume' in sys.argv
    main(resume=resume)
