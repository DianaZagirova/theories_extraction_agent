"""
Stage 2: Group Normalization and Standardization

This stage processes the output from Stage 1.5 to further normalize and group theory names.

Process:
1. Extract unique names that need normalization:
   - Mapped theories from stage 1.5 that are NOT in the initial ontology
   - Unmapped theories (original names)
2. Group names alphabetically by first letter/symbol
3. Create batches (max 100 unique names per batch)
4. Use LLM to group similar theories and standardize names
5. Track existing groups across iterations to maintain consistency
6. Validate that all input names appear in output

Input: output/stage1_5_llm_mapped.json
Output: output/stage2_grouped_theories.json
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import time
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient

SAVE_PROMPT_FOR_DEBUG = False


class Stage2GroupNormalizer:
    """
    Stage 2: Group and standardize theory names using LLM.
    
    Strategy:
    1. Extract unique names needing normalization
    2. Group alphabetically by first character
    3. Create batches (≤100 names per batch)
    4. Use LLM to group and standardize names
    5. Track existing groups to maintain consistency
    """
    
    def __init__(self, 
                 stage1_5_path: str = 'output/stage1_5_llm_mapped.json',
                 ontology_path: str = 'ontology/groups_ontology_alliases.json'):
        """Initialize Stage 2 normalizer."""
        self.stage1_5_path = Path(stage1_5_path)
        self.ontology_path = Path(ontology_path)
        
        # Load initial ontology theories
        self.initial_ontology_theories = self._load_initial_ontology()
        
        # Initialize LLM client
        self.use_module = os.getenv('USE_MODULE_FILTERING_LLM_STAGE2', 'azure')
        if self.use_module == 'openai':
            self.llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY2'))
        else:
            self.llm = AzureOpenAIClient()
        self.model = 'gpt-4.1-mini'
        
        # Track existing groups across batches
        self.existing_groups: Dict[str, Set[str]] = {}  # group_name -> set of theory names
        
        # Statistics
        self.stats = {
            'total_input_names': 0,
            'total_batches': 0,
            'total_groups_created': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
    
    def _load_initial_ontology(self) -> Set[str]:
        """Load theory names from initial ontology."""
        print(f"📂 Loading initial ontology from {self.ontology_path}...")
        
        with open(self.ontology_path, 'r') as f:
            data = json.load(f)
        
        theories = set()
        for category, subcats in data['TheoriesOfAging'].items():
            for subcat, theory_list in subcats.items():
                for theory in theory_list:
                    theories.add(theory['name'])
        
        print(f"✓ Loaded {len(theories)} theories from initial ontology")
        return theories
    
    def _load_stage1_5_data(self) -> Tuple[List[str], List[str]]:
        """
        Load stage 1.5 data and extract unique names needing normalization.
        
        Returns:
            Tuple of (mapped_not_in_ontology, unmapped_original_names)
        """
        print(f"\n📂 Loading Stage 1.5 data from {self.stage1_5_path}...")
        
        with open(self.stage1_5_path, 'r') as f:
            data = json.load(f)
        
        mapped_theories = data.get('mapped_theories', [])
        unmapped_theories = data.get('unmapped_theories', [])
        
        print(f"  Loaded {len(mapped_theories)} mapped theories")
        print(f"  Loaded {len(unmapped_theories)} unmapped theories")
        
        # Extract mapped names NOT in initial ontology
        mapped_not_in_ontology = []
        for theory in mapped_theories:
            mapped_name = theory.get('mapped_name')
            if mapped_name and mapped_name not in self.initial_ontology_theories:
                mapped_not_in_ontology.append(mapped_name)
        
        # Extract original names from unmapped theories
        unmapped_original_names = []
        for theory in unmapped_theories:
            original_name = theory.get('original_name', '')
            if original_name:
                unmapped_original_names.append(original_name)
        
        print(f"\n📊 Extraction results:")
        print(f"  Mapped (not in ontology): {len(mapped_not_in_ontology)}")
        print(f"  Unmapped (original names): {len(unmapped_original_names)}")
        
        return mapped_not_in_ontology, unmapped_original_names
    
    def _get_unique_names(self, mapped_names: List[str], unmapped_names: List[str]) -> List[str]:
        """Get unique names from both lists."""
        all_names = mapped_names + unmapped_names
        unique_names = sorted(set(all_names))
        
        print(f"\n📊 Total unique names to process: {len(unique_names)}")
        return unique_names
    
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
        Create batches from alphabetical groups.
        
        Smart batching:
        - If a group has ≤ max_batch_size names, it's a single batch
        - If a group has > max_batch_size names, split it
        - Combine small consecutive groups to reach max_batch_size when possible
        - If current batch is smaller than max_batch_size, keep adding groups
        
        Returns:
            List of (batch_id, names_list) tuples
        """
        batches = []
        current_batch = []
        current_batch_chars = []
        
        for char, names_list in alphabetical_groups.items():
            # If this group alone exceeds max_batch_size, split it
            if len(names_list) > max_batch_size:
                # First, flush current batch if any
                if current_batch:
                    batch_id = f"batch_{'_'.join(current_batch_chars)}"
                    batches.append((batch_id, current_batch))
                    current_batch = []
                    current_batch_chars = []
                
                # Split large group into multiple batches
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
    
    def _create_grouping_prompt(self, names: List[str], batch_id: str) -> str:
        """
        Create prompt for LLM to group and standardize theory names.
        
        Args:
            names: List of theory names to group
            batch_id: Batch identifier
            
        Returns:
            Prompt string
        """
        # Build list of existing groups
        existing_groups_text = ""
        if self.existing_groups:
            existing_groups_text = "\n# EXISTING GROUPS (from previous batches)\n"
            existing_groups_text += "Use these group names when appropriate to maintain consistency:\n"
            for group_name in sorted(self.existing_groups.keys()):
                existing_groups_text += f"- {group_name}\n"
        
        # Build list of initial ontology theories
        ontology_text = "\n# INITIAL ONTOLOGY THEORIES\n"
        ontology_text += "These are already standardized and should NOT be changed (you can use these name):\n"
        for theory_name in sorted(self.initial_ontology_theories):
            ontology_text += f"- {theory_name}\n"
        
        # Build list of names to process
        names_text = "\n# THEORY NAMES TO GROUP AND STANDARDIZE\n"
        for i, name in enumerate(names, 1):
            names_text += f"{i}. {name}\n"


       
        prompt = f"""# TASK
You are given a list of possible aging theory names. 
1. Analyze the whole input and it there are same theory written in a different way but refering to the same concept, group them together.
2. Create a standardized name for each such case 
3. Ensure consistency with existing names with previous batches

# INSTRUCTIONS
1. Identify theories that mean the same 
   - Example: "Insulin/Igf-1 Signaling", "Insulin/Igf-1 Signaling Pathway", "Insulin/Igf-1 Signaling Theory" → same name
   
2. Create a clear, standardized name for each theory
- Try to reuse the existing names, if possible
- Not too generic (preserve specific meaning)
- Not too specific (allow reasonable clustering)
   
3. If a theory matches an existing name, use that name
   - Check the EXISTING GROUPS list below
   - Reuse names when theories are the same across batches
   
4. Note
   - If a name doesn't fit any another theory's name, create a new individual name for it
   - Single theories are accepted as individual clusters

{existing_groups_text}{ontology_text}{names_text}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):

{{
  "mappings": {{
    "Theory Name 1": "Standardized Name 1",
    "Theory Name 2": "Standardized Name 1",
    "Theory Name 3": "Standardized Name 2",
    ...
  }}
}}

# CRITICAL REQUIREMENTS
- You must include ALL {len(names)} initial names in the output
"""
        
        if SAVE_PROMPT_FOR_DEBUG:
            debug_path = f'prompt_stage2_{batch_id}.txt'
            with open(debug_path, 'w') as f:
                f.write(prompt)
            print(f"  💾 Debug prompt saved to {debug_path}")
        
        return prompt
    
    def _process_batch(self, batch_id: str, names: List[str], 
                      batch_num: int, total_batches: int) -> Tuple[Dict[str, str], Dict]:
        """
        Process a single batch using LLM.
        
        Args:
            batch_id: Batch identifier
            names: List of theory names
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            Tuple of (mappings_dict, batch_metadata)
            - mappings_dict: Dictionary mapping initial_name -> mapped_name
            - batch_metadata: Dictionary with batch processing metadata
        """
        print(f"\n🔄 Processing batch {batch_num}/{total_batches}: {batch_id}")
        print(f"  Input: {len(names)} names")
        
        # Create prompt
        prompt = self._create_grouping_prompt(names, batch_id)
        
        # Call LLM
        try:
            response = self.llm.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in aging biology"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=17000
            )
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            
            # Calculate cost (for gpt-4o-mini: $0.150/1M input, $0.600/1M output)
            cost = (input_tokens / 1_000_000) * 0.150 + (output_tokens / 1_000_000) * 0.600
            
            self.stats['total_input_tokens'] += input_tokens
            self.stats['total_output_tokens'] += output_tokens
            self.stats['total_cost'] += cost
            
            print(f"  💰 Tokens: {input_tokens:,} in / {output_tokens:,} out | Cost: ${cost:.4f}")
            
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
                mappings = data.get('mappings', {})
                
                # Convert mappings to groups format for internal processing
                groups = {}
                for initial_name, mapped_name in mappings.items():
                    if mapped_name not in groups:
                        groups[mapped_name] = []
                    groups[mapped_name].append(initial_name)
                
                print(f"  ✓ Created {len(groups)} PROCESSED NAMES from {len(mappings)} INITIAL NAMES")
                
                # Validate: check all input names are present
                output_names = set(mappings.keys())
                input_names_set = set(names)
                missing_names = input_names_set - output_names
                extra_names = output_names - input_names_set
                
                if missing_names:
                    print(f"  ⚠️  WARNING: {len(missing_names)} names missing from output!")
                    print(f"     Missing: {list(missing_names)[:5]}...")
                    # Add missing names as individual mappings
                    for name in missing_names:
                        mappings[name] = name
                        groups[name] = [name]
                        print(f"     Added missing name as individual mapping: {name}")
                
                if extra_names:
                    print(f"  ⚠️  WARNING: {len(extra_names)} extra names in output!")
                    print(f"     Extra: {list(extra_names)[:5]}...")
                
                # Update existing groups
                for group_name, group_members in groups.items():
                    if group_name not in self.existing_groups:
                        self.existing_groups[group_name] = set()
                    self.existing_groups[group_name].update(group_members)
                
                # Create batch metadata
                batch_metadata = {
                    'batch_id': batch_id,
                    'total_input_names': len(names),
                    'total_groups': len(groups),
                    'missing_names_count': len(missing_names),
                    'extra_names_count': len(extra_names)
                }
                
                return mappings, batch_metadata
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing error: {e}")
                print(f"  Response length: {len(response_text)} chars")
                print(f"  Last 200 chars: ...{response_text[-200:]}")
                # Return empty dict on error
                return {}, {'batch_id': batch_id, 'error': 'JSON parsing error'}
                
        except Exception as e:
            print(f"  ❌ Error calling LLM: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()[:500]}")
            return {}, {'batch_id': batch_id, 'error': str(e)}
    
    def _save_intermediate_results(self, all_mappings: Dict[str, str],
                                   batch_metadata_list: List[Dict],
                                   output_path: str,
                                   batches_completed: int,
                                   is_final: bool = False):
        """Save intermediate or final mapping results."""
        # Count unique mapped names (groups)
        unique_mapped_names = len(set(all_mappings.values()))
        
        # Create output structure
        output_data = {
            'metadata': {
                'stage': 'stage2_group_normalization',
                'status': 'complete' if is_final else 'in_progress',
                'batches_completed': batches_completed,
                'total_batches': self.stats['total_batches'],
                'total_mappings': len(all_mappings),
                'unique_mapped_names': unique_mapped_names,
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_cost': self.stats['total_cost'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'mappings': all_mappings,
            'initial_ontology_theories': sorted(list(self.initial_ontology_theories)),
            'batch_metadata': batch_metadata_list
        }
        
        # Save to file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if not is_final:
            print(f"    💾 Checkpoint saved: {len(all_mappings)} mappings, {unique_mapped_names} unique names")
    
    def _save_results(self, all_mappings: Dict[str, str], 
                     output_path: str,
                     original_unique_names: List[str],
                     batch_metadata_list: List[Dict] = None):
        """Save final mapping results."""
        # Validate all names are present
        output_names = set(all_mappings.keys())
        input_names_set = set(original_unique_names)
        missing_names = input_names_set - output_names
        
        if missing_names:
            print(f"\n⚠️  WARNING: {len(missing_names)} names missing from final output!")
            print(f"   Adding them as individual mappings...")
            for name in missing_names:
                all_mappings[name] = name
        
        # Count unique mapped names (groups)
        unique_mapped_names = len(set(all_mappings.values()))
        
        # Create output structure
        output_data = {
            'metadata': {
                'stage': 'stage2_group_normalization',
                'total_input_names': len(original_unique_names),
                'total_mappings': len(all_mappings),
                'unique_mapped_names': unique_mapped_names,
                'total_batches': self.stats['total_batches'],
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_cost': self.stats['total_cost'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'mappings': all_mappings,
            'initial_ontology_theories': sorted(list(self.initial_ontology_theories)),
            'batch_metadata': batch_metadata_list if batch_metadata_list else []
        }
        
        # Save to file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"\n📊 Final Statistics:")
        print(f"  Total input names: {len(original_unique_names)}")
        print(f"  Total mappings: {len(all_mappings)}")
        print(f"  Unique mapped names (groups): {unique_mapped_names}")
        print(f"  Total batches processed: {self.stats['total_batches']}")
        print(f"  Total tokens: {self.stats['total_input_tokens']:,} in / {self.stats['total_output_tokens']:,} out")
        print(f"  Total cost: ${self.stats['total_cost']:.4f}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> tuple:
        """
        Load checkpoint from previous run.
        
        Returns:
            Tuple of (all_mappings, batch_metadata_list, batches_completed, stats)
        """
        if not os.path.exists(checkpoint_path):
            return None, None, 0, None
        
        print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        
        # Check if it's complete
        if metadata.get('status') == 'complete':
            print("  ⚠️  Checkpoint is already complete!")
            return None, None, 0, None
        
        all_mappings = data.get('mappings', {})
        batch_metadata_list = data.get('batch_metadata', [])
        batches_completed = metadata.get('batches_completed', 0)
        
        # Restore stats
        stats = {
            'total_input_names': metadata.get('total_input_names', 0),
            'total_batches': metadata.get('total_batches', 0),
            'total_groups_created': 0,
            'total_input_tokens': metadata.get('total_input_tokens', 0),
            'total_output_tokens': metadata.get('total_output_tokens', 0),
            'total_cost': metadata.get('total_cost', 0.0)
        }
        
        print(f"  ✓ Loaded checkpoint:")
        print(f"    - Batches completed: {batches_completed}/{stats['total_batches']}")
        print(f"    - Mappings: {len(all_mappings)}")
        print(f"    - Unique names: {len(set(all_mappings.values()))}")
        print(f"    - Cost so far: ${stats['total_cost']:.4f}")
        
        return all_mappings, batch_metadata_list, batches_completed, stats
    
    def run(self, output_path: str = 'output/stage2_grouped_theories.json',
            max_batch_size: int = 400,
            resume_from_checkpoint: bool = False):
        """
        Run Stage 2 normalization.
        
        Args:
            output_path: Path to save output
            max_batch_size: Maximum names per batch
        """
        print("=" * 80)
        print("STAGE 2: GROUP NORMALIZATION AND STANDARDIZATION")
        print("=" * 80)
        
        # Try to load checkpoint if requested
        start_batch = 0
        all_mappings = {}
        batch_metadata_list = []
        
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint(output_path)
            if checkpoint_data[0] is not None:
                all_mappings, batch_metadata_list, start_batch, loaded_stats = checkpoint_data
                # Restore stats
                self.stats.update(loaded_stats)
                print(f"\n🔄 Resuming from batch {start_batch + 1}")
        
        # Load data
        mapped_names, unmapped_names = self._load_stage1_5_data()
        unique_names = self._get_unique_names(mapped_names, unmapped_names)
        
        if not resume_from_checkpoint or start_batch == 0:
            self.stats['total_input_names'] = len(unique_names)
        
        # Group by first character
        alphabetical_groups = self._group_by_first_char(unique_names)
        
        # Create batches
        batches = self._create_batches(alphabetical_groups, max_batch_size)
        
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
                batch_mappings, batch_metadata = self._process_batch(batch_id, batch_names, i, len(batches))
                
                # Store batch metadata
                batch_metadata_list.append(batch_metadata)
                
                # Merge mappings
                all_mappings.update(batch_mappings)
                
                # Save intermediate results: after batch 1 and then every 5 batches
                if i == 1 or i % 5 == 0 or i == len(batches):
                    self._save_intermediate_results(
                        all_mappings, 
                        batch_metadata_list, 
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
        self._save_results(all_mappings, output_path, unique_names, batch_metadata_list)
        
        print("\n✅ Stage 2 complete!")


def main(resume: bool = False):
    """
    Main entry point.
    
    Args:
        resume: If True, resume from checkpoint
    """
    normalizer = Stage2GroupNormalizer(
        stage1_5_path='output/stage1_5_llm_mapped.json',
        ontology_path='ontology/groups_ontology_alliases.json'
    )
    
    normalizer.run(
        output_path='output/stage2_grouped_theories.json',
        max_batch_size=200,  # Use larger batches for efficiency
        resume_from_checkpoint=resume
    )


if __name__ == '__main__':
    import sys
    # Check if --resume flag is passed
    resume = '--resume' in sys.argv
    main(resume=resume)
