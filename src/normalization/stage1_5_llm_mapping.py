"""
Stage 2: LLM-Based Mapping to Canonical Theories

Maps unmatched theories from Stage 1 to canonical theories using LLM.
This stage sits between Stage 1 (fuzzy matching) and Stage 2 (full extraction).

Strategy:
1. Take theories that Stage 1 couldn't match
2. Use LLM to validate if it's a real aging theory
3. Try to map to canonical theory from ontology
4. Process in batches for efficiency
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from tqdm import tqdm
import re
from collections import defaultdict
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_integration import AzureOpenAIClient, OpenAIClient

# Constants
MAX_TOKENS = 16000
MAX_BATCH_CHARS = 50000  # Approximate character limit per batch (conservative estimate)
DEFAULT_BATCH_SIZE = 30  # Fallback if clustering produces large batches

# Pricing (per 1M tokens)
PRICING = {
    'gpt-4.1-mini': {
        'input': 0.04,   # $0.04 per 1M input tokens
        'output': 1.60   # $1.60 per 1M output tokens
    },
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.60
    }
}


@dataclass
class MappingResult:
    """Result of LLM mapping."""
    theory_id: str
    original_name: str
    is_valid_theory: bool
    validation_reasoning: str
    is_mapped: bool
    novelty_reasoning: str
    canonical_name: Optional[str] = None
    mapping_confidence: float = 0.0
    is_novel: bool = False    
    proposed_name: Optional[str] = None


class CheckpointDB:
    """Database for checkpointing LLM mapping results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()  # Thread-safe operations
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mapping_results (
                theory_id TEXT PRIMARY KEY,
                original_name TEXT,
                is_valid_theory INTEGER,
                validation_reasoning TEXT,
                is_mapped INTEGER,
                novelty_reasoning TEXT,
                canonical_name TEXT,
                mapping_confidence REAL,
                is_novel INTEGER,
                proposed_name TEXT,
                batch_number INTEGER,
                timestamp TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_metadata (
                batch_number INTEGER PRIMARY KEY,
                theory_count INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL,
                timestamp TEXT,
                completed INTEGER DEFAULT 1
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        self.conn.commit()
    
    def save_result(self, result: MappingResult, batch_number: int):
        """Save a single mapping result."""
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO mapping_results 
                (theory_id, original_name, is_valid_theory, validation_reasoning, 
                 is_mapped, novelty_reasoning, canonical_name, mapping_confidence,
                 is_novel, proposed_name, batch_number, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.theory_id,
                result.original_name,
                1 if result.is_valid_theory else 0,
                result.validation_reasoning,
                1 if result.is_mapped else 0,
                result.novelty_reasoning,
                result.canonical_name,
                result.mapping_confidence,
                1 if result.is_novel else 0,
                result.proposed_name,
                batch_number,
                time.strftime('%Y-%m-%d %H:%M:%S')
            ))
            self.conn.commit()
    
    def save_batch_metadata(self, batch_number: int, theory_count: int, 
                           input_tokens: int, output_tokens: int, cost: float):
        """Save batch processing metadata."""
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO batch_metadata 
                (batch_number, theory_count, input_tokens, output_tokens, cost, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                batch_number,
                theory_count,
                input_tokens,
                output_tokens,
                cost,
                time.strftime('%Y-%m-%d %H:%M:%S')
            ))
            self.conn.commit()
    
    def get_processed_theory_ids(self) -> set:
        """Get set of already processed theory IDs."""
        cursor = self.conn.execute("SELECT theory_id FROM mapping_results")
        return {row[0] for row in cursor.fetchall()}
    
    def get_completed_batches(self) -> set:
        """Get set of completed batch numbers."""
        cursor = self.conn.execute("SELECT batch_number FROM batch_metadata WHERE completed = 1")
        return {row[0] for row in cursor.fetchall()}
    
    def get_all_results(self) -> List[MappingResult]:
        """Retrieve all mapping results."""
        cursor = self.conn.execute("""
            SELECT theory_id, original_name, is_valid_theory, validation_reasoning,
                   is_mapped, novelty_reasoning, canonical_name, mapping_confidence,
                   is_novel, proposed_name
            FROM mapping_results
            ORDER BY batch_number, theory_id
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append(MappingResult(
                theory_id=row[0],
                original_name=row[1],
                is_valid_theory=bool(row[2]),
                validation_reasoning=row[3],
                is_mapped=bool(row[4]),
                novelty_reasoning=row[5],
                canonical_name=row[6],
                mapping_confidence=row[7],
                is_novel=bool(row[8]),
                proposed_name=row[9]
            ))
        return results
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics."""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_valid_theory = 1 THEN 1 ELSE 0 END) as valid,
                SUM(CASE WHEN is_mapped = 1 THEN 1 ELSE 0 END) as mapped,
                SUM(CASE WHEN is_novel = 1 THEN 1 ELSE 0 END) as novel,
                SUM(CASE WHEN is_valid_theory = 0 THEN 1 ELSE 0 END) as invalid
            FROM mapping_results
        """)
        row = cursor.fetchone()
        
        cursor2 = self.conn.execute("""
            SELECT 
                COUNT(*) as batches,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(cost) as total_cost
            FROM batch_metadata
        """)
        batch_row = cursor2.fetchone()
        
        return {
            'total_processed': row[0],
            'valid_theories': row[1],
            'mapped_to_canonical': row[2],
            'novel_theories': row[3],
            'invalid_theories': row[4],
            'batch_count': batch_row[0],
            'total_input_tokens': batch_row[1] or 0,
            'total_output_tokens': batch_row[2] or 0,
            'total_cost': batch_row[3] or 0.0
        }
    
    def save_metadata(self, key: str, value: str):
        """Save run metadata."""
        self.conn.execute("""
            INSERT OR REPLACE INTO run_metadata (key, value)
            VALUES (?, ?)
        """, (key, value))
        self.conn.commit()
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get run metadata."""
        cursor = self.conn.execute("SELECT value FROM run_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class LLMMapper:
    """Map theories to canonical names using LLM."""
    
    def __init__(self, ontology_path: str = 'ontology/groups_ontology_alliases.json',
                 mechanisms_path: str = 'ontology/group_ontology_mechanisms.json'):
        self.ontology_path = Path(ontology_path)
        self.mechanisms_path = Path(mechanisms_path)
        self.canonical_theories = self.load_ontology()
        
        self.stats = {
            'total_processed': 0,
            'valid_theories': 0,
            'invalid_theories': 0,
            'mapped_to_canonical': 0,
            'novel_theories': 0,
            'batch_count': 0,
            'clusters_found': 0,
            'avg_cluster_size': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0
        }
        self.use_module_norm = os.getenv('USE_MODULE_FILTERING_LLM', 'openai')
        if self.use_module_norm == 'openai':
            self.llm = OpenAIClient()
        else:
            self.llm = AzureOpenAIClient()
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
       
       
    
    def load_ontology(self) -> Dict[str, Dict]:
        """Load canonical theories with mechanisms from ontology."""
        print(f"📂 Loading ontology from {self.ontology_path} and {self.mechanisms_path}...")
        
        # Load aliases
        with open(self.ontology_path, 'r') as f:
            aliases_data = json.load(f)
        
        # Load mechanisms (NEW FORMAT: array of objects with lowercase keys)
        with open(self.mechanisms_path, 'r') as f:
            mechanisms_list = json.load(f)
        
        # Convert mechanisms list to dict for lookup
        mechanisms_dict = {item['theory_name']: item for item in mechanisms_list}
        
        # Build canonical theories dict
        canonical = {}
        for category, subcats in aliases_data['TheoriesOfAging'].items():
            for subcat, theories in subcats.items():
                for theory in theories:
                    name = theory['name']
                    mech_data = mechanisms_dict.get(name, {})
                    
                    canonical[name] = {
                        'name': name,
                        'aliases': theory.get('aliases', []),
                        'abbreviations': theory.get('abbreviations', []),
                        'category': category,
                        'subcategory': subcat,
                        'mechanisms': mech_data.get('mechanisms', []),
                        'key_players': mech_data.get('key_players', []),
                        'pathways': mech_data.get('pathways', [])
                    }
        
        print(f"✓ Loaded {len(canonical)} canonical theories from ontology")
        print(f"✓ Loaded {len(mechanisms_dict)} theories with mechanisms")
        return canonical
    
    def _normalize_for_clustering(self, name: str) -> str:
        """Normalize theory name for clustering similar theories."""
        normalized = name.lower().strip()
        
        # Remove common words
        for word in ['theory', 'hypothesis', 'model', 'framework', 'of', 'aging', 'ageing', 'the', 'a', 'an']:
            normalized = re.sub(r'\b' + word + r'\b', '', normalized)
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _cluster_theories(self, theories: List[Dict]) -> List[List[Dict]]:
        """
        Cluster theories by normalized name similarity.
        Theories with similar core concepts are grouped together.
        
        Returns:
            List of clusters, where each cluster is a list of theories
        """
        # Group by normalized name
        clusters_dict = defaultdict(list)
        
        for theory in theories:
            name = theory.get('name', theory.get('original_name', ''))
            normalized = self._normalize_for_clustering(name)
            
            # Use first 3 significant words as cluster key
            words = normalized.split()[:3]
            cluster_key = ' '.join(words) if words else 'misc'
            
            clusters_dict[cluster_key].append(theory)
        
        # Sort clusters deterministically:
        # 1. By size (largest first)
        # 2. By cluster key (alphabetically) for same-sized clusters
        # 3. Within each cluster, sort by processing_order for reproducibility
        for cluster in clusters_dict.values():
            cluster.sort(key=lambda t: t.get('_processing_order', 0))
        
        clusters = sorted(
            clusters_dict.values(), 
            key=lambda c: (-len(c), min(t.get('_processing_order', 0) for t in c))
        )
        
        self.stats['clusters_found'] = len(clusters)
        if clusters:
            self.stats['avg_cluster_size'] = sum(len(c) for c in clusters) / len(clusters)
        
        return clusters
    
    def _estimate_batch_size(self, theories: List[Dict]) -> int:
        """Estimate how many characters a batch of theories will use."""
        total_chars = 0
        for theory in theories:
            # Estimate: theory_id + name + concept_text
            total_chars += len(theory.get('theory_id', ''))
            total_chars += len(theory.get('name', theory.get('original_name', '')))
            concept_text = theory.get('concept_text', '')
            total_chars += min(len(concept_text), 1900)  # Truncated at 1900
            total_chars += 100  # Overhead for formatting
        
        return total_chars
    
    def _create_smart_batches(self, theories: List[Dict]) -> List[List[Dict]]:
        """
        Create batches using smart clustering strategy:
        1. Cluster similar theories together
        2. Pack clusters into batches respecting size limits
        3. Split large clusters if needed
        
        Returns:
            List of batches, where each batch is a list of theories
        """
        print("\n🧩 Clustering theories by similarity...")
        clusters = self._cluster_theories(theories)
        
        print(f"✓ Found {len(clusters)} clusters")
        print(f"  Average cluster size: {self.stats['avg_cluster_size']:.1f}")
        print(f"  Largest cluster: {max(len(c) for c in clusters)} theories")
        
        # Pack clusters into batches
        batches = []
        current_batch = []
        current_size = 0
        
        for cluster in clusters:
            cluster_size = self._estimate_batch_size(cluster)
            
            # If cluster alone exceeds limit, split it
            if cluster_size > MAX_BATCH_CHARS:
                # Flush current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                
                # Split large cluster into sub-batches
                sub_batch = []
                sub_size = 0
                for theory in cluster:
                    theory_size = self._estimate_batch_size([theory])
                    
                    if sub_size + theory_size > MAX_BATCH_CHARS and sub_batch:
                        batches.append(sub_batch)
                        sub_batch = []
                        sub_size = 0
                    
                    sub_batch.append(theory)
                    sub_size += theory_size
                
                if sub_batch:
                    batches.append(sub_batch)
            
            # If adding cluster to current batch exceeds limit, start new batch
            elif current_size + cluster_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = cluster.copy()
                current_size = cluster_size
            
            # Otherwise, add cluster to current batch
            else:
                current_batch.extend(cluster)
                current_size += cluster_size
        
        # Add remaining theories
        if current_batch:
            batches.append(current_batch)
        
        print(f"\n✓ Created {len(batches)} smart batches")
        for i, batch in enumerate(batches, 1):
            size = self._estimate_batch_size(batch)
            print(f"  Batch {i}: {len(batch)} theories (~{size:,} chars)")
        
        return batches
    
    def create_batch_prompt(self, theories: List[Dict]) -> str:
        """Create prompt for batch of theories."""
        
        # Build canonical theories reference
        canonical_ref = []
        for name, data in self.canonical_theories.items():
            mechanisms_str = "; ".join(data['mechanisms']) if data['mechanisms'] else "N/A"
            canonical_ref.append(f"""
                # {name}
                    Category: {data['category']} / {data['subcategory']}
                    Aliases: {data['aliases']}
                    Key mechanisms:
                    {mechanisms_str}
                """)
        
        canonical_theories_text = "\n".join(canonical_ref)
        
        # Build theories to map
        theories_to_map = []
        for i, theory in enumerate(theories, 1):
            confidence_note = ""
            if theory.get('confidence_is_theory') == 'medium':
                confidence_note = "**MEDIUM CONFIDENCE - Validate carefully**"
            
            concept_text = theory.get('concept_text', 'N/A')
            if len(concept_text) > 1900:
                concept_text = concept_text[:1900] + "..."
            
            # Get theory name - handle both 'name' and 'original_name' fields
            theory_name = theory.get('name') or theory.get('original_name') or theory.get('theory_name', 'Unknown')
            theory_id = theory.get('theory_id', 'UNKNOWN')
            
            theories_to_map.append(f"{theory_id}:{theory_name}. {confidence_note}\nConcept: {concept_text}")
                    
        theories_text = "\n".join(theories_to_map)
        
        prompt = f"""Your task is to:
1. Validate if each theory is a valid aging theory
2. Map valid theories to canonical theories from ontology, when possible.

# INSTRUCTIONS
For each theory, think step by step:

1. **VALIDATE**: Is this a valid theory of aging?
   - an "aging theory" is a proposal, model, hypothesis, or mechanism that tries to explain WHY or HOW biological or psychosocial aging occurs at a general, organism-level scale.
   - **Generalizability**: The theory must attempt to explain aging as a fundamental process, not just describe a narrow phenomenon. Addresses aging broadly, not in the context of a specific disease/ organ/ pathway/ etc.
   - **Causal explanation**: Must propose mechanisms or reasons for aging, not just describe patterns or correlations Could be mathematical/computational model/epigenetic clocks but has to discuss causal mechanisms.
   - If marked with MEDIUM CONFIDENCE, be extra careful in validation.
   
2. **MAP**: If valid:
   - Compare the theory's concept to listed theories' mechanisms
   - Consider semantic/mechanistic similarity, not just name matching
   - If a theory contains additional/different core mechanisms that are in listed theories, mark as novel
   
3. **CLASSIFY**:
   - If matches canonical theory → output canonical name + confidence (0.0-1.0)
   - If valid but doesn't match → mark as NOVEL + propose a clear descriptive name

# CANONICAL THEORIES (not exhaustive)
{canonical_theories_text}

# THEORIES TO VALIDATE AND MAP
{theories_text}

# OUTPUT FORMAT
Respond with ONLY valid JSON (no markdown, no extra text):

{{
  "mappings": [
    {{
      "theory_id": "T000001",
      "original_name": "...",
      "is_valid_theory": true/false,
      "validation_reasoning": "Brief explanation",
      "is_mapped": true/false - meaning it matches any listed theory,
      "canonical_name": "Exact canonical name or null",
      "mapping_confidence": 0.0-1.0,
      "is_novel": true/false - meaning it doesn't match any listed theory,
      "novelty_reasoning": "Brief explanation why it not assined to any listed theory",
      "proposed_name": "Clear name for this theory (avoid too generic and too specific names) or null"
    }},
    ...
  ]
}}

# IMPORTANT
- Include ALL {len(theories)} theories in output
- Be conservative: only map if confidence >= 0.8"""
        
        return prompt
    
    def parse_llm_response(self, response: str, theories: List[Dict]) -> List[MappingResult]:
        """Parse LLM response into mapping results."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            mappings = data.get('mappings', [])
            
            # Convert to MappingResult objects
            results = []
            for mapping in mappings:
                result = MappingResult(
                    theory_id=mapping['theory_id'],
                    original_name=mapping['original_name'],
                    is_valid_theory=mapping['is_valid_theory'],
                    validation_reasoning=mapping.get('validation_reasoning', ''),
                    is_mapped=mapping.get('is_mapped', False),
                    canonical_name=mapping.get('canonical_name'),
                    mapping_confidence=mapping.get('mapping_confidence', 0.0),
                    is_novel=mapping.get('is_novel', False),
                    novelty_reasoning=mapping.get('novelty_reasoning', ''),
                    proposed_name=mapping.get('proposed_name')
                )
                results.append(result)
            
            # Verify all theories are in output
            result_ids = {r.theory_id for r in results}
            input_ids = {t['theory_id'] for t in theories}
            
            missing = input_ids - result_ids
            extra = result_ids - input_ids
            
            if missing or extra:
                print(f"\n⚠️  Theory ID mismatch detected!")
                if missing:
                    print(f"   Missing {len(missing)} theories: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                if extra:
                    print(f"   Extra {len(extra)} theories: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
                
                # Remove extra theories (LLM hallucinated IDs)
                if extra:
                    results = [r for r in results if r.theory_id not in extra]
                    print(f"   Removed {len(extra)} extra theories")
                
                # Add missing theories as unprocessed
                if missing:
                    for theory in theories:
                        if theory['theory_id'] in missing:
                            theory_name = theory.get('name') or theory.get('original_name') or theory.get('theory_name', 'Unknown')
                            results.append(MappingResult(
                                theory_id=theory['theory_id'],
                                original_name=theory_name,
                                is_valid_theory=False,
                                validation_reasoning='LLM did not return this theory in response',
                                is_mapped=False,
                                novelty_reasoning='Not processed'
                            ))
                    print(f"   Added {len(missing)} missing theories as unprocessed")
            
            # Final verification
            final_result_ids = {r.theory_id for r in results}
            if final_result_ids != input_ids:
                raise ValueError(f"Critical error: Result IDs still don't match input IDs after correction!")
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing LLM response: {e}")
            print(f"Response: {response[:500]}...")
            
            # Return all as unprocessed
            return [
                MappingResult(
                    theory_id=t['theory_id'],
                    original_name=t['theory_name'],
                    is_valid_theory=False,
                    validation_reasoning='Failed to parse LLM response',
                    is_mapped=False
                )
                for t in theories
            ]
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on token usage and model pricing."""
        pricing = PRICING.get(model, PRICING['gpt-4.1-mini'])
        
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        return input_cost + output_cost
    
    def map_batch(self, theories: List[Dict], checkpoint_db: Optional[CheckpointDB] = None, 
                  batch_number: int = 0) -> Tuple[List[MappingResult], int, int, float]:
        """
        Map a batch of theories.
        
        Returns:
            Tuple of (results, input_tokens, output_tokens, cost)
        """
        print(f"\n  Processing batch of {len(theories)} theories...")
        
        # Debug: print first few theory IDs in batch
        theory_ids = [t.get('theory_id') for t in theories[:3]]
        print(f"  First theories in batch: {', '.join(theory_ids)}")
        
        # Create prompt
        prompt = self.create_batch_prompt(theories)
        
        # Get LLM response
        try:
            response = self.llm.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"""You are a biologist with expertise in aging and senescence. 
                    """},
                {"role": "user", "content": prompt}
                ],
                temperature = 0.2,
                max_tokens=MAX_TOKENS
            )
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            batch_cost = self._calculate_cost(input_tokens, output_tokens, self.model)
            
            # Update stats
            self.stats['total_input_tokens'] += input_tokens
            self.stats['total_output_tokens'] += output_tokens
            self.stats['total_cost'] += batch_cost
            
            print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out | Cost: ${batch_cost:.4f}")
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            results = self.parse_llm_response(response_text, theories)
            
            # Verify all input theories are in results
            result_ids = {r.theory_id for r in results}
            input_ids = {t['theory_id'] for t in theories}
            if result_ids != input_ids:
                print(f"  ⚠️  Warning: {len(input_ids - result_ids)} theories missing, {len(result_ids - input_ids)} extra")
            else:
                print(f"  ✓ All {len(theories)} theories returned")
            
            # Save to checkpoint database
            if checkpoint_db:
                for result in results:
                    checkpoint_db.save_result(result, batch_number)
                checkpoint_db.save_batch_metadata(
                    batch_number, len(theories), input_tokens, output_tokens, batch_cost
                )
            
            # Update stats
            for result in results:
                if result.is_valid_theory:
                    self.stats['valid_theories'] += 1
                    if result.is_mapped:
                        self.stats['mapped_to_canonical'] += 1
                    elif result.is_novel:
                        self.stats['novel_theories'] += 1
                else:
                    self.stats['invalid_theories'] += 1
            
            return results, input_tokens, output_tokens, batch_cost
            
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            
            # Return all as unprocessed
            results = [
                MappingResult(
                    theory_id=t['theory_id'],
                    original_name=t.get('name') or t.get('original_name') or t.get('theory_name', 'Unknown'),
                    is_valid_theory=False,
                    validation_reasoning=f'LLM error: {str(e)}',
                    is_mapped=False,
                    novelty_reasoning='Error during processing'
                )
                for t in theories
            ]
            return results, 0, 0, 0.0
    
    def process_unmatched_theories(self, 
                                   stage1_output_path: str,
                                   output_path: str,
                                   use_smart_batching: bool = True,
                                   batch_size: int = 25,
                                   max_theories: Optional[int] = None,
                                   random_seed: Optional[int] = None,
                                   max_workers: int = 4) -> Dict:
        """
        Process unmatched theories from Stage 1.
        
        Args:
            stage1_output_path: Path to Stage 1 output
            output_path: Path to save results
            use_smart_batching: Use clustering-based smart batching (default True)
            batch_size: Number of theories per batch if not using smart batching (default 30)
            max_theories: Maximum theories to process (for testing)
            random_seed: Random seed for reproducibility (if None, uses deterministic ordering)
            max_workers: Number of parallel workers for batch processing (default 5)
        """
        import time
        import hashlib
        
        print("🚀 Starting Stage 1.5: LLM-Based Mapping\n")
        
        # Record run metadata for reproducibility
        run_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Load Stage 1 output
        print(f"📂 Loading Stage 1 output from {stage1_output_path}...")
        with open(stage1_output_path, 'r') as f:
            stage1_data = json.load(f)
        
        unmatched_theories = stage1_data.get('unmatched_theories', [])
        
        # Add processing order index to each theory for reproducibility
        for idx, theory in enumerate(unmatched_theories):
            theory['_processing_order'] = idx
        
        if max_theories:
            unmatched_theories = unmatched_theories[:max_theories]
        
        # Keep a copy of all theories for lookup later
        all_unmatched_theories = unmatched_theories.copy()
        
        print(f"✓ Found {len(unmatched_theories)} unmatched theories to process")
        
        # Initialize checkpoint database
        checkpoint_path = output_path.replace('.json', '_checkpoint.db')
        print(f"📁 Checkpoint database: {checkpoint_path}")
        
        # Check if checkpoint exists
        import os
        if os.path.exists(checkpoint_path):
            print(f"   ✓ Checkpoint file exists - will resume from previous run")
        else:
            print(f"   ℹ️  No checkpoint found - starting fresh")
        
        checkpoint_db = CheckpointDB(checkpoint_path)
        
        # Check for existing progress
        processed_ids = checkpoint_db.get_processed_theory_ids()
        if processed_ids:
            print(f"\n♻️  Found existing checkpoint with {len(processed_ids)} processed theories")
            print(f"   Resuming from checkpoint...")
            
            # Filter out already processed theories
            unmatched_theories = [t for t in unmatched_theories if t['theory_id'] not in processed_ids]
            print(f"   {len(unmatched_theories)} theories remaining to process")
            
            # Load existing stats from DB
            db_stats = checkpoint_db.get_stats()
            self.stats.update(db_stats)
        
        if not unmatched_theories:
            print("\n✅ All theories already processed! Skipping LLM calls.")
            print("   Loading results from checkpoint database...")
            all_results = checkpoint_db.get_all_results()
            
            # Update stats from database
            db_stats = checkpoint_db.get_stats()
            self.stats.update(db_stats)
            
            print(f"   Loaded {len(all_results)} results from checkpoint")
            batch_assignments = []  # No new batches processed
        else:
            # Create batches using smart clustering or simple chunking
            if use_smart_batching:
                print("\n🧠 Using smart batching with clustering...")
                batches = self._create_smart_batches(unmatched_theories)
            else:
                print(f"\n📦 Using simple batching (batch_size={batch_size})...")
                batches = [unmatched_theories[i:i+batch_size] 
                          for i in range(0, len(unmatched_theories), batch_size)]
                print(f"✓ Created {len(batches)} batches")
            
            print(f"\n🔄 Processing {len(batches)} batches with {max_workers} parallel workers...")
            
            # Track batch assignments for reproducibility
            batch_assignments = []
            
            all_results = []
            starting_batch_num = checkpoint_db.get_stats()['batch_count']
            
            # Process batches in parallel
            def process_single_batch(batch_idx, batch):
                """Process a single batch (for parallel execution)."""
                actual_batch_num = starting_batch_num + batch_idx
                
                # Record which theories are in this batch
                batch_theory_ids = [t.get('theory_id') for t in batch]
                batch_assignment = {
                    'batch_number': actual_batch_num,
                    'theory_ids': batch_theory_ids,
                    'theory_count': len(batch)
                }
                
                # Add batch number to each theory for tracking
                for theory in batch:
                    theory['_batch_number'] = actual_batch_num
                
                results, input_tok, output_tok, cost = self.map_batch(
                    batch, checkpoint_db, actual_batch_num
                )
                
                return results, batch_assignment
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                futures = {
                    executor.submit(process_single_batch, idx, batch): idx 
                    for idx, batch in enumerate(batches, 1)
                }
                
                # Process completed batches with progress bar
                with tqdm(total=len(batches), desc="Processing batches") as pbar:
                    for future in as_completed(futures):
                        try:
                            results, batch_assignment = future.result()
                            all_results.extend(results)
                            batch_assignments.append(batch_assignment)
                            self.stats['batch_count'] += 1
                            self.stats['total_processed'] += len(results)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n❌ Error processing batch: {e}")
                            pbar.update(1)
            
            # Load all results from DB (including previously processed)
            all_results = checkpoint_db.get_all_results()
        
        # Final verification: Check all input theories have results
        print(f"\n🔍 Final verification...")
        result_ids = {r.theory_id for r in all_results}
        input_ids = {t['theory_id'] for t in all_unmatched_theories}
        
        missing_theories = input_ids - result_ids
        extra_theories = result_ids - input_ids
        
        if missing_theories:
            print(f"❌ ERROR: {len(missing_theories)} theories missing from results!")
            print(f"   Missing IDs: {list(missing_theories)[:10]}{'...' if len(missing_theories) > 10 else ''}")
            raise ValueError(f"Missing {len(missing_theories)} theories in final results!")
        
        if extra_theories:
            print(f"⚠️  Warning: {len(extra_theories)} extra theories in results (will be ignored)")
            print(f"   Extra IDs: {list(extra_theories)[:10]}{'...' if len(extra_theories) > 10 else ''}")
            # Filter out extra theories
            all_results = [r for r in all_results if r.theory_id in input_ids]
        
        print(f"✓ Verified: All {len(all_unmatched_theories)} input theories have results")
        
        # Separate mapped, novel, and still unmatched
        mapped_theories = []
        novel_theories = []
        still_unmatched = []
        invalid_theories = []
        
        # Create lookup for original theory data (use all theories, not just remaining)
        theory_lookup = {t['theory_id']: t for t in all_unmatched_theories}
        
        for result in all_results:
            # Verify theory_id exists in lookup
            if result.theory_id not in theory_lookup:
                print(f"⚠️  Warning: Result theory_id '{result.theory_id}' not found in original theories!")
                print(f"   Result name: {result.original_name}")
                continue
            
            original_theory = theory_lookup[result.theory_id]
            
            # Verify the names match (sanity check)
            original_name = original_theory.get('name', original_theory.get('original_name', ''))
            if original_name != result.original_name:
                print(f"⚠️  Warning: Name mismatch for {result.theory_id}!")
                print(f"   Original: {original_name}")
                print(f"   LLM returned: {result.original_name}")
            
            # Add mapping result to theory
            theory_with_result = original_theory.copy()
            theory_with_result['stage1_5_result'] = {
                'is_valid_theory': result.is_valid_theory,
                'validation_reasoning': result.validation_reasoning,
                'is_mapped': result.is_mapped,
                'canonical_name': result.canonical_name,
                'mapping_confidence': result.mapping_confidence,
                'is_novel': result.is_novel,
                'novelty_reasoning': result.novelty_reasoning,
                'proposed_name': result.proposed_name
            }
            
            if not result.is_valid_theory:
                invalid_theories.append(theory_with_result)
            elif result.is_mapped and result.canonical_name:
                # Add match_result for consistency with Stage 1
                theory_with_result['match_result'] = {
                    'matched': True,
                    'canonical_name': result.canonical_name,
                    'match_type': 'llm_mapping',
                    'confidence': result.mapping_confidence,
                    'score': result.mapping_confidence
                }
                mapped_theories.append(theory_with_result)
            elif result.is_novel:
                novel_theories.append(theory_with_result)
            else:
                still_unmatched.append(theory_with_result)
        
        # Save results
        print(f"\n💾 Saving results to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create reproducibility metadata
        import hashlib
        input_hash = hashlib.md5(json.dumps([t.get('theory_id') for t in unmatched_theories], sort_keys=True).encode()).hexdigest()
        
        output_data = {
            'metadata': {
                'stage': 'stage1_5_llm_mapping',
                'timestamp': run_timestamp,
                'model': self.model,
                'use_smart_batching': use_smart_batching,
                'batch_size': batch_size if not use_smart_batching else 'dynamic',
                'random_seed': random_seed,
                'input_hash': input_hash,
                'input_count': len(unmatched_theories),
                'statistics': self.stats,
                'reproducibility': {
                    'batch_assignments': batch_assignments,
                    'clustering_enabled': use_smart_batching,
                    'processing_order': 'deterministic',
                    'note': 'Theories are processed in deterministic order based on input sequence. Batch assignments are recorded for reproducibility.'
                }
            },
            'mapped_theories': mapped_theories,
            'novel_theories': novel_theories,
            'still_unmatched': still_unmatched,
            'invalid_theories': invalid_theories
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
        print(f"✓ Checkpoint database: {checkpoint_path}")
        
        # Save final metadata to checkpoint
        checkpoint_db.save_metadata('output_path', output_path)
        checkpoint_db.save_metadata('completed', 'true')
        checkpoint_db.save_metadata('completion_time', time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Print statistics
        self.print_statistics()
        
        # Print sample results
        self.print_sample_results(mapped_theories, novel_theories)
        
        # Close checkpoint database
        checkpoint_db.close()
        
        return output_data
    
    def print_statistics(self):
        """Print mapping statistics."""
        print("\n" + "="*80)
        print("STAGE 1.5: LLM MAPPING STATISTICS")
        print("="*80)
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Batches: {self.stats['batch_count']}")
        
        if self.stats.get('clusters_found', 0) > 0:
            print(f"Clusters found: {self.stats['clusters_found']}")
            print(f"Average cluster size: {self.stats['avg_cluster_size']:.1f}")
        
        print(f"\nValidation:")
        print(f"  Valid theories: {self.stats['valid_theories']} ({self.stats['valid_theories']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Invalid theories: {self.stats['invalid_theories']} ({self.stats['invalid_theories']/self.stats['total_processed']*100:.1f}%)")
        print(f"\nMapping:")
        print(f"  Mapped to canonical: {self.stats['mapped_to_canonical']} ({self.stats['mapped_to_canonical']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Novel theories: {self.stats['novel_theories']} ({self.stats['novel_theories']/self.stats['total_processed']*100:.1f}%)")
        print(f"  Still unmatched: {self.stats['total_processed'] - self.stats['mapped_to_canonical'] - self.stats['novel_theories'] - self.stats['invalid_theories']}")
        
        print(f"\n💰 Cost Analysis:")
        print(f"  Model: {self.model}")
        print(f"  Input tokens: {self.stats['total_input_tokens']:,}")
        print(f"  Output tokens: {self.stats['total_output_tokens']:,}")
        print(f"  Total tokens: {self.stats['total_input_tokens'] + self.stats['total_output_tokens']:,}")
        print(f"  Total cost: ${self.stats['total_cost']:.4f}")
        
        if self.stats['batch_count'] > 0:
            avg_cost_per_batch = self.stats['total_cost'] / self.stats['batch_count']
            avg_cost_per_theory = self.stats['total_cost'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
            print(f"  Average cost per batch: ${avg_cost_per_batch:.4f}")
            print(f"  Average cost per theory: ${avg_cost_per_theory:.4f}")
        
        print("="*80)
    
    def print_sample_results(self, mapped: List[Dict], novel: List[Dict], n: int = 5):
        """Print sample mapping results."""
        print(f"\n📊 Sample Mapped Theories (top {n}):")
        print("-" * 80)
        for i, theory in enumerate(mapped[:n], 1):
            result = theory['stage1_5_result']
            # Get theory name - handle both 'name' and 'original_name' fields
            theory_name = theory.get('name') or theory.get('original_name') or theory.get('theory_name', 'Unknown')
            print(f"\n{i}. {theory_name}")
            print(f"   → Mapped to: {result['canonical_name']}")
            print(f"   Confidence: {result['mapping_confidence']:.2f}")
            print(f"   Reasoning: {result['validation_reasoning'][:100]}...")
        
        if novel:
            print(f"\n📊 Sample Novel Theories (top {min(n, len(novel))}):")
            print("-" * 80)
            for i, theory in enumerate(novel[:n], 1):
                result = theory['stage1_5_result']
                # Get theory name - handle both 'name' and 'original_name' fields
                theory_name = theory.get('name') or theory.get('original_name') or theory.get('theory_name', 'Unknown')
                print(f"\n{i}. {theory_name}")
                print(f"   → Novel: {result['proposed_name']}")
                print(f"   Reasoning: {result['validation_reasoning'][:100]}...")


def main():
    """Run Stage 1.5 LLM mapping."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Stage 1.5: LLM-based mapping to canonical theories')
    parser.add_argument('--input', default='output/stage1_fuzzy_matched.json', help='Stage 1 output')
    parser.add_argument('--output', default='output/stage1_5_llm_mapped.json', help='Output file')
    parser.add_argument('--batch-size', type=int, default=25, help='Theories per batch (default: 25)')
    parser.add_argument('--max-theories', type=int, default=None, help='Max theories (for testing)')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--reset', action='store_true', help='Reset checkpoint database and start from scratch')
    
    args = parser.parse_args()
    
    # Handle checkpoint reset
    checkpoint_path = args.output.replace('.json', '_checkpoint.db')
    if args.reset:
        if os.path.exists(checkpoint_path):
            print(f"🗑️  Removing existing checkpoint: {checkpoint_path}")
            os.remove(checkpoint_path)
            print("✓ Checkpoint database deleted - starting fresh")
        else:
            print(f"ℹ️  No checkpoint found at {checkpoint_path} - will start fresh anyway")
    
    # Initialize mapper
    mapper = LLMMapper()
    
    # Process theories
    mapper.process_unmatched_theories(
        stage1_output_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        max_theories=args.max_theories,
        max_workers=args.max_workers
    )
    
    print("\n✅ Stage 1.5 complete!")
    print(f"\nNext step: Run Stage 2 on still_unmatched theories from {args.output}")


if __name__ == '__main__':
    main()
