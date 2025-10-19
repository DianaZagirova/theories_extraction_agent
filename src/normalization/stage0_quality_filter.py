"""
Stage 0: Quality Filtering and Data Enrichment
Filters theories by confidence and enriches with concept data.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.theory import THEORY_CRITERIA

@dataclass
class Theory:
    """Represents a single theory with all metadata."""
    name: str
    key_concepts: List[Dict] = field(default_factory=list)
    description: str = ""
    evidence: str = ""
    confidence_is_theory: str = "unknown"
    mode: str = ""
    criteria_reasoning: str = ""
    paper_focus: int = 0
    
    # Paper metadata
    doi: str = ""
    pmid: str = ""
    paper_title: str = ""
    timestamp: str = ""
    
    # Processing metadata
    theory_id: Optional[str] = None
    enriched_text: Optional[str] = None
    concept_text: Optional[str] = None
    is_validated: bool = False
    validation_reason: str = ""
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class QualityFilter:
    """Filters theories based on confidence and validates medium confidence theories."""
    
    def __init__(self, llm_client=None):
        """
        Initialize quality filter.
        
        Args:
            llm_client: Optional LLM client for validating medium confidence theories
        """
        self.llm_client = llm_client
        self.stats = {
            'total_input': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'medium_validated': 0,
            'medium_rejected': 0,
            'total_output': 0
        }
    
    def load_theories(self, json_path: str) -> List[Theory]:
        """Load theories from JSON file."""
        print(f"📂 Loading theories from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        theories = []
        theory_counter = 0
        
        for paper in data.get('results', []):
            if not paper.get('theories'):
                continue
            
            for theory_data in paper['theories']:
                theory_counter += 1
                theory = Theory(
                    theory_id=f"T{theory_counter:06d}",
                    name=theory_data.get('name', ''),
                    key_concepts=theory_data.get('key_concepts', []),
                    description=theory_data.get('description', ''),
                    evidence=theory_data.get('evidence', ''),
                    confidence_is_theory=theory_data.get('confidence_is_theory', 'unknown'),
                    mode=theory_data.get('mode', ''),
                    criteria_reasoning=theory_data.get('criteria_reasoning', ''),
                    paper_focus=theory_data.get('paper_focus', 0),
                    doi=paper.get('doi', ''),
                    pmid=paper.get('pmid', ''),
                    paper_title=paper.get('title', ''),
                    timestamp=paper.get('timestamp', '')
                )
                theories.append(theory)
        
        self.stats['total_input'] = len(theories)
        print(f"✓ Loaded {len(theories)} theories from {len(data.get('results', []))} papers")
        
        return theories
    
    def filter_by_confidence(self, theories: List[Theory], 
                            validate_medium: bool = False) -> List[Theory]:
        """
        Filter theories by confidence level.
        
        Args:
            theories: List of theories to filter
            validate_medium: If True, use LLM to validate medium confidence theories
        
        Returns:
            Filtered list of theories
        """
        print(f"\n🔍 Filtering {len(theories)} theories by confidence...")
        
        high_conf = []
        medium_conf = []
        low_conf = []
        unknown_conf = []
        
        for theory in theories:
            conf = theory.confidence_is_theory.lower()
            if conf == 'high':
                high_conf.append(theory)
            elif conf == 'medium':
                medium_conf.append(theory)
            elif conf == 'low':
                low_conf.append(theory)
            else:
                unknown_conf.append(theory)
        
        self.stats['high_confidence'] = len(high_conf)
        self.stats['medium_confidence'] = len(medium_conf)
        self.stats['low_confidence'] = len(low_conf)
        
        print(f"  High confidence: {len(high_conf)}")
        print(f"  Medium confidence: {len(medium_conf)}")
        print(f"  Low confidence: {len(low_conf)}")
        print(f"  Unknown confidence: {len(unknown_conf)}")
        
        # Start with high confidence theories
        filtered = high_conf.copy()
        
        # Validate medium confidence if requested
        if validate_medium and self.llm_client and medium_conf:
            print(f"\n🤖 Validating {len(medium_conf)} medium confidence theories with LLM...")
            validated_medium = self._validate_medium_confidence(medium_conf)
            filtered.extend(validated_medium)
            self.stats['medium_validated'] = len(validated_medium)
            self.stats['medium_rejected'] = len(medium_conf) - len(validated_medium)
            print(f"  ✓ Validated: {len(validated_medium)}")
            print(f"  ✗ Rejected: {len(medium_conf) - len(validated_medium)}")
        else:
            # Without validation, keep medium confidence
            filtered.extend(medium_conf)
            self.stats['medium_validated'] = len(medium_conf)
        
        # Add unknown as medium (conservative)
        filtered.extend(unknown_conf)
        
        self.stats['total_output'] = len(filtered)
        
        print(f"\n✓ Filtered: {len(filtered)} theories kept, {len(low_conf)} removed")
        
        return filtered
    
    def _validate_medium_confidence(self, theories: List[Theory]) -> List[Theory]:
        """Validate medium confidence theories using LLM."""
        validated = []
        
        for i, theory in enumerate(theories):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(theories)}")
            
            is_valid, reason = self._validate_single_theory(theory)
            
            if is_valid:
                theory.is_validated = True
                theory.validation_reason = reason
                validated.append(theory)
            else:
                theory.is_validated = False
                theory.validation_reason = reason
        
        return validated
    
    def _validate_single_theory(self, theory: Theory) -> tuple[bool, str]:
        """Validate a single theory using LLM."""
        if not self.llm_client:
            return True, "No LLM validation"
        
        # Format concepts
        concepts_text = "\n".join([
            f"  - {c.get('concept', '')}: {c.get('description', '')}"
            for c in theory.key_concepts
        ])
        
        prompt = f"""Given the definition of valid aging theory, evaluate if the following theory is valid.

{THEORY_CRITERIA}

# POTENTIAL THEORY
- Name: {theory.name}
- Key Concepts: {concepts_text}

#OUTPUT FORMAT:
Return STRICTLY a single-line JSON object with keys:
"status" - must be one of: "valid", "not_valid".
"reason" - must be a short one-sentence justification.

#EXAMPLE:
{"status": "not_valid", "reason": "Describes a biomarker correlation without causal mechanism."}

#ANSWER:
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            
            def _extract_json_text(raw: str) -> str:
                text = raw.strip()
                if text.startswith('```'):
                    parts = text.split('```')
                    # choose the first fenced block content if present
                    if len(parts) >= 2:
                        text = parts[1]
                    # strip optional leading 'json'
                    if text.lstrip().startswith('json'):
                        text = text.lstrip()[4:]
                return text.strip()

            def _parse_response(resp_text: str):
                clean_text = _extract_json_text(resp_text)
                data = json.loads(clean_text)
                status = str(data.get('status', '')).lower().strip()
                reason = str(data.get('reason', '')).strip()
                return status, reason

            try:
                raw = response.choices[0].message.content
                status, reason = _parse_response(raw)
            except Exception:
                # Retry once after a short delay
                time.sleep(0.6)
                response_retry = self.llm_client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150
                )
                raw = response_retry.choices[0].message.content
                try:
                    status, reason = _parse_response(raw)
                except Exception:
                    # Final fallback: mark as valid to avoid dropping theories due to formatting issues
                    fallback_preview = (raw or "").strip().replace("\n", " ")[:160]
                    return True, f"Non-JSON LLM response; treated as valid. Preview: {fallback_preview}"

            is_valid = status == 'valid'
            if not reason:
                reason = 'No reason provided'
            return is_valid, reason
            
        except Exception as e:
            print(f"  Warning: LLM validation failed for {theory.theory_id}: {e}")
            return True, f"LLM validation failed: {e}"
    
    def enrich_theories(self, theories: List[Theory]) -> List[Theory]:
        """Enrich theories with combined text representations."""
        print(f"\n📝 Enriching {len(theories)} theories with concept data...")
        
        for theory in theories:
            # Extract concept text
            concept_parts = []
            for concept in theory.key_concepts:
                concept_name = concept.get('concept', '')
                concept_desc = concept.get('description', '')
                if concept_name:
                    concept_parts.append(f"{concept_name}: {concept_desc}")
            
            theory.concept_text = " | ".join(concept_parts)
            
            # Create enriched full text
            parts = [theory.name]
            if theory.concept_text:
                parts.append(theory.concept_text)
            if theory.description:
                parts.append(theory.description)
            
            theory.enriched_text = ". ".join(parts)
        
        print(f"✓ Enriched {len(theories)} theories")
        return theories
    
    def save_filtered_theories(self, theories: List[Theory], output_path: str):
        """Save filtered theories to JSON."""
        print(f"\n💾 Saving {len(theories)} filtered theories to {output_path}...")
        
        output_data = {
            'metadata': {
                'stage': 'stage0_quality_filter',
                'statistics': self.stats,
                'total_theories': len(theories)
            },
            'theories': [t.to_dict() for t in theories]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print filtering statistics."""
        print("\n" + "="*60)
        print("STAGE 0: QUALITY FILTERING STATISTICS")
        print("="*60)
        print(f"Input theories: {self.stats['total_input']}")
        print(f"  High confidence: {self.stats['high_confidence']} ({self.stats['high_confidence']/self.stats['total_input']*100:.1f}%)")
        print(f"  Medium confidence: {self.stats['medium_confidence']} ({self.stats['medium_confidence']/self.stats['total_input']*100:.1f}%)")
        print(f"  Low confidence: {self.stats['low_confidence']} ({self.stats['low_confidence']/self.stats['total_input']*100:.1f}%)")
        print(f"\nMedium confidence validation:")
        print(f"  Validated: {self.stats['medium_validated']}")
        print(f"  Rejected: {self.stats['medium_rejected']}")
        print(f"\nOutput theories: {self.stats['total_output']}")
        print(f"Removed: {self.stats['total_input'] - self.stats['total_output']}")
        print("="*60)


def main():
    """Run Stage 0 quality filtering."""
    from src.core.llm_integration import AzureOpenAIClient
    
    # Initialize
    print("🚀 Starting Stage 0: Quality Filtering\n")
    
    # Load LLM client (optional, for medium confidence validation)
    try:
        llm_client = AzureOpenAIClient()
        print("✓ LLM client initialized for validation\n")
    except Exception as e:
        print(f"⚠ LLM client not available: {e}")
        print("  Continuing without medium confidence validation\n")
        llm_client = None
    
    # Initialize filter
    filter_engine = QualityFilter(llm_client=llm_client)
    
    # Load theories
    theories = filter_engine.load_theories('theories_per_paper.json')
    
    # Filter by confidence
    filtered = filter_engine.filter_by_confidence(
        theories, 
        validate_medium=False  # Set to True to enable LLM validation
    )
    
    # Enrich with concept data
    enriched = filter_engine.enrich_theories(filtered)
    
    # Save results
    filter_engine.save_filtered_theories(
        enriched, 
        'output/stage0_filtered_theories.json'
    )
    
    # Print statistics
    filter_engine.print_statistics()
    
    print("\n✅ Stage 0 complete!")


if __name__ == '__main__':
    main()
