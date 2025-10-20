"""
Stage 3: Theory Grouping by Shared Mechanisms

Identifies theories that share the same mechanisms and groups them together.
This combines:
1. Theories matched in Stage 1 (fuzzy matching to canonical names)
2. Valid theories from Stage 2 (LLM extraction)

Groups theories by:
- Exact mechanism match (same mechanisms list)
- High overlap (>80% shared mechanisms)
- Shared key players and pathways
"""

import json
import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TheoryGroup:
    """Represents a group of theories sharing the same mechanisms."""
    group_id: str
    canonical_name: Optional[str]  # From Stage 1 if matched
    representative_name: str  # Most common or descriptive name
    theory_ids: List[str] = field(default_factory=list)
    theory_count: int = 0
    
    # Shared characteristics
    primary_category: Optional[str] = None
    secondary_category: Optional[str] = None
    shared_mechanisms: List[str] = field(default_factory=list)
    shared_key_players: List[str] = field(default_factory=list)
    shared_pathways: List[str] = field(default_factory=list)
    
    # Classification
    level_of_explanation: Optional[str] = None
    type_of_cause: Optional[str] = None
    temporal_focus: Optional[str] = None
    adaptiveness: Optional[str] = None
    
    # Source tracking
    source: str = 'stage2'  # 'stage1' or 'stage2' or 'both'
    
    def to_dict(self):
        return {
            'group_id': self.group_id,
            'canonical_name': self.canonical_name,
            'representative_name': self.representative_name,
            'theory_ids': self.theory_ids,
            'theory_count': self.theory_count,
            'primary_category': self.primary_category,
            'secondary_category': self.secondary_category,
            'shared_mechanisms': self.shared_mechanisms,
            'shared_key_players': self.shared_key_players,
            'shared_pathways': self.shared_pathways,
            'level_of_explanation': self.level_of_explanation,
            'type_of_cause': self.type_of_cause,
            'temporal_focus': self.temporal_focus,
            'adaptiveness': self.adaptiveness,
            'source': self.source
        }


class TheoryGrouper:
    """Groups theories by shared mechanisms."""
    
    def __init__(self, exact_match_threshold: float = 1.0, 
                 high_overlap_threshold: float = 0.8):
        """
        Initialize theory grouper.
        
        Args:
            exact_match_threshold: Jaccard similarity for exact match (1.0 = 100%)
            high_overlap_threshold: Jaccard similarity for high overlap (0.8 = 80%)
        """
        self.exact_match_threshold = exact_match_threshold
        self.high_overlap_threshold = high_overlap_threshold

         
        self.groups = []
        self.theory_to_group = {}  # theory_id -> group_id
        
        self.stats = {
            'total_theories': 0,
            'stage1_matched': 0,
            'stage2_valid': 0,
            'total_groups': 0,
            'exact_match_groups': 0,
            'high_overlap_groups': 0,
            'singleton_groups': 0,
            'avg_group_size': 0.0
        }
    
    def load_data(self, stage1_path: str, stage2_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load data from Stage 1 and Stage 2."""
        print(f"📂 Loading data from Stage 1 and Stage 2...")
        
        # Load Stage 1 (fuzzy matched)
        with open(stage1_path, 'r') as f:
            stage1_data = json.load(f)
        
        stage1_matched = stage1_data.get('matched_theories', [])
        
        # Load Stage 2 (LLM extracted)
        with open(stage2_path, 'r') as f:
            stage2_data = json.load(f)
        
        stage2_valid = stage2_data.get('valid_theories', [])
        
        print(f"✓ Loaded {len(stage1_matched)} theories from Stage 1 (fuzzy matched)")
        print(f"✓ Loaded {len(stage2_valid)} theories from Stage 2 (LLM validated)")
        
        self.stats['stage1_matched'] = len(stage1_matched)
        self.stats['stage2_valid'] = len(stage2_valid)
        self.stats['total_theories'] = len(stage1_matched) + len(stage2_valid)
        
        return stage1_matched, stage2_valid
    
    def _normalize_list(self, items: List[str]) -> Set[str]:
        """Normalize list of items for comparison."""
        return set(item.lower().strip() for item in items if item)
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_mechanism_signature(self, theory: Dict) -> Dict[str, Set[str]]:
        """
        Compute mechanism signature for a theory.
        
        Returns dict with normalized sets of:
        - mechanisms
        - key_players
        - pathways
        """
        # Check if from Stage 1 or Stage 2
        if 'match_result' in theory and theory.get('match_result', {}).get('matched'):
            # Stage 1 matched theory - no extracted mechanisms
            return {
                'mechanisms': set(),
                'key_players': set(),
                'pathways': set(),
                'has_data': False
            }
        
        # Stage 2 theory with extracted metadata
        metadata = theory.get('stage2_metadata', {})
        
        return {
            'mechanisms': self._normalize_list(metadata.get('mechanisms', [])),
            'key_players': self._normalize_list(metadata.get('key_players', [])),
            'pathways': self._normalize_list(metadata.get('pathways', [])),
            'has_data': True
        }
    
    def _signatures_match(self, sig1: Dict, sig2: Dict, threshold: float) -> bool:
        """Check if two signatures match above threshold."""
        # If either has no data, can't match
        if not sig1['has_data'] or not sig2['has_data']:
            return False
        
        # Calculate similarities
        mech_sim = self._jaccard_similarity(sig1['mechanisms'], sig2['mechanisms'])
        player_sim = self._jaccard_similarity(sig1['key_players'], sig2['key_players'])
        pathway_sim = self._jaccard_similarity(sig1['pathways'], sig2['pathways'])
        
        # Weighted average (mechanisms are most important)
        if sig1['mechanisms'] and sig2['mechanisms']:
            # If both have mechanisms, weight them heavily
            avg_sim = (mech_sim * 0.6 + player_sim * 0.2 + pathway_sim * 0.2)
        elif sig1['key_players'] and sig2['key_players']:
            # If no mechanisms but have key players
            avg_sim = (player_sim * 0.7 + pathway_sim * 0.3)
        elif sig1['pathways'] and sig2['pathways']:
            # Only pathways
            avg_sim = pathway_sim
        else:
            avg_sim = 0.0
        
        return avg_sim >= threshold
    
    def group_theories(self, stage1_theories: List[Dict], stage2_theories: List[Dict]) -> List[TheoryGroup]:
        """
        Group theories by shared mechanisms.
        
        Strategy:
        1. Group Stage 1 theories by canonical name
        2. Group Stage 2 theories by mechanism similarity
        3. Try to merge Stage 2 groups with Stage 1 groups if mechanisms match
        """
        print(f"\n🔄 Grouping theories by shared mechanisms...")
        
        groups = []
        group_counter = 0
        
        # Step 1: Group Stage 1 theories by canonical name
        print(f"  Step 1: Grouping Stage 1 theories by canonical name...")
        canonical_groups = defaultdict(list)
        
        for theory in stage1_theories:
            match_result = theory.get('match_result', {})
            if match_result.get('matched'):
                canonical_name = match_result.get('canonical_name')
                canonical_groups[canonical_name].append(theory)
        
        for canonical_name, theories in canonical_groups.items():
            group_counter += 1
            
            # Get representative metadata (from first theory if available)
            rep_theory = theories[0]
            
            group = TheoryGroup(
                group_id=f"G{group_counter:04d}",
                canonical_name=canonical_name,
                representative_name=canonical_name,
                theory_ids=[t['theory_id'] for t in theories],
                theory_count=len(theories),
                source='stage1'
            )
            
            groups.append(group)
            for t in theories:
                self.theory_to_group[t['theory_id']] = group.group_id
        
        print(f"    Created {len(canonical_groups)} groups from Stage 1")
        
        # Step 2: Group Stage 2 theories by mechanism similarity
        print(f"  Step 2: Grouping Stage 2 theories by mechanism similarity...")
        
        ungrouped_stage2 = []
        for theory in stage2_theories:
            sig = self._compute_mechanism_signature(theory)
            if sig['has_data']:
                ungrouped_stage2.append((theory, sig))
        
        # Greedy clustering
        while ungrouped_stage2:
            # Take first theory as seed
            seed_theory, seed_sig = ungrouped_stage2.pop(0)
            cluster = [seed_theory]
            cluster_sigs = [seed_sig]
            
            # Find all theories that match seed
            remaining = []
            for theory, sig in ungrouped_stage2:
                if self._signatures_match(seed_sig, sig, self.high_overlap_threshold):
                    cluster.append(theory)
                    cluster_sigs.append(sig)
                else:
                    remaining.append((theory, sig))
            
            ungrouped_stage2 = remaining
            
            # Create group
            group_counter += 1
            
            # Compute shared characteristics
            metadata = seed_theory.get('stage2_metadata', {})
            
            # Find shared mechanisms, key_players, pathways
            all_mechanisms = [sig['mechanisms'] for sig in cluster_sigs]
            all_players = [sig['key_players'] for sig in cluster_sigs]
            all_pathways = [sig['pathways'] for sig in cluster_sigs]
            
            shared_mechanisms = list(set.intersection(*all_mechanisms) if all_mechanisms else set())
            shared_players = list(set.intersection(*all_players) if all_players else set())
            shared_pathways = list(set.intersection(*all_pathways) if all_pathways else set())
            
            # Get most common name
            names = [t['original_name'] for t in cluster]
            name_counter = Counter(names)
            representative_name = name_counter.most_common(1)[0][0] if names else "Unknown"
            
            group = TheoryGroup(
                group_id=f"G{group_counter:04d}",
                canonical_name=None,
                representative_name=representative_name,
                theory_ids=[t['theory_id'] for t in cluster],
                theory_count=len(cluster),
                primary_category=metadata.get('primary_category'),
                secondary_category=metadata.get('secondary_category'),
                shared_mechanisms=shared_mechanisms,
                shared_key_players=shared_players,
                shared_pathways=shared_pathways,
                level_of_explanation=metadata.get('level_of_explanation'),
                type_of_cause=metadata.get('type_of_cause'),
                temporal_focus=metadata.get('temporal_focus'),
                adaptiveness=metadata.get('adaptiveness'),
                source='stage2'
            )
            
            groups.append(group)
            for t in cluster:
                self.theory_to_group[t['theory_id']] = group.group_id
        
        print(f"    Created {group_counter - len(canonical_groups)} groups from Stage 2")
        
        self.groups = groups
        self.stats['total_groups'] = len(groups)
        self.stats['singleton_groups'] = sum(1 for g in groups if g.theory_count == 1)
        self.stats['avg_group_size'] = sum(g.theory_count for g in groups) / len(groups) if groups else 0
        
        print(f"✓ Created {len(groups)} theory groups")
        print(f"  Singleton groups: {self.stats['singleton_groups']}")
        print(f"  Avg group size: {self.stats['avg_group_size']:.1f}")
        
        return groups
    
    def save_results(self, stage1_theories: List[Dict], stage2_theories: List[Dict], 
                     output_path: str):
        """Save grouping results."""
        print(f"\n💾 Saving results to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Combine all theories
        all_theories = stage1_theories + stage2_theories
        
        # Add group_id to each theory
        for theory in all_theories:
            theory['group_id'] = self.theory_to_group.get(theory['theory_id'], None)
        
        data = {
            'metadata': {
                'stage': 'stage3_theory_grouping',
                'approach': 'mechanism-based grouping',
                'statistics': self.stats,
                'thresholds': {
                    'exact_match': self.exact_match_threshold,
                    'high_overlap': self.high_overlap_threshold
                }
            },
            'groups': [g.to_dict() for g in self.groups],
            'theories': all_theories
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print grouping statistics."""
        print("\n" + "="*60)
        print("STAGE 3: THEORY GROUPING STATISTICS")
        print("="*60)
        print(f"Input:")
        print(f"  Stage 1 matched: {self.stats['stage1_matched']}")
        print(f"  Stage 2 valid: {self.stats['stage2_valid']}")
        print(f"  Total theories: {self.stats['total_theories']}")
        print(f"\nGrouping results:")
        print(f"  Total groups: {self.stats['total_groups']}")
        print(f"  Singleton groups: {self.stats['singleton_groups']}")
        print(f"  Multi-theory groups: {self.stats['total_groups'] - self.stats['singleton_groups']}")
        print(f"  Avg group size: {self.stats['avg_group_size']:.1f}")
        print(f"\nCompression:")
        compression = (1 - self.stats['total_groups'] / self.stats['total_theories']) * 100
        print(f"  {compression:.1f}% reduction ({self.stats['total_theories']} → {self.stats['total_groups']} groups)")
        print("="*60)
    
    def print_sample_groups(self, n: int = 10):
        """Print sample groups."""
        print(f"\n📊 Sample Groups (largest {n}):")
        print("-" * 80)
        
        # Sort by size
        sorted_groups = sorted(self.groups, key=lambda g: g.theory_count, reverse=True)
        
        for i, group in enumerate(sorted_groups[:n], 1):
            print(f"\n{i}. {group.representative_name}")
            print(f"   Group ID: {group.group_id}")
            print(f"   Theory count: {group.theory_count}")
            print(f"   Source: {group.source}")
            
            if group.canonical_name:
                print(f"   Canonical: {group.canonical_name}")
            
            if group.primary_category:
                print(f"   Category: {group.primary_category} / {group.secondary_category}")
            
            if group.shared_mechanisms:
                print(f"   Shared mechanisms ({len(group.shared_mechanisms)}): {', '.join(list(group.shared_mechanisms)[:3])}...")
            
            if group.shared_key_players:
                print(f"   Shared key players ({len(group.shared_key_players)}): {', '.join(list(group.shared_key_players)[:5])}...")


def main():
    """Run Stage 3 theory grouping."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 3: Theory grouping by shared mechanisms')
    parser.add_argument('--stage1', default='output/stage1_fuzzy_matched.json', help='Stage 1 input')
    parser.add_argument('--stage2', default='output/stage2_llm_extracted.json', help='Stage 2 input')
    parser.add_argument('--output', default='output/stage3_theory_groups.json', help='Output file')
    parser.add_argument('--exact-threshold', type=float, default=1.0, help='Exact match threshold')
    parser.add_argument('--overlap-threshold', type=float, default=0.8, help='High overlap threshold')
    
    args = parser.parse_args()
    
    print("🚀 Starting Stage 3: Theory Grouping\n")
    
    # Initialize grouper
    grouper = TheoryGrouper(
        exact_match_threshold=args.exact_threshold,
        high_overlap_threshold=args.overlap_threshold
    )
    
    # Load data
    stage1_theories, stage2_theories = grouper.load_data(args.stage1, args.stage2)
    
    # Group theories
    groups = grouper.group_theories(stage1_theories, stage2_theories)
    
    # Save results
    grouper.save_results(stage1_theories, stage2_theories, args.output)
    
    # Print statistics
    grouper.print_statistics()
    
    # Print sample groups
    grouper.print_sample_groups(n=10)
    
    print("\n✅ Stage 3 complete!")
    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
