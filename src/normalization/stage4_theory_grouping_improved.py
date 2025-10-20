"""
Stage 4: Theory Grouping by Shared Mechanisms (IMPROVED)

Groups theories by shared mechanisms using improved Stage 3 output.

Key Improvements:
1. Works with Stage 3 improved output (all theories have mechanisms)
2. Groups by canonical name first
3. Clusters novel/unmatched by mechanism similarity
4. Attempts to merge novel groups with canonical groups
"""

import json
import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TheoryGroup:
    """Represents a group of theories sharing mechanisms."""
    group_id: str
    canonical_name: Optional[str]
    representative_name: str
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
    source: str = 'canonical'  # 'canonical', 'novel', or 'mixed'
    mechanism_source: str = 'canonical'  # 'canonical' or 'extracted'
    
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
            'source': self.source,
            'mechanism_source': self.mechanism_source
        }


class ImprovedTheoryGrouper:
    """Groups theories by shared mechanisms using improved data."""
    
    def __init__(self, 
                 high_overlap_threshold: float = 0.6,
                 merge_threshold: float = 0.5):
        """
        Initialize grouper.
        
        Args:
            high_overlap_threshold: Similarity for clustering novel theories
            merge_threshold: Similarity for merging novel with canonical groups
        """
        self.high_overlap_threshold = high_overlap_threshold
        self.merge_threshold = merge_threshold
        
        self.groups = []
        self.theory_to_group = {}
        
        self.stats = {
            'total_theories': 0,
            'canonical_groups': 0,
            'novel_groups': 0,
            'merged_groups': 0,
            'singleton_groups': 0,
            'avg_group_size': 0.0
        }
    
    def _normalize_list(self, items: List[str]) -> Set[str]:
        """Normalize list for comparison with better tokenization."""
        normalized = set()
        for item in items:
            if not item:
                continue
            # Convert to lowercase and strip
            item = item.lower().strip()
            # Remove common words that don't add meaning
            stop_words = ['the', 'a', 'an', 'of', 'in', 'to', 'and', 'or', 'with', 'by']
            words = item.split()
            words = [w for w in words if w not in stop_words]
            # Also add individual significant words for better matching
            normalized.add(' '.join(words))
            # Add individual words for partial matching
            for word in words:
                if len(word) > 3:  # Only add words longer than 3 chars
                    normalized.add(word)
        return normalized
    
    def _normalize_theory_name(self, name: str) -> str:
        """Normalize theory name for comparison."""
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove common suffixes/prefixes
        removals = [
            'theory', 'hypothesis', 'model', 'of aging', 'aging', 
            'the ', 'a ', 'an '
        ]
        for removal in removals:
            name = name.replace(removal, '')
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two theory names."""
        norm1 = self._normalize_theory_name(name1)
        norm2 = self._normalize_theory_name(name2)
        
        # Use SequenceMatcher for fuzzy string matching
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_mechanism_signature(self, theory: Dict) -> Dict[str, Set[str]]:
        """Compute comprehensive signature for a theory using all metadata."""
        metadata = theory.get('stage3_metadata', {})
        
        return {
            'mechanisms': self._normalize_list(metadata.get('mechanisms', [])),
            'key_players': self._normalize_list(metadata.get('key_players', [])),
            'pathways': self._normalize_list(metadata.get('pathways', [])),
            'level_of_explanation': metadata.get('level_of_explanation'),
            'type_of_cause': metadata.get('type_of_cause'),
            'temporal_focus': metadata.get('temporal_focus'),
            'adaptiveness': metadata.get('adaptiveness'),
            'has_data': bool(metadata.get('mechanisms') or metadata.get('key_players'))
        }
    
    def _calculate_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate multi-dimensional weighted similarity between two signatures."""
        if not sig1['has_data'] or not sig2['has_data']:
            return 0.0
        
        # 1. Calculate content similarity (mechanisms, key players, pathways)
        mech_sim = self._jaccard_similarity(sig1['mechanisms'], sig2['mechanisms'])
        player_sim = self._jaccard_similarity(sig1['key_players'], sig2['key_players'])
        pathway_sim = self._jaccard_similarity(sig1['pathways'], sig2['pathways'])
        
        # 2. Calculate categorical similarity (level, type, temporal, adaptiveness)
        categorical_matches = 0
        categorical_total = 0
        
        # Level of explanation match
        if sig1['level_of_explanation'] and sig2['level_of_explanation']:
            categorical_total += 1
            if sig1['level_of_explanation'] == sig2['level_of_explanation']:
                categorical_matches += 1
        
        # Type of cause match
        if sig1['type_of_cause'] and sig2['type_of_cause']:
            categorical_total += 1
            if sig1['type_of_cause'] == sig2['type_of_cause']:
                categorical_matches += 1
        
        # Temporal focus match
        if sig1['temporal_focus'] and sig2['temporal_focus']:
            categorical_total += 1
            if sig1['temporal_focus'] == sig2['temporal_focus']:
                categorical_matches += 1
        
        # Adaptiveness match
        if sig1['adaptiveness'] and sig2['adaptiveness']:
            categorical_total += 1
            if sig1['adaptiveness'] == sig2['adaptiveness']:
                categorical_matches += 1
        
        categorical_sim = categorical_matches / categorical_total if categorical_total > 0 else 0.0
        
        # 3. Weighted combination
        # Content similarity (80%): mechanisms, key players, pathways
        # Categorical similarity (20%): level, type, temporal, adaptiveness
        
        if sig1['mechanisms'] and sig2['mechanisms']:
            # If both have mechanisms, weight them most heavily
            content_sim = mech_sim * 0.5 + player_sim * 0.3 + pathway_sim * 0.2
        elif sig1['key_players'] and sig2['key_players']:
            # If no mechanisms but have key players
            content_sim = player_sim * 0.6 + pathway_sim * 0.4
        elif sig1['pathways'] and sig2['pathways']:
            # Only pathways
            content_sim = pathway_sim
        else:
            content_sim = 0.0
        
        # Combine content and categorical similarity
        # Give more weight to content (actual mechanisms) than categories
        final_sim = content_sim * 0.8 + categorical_sim * 0.2
        
        return final_sim
    
    def _get_similarity_breakdown(self, sig1: Dict, sig2: Dict) -> Dict:
        """Get detailed breakdown of similarity scores for analysis."""
        if not sig1['has_data'] or not sig2['has_data']:
            return {'total': 0.0, 'breakdown': {}}
        
        mech_sim = self._jaccard_similarity(sig1['mechanisms'], sig2['mechanisms'])
        player_sim = self._jaccard_similarity(sig1['key_players'], sig2['key_players'])
        pathway_sim = self._jaccard_similarity(sig1['pathways'], sig2['pathways'])
        
        categorical_matches = []
        if sig1['level_of_explanation'] and sig2['level_of_explanation']:
            match = sig1['level_of_explanation'] == sig2['level_of_explanation']
            categorical_matches.append(('level_of_explanation', match))
        
        if sig1['type_of_cause'] and sig2['type_of_cause']:
            match = sig1['type_of_cause'] == sig2['type_of_cause']
            categorical_matches.append(('type_of_cause', match))
        
        if sig1['temporal_focus'] and sig2['temporal_focus']:
            match = sig1['temporal_focus'] == sig2['temporal_focus']
            categorical_matches.append(('temporal_focus', match))
        
        if sig1['adaptiveness'] and sig2['adaptiveness']:
            match = sig1['adaptiveness'] == sig2['adaptiveness']
            categorical_matches.append(('adaptiveness', match))
        
        categorical_sim = sum(1 for _, match in categorical_matches if match) / len(categorical_matches) if categorical_matches else 0.0
        
        # Calculate content similarity
        if sig1['mechanisms'] and sig2['mechanisms']:
            content_sim = mech_sim * 0.5 + player_sim * 0.3 + pathway_sim * 0.2
        elif sig1['key_players'] and sig2['key_players']:
            content_sim = player_sim * 0.6 + pathway_sim * 0.4
        elif sig1['pathways'] and sig2['pathways']:
            content_sim = pathway_sim
        else:
            content_sim = 0.0
        
        final_sim = content_sim * 0.8 + categorical_sim * 0.2
        
        return {
            'total': final_sim,
            'content_similarity': content_sim,
            'categorical_similarity': categorical_sim,
            'breakdown': {
                'mechanisms': mech_sim,
                'key_players': player_sim,
                'pathways': pathway_sim,
                'categorical_matches': dict(categorical_matches)
            }
        }
    
    def group_theories(self, theories_with_mechanisms: List[Dict]) -> List[TheoryGroup]:
        """
        Group theories by shared mechanisms.
        
        Strategy:
        1. Group theories with canonical names
        2. Cluster novel/unmatched theories by mechanism similarity
        3. Attempt to merge novel clusters with canonical groups
        """
        print(f"\n🔄 Grouping {len(theories_with_mechanisms)} theories...")
        
        self.stats['total_theories'] = len(theories_with_mechanisms)
        
        groups = []
        group_counter = 0
        
        # Step 1: Group by canonical name
        print(f"  Step 1: Grouping theories by canonical name...")
        canonical_groups = defaultdict(list)
        unmapped_theories = []
        
        for theory in theories_with_mechanisms:
            canonical_name = theory.get('match_result', {}).get('canonical_name')
            if canonical_name:
                canonical_groups[canonical_name].append(theory)
            else:
                unmapped_theories.append(theory)
        
        # Create canonical groups
        for canonical_name, theories in canonical_groups.items():
            group_counter += 1
            
            # Get shared mechanisms (should all be the same canonical mechanisms)
            first_theory = theories[0]
            metadata = first_theory.get('stage3_metadata', {})
            
            group = TheoryGroup(
                group_id=f"G{group_counter:04d}",
                canonical_name=canonical_name,
                representative_name=canonical_name,
                theory_ids=[t['theory_id'] for t in theories],
                theory_count=len(theories),
                shared_mechanisms=metadata.get('mechanisms', []),
                shared_key_players=metadata.get('key_players', []),
                shared_pathways=metadata.get('pathways', []),
                source='canonical',
                mechanism_source='canonical'
            )
            
            groups.append(group)
            for t in theories:
                self.theory_to_group[t['theory_id']] = group.group_id
        
        self.stats['canonical_groups'] = len(canonical_groups)
        print(f"    Created {len(canonical_groups)} canonical groups")
        
        # Step 2: Cluster unmapped theories by NAME + mechanism similarity
        print(f"  Step 2: Clustering {len(unmapped_theories)} unmapped theories...")
        
        if unmapped_theories:
            # Compute signatures with names
            theories_with_sigs = []
            for theory in unmapped_theories:
                sig = self._compute_mechanism_signature(theory)
                if sig['has_data']:
                    # Get theory name (prefer proposed_name from Stage 1.5)
                    stage1_5_result = theory.get('stage1_5_result', {})
                    theory_name = stage1_5_result.get('proposed_name') or theory.get('name') or theory.get('original_name', 'Unknown')
                    sig['theory_name'] = theory_name
                    theories_with_sigs.append((theory, sig))
            
            # Greedy clustering with NAME + MECHANISM similarity
            while theories_with_sigs:
                seed_theory, seed_sig = theories_with_sigs.pop(0)
                cluster = [seed_theory]
                cluster_sigs = [seed_sig]
                
                # Find similar theories (by name AND mechanisms)
                remaining = []
                for theory, sig in theories_with_sigs:
                    # Calculate name similarity
                    name_sim = self._calculate_name_similarity(seed_sig['theory_name'], sig['theory_name'])
                    
                    # Calculate mechanism similarity
                    mech_sim = self._calculate_similarity(seed_sig, sig)
                    
                    # Combined similarity (weight name more heavily for novel theories)
                    # Use 70% name, 30% mechanism to prioritize name matching
                    combined_sim = name_sim * 0.7 + mech_sim * 0.3
                    
                    if combined_sim >= self.high_overlap_threshold:
                        cluster.append(theory)
                        cluster_sigs.append(sig)
                    else:
                        remaining.append((theory, sig))
                
                theories_with_sigs = remaining
                
                # Create group
                group_counter += 1
                
                # Compute shared characteristics
                all_mechanisms = [sig['mechanisms'] for sig in cluster_sigs]
                all_players = [sig['key_players'] for sig in cluster_sigs]
                all_pathways = [sig['pathways'] for sig in cluster_sigs]
                
                shared_mechanisms = list(set.intersection(*all_mechanisms) if all_mechanisms else set())
                shared_players = list(set.intersection(*all_players) if all_players else set())
                shared_pathways = list(set.intersection(*all_pathways) if all_pathways else set())
                
                # Get representative name (normalized)
                # Collect all names from cluster
                all_names = []
                for t in cluster:
                    stage1_5_result = t.get('stage1_5_result', {})
                    name = stage1_5_result.get('proposed_name') or t.get('name') or t.get('original_name', 'Unknown')
                    all_names.append(name)
                
                # Find most common name
                name_counter = Counter(all_names)
                representative_name = name_counter.most_common(1)[0][0] if all_names else "Unknown"
                
                # Store all variant names for reference
                variant_names = list(set(all_names))
                
                # Get metadata from first theory
                first_metadata = cluster[0].get('stage3_metadata', {})
                
                group = TheoryGroup(
                    group_id=f"G{group_counter:04d}",
                    canonical_name=None,
                    representative_name=representative_name,
                    theory_ids=[t['theory_id'] for t in cluster],
                    theory_count=len(cluster),
                    shared_mechanisms=shared_mechanisms,
                    shared_key_players=shared_players,
                    shared_pathways=shared_pathways,
                    level_of_explanation=first_metadata.get('level_of_explanation'),
                    type_of_cause=first_metadata.get('type_of_cause'),
                    temporal_focus=first_metadata.get('temporal_focus'),
                    adaptiveness=first_metadata.get('adaptiveness'),
                    source='novel',
                    mechanism_source='extracted'
                )
                
                # Store variant names in group (for debugging/analysis)
                group.primary_category = f"Novel ({len(variant_names)} variants)"
                group.secondary_category = ', '.join(variant_names[:3]) + ('...' if len(variant_names) > 3 else '')
                
                groups.append(group)
                for t in cluster:
                    self.theory_to_group[t['theory_id']] = group.group_id
            
            novel_groups = group_counter - len(canonical_groups)
            self.stats['novel_groups'] = novel_groups
            print(f"    Created {novel_groups} novel groups")
        
        # Step 3: Attempt to merge novel groups with canonical groups
        print(f"  Step 3: Attempting to merge novel groups with canonical groups...")
        
        canonical_group_list = [g for g in groups if g.source == 'canonical']
        novel_group_list = [g for g in groups if g.source == 'novel']
        
        merged_count = 0
        for novel_group in novel_group_list:
            novel_sig = {
                'mechanisms': self._normalize_list(novel_group.shared_mechanisms),
                'key_players': self._normalize_list(novel_group.shared_key_players),
                'pathways': self._normalize_list(novel_group.shared_pathways),
                'has_data': True
            }
            
            best_match = None
            best_similarity = 0.0
            
            for canonical_group in canonical_group_list:
                canonical_sig = {
                    'mechanisms': self._normalize_list(canonical_group.shared_mechanisms),
                    'key_players': self._normalize_list(canonical_group.shared_key_players),
                    'pathways': self._normalize_list(canonical_group.shared_pathways),
                    'has_data': True
                }
                
                similarity = self._calculate_similarity(novel_sig, canonical_sig)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = canonical_group
            
            if best_match and best_similarity >= self.merge_threshold:
                # Merge novel group into canonical group
                best_match.theory_ids.extend(novel_group.theory_ids)
                best_match.theory_count += novel_group.theory_count
                best_match.source = 'mixed'
                
                # Update theory_to_group mapping
                for theory_id in novel_group.theory_ids:
                    self.theory_to_group[theory_id] = best_match.group_id
                
                # Remove novel group
                groups.remove(novel_group)
                merged_count += 1
        
        self.stats['merged_groups'] = merged_count
        print(f"    Merged {merged_count} novel groups into canonical groups")
        
        self.groups = groups
        self.stats['singleton_groups'] = sum(1 for g in groups if g.theory_count == 1)
        self.stats['avg_group_size'] = sum(g.theory_count for g in groups) / len(groups) if groups else 0
        
        print(f"✓ Created {len(groups)} final groups")
        
        return groups
    
    def save_results(self, theories: List[Dict], output_path: str):
        """Save grouping results."""
        print(f"\n💾 Saving results to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add group_id to each theory
        for theory in theories:
            theory['group_id'] = self.theory_to_group.get(theory['theory_id'], None)
        
        data = {
            'metadata': {
                'stage': 'stage4_improved_grouping',
                'statistics': self.stats,
                'thresholds': {
                    'high_overlap': self.high_overlap_threshold,
                    'merge': self.merge_threshold
                }
            },
            'groups': [g.to_dict() for g in self.groups],
            'theories': theories
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print grouping statistics."""
        print("\n" + "="*80)
        print("STAGE 4: IMPROVED GROUPING STATISTICS")
        print("="*80)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"\nGroups created:")
        print(f"  Canonical groups: {self.stats['canonical_groups']}")
        print(f"  Novel groups: {self.stats['novel_groups']}")
        print(f"  Merged groups: {self.stats['merged_groups']}")
        print(f"  Total final groups: {len(self.groups)}")
        print(f"\nGroup characteristics:")
        print(f"  Singleton groups: {self.stats['singleton_groups']}")
        print(f"  Multi-theory groups: {len(self.groups) - self.stats['singleton_groups']}")
        print(f"  Avg group size: {self.stats['avg_group_size']:.1f}")
        print(f"\nCompression:")
        compression = (1 - len(self.groups) / self.stats['total_theories']) * 100
        print(f"  {compression:.1f}% reduction ({self.stats['total_theories']} → {len(self.groups)} groups)")
        print("="*80)
    
    def print_sample_groups(self, n: int = 10):
        """Print sample groups."""
        print(f"\n📊 Sample Groups (largest {n}):")
        print("-" * 80)
        
        sorted_groups = sorted(self.groups, key=lambda g: g.theory_count, reverse=True)
        
        for i, group in enumerate(sorted_groups[:n], 1):
            print(f"\n{i}. {group.representative_name}")
            print(f"   Group ID: {group.group_id}")
            print(f"   Theory count: {group.theory_count}")
            print(f"   Source: {group.source} ({group.mechanism_source} mechanisms)")
            
            if group.canonical_name:
                print(f"   Canonical: {group.canonical_name}")
            
            if group.shared_mechanisms:
                print(f"   Mechanisms ({len(group.shared_mechanisms)}): {len(group.shared_mechanisms)} total")
                print(f"   Sample: {group.shared_mechanisms[0][:80]}...")
            
            if group.shared_key_players:
                print(f"   Key players ({len(group.shared_key_players)}): {', '.join(list(group.shared_key_players)[:5])}...")


def main():
    """Run improved Stage 4 grouping."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 4: Improved theory grouping')
    parser.add_argument('--input', default='output/stage3_extracted_improved_TEST.json', help='Stage 3 improved output')
    parser.add_argument('--output', default='output/stage4_groups_improved.json', help='Output file')
    parser.add_argument('--overlap-threshold', type=float, default=0.7, help='Clustering threshold')
    parser.add_argument('--merge-threshold', type=float, default=0.6, help='Merge threshold')
    
    args = parser.parse_args()
    
    print("🚀 Starting Stage 4: Improved Theory Grouping\n")
    
    # Load Stage 3 output
    print(f"📂 Loading Stage 3 output from {args.input}...")
    with open(args.input, 'r') as f:
        stage3_data = json.load(f)
    
    theories = stage3_data.get('theories_with_mechanisms', [])
    print(f"✓ Loaded {len(theories)} theories with mechanisms")
    
    # Initialize grouper
    grouper = ImprovedTheoryGrouper(
        high_overlap_threshold=args.overlap_threshold,
        merge_threshold=args.merge_threshold
    )
    
    # Group theories
    groups = grouper.group_theories(theories)
    
    # Save results
    grouper.save_results(theories, args.output)
    
    # Print statistics
    grouper.print_statistics()
    
    # Print sample groups
    grouper.print_sample_groups(n=10)
    
    print("\n✅ Stage 4 complete!")
    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
