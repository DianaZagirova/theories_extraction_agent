"""
Stage 3: Mechanism-Based Clustering
Cluster theories based on extracted biological mechanisms, not embeddings.
"""

import json
import os
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class MechanismCluster:
    """Represents a cluster of theories based on mechanisms."""
    cluster_id: str
    level: str  # 'family', 'parent', or 'child'
    name: str
    theory_ids: List[str] = field(default_factory=list)
    mechanism_signature: Dict = field(default_factory=dict)
    parent_cluster_id: str = None
    child_cluster_ids: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'cluster_id': self.cluster_id,
            'level': self.level,
            'name': self.name,
            'theory_ids': self.theory_ids,
            'theory_count': len(self.theory_ids),
            'mechanism_signature': self.mechanism_signature,
            'parent_cluster_id': self.parent_cluster_id,
            'child_cluster_ids': self.child_cluster_ids
        }


class MechanismClusterer:
    """Cluster theories based on biological mechanisms."""
    
    def __init__(self):
        self.families = []
        self.parents = []
        self.children = []
        self.taxonomy = {}
        
        self.stats = {
            'total_theories': 0,
            'num_families': 0,
            'num_parents': 0,
            'num_children': 0,
            'theories_without_mechanisms': 0
        }
    
    def load_data(self, theories_path: str, mechanisms_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load theories and mechanisms."""
        print(f"📂 Loading data...")
        
        with open(theories_path, 'r') as f:
            theories_data = json.load(f)
        theories = theories_data['theories']
        
        with open(mechanisms_path, 'r') as f:
            mechanisms_data = json.load(f)
        mechanisms = mechanisms_data['mechanisms']
        
        print(f"✓ Loaded {len(theories)} theories and {len(mechanisms)} mechanism extractions")
        
        self.stats['total_theories'] = len(theories)
        
        return theories, mechanisms
    
    def build_taxonomy(self, mechanisms: List[Dict]):
        """Build hierarchical taxonomy from mechanisms."""
        print(f"\n🔄 Building mechanism taxonomy...")
        
        taxonomy = {
            'primary_categories': defaultdict(int),
            'secondary_categories': defaultdict(lambda: defaultdict(int)),
            'specific_mechanisms': defaultdict(lambda: defaultdict(int)),
            'pathways': defaultdict(int),
            'molecules': defaultdict(int),
            'biological_levels': defaultdict(int),
            'mechanism_types': defaultdict(int)
        }
        
        for mech in mechanisms:
            # Count primary categories
            primary = mech['primary_category']
            taxonomy['primary_categories'][primary] += 1
            
            # Count secondary categories
            for secondary in mech['secondary_categories']:
                taxonomy['secondary_categories'][primary][secondary] += 1
            
            # Count specific mechanisms
            for mechanism in mech['specific_mechanisms']:
                for secondary in mech['secondary_categories']:
                    key = f"{primary}/{secondary}"
                    taxonomy['specific_mechanisms'][key][mechanism] += 1
            
            # Count pathways
            for pathway in mech['pathways']:
                taxonomy['pathways'][pathway] += 1
            
            # Count molecules
            for molecule in mech['molecules']:
                taxonomy['molecules'][molecule] += 1
            
            # Count biological levels
            taxonomy['biological_levels'][mech['biological_level']] += 1
            
            # Count mechanism types
            taxonomy['mechanism_types'][mech['mechanism_type']] += 1
        
        self.taxonomy = taxonomy
        
        print(f"✓ Built taxonomy:")
        print(f"  Primary categories: {len(taxonomy['primary_categories'])}")
        print(f"  Secondary categories: {sum(len(v) for v in taxonomy['secondary_categories'].values())}")
        print(f"  Specific mechanisms: {sum(len(v) for v in taxonomy['specific_mechanisms'].values())}")
        print(f"  Pathways: {len(taxonomy['pathways'])}")
        
        return taxonomy
    
    def cluster_level1_families(self, theories: List[Dict], 
                                mechanisms: List[Dict]) -> List[MechanismCluster]:
        """
        Level 1: Families = Secondary Category
        Group theories by their secondary mechanism category.
        """
        print(f"\n🔄 Level 1: Creating theory families (by secondary category)...")
        
        # Create mechanism lookup
        mech_lookup = {m['theory_id']: m for m in mechanisms}
        
        # Group by secondary category
        family_groups = defaultdict(list)
        
        for theory in theories:
            mech = mech_lookup.get(theory['theory_id'])
            if not mech:
                self.stats['theories_without_mechanisms'] += 1
                continue
            
            # Use first secondary category as primary grouping
            if mech['secondary_categories']:
                secondary = mech['secondary_categories'][0]
                primary = mech['primary_category']
                family_key = f"{primary}/{secondary}"
                family_groups[family_key].append(theory['theory_id'])
            else:
                # Fallback to primary category only
                family_groups[mech['primary_category']].append(theory['theory_id'])
        
        # Create family clusters
        families = []
        for i, (family_key, theory_ids) in enumerate(sorted(family_groups.items()), 1):
            # Get mechanism signature
            family_mechanisms = [mech_lookup[tid] for tid in theory_ids if tid in mech_lookup]
            
            signature = {
                'primary_category': family_mechanisms[0]['primary_category'] if family_mechanisms else 'Unknown',
                'secondary_category': family_key.split('/')[-1] if '/' in family_key else family_key,
                'common_mechanisms': self._get_common_items([m['specific_mechanisms'] for m in family_mechanisms], min_count=2),
                'common_pathways': self._get_common_items([m['pathways'] for m in family_mechanisms], min_count=2)
            }
            
            family = MechanismCluster(
                cluster_id=f"F{i:03d}",
                level='family',
                name=self._generate_family_name(family_key, signature),
                theory_ids=theory_ids,
                mechanism_signature=signature
            )
            families.append(family)
        
        self.families = families
        self.stats['num_families'] = len(families)
        
        print(f"✓ Created {len(families)} theory families")
        print(f"  Avg theories per family: {sum(len(f.theory_ids) for f in families)/len(families):.1f}")
        
        return families
    
    def cluster_level2_parents(self, theories: List[Dict],
                              mechanisms: List[Dict],
                              families: List[MechanismCluster]) -> List[MechanismCluster]:
        """
        Level 2: Parents = Specific Mechanism
        Within each family, group by specific mechanisms.
        """
        print(f"\n🔄 Level 2: Creating parent theories (by specific mechanism)...")
        
        mech_lookup = {m['theory_id']: m for m in mechanisms}
        
        all_parents = []
        parent_counter = 0
        
        for family in families:
            # Group theories by specific mechanisms
            parent_groups = defaultdict(list)
            
            for theory_id in family.theory_ids:
                mech = mech_lookup.get(theory_id)
                if not mech or not mech['specific_mechanisms']:
                    # Put in "Other" group
                    parent_groups['Other'].append(theory_id)
                    continue
                
                # Use first specific mechanism as primary grouping
                mechanism = mech['specific_mechanisms'][0]
                parent_groups[mechanism].append(theory_id)
            
            # Create parent clusters
            for mechanism, theory_ids in sorted(parent_groups.items()):
                parent_counter += 1
                
                # Get mechanism signature
                parent_mechanisms = [mech_lookup[tid] for tid in theory_ids if tid in mech_lookup]
                
                signature = {
                    'specific_mechanism': mechanism,
                    'common_pathways': self._get_common_items([m['pathways'] for m in parent_mechanisms], min_count=1),
                    'common_molecules': self._get_common_items([m['molecules'] for m in parent_mechanisms], min_count=1),
                    'mechanism_types': list(set(m['mechanism_type'] for m in parent_mechanisms))
                }
                
                parent = MechanismCluster(
                    cluster_id=f"P{parent_counter:04d}",
                    level='parent',
                    name=self._generate_parent_name(mechanism, signature),
                    theory_ids=theory_ids,
                    mechanism_signature=signature,
                    parent_cluster_id=family.cluster_id
                )
                all_parents.append(parent)
                family.child_cluster_ids.append(parent.cluster_id)
        
        self.parents = all_parents
        self.stats['num_parents'] = len(all_parents)
        
        print(f"✓ Created {len(all_parents)} parent theories")
        print(f"  Avg parents per family: {len(all_parents)/len(families):.1f}")
        
        return all_parents
    
    def cluster_level3_children(self, theories: List[Dict],
                               mechanisms: List[Dict],
                               parents: List[MechanismCluster]) -> List[MechanismCluster]:
        """
        Level 3: Children = Pathway/Molecule
        Within each parent, group by pathways or molecules.
        """
        print(f"\n🔄 Level 3: Creating child theories (by pathway/molecule)...")
        
        mech_lookup = {m['theory_id']: m for m in mechanisms}
        
        all_children = []
        child_counter = 0
        
        for parent in parents:
            # If parent has few theories, don't split further
            if len(parent.theory_ids) <= 3:
                child_counter += 1
                child = MechanismCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    name=parent.name,
                    theory_ids=parent.theory_ids,
                    mechanism_signature=parent.mechanism_signature,
                    parent_cluster_id=parent.cluster_id
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
                continue
            
            # Group by pathway or molecule
            child_groups = defaultdict(list)
            
            for theory_id in parent.theory_ids:
                mech = mech_lookup.get(theory_id)
                if not mech:
                    child_groups['Other'].append(theory_id)
                    continue
                
                # Try pathways first
                if mech['pathways']:
                    pathway = mech['pathways'][0]
                    child_groups[pathway].append(theory_id)
                # Then molecules
                elif mech['molecules']:
                    molecule = mech['molecules'][0]
                    child_groups[molecule].append(theory_id)
                # Then mechanism type
                elif mech['mechanism_type']:
                    child_groups[mech['mechanism_type']].append(theory_id)
                else:
                    child_groups['Other'].append(theory_id)
            
            # Create child clusters
            for key, theory_ids in sorted(child_groups.items()):
                child_counter += 1
                
                # Get mechanism signature
                child_mechanisms = [mech_lookup[tid] for tid in theory_ids if tid in mech_lookup]
                
                signature = {
                    'pathway_or_molecule': key,
                    'mechanism_types': list(set(m['mechanism_type'] for m in child_mechanisms)),
                    'biological_levels': list(set(m['biological_level'] for m in child_mechanisms))
                }
                
                child = MechanismCluster(
                    cluster_id=f"C{child_counter:04d}",
                    level='child',
                    name=self._generate_child_name(key, parent.name, signature),
                    theory_ids=theory_ids,
                    mechanism_signature=signature,
                    parent_cluster_id=parent.cluster_id
                )
                all_children.append(child)
                parent.child_cluster_ids.append(child.cluster_id)
        
        self.children = all_children
        self.stats['num_children'] = len(all_children)
        
        print(f"✓ Created {len(all_children)} child theories")
        print(f"  Avg children per parent: {len(all_children)/len(parents):.1f}")
        
        return all_children
    
    def _get_common_items(self, item_lists: List[List[str]], min_count: int = 2) -> List[str]:
        """Get items that appear in at least min_count lists."""
        if not item_lists:
            return []
        
        # Flatten and count
        all_items = [item for items in item_lists for item in items]
        counter = Counter(all_items)
        
        # Return items that appear at least min_count times
        return [item for item, count in counter.most_common() if count >= min_count]
    
    def _generate_family_name(self, family_key: str, signature: Dict) -> str:
        """Generate descriptive name for family."""
        secondary = signature['secondary_category']
        
        # Clean up name
        name = secondary.replace('_', ' ').title()
        
        # Add common mechanisms if available
        if signature['common_mechanisms']:
            mechanisms = ', '.join(signature['common_mechanisms'][:2])
            name = f"{name} ({mechanisms})"
        
        return name
    
    def _generate_parent_name(self, mechanism: str, signature: Dict) -> str:
        """Generate descriptive name for parent."""
        name = mechanism.replace('_', ' ').title()
        
        # Add pathways if available
        if signature['common_pathways']:
            pathways = ', '.join(signature['common_pathways'][:2])
            name = f"{name} - {pathways}"
        
        return name
    
    def _generate_child_name(self, key: str, parent_name: str, signature: Dict) -> str:
        """Generate descriptive name for child."""
        if key == 'Other':
            return f"{parent_name} (Other)"
        
        name = key.replace('_', ' ').title()
        
        # Add mechanism type if available
        if signature['mechanism_types'] and signature['mechanism_types'][0] != 'Unknown':
            mech_type = signature['mechanism_types'][0].replace('_', ' ').title()
            name = f"{name} ({mech_type})"
        
        return name
    
    def save_clusters(self, theories: List[Dict], output_path: str):
        """Save clustering results."""
        print(f"\n💾 Saving clusters to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'metadata': {
                'stage': 'stage3_mechanism_clustering',
                'approach': 'mechanism-based (not embedding-based)',
                'statistics': self.stats,
                'taxonomy_summary': {
                    'primary_categories': len(self.taxonomy['primary_categories']),
                    'pathways': len(self.taxonomy['pathways']),
                    'molecules': len(self.taxonomy['molecules'])
                }
            },
            'theories': theories,
            'families': [f.to_dict() for f in self.families],
            'parents': [p.to_dict() for p in self.parents],
            'children': [c.to_dict() for c in self.children],
            'taxonomy': {
                'primary_categories': dict(self.taxonomy['primary_categories']),
                'secondary_categories': {k: dict(v) for k, v in self.taxonomy['secondary_categories'].items()},
                'pathways': dict(sorted(self.taxonomy['pathways'].items(), key=lambda x: x[1], reverse=True)[:50]),
                'molecules': dict(sorted(self.taxonomy['molecules'].items(), key=lambda x: x[1], reverse=True)[:50])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to {output_path}")
    
    def print_statistics(self):
        """Print clustering statistics."""
        print("\n" + "="*60)
        print("MECHANISM-BASED CLUSTERING STATISTICS")
        print("="*60)
        print(f"Total theories: {self.stats['total_theories']}")
        print(f"Theories without mechanisms: {self.stats['theories_without_mechanisms']}")
        print(f"\nLevel 1 - Theory Families: {self.stats['num_families']}")
        print(f"  Avg theories per family: {sum(len(f.theory_ids) for f in self.families)/len(self.families):.1f}")
        print(f"\nLevel 2 - Parent Theories: {self.stats['num_parents']}")
        print(f"  Avg parents per family: {self.stats['num_parents']/self.stats['num_families']:.1f}")
        print(f"\nLevel 3 - Child Theories: {self.stats['num_children']}")
        print(f"  Avg children per parent: {self.stats['num_children']/self.stats['num_parents']:.1f}")
        print(f"\nCompression ratio: {self.stats['total_theories']/self.stats['num_children']:.1f}:1")
        print("="*60)


def main():
    """Run mechanism-based clustering."""
    print("🚀 Starting Stage 3: Mechanism-Based Clustering\n")
    
    # Initialize clusterer
    clusterer = MechanismClusterer()
    
    # Load data
    theories, mechanisms = clusterer.load_data(
        'output/stage1_embeddings.json',
        'output/stage2_mechanisms.json'
    )
    
    # Build taxonomy
    taxonomy = clusterer.build_taxonomy(mechanisms)
    
    # Level 1: Families (by secondary category)
    families = clusterer.cluster_level1_families(theories, mechanisms)
    
    # Level 2: Parents (by specific mechanism)
    parents = clusterer.cluster_level2_parents(theories, mechanisms, families)
    
    # Level 3: Children (by pathway/molecule)
    children = clusterer.cluster_level3_children(theories, mechanisms, parents)
    
    # Save results
    clusterer.save_clusters(theories, 'output/stage3_mechanism_clusters.json')
    
    # Print statistics
    clusterer.print_statistics()
    
    # Print sample families
    print("\n📊 Sample Families:")
    for i, family in enumerate(families[:5], 1):
        print(f"\n{i}. {family.name} ({len(family.theory_ids)} theories)")
        print(f"   Signature: {family.mechanism_signature['secondary_category']}")
        if family.mechanism_signature['common_mechanisms']:
            print(f"   Common mechanisms: {', '.join(family.mechanism_signature['common_mechanisms'][:3])}")
    
    print("\n✅ Stage 3 (Mechanism-Based Clustering) complete!")
    print("\nNext step: Review output/stage3_mechanism_clusters.json")


if __name__ == '__main__':
    main()
