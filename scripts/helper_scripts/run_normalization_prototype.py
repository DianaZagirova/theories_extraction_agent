"""
Theory Normalization Prototype Runner
Tests the pipeline on a small subset of theories for threshold tuning.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.normalization.stage0_quality_filter import QualityFilter
from src.normalization.stage1_embedding import EmbeddingGenerator
from src.normalization.stage2_clustering import HierarchicalClusterer
from src.normalization.stage3_llm_validation import LLMValidator
from src.normalization.stage4_ontology_matching import OntologyMatcher
from src.core.llm_integration import AzureOpenAIClient


class NormalizationPrototype:
    """Runs normalization pipeline on a subset for testing."""
    
    def __init__(self, subset_size: int = 200):
        """
        Initialize prototype runner.
        
        Args:
            subset_size: Number of theories to process in prototype
        """
        self.subset_size = subset_size
        self.output_dir = 'output/prototype'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize LLM client
        try:
            self.llm_client = AzureOpenAIClient()
            print("✓ LLM client initialized\n")
        except Exception as e:
            print(f"⚠ Warning: LLM client not available: {e}")
            print("  Some features will be limited\n")
            self.llm_client = None
    
    def create_subset(self, input_path: str) -> str:
        """Create a subset of theories for testing."""
        print(f"📂 Creating subset of {self.subset_size} theories...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Take first N papers with theories
        subset_results = []
        theory_count = 0
        
        for paper in data['results']:
            if paper.get('theories'):
                subset_results.append(paper)
                theory_count += len(paper['theories'])
                
                if theory_count >= self.subset_size:
                    break
        
        # Create subset file
        subset_data = {
            'metadata': {
                'subset': True,
                'subset_size': theory_count,
                'original_file': input_path
            },
            'results': subset_results
        }
        
        subset_path = f'{self.output_dir}/subset_theories.json'
        with open(subset_path, 'w') as f:
            json.dump(subset_data, f, indent=2)
        
        print(f"✓ Created subset with {theory_count} theories from {len(subset_results)} papers")
        print(f"  Saved to: {subset_path}\n")
        
        return subset_path
    
    def run_stage0(self, input_path: str) -> str:
        """Run Stage 0: Quality filtering."""
        print("="*70)
        print("STAGE 0: QUALITY FILTERING")
        print("="*70)
        
        filter_engine = QualityFilter(llm_client=self.llm_client)
        
        theories = filter_engine.load_theories(input_path)
        filtered = filter_engine.filter_by_confidence(theories, validate_medium=False)
        enriched = filter_engine.enrich_theories(filtered)
        
        output_path = f'{self.output_dir}/stage0_filtered.json'
        filter_engine.save_filtered_theories(enriched, output_path)
        filter_engine.print_statistics()
        
        return output_path
    
    def run_stage1(self, input_path: str, use_local: bool = False) -> str:
        """Run Stage 1: Embedding generation."""
        print("\n" + "="*70)
        print("STAGE 1: EMBEDDING GENERATION")
        print("="*70)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        theories = data['theories']
        
        # Use local model if requested or if OpenAI not available
        if use_local:
            print("🔧 Using local embeddings (sentence-transformers)")
            use_openai = False
            llm_client = None
        else:
            use_openai = self.llm_client is not None
            llm_client = self.llm_client
            if use_openai:
                print("🔧 Attempting OpenAI embeddings (will fallback to local if fails)")
        
        generator = EmbeddingGenerator(use_openai=use_openai, llm_client=llm_client)
        
        embeddings = generator.generate_embeddings(theories, batch_size=50)
        
        output_path = f'{self.output_dir}/stage1_embeddings.json'
        generator.save_embeddings(embeddings, theories, output_path)
        
        return output_path
    
    def run_stage2(self, input_path: str, 
                   family_threshold: float = 0.7,
                   parent_threshold: float = 0.5,
                   child_threshold: float = 0.4) -> str:
        """Run Stage 2: Hierarchical clustering."""
        print("\n" + "="*70)
        print("STAGE 2: HIERARCHICAL CLUSTERING")
        print("="*70)
        
        clusterer = HierarchicalClusterer(
            family_threshold=family_threshold,
            parent_threshold=parent_threshold,
            child_threshold=child_threshold
        )
        
        theories, name_emb, semantic_emb, detailed_emb = clusterer.load_embeddings(input_path)
        
        with open(input_path, 'r') as f:
            emb_data = json.load(f)
        embeddings_list = emb_data['embeddings']
        
        families = clusterer.cluster_level1_families(theories, name_emb)
        parents = clusterer.cluster_level2_parents(theories, semantic_emb, families)
        children = clusterer.cluster_level3_children(theories, detailed_emb, parents, embeddings_list)
        
        output_path = f'{self.output_dir}/stage2_clusters.json'
        clusterer.save_clusters(theories, output_path)
        clusterer.print_statistics()
        
        return output_path
    
    def run_stage3(self, input_path: str) -> str:
        """Run Stage 3: LLM validation."""
        print("\n" + "="*70)
        print("STAGE 3: LLM VALIDATION")
        print("="*70)
        
        if not self.llm_client:
            print("⚠ Skipping Stage 3 - LLM client not available")
            return input_path
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        theories = data['theories']
        families = data['families']
        parents = data['parents']
        children = data['children']
        
        validator = LLMValidator(self.llm_client)
        
        # Validate only a sample for prototype
        sample_size = min(10, len(children))
        print(f"   Validating sample of {sample_size} children (prototype mode)")
        
        children_sample = children[:sample_size]
        children_sample = validator.validate_and_name_clusters(children_sample, theories, 'child')
        
        # Update original list
        for i, validated in enumerate(children_sample):
            children[i] = validated
        
        output_path = f'{self.output_dir}/stage3_validated.json'
        validator.save_results(families, parents, children, theories, output_path)
        
        return output_path
    
    def run_stage4(self, input_path: str) -> str:
        """Run Stage 4: Ontology matching."""
        print("\n" + "="*70)
        print("STAGE 4: ONTOLOGY MATCHING")
        print("="*70)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        theories = data['theories']
        families = data['families']
        parents = data['parents']
        children = data['children']
        
        matcher = OntologyMatcher('ontology/initial_ontology.json', llm_client=self.llm_client)
        children = matcher.match_theories(children, theories)
        
        output_path = f'{self.output_dir}/stage4_final.json'
        matcher.save_results(families, parents, children, theories, output_path)
        
        return output_path
    
    def generate_report(self, final_path: str):
        """Generate summary report."""
        print("\n" + "="*70)
        print("PROTOTYPE SUMMARY REPORT")
        print("="*70)
        
        with open(final_path, 'r') as f:
            data = json.load(f)
        
        theories = data['theories']
        families = data['families']
        parents = data['parents']
        children = data['children']
        
        print(f"\n📊 Pipeline Results:")
        print(f"   Input theories: {len(theories)}")
        print(f"   Theory families: {len(families)}")
        print(f"   Parent theories: {len(parents)}")
        print(f"   Child theories (normalized): {len(children)}")
        print(f"   Compression ratio: {len(theories)/len(children):.1f}:1")
        
        # Ontology matching stats
        if 'statistics' in data['metadata']:
            stats = data['metadata']['statistics']
            print(f"\n📚 Ontology Matching:")
            print(f"   Exact matches: {stats.get('exact_matches', 0)}")
            print(f"   Partial matches: {stats.get('partial_matches', 0)}")
            print(f"   Novel theories: {stats.get('novel_theories', 0)}")
        
        # Sample normalized theories
        print(f"\n🔬 Sample Normalized Theories:")
        for i, child in enumerate(children[:5], 1):
            print(f"\n{i}. {child.get('canonical_name', 'Unnamed')}")
            print(f"   Original theories: {len(child['theory_ids'])}")
            print(f"   Ontology match: {child.get('ontology_match', 'None')} ({child.get('ontology_confidence', 0):.2f})")
            if child.get('alternative_names'):
                print(f"   Variants: {child['alternative_names'][:2]}")
        
        print("\n" + "="*70)
        print(f"✅ Prototype complete! Results saved to: {self.output_dir}/")
        print("="*70)
    
    def run_full_pipeline(self, input_path: str,
                         family_threshold: float = 0.7,
                         parent_threshold: float = 0.5,
                         child_threshold: float = 0.4,
                         use_local_embeddings: bool = False):
        """Run complete normalization pipeline on subset."""
        print("\n🚀 STARTING THEORY NORMALIZATION PROTOTYPE")
        print(f"   Subset size: {self.subset_size} theories")
        print(f"   Thresholds: Family={family_threshold}, Parent={parent_threshold}, Child={child_threshold}")
        print(f"   Embeddings: {'Local (sentence-transformers)' if use_local_embeddings else 'OpenAI (with local fallback)'}\n")
        
        # Create subset
        subset_path = self.create_subset(input_path)
        
        # Run pipeline stages
        stage0_output = self.run_stage0(subset_path)
        stage1_output = self.run_stage1(stage0_output, use_local=use_local_embeddings)
        stage2_output = self.run_stage2(stage1_output, family_threshold, parent_threshold, child_threshold)
        stage3_output = self.run_stage3(stage2_output)
        stage4_output = self.run_stage4(stage3_output)
        
        # Generate report
        self.generate_report(stage4_output)
        
        return stage4_output


def main():
    """Run prototype with different threshold configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run theory normalization prototype')
    parser.add_argument('--subset-size', type=int, default=200, help='Number of theories to process')
    parser.add_argument('--family-threshold', type=float, default=0.7, help='Family clustering threshold')
    parser.add_argument('--parent-threshold', type=float, default=0.5, help='Parent clustering threshold')
    parser.add_argument('--child-threshold', type=float, default=0.4, help='Child clustering threshold')
    parser.add_argument('--input', type=str, default='theories_per_paper_checkpoint.json', help='Input JSON file')
    parser.add_argument('--use-local', action='store_true', help='Use local embeddings (sentence-transformers) instead of OpenAI')
    
    args = parser.parse_args()
    
    # Run prototype
    prototype = NormalizationPrototype(subset_size=args.subset_size)
    
    prototype.run_full_pipeline(
        args.input,
        family_threshold=args.family_threshold,
        parent_threshold=args.parent_threshold,
        child_threshold=args.child_threshold,
        use_local_embeddings=args.use_local
    )


if __name__ == '__main__':
    main()
