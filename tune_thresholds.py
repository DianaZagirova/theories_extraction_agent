"""
Threshold Tuning Script
Tests different threshold combinations to find optimal clustering parameters.
"""

import subprocess
import json
import os
from typing import List, Dict


class ThresholdTuner:
    """Tests different threshold combinations."""
    
    def __init__(self):
        self.results = []
    
    def test_configuration(self, family_t: float, parent_t: float, child_t: float, subset_size: int = 200):
        """Test a specific threshold configuration."""
        print(f"\n{'='*70}")
        print(f"Testing: Family={family_t}, Parent={parent_t}, Child={child_t}")
        print(f"{'='*70}")
        
        # Run prototype with these thresholds
        cmd = [
            'python3', 'run_normalization_prototype.py',
            '--subset-size', str(subset_size),
            '--family-threshold', str(family_t),
            '--parent-threshold', str(parent_t),
            '--child-threshold', str(child_t)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse results
            output_path = 'output/prototype/stage4_final.json'
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                stats = {
                    'family_threshold': family_t,
                    'parent_threshold': parent_t,
                    'child_threshold': child_t,
                    'input_theories': len(data['theories']),
                    'num_families': len(data['families']),
                    'num_parents': len(data['parents']),
                    'num_children': len(data['children']),
                    'compression_ratio': len(data['theories']) / len(data['children']) if data['children'] else 0,
                    'ontology_stats': data['metadata'].get('statistics', {})
                }
                
                self.results.append(stats)
                
                print(f"\n✓ Results:")
                print(f"   Families: {stats['num_families']}")
                print(f"   Parents: {stats['num_parents']}")
                print(f"   Children: {stats['num_children']}")
                print(f"   Compression: {stats['compression_ratio']:.1f}:1")
                
                return stats
            else:
                print(f"❌ Output file not found")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_grid_search(self, subset_size: int = 200):
        """Run grid search over threshold combinations."""
        print("🔍 Starting threshold grid search...\n")
        
        # Define threshold ranges
        family_thresholds = [0.6, 0.7, 0.8]
        parent_thresholds = [0.4, 0.5, 0.6]
        child_thresholds = [0.3, 0.4, 0.5]
        
        total_configs = len(family_thresholds) * len(parent_thresholds) * len(child_thresholds)
        print(f"Testing {total_configs} configurations\n")
        
        config_num = 0
        for family_t in family_thresholds:
            for parent_t in parent_thresholds:
                for child_t in child_thresholds:
                    config_num += 1
                    print(f"\n[{config_num}/{total_configs}]")
                    self.test_configuration(family_t, parent_t, child_t, subset_size)
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save tuning results."""
        output_path = 'output/threshold_tuning_results.json'
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of best configurations."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*70)
        print("THRESHOLD TUNING SUMMARY")
        print("="*70)
        
        # Find best by compression ratio (closest to target)
        target_compression = 40  # 14K theories → 350 normalized
        
        best_by_compression = min(
            self.results,
            key=lambda x: abs(x['compression_ratio'] - target_compression)
        )
        
        print(f"\n🏆 Best Configuration (closest to {target_compression}:1 compression):")
        print(f"   Family threshold: {best_by_compression['family_threshold']}")
        print(f"   Parent threshold: {best_by_compression['parent_threshold']}")
        print(f"   Child threshold: {best_by_compression['child_threshold']}")
        print(f"   Compression ratio: {best_by_compression['compression_ratio']:.1f}:1")
        print(f"   Families: {best_by_compression['num_families']}")
        print(f"   Parents: {best_by_compression['num_parents']}")
        print(f"   Children: {best_by_compression['num_children']}")
        
        # Show all results sorted by compression
        print(f"\n📊 All Configurations (sorted by compression ratio):")
        print(f"{'Family':<8} {'Parent':<8} {'Child':<8} {'Families':<10} {'Parents':<10} {'Children':<10} {'Ratio':<8}")
        print("-" * 70)
        
        for result in sorted(self.results, key=lambda x: x['compression_ratio']):
            print(f"{result['family_threshold']:<8.1f} {result['parent_threshold']:<8.1f} {result['child_threshold']:<8.1f} "
                  f"{result['num_families']:<10} {result['num_parents']:<10} {result['num_children']:<10} "
                  f"{result['compression_ratio']:<8.1f}")
        
        print("="*70)


def main():
    """Run threshold tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune clustering thresholds')
    parser.add_argument('--subset-size', type=int, default=200, help='Subset size for testing')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer configurations')
    
    args = parser.parse_args()
    
    tuner = ThresholdTuner()
    
    if args.quick:
        # Quick test with 3 configurations
        print("🚀 Quick threshold test (3 configurations)\n")
        tuner.test_configuration(0.7, 0.5, 0.4, args.subset_size)
        tuner.test_configuration(0.6, 0.4, 0.3, args.subset_size)
        tuner.test_configuration(0.8, 0.6, 0.5, args.subset_size)
        tuner.save_results()
        tuner.print_summary()
    else:
        # Full grid search
        tuner.run_grid_search(args.subset_size)


if __name__ == '__main__':
    main()
