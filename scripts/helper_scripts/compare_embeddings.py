"""
Compare Basic vs Advanced Embedding Systems
Helps evaluate which system works better for your use case.
"""

import json
import sys
from pathlib import Path


def load_embeddings(path):
    """Load embeddings from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_features(embeddings_data):
    """Analyze feature extraction quality."""
    embeddings = embeddings_data['embeddings']
    
    stats = {
        'total_theories': len(embeddings),
        'mechanisms_found': 0,
        'receptors_found': 0,
        'pathways_found': 0,
        'entities_found': 0,
        'keywords_found': 0,
        'parent_candidates': 0,
        'child_candidates': 0,
        'avg_specificity': 0.0,
        'biological_levels': {}
    }
    
    specificity_scores = []
    
    for emb in embeddings:
        features = emb.get('concept_features', {})
        hierarchical = emb.get('hierarchical_features', {})
        
        # Count features
        if features.get('mechanisms'):
            stats['mechanisms_found'] += 1
        if features.get('receptors'):
            stats['receptors_found'] += 1
        if features.get('pathways'):
            stats['pathways_found'] += 1
        if features.get('entities'):
            stats['entities_found'] += 1
        if features.get('keywords'):
            stats['keywords_found'] += 1
        
        # Hierarchical features
        if hierarchical.get('is_parent_candidate'):
            stats['parent_candidates'] += 1
        if hierarchical.get('is_child_candidate'):
            stats['child_candidates'] += 1
        
        # Specificity
        spec = features.get('specificity_score', 0.5)
        specificity_scores.append(spec)
        
        # Biological level
        level = features.get('biological_level', 'unknown')
        stats['biological_levels'][level] = stats['biological_levels'].get(level, 0) + 1
    
    if specificity_scores:
        stats['avg_specificity'] = sum(specificity_scores) / len(specificity_scores)
    
    return stats


def compare_systems(basic_path, advanced_path):
    """Compare basic and advanced embedding systems."""
    print("="*70)
    print("EMBEDDING SYSTEM COMPARISON")
    print("="*70)
    
    # Load data
    print("\n📂 Loading embeddings...")
    basic_data = load_embeddings(basic_path)
    advanced_data = load_embeddings(advanced_path)
    
    print(f"✓ Basic: {len(basic_data['embeddings'])} theories")
    print(f"✓ Advanced: {len(advanced_data['embeddings'])} theories")
    
    # Analyze
    print("\n🔍 Analyzing feature extraction...")
    basic_stats = analyze_features(basic_data)
    advanced_stats = analyze_features(advanced_data)
    
    # Print comparison
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPARISON")
    print("="*70)
    
    print(f"\n{'Feature':<30} {'Basic':<15} {'Advanced':<15} {'Improvement':<15}")
    print("-"*70)
    
    def print_metric(name, basic_val, adv_val, is_percentage=True):
        if is_percentage:
            basic_pct = (basic_val / basic_stats['total_theories'] * 100) if basic_stats['total_theories'] > 0 else 0
            adv_pct = (adv_val / advanced_stats['total_theories'] * 100) if advanced_stats['total_theories'] > 0 else 0
            improvement = adv_pct - basic_pct
            print(f"{name:<30} {basic_pct:>6.1f}% ({basic_val:>3}) {adv_pct:>6.1f}% ({adv_val:>3}) {improvement:>+6.1f}%")
        else:
            improvement = adv_val - basic_val
            print(f"{name:<30} {basic_val:>14.3f} {adv_val:>14.3f} {improvement:>+14.3f}")
    
    print_metric("Mechanisms extracted", basic_stats['mechanisms_found'], advanced_stats['mechanisms_found'])
    print_metric("Receptors extracted", basic_stats['receptors_found'], advanced_stats['receptors_found'])
    print_metric("Pathways extracted", basic_stats['pathways_found'], advanced_stats['pathways_found'])
    print_metric("Entities extracted (NER)", basic_stats['entities_found'], advanced_stats['entities_found'])
    print_metric("Keywords extracted", basic_stats['keywords_found'], advanced_stats['keywords_found'])
    print_metric("Parent candidates", basic_stats['parent_candidates'], advanced_stats['parent_candidates'])
    print_metric("Child candidates", basic_stats['child_candidates'], advanced_stats['child_candidates'])
    print_metric("Avg specificity score", basic_stats['avg_specificity'], advanced_stats['avg_specificity'], is_percentage=False)
    
    # Biological levels
    print("\n" + "="*70)
    print("BIOLOGICAL LEVEL DETECTION")
    print("="*70)
    
    all_levels = set(basic_stats['biological_levels'].keys()) | set(advanced_stats['biological_levels'].keys())
    
    print(f"\n{'Level':<20} {'Basic':<15} {'Advanced':<15}")
    print("-"*50)
    
    for level in sorted(all_levels):
        basic_count = basic_stats['biological_levels'].get(level, 0)
        adv_count = advanced_stats['biological_levels'].get(level, 0)
        basic_pct = (basic_count / basic_stats['total_theories'] * 100) if basic_stats['total_theories'] > 0 else 0
        adv_pct = (adv_count / advanced_stats['total_theories'] * 100) if advanced_stats['total_theories'] > 0 else 0
        print(f"{level:<20} {basic_pct:>6.1f}% ({basic_count:>3}) {adv_pct:>6.1f}% ({adv_count:>3})")
    
    # Sample comparison
    print("\n" + "="*70)
    print("SAMPLE FEATURE COMPARISON (First 3 Theories)")
    print("="*70)
    
    for i in range(min(3, len(basic_data['embeddings']))):
        basic_emb = basic_data['embeddings'][i]
        advanced_emb = advanced_data['embeddings'][i]
        
        theory_id = basic_emb['theory_id']
        theory = next((t for t in basic_data['theories'] if t['theory_id'] == theory_id), {})
        theory_name = theory.get('name', 'Unknown')
        
        print(f"\n{i+1}. {theory_name}")
        print("-"*70)
        
        basic_features = basic_emb.get('concept_features', {})
        adv_features = advanced_emb.get('concept_features', {})
        
        print(f"  Mechanisms:")
        print(f"    Basic:    {basic_features.get('mechanisms', [])}")
        print(f"    Advanced: {adv_features.get('mechanisms', [])}")
        
        print(f"  Entities (NER):")
        print(f"    Basic:    {basic_features.get('entities', {})}")
        print(f"    Advanced: {adv_features.get('entities', {})}")
        
        print(f"  Keywords:")
        print(f"    Basic:    {basic_features.get('keywords', [])}")
        adv_keywords = adv_features.get('keywords', [])
        if adv_keywords:
            print(f"    Advanced: {[kw[0] for kw in adv_keywords[:3]]}")  # Top 3
        else:
            print(f"    Advanced: []")
        
        print(f"  Specificity:")
        print(f"    Basic:    {basic_features.get('specificity_score', 0.5):.3f}")
        print(f"    Advanced: {adv_features.get('specificity_score', 0.5):.3f}")
        
        adv_hierarchical = advanced_emb.get('hierarchical_features', {})
        print(f"  Hierarchical:")
        print(f"    Parent candidate: {adv_hierarchical.get('is_parent_candidate', False)}")
        print(f"    Child candidate:  {adv_hierarchical.get('is_child_candidate', False)}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Calculate improvement score
    improvement_score = 0
    if advanced_stats['mechanisms_found'] > basic_stats['mechanisms_found']:
        improvement_score += 2
    if advanced_stats['entities_found'] > basic_stats['entities_found']:
        improvement_score += 3
    if advanced_stats['keywords_found'] > basic_stats['keywords_found']:
        improvement_score += 2
    if abs(advanced_stats['avg_specificity'] - 0.5) > abs(basic_stats['avg_specificity'] - 0.5):
        improvement_score += 2
    if advanced_stats['parent_candidates'] > 0 or advanced_stats['child_candidates'] > 0:
        improvement_score += 3
    
    print(f"\nImprovement Score: {improvement_score}/12")
    
    if improvement_score >= 8:
        print("\n✅ STRONG RECOMMENDATION: Use Advanced System")
        print("   - Significantly better feature extraction")
        print("   - Better hierarchical detection")
        print("   - Worth the extra computational cost")
    elif improvement_score >= 5:
        print("\n⚠️  MODERATE RECOMMENDATION: Use Advanced System")
        print("   - Noticeable improvements in feature extraction")
        print("   - Consider for production use")
        print("   - May use basic system for quick prototyping")
    else:
        print("\n⚠️  WEAK IMPROVEMENT: Consider Basic System")
        print("   - Limited improvements over basic system")
        print("   - Basic system may be sufficient")
        print("   - Use advanced only if you need specific features")
    
    print("\n" + "="*70)


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare basic vs advanced embedding systems')
    parser.add_argument('--basic', type=str, default='output/stage1_embeddings.json',
                       help='Path to basic embeddings JSON')
    parser.add_argument('--advanced', type=str, default='output/stage1_embeddings_advanced.json',
                       help='Path to advanced embeddings JSON')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.basic).exists():
        print(f"❌ Error: Basic embeddings not found at {args.basic}")
        print(f"   Run: python run_normalization_prototype.py --subset-size 50 --use-local")
        sys.exit(1)
    
    if not Path(args.advanced).exists():
        print(f"❌ Error: Advanced embeddings not found at {args.advanced}")
        print(f"   Run: python src/normalization/stage1_embedding_advanced.py")
        sys.exit(1)
    
    # Compare
    compare_systems(args.basic, args.advanced)


if __name__ == '__main__':
    main()
