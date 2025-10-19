#!/usr/bin/env python3
"""
Test Environment Setup for Advanced Embedding System
Verifies all required packages are installed and working.
"""

import sys
from typing import Dict, List, Tuple


def test_package(package_name: str, import_statement: str, test_code: str = None) -> Tuple[bool, str]:
    """Test if a package is installed and working."""
    try:
        exec(import_statement)
        if test_code:
            exec(test_code)
        return True, "OK"
    except ImportError as e:
        return False, f"Not installed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run environment tests."""
    print("="*70)
    print("TESTING ADVANCED EMBEDDING SYSTEM ENVIRONMENT")
    print("="*70)
    
    tests = [
        # Core PyTorch
        ("PyTorch", "import torch", "torch.__version__"),
        ("PyTorch CUDA", "import torch", "assert torch.cuda.is_available(), 'CUDA not available'"),
        ("TorchVision", "import torchvision", None),
        ("TorchAudio", "import torchaudio", None),
        
        # NLP Libraries
        ("NumPy", "import numpy", None),
        ("Pandas", "import pandas", None),
        ("Scikit-learn", "import sklearn", None),
        ("Transformers", "import transformers", None),
        ("Sentence Transformers", "from sentence_transformers import SentenceTransformer", None),
        ("KeyBERT", "from keybert import KeyBERT", None),
        ("YAKE", "import yake", None),
        ("BERTopic", "from bertopic import BERTopic", None),
        ("spaCy", "import spacy", None),
        
        # OpenAI
        ("OpenAI", "import openai", None),
        
        # Utilities
        ("Requests", "import requests", None),
        ("Python-dotenv", "import dotenv", None),
        ("tqdm", "import tqdm", None),
        ("Accelerate", "import accelerate", None),
        
        # Jupyter
        ("JupyterLab", "import jupyterlab", None),
        ("IPyKernel", "import ipykernel", None),
        ("IPyWidgets", "import ipywidgets", None),
    ]
    
    print("\n📦 Testing Package Installation:")
    print("-"*70)
    
    results = []
    max_name_len = max(len(name) for name, _, _ in tests)
    
    for name, import_stmt, test_code in tests:
        success, message = test_package(name, import_stmt, test_code)
        results.append((name, success, message))
        
        status = "✅" if success else "❌"
        print(f"{status} {name:<{max_name_len}} : {message}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    failed = total - passed
    
    print(f"\nTotal tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\n⚠️  Some packages are missing. Install with:")
        print("   pip install -r requirements_advanced.txt")
        
        # Check for critical failures
        critical = ["PyTorch", "Sentence Transformers", "Transformers", "NumPy"]
        critical_failures = [name for name, success, _ in results if not success and name in critical]
        
        if critical_failures:
            print(f"\n❌ CRITICAL: Missing required packages: {', '.join(critical_failures)}")
            return 1
    
    # Test spaCy model
    print("\n" + "="*70)
    print("TESTING SPACY MODEL")
    print("="*70)
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' is installed")
        except OSError:
            print("❌ spaCy model 'en_core_web_sm' not found")
            print("   Download with: python -m spacy download en_core_web_sm")
            return 1
    except ImportError:
        print("❌ spaCy not installed")
        return 1
    
    # Test model loading
    print("\n" + "="*70)
    print("TESTING MODEL LOADING")
    print("="*70)
    
    print("\n🔄 Testing sentence-transformers model loading...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        test_embedding = model.encode("test sentence")
        print(f"✅ Model loaded successfully (embedding dim: {len(test_embedding)})")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return 1
    
    print("\n🔄 Testing KeyBERT...")
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords("test sentence", top_n=1)
        print(f"✅ KeyBERT working (extracted {len(keywords)} keywords)")
    except Exception as e:
        print(f"❌ KeyBERT failed: {e}")
        return 1
    
    # Test advanced embedding system
    print("\n" + "="*70)
    print("TESTING ADVANCED EMBEDDING SYSTEM")
    print("="*70)
    
    print("\n🔄 Importing advanced embedding modules...")
    try:
        from src.normalization.stage1_embedding_advanced import (
            AdvancedConceptFeatureExtractor,
            MultiModelEmbeddingGenerator
        )
        print("✅ Advanced embedding modules imported successfully")
        
        print("\n🔄 Testing feature extractor...")
        extractor = AdvancedConceptFeatureExtractor()
        
        test_theory = {
            'name': 'CB1 receptor-mediated mitochondrial dysfunction theory',
            'concept_text': 'Involves CB1 receptors and mitochondrial quality control',
            'description': 'A theory about aging mechanisms'
        }
        
        features = extractor.extract_features(test_theory)
        print(f"✅ Feature extraction working")
        print(f"   - Mechanisms found: {len(features.get('mechanisms', []))}")
        print(f"   - Receptors found: {len(features.get('receptors', []))}")
        print(f"   - Specificity score: {features.get('specificity_score', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Advanced embedding system failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("   Your environment is ready for advanced embedding system.")
        print("\n🚀 Next steps:")
        print("   1. Test on sample data:")
        print("      python src/normalization/stage1_embedding_advanced.py")
        print("   2. Run prototype:")
        print("      python run_normalization_prototype.py --subset-size 50 --use-local")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Install missing packages:")
        print("   pip install -r requirements_advanced.txt")
        print("   python -m spacy download en_core_web_sm")
        return 1


if __name__ == '__main__':
    sys.exit(main())
