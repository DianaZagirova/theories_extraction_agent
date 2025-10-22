# Installation Guide - Advanced Embedding System

## Prerequisites

- Python 3.8+
- CUDA 12.1 (for GPU acceleration)
- ~5GB disk space for models
- ~3GB RAM for model loading

## Installation Steps

### 1. Install Python Packages

```bash
# Install all requirements (based on your verified working setup)
pip install -r requirements_advanced.txt
```

**What gets installed:**
- ✅ PyTorch 2.1+ with CUDA 12.1 support (already in your env)
- ✅ sentence-transformers, KeyBERT, YAKE, BERTopic (already in your env)
- ✅ transformers 4.30-4.50 (already in your env)
- ✅ NumPy, Pandas, scikit-learn (already in your env)
- ✅ JupyterLab, IPyKernel, IPyWidgets (already in your env)
- 🆕 spaCy 3.5+ (NEW - for linguistic analysis)
- 🆕 tqdm (NEW - for progress bars)

**Time:** ~2 minutes (most packages already installed)

### 2. Download spaCy Model

```bash
# Required for linguistic analysis
python -m spacy download en_core_web_sm
```

**Time:** ~30 seconds  
**Size:** ~50MB

### 3. Verify Installation

```bash
# Run comprehensive environment test
python scripts/test_environment.py
```

**Expected output:**
```
✅ PyTorch : OK
✅ PyTorch CUDA : OK
✅ Sentence Transformers : OK
✅ KeyBERT : OK
✅ spaCy : OK
...
✅ ALL TESTS PASSED!
```

### 4. Test Advanced System

```bash
# Quick test (will download models on first run)
python -c "from src.normalization.stage1_embedding_advanced import AdvancedConceptFeatureExtractor; print('✅ Advanced system ready!')"
```

## What Gets Downloaded (First Run Only)

When you first run the advanced system, these models will be downloaded automatically:

| Model | Size | Purpose | Cache Location |
|-------|------|---------|----------------|
| all-mpnet-base-v2 | 420MB | General embeddings | `.cache/embedding_models/general_model/` |
| S-PubMedBert-MS-MARCO | 440MB | Biomedical embeddings | `.cache/embedding_models/biomedical_model/` |
| biomedical-ner-all | 500MB | Entity extraction | `.cache/nlp_models/ner_model/` |
| en_core_web_sm | 50MB | Linguistic analysis | spaCy cache |

**Total:** ~1.5GB (one-time download)  
**Time:** ~5-10 minutes on first run

## Quick Verification Commands

### Test PyTorch + CUDA
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Test Sentence Transformers
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2'); print('✅ OK')"
```

### Test KeyBERT
```bash
python -c "from keybert import KeyBERT; KeyBERT(); print('✅ OK')"
```

### Test spaCy
```bash
python -c "import spacy; spacy.load('en_core_web_sm'); print('✅ OK')"
```

### Test Advanced System
```bash
python test_environment.py
```

## Troubleshooting

### Issue: spaCy model not found

**Error:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: CUDA not available

**Error:**
```
AssertionError: CUDA not available
```

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Transformers version conflict

**Error:**
```
ERROR: transformers 4.50.0 has requirement ..., but you have transformers 4.30.0
```

**Solution:**
```bash
# Ensure transformers is in correct range
pip install "transformers>=4.30.0,<4.50.0" --force-reinstall
```

### Issue: Out of memory during model loading

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In stage1_embedding_advanced.py, reduce batch size
generator.generate_embeddings(theories, batch_size=16)  # Default: 32

# Or disable biomedical models
generator = MultiModelEmbeddingGenerator(use_biomedical=False)
```

### Issue: Models downloading slowly

**Solution:**
```bash
# Pre-download models manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')"
```

## Minimal Installation (If Issues Persist)

If you encounter persistent issues, you can install a minimal version:

```bash
# Install only core packages
pip install sentence-transformers transformers torch numpy

# Skip advanced features
# - No KeyBERT (automatic keyword extraction)
# - No spaCy (linguistic analysis)
# - No biomedical NER (entity extraction)
```

The system will gracefully degrade and still work with reduced features.

## Verification Checklist

- [ ] PyTorch installed with CUDA support
- [ ] sentence-transformers working
- [ ] KeyBERT working
- [ ] spaCy installed
- [ ] spaCy model 'en_core_web_sm' downloaded
- [ ] `test_environment.py` passes all tests
- [ ] Advanced system imports successfully

## Post-Installation

### Test on Sample Data

```bash
# Create sample theories file (if needed)
# Then run:
python src/normalization/stage1_embedding_advanced.py
```

### Run Full Pipeline

```bash
# Test on 50 theories
python run_normalization_prototype.py --subset-size 50 --use-local
```

### Compare Basic vs Advanced

```bash
# After running both systems
python compare_embeddings.py
```

## Environment Summary

After installation, your environment will have:

✅ **PyTorch 2.1+** with CUDA 12.1  
✅ **sentence-transformers 2.2+** for embeddings  
✅ **KeyBERT 0.8+** for keyword extraction  
✅ **spaCy 3.5+** for linguistic analysis  
✅ **transformers 4.30-4.50** for NER  
✅ **All verified working packages** from your base setup  

**Total disk usage:** ~3GB (packages + models)  
**RAM usage:** ~3GB (when all models loaded)  

## Next Steps

1. ✅ Verify installation: `python test_environment.py`
2. 🧪 Test advanced system: `python src/normalization/stage1_embedding_advanced.py`
3. 🚀 Run prototype: `python run_normalization_prototype.py --subset-size 50 --use-local`
4. 📊 Compare results: `python compare_embeddings.py`

## Support

If you encounter issues:

1. Check `test_environment.py` output for specific failures
2. Review error messages carefully
3. Ensure CUDA 12.1 is properly installed
4. Verify all packages in `requirements_advanced.txt` are installed

**Your environment is based on a verified working setup, so installation should be smooth!** ✅
