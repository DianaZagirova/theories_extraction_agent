"""
Test script to diagnose why doubted papers are not appearing in theories.db.
Samples 5 doubted papers and checks content availability at each stage.
"""
import sqlite3
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.text_preprocessor import TextPreprocessor

EVAL_DB = '/home/diana.z/hack/llm_judge/data/evaluations.db'
PAPERS_DB = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'

def test_doubted_papers():
    """Test 5 doubted papers to see why they're skipped."""
    
    print("="*70)
    print("TESTING DOUBTED PAPERS PROCESSING")
    print("="*70)
    
    # 1. Get sample of doubted papers
    print("\n1. Fetching 5 doubted papers from evaluations.db...")
    eval_conn = sqlite3.connect(EVAL_DB)
    eval_cursor = eval_conn.cursor()
    
    eval_cursor.execute("""
        SELECT doi, pmid, title, result, confidence_score
        FROM paper_evaluations
        WHERE result = 'doubted'
          AND doi IS NOT NULL 
          AND doi != ''
        LIMIT 5
    """)
    
    doubted_papers = eval_cursor.fetchall()
    eval_conn.close()
    
    if not doubted_papers:
        print("❌ No doubted papers found!")
        return
    
    print(f"✓ Found {len(doubted_papers)} doubted papers\n")
    
    # 2. Check each paper's content in papers.db
    papers_conn = sqlite3.connect(PAPERS_DB)
    papers_cursor = papers_conn.cursor()
    preprocessor = TextPreprocessor()
    
    for i, (doi, pmid, title, result, conf_score) in enumerate(doubted_papers, 1):
        print(f"\n{'='*70}")
        print(f"PAPER {i}/5")
        print(f"{'='*70}")
        print(f"DOI: {doi}")
        print(f"PMID: {pmid}")
        print(f"Title: {title[:80]}...")
        print(f"Result: {result}")
        print(f"Confidence: {conf_score}")
        
        # Fetch from papers.db
        papers_cursor.execute("""
            SELECT doi, pmid, title, abstract, full_text, full_text_sections
            FROM papers
            WHERE doi = ?
        """, (doi,))
        
        paper_data = papers_cursor.fetchone()
        
        if not paper_data:
            print("\n❌ SKIP REASON: Paper not found in papers.db")
            continue
        
        _, pmid_db, title_db, abstract, full_text, full_text_sections = paper_data
        
        # Check content availability
        print("\n--- Content Check ---")
        
        # Abstract
        has_abstract = bool(abstract and str(abstract).strip())
        print(f"Abstract: {'✓ Present' if has_abstract else '❌ Missing'}")
        if has_abstract:
            print(f"  Length: {len(str(abstract))} chars")
        
        # Full text
        has_full_text = bool(full_text and str(full_text).strip())
        print(f"Full text: {'✓ Present' if has_full_text else '❌ Missing'}")
        if has_full_text:
            print(f"  Length: {len(str(full_text))} chars")
        
        # Full text sections
        has_sections = False
        if full_text_sections and str(full_text_sections).strip():
            try:
                sections_obj = json.loads(full_text_sections) if isinstance(full_text_sections, str) else full_text_sections
                if isinstance(sections_obj, (list, dict)) and len(sections_obj) > 0:
                    has_sections = True
                    print(f"Full text sections: ✓ Present")
                    if isinstance(sections_obj, list):
                        print(f"  Sections count: {len(sections_obj)}")
                    elif isinstance(sections_obj, dict):
                        print(f"  Section keys: {list(sections_obj.keys())[:5]}")
                else:
                    print(f"Full text sections: ❌ Empty structure")
            except Exception as e:
                print(f"Full text sections: ⚠️  Parse error: {e}")
                has_sections = True  # Code treats parse errors as "has sections"
        else:
            print(f"Full text sections: ❌ Missing")
        
        # Check if would pass content gate
        print("\n--- Pipeline Gate Check ---")
        passes_gate = has_full_text or has_sections
        print(f"Passes content gate (full_text OR sections): {'✓ YES' if passes_gate else '❌ NO'}")
        
        if not passes_gate:
            print("\n❌ SKIP REASON: No full_text and no full_text_sections")
            print("   → Paper would be skipped before DB upsert")
            continue
        
        # Try preprocessing
        print("\n--- Preprocessing Test ---")
        try:
            processed_text = preprocessor.preprocess(full_text, full_text_sections, abstract)
            if processed_text:
                print(f"✓ Preprocessing successful")
                print(f"  Processed length: {len(processed_text)} chars")
                print(f"  First 200 chars: {processed_text[:200]}...")
                print("\n✓ WOULD BE PROCESSED: Paper should appear in theories.db")
            else:
                print("❌ SKIP REASON: Preprocessing returned empty text")
                print("   → Paper would be skipped before DB upsert")
        except Exception as e:
            print(f"❌ SKIP REASON: Preprocessing error: {e}")
            print("   → Paper would be skipped before DB upsert")
    
    papers_conn.close()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCommon skip reasons:")
    print("1. Paper not in papers.db")
    print("2. Missing both full_text AND full_text_sections")
    print("3. Preprocessing returns empty/fails")
    print("\nTo include these papers in theories.db:")
    print("- Option A: Upsert paper_results before content checks")
    print("- Option B: Allow abstract-only processing as fallback")
    print("- Option C: Fetch missing full texts for doubted papers")

if __name__ == "__main__":
    test_doubted_papers()
