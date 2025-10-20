"""
Extract aging theories from validated papers with strict criteria.

Pipeline:
1. Load validated papers (valid/doubted/not_valid ≤7) from evaluations.db
2. For each paper, retrieve full text from papers.db
3. Use multi-stage LLM analysis:
   - Stage 1: Extract theories with detailed criteria checking
   - Stage 2: Validate extracted theories against criteria
4. Output structured theory data per paper

Techniques:
- Full text analysis (not just chunks) for completeness
- Chain-of-thought prompting for better reasoning
- Structured output with validation
- Batch processing with progress tracking
"""
import sys
import os
from pathlib import Path
import sqlite3
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
from datetime import datetime
import time
import asyncio
from collections import deque

from dotenv import load_dotenv
load_dotenv()

# Environment-configured limits (cast to proper types)
try:
    BACK_LIMIT = int(os.getenv('BACK_LIMIT', '20000'))
except Exception:
    BACK_LIMIT = 20000
try:
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))
except Exception:
    MAX_TOKENS = 4000
try:
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
except Exception:
    TEMPERATURE = 0.2

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_integration import OpenAIClient
from src.core.text_preprocessor import TextPreprocessor
from prompts.theory import THEORY_CRITERIA


class RateLimiter:
    """Rate limiter for API calls respecting TPM and RPM limits."""
    
    def __init__(self, max_tokens_per_minute: int, max_requests_per_minute: int, tokens_per_day: int):
        self.max_tpm = max_tokens_per_minute
        self.max_rpm = max_requests_per_minute
        self.max_tpd = tokens_per_day
        
        self.token_timestamps = deque()
        self.request_timestamps = deque()
        self.daily_tokens = 0
        self.day_start = time.time()
        import threading
        self._lock = threading.Lock()
    
    async def acquire(self, estimated_tokens: int):
        """Wait until we can make a request within rate limits."""
        # Non-blocking check - just track usage, don't wait
        # The batch size controls concurrency naturally
        with self._lock:
            now = time.time()
            
            # Reset daily counter if new day
            if now - self.day_start > 86400:
                self.daily_tokens = 0
                self.day_start = now
            
            # Remove old timestamps (older than 1 minute)
            cutoff = now - 60
            while self.token_timestamps and self.token_timestamps[0][0] < cutoff:
                self.token_timestamps.popleft()
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.popleft()
            
            # Record this request
            self.token_timestamps.append((now, estimated_tokens))
            self.request_timestamps.append(now)
            self.daily_tokens += estimated_tokens
            
            # Calculate current usage for monitoring
            current_tokens = sum(t[1] for t in self.token_timestamps)
            current_requests = len(self.request_timestamps)
        
        # Only wait if we're way over limits (safety check)
        if current_tokens > self.max_tpm * 1.5 or current_requests > self.max_rpm * 1.5:
            await asyncio.sleep(2)  # Brief pause if severely over limit


@dataclass
class AgingTheory:
    """Structured aging theory data."""
    name: str
    key_concepts: List[Dict[str, str]]  # [{"concept": "...", "description": "..."}]
    confidence_is_theory: str  # "high", "medium", "low"
    mode: str  # "propose", "discuss", "critique", "review", "test", "formalize"
    evidence: str
    criteria_reasoning: str
    paper_focus: int


@dataclass
class PaperTheoryExtraction:
    """Results for one paper."""
    doi: str
    pmid: Optional[str]
    title: str
    validation_result: str  # from evaluations.db
    confidence_score: int
    contains_theory: bool
    theories: List[AgingTheory]
    timestamp: str


class TheoryExtractionPipeline:
    """Pipeline for extracting theories from validated papers."""
    
    # Theory criteria as a constant for prompts
    
    
    def __init__(self):
        if os.getenv('MODE') == "full_texts":
            api_key = os.getenv('OPENAI_API_KEY2')
        else:
            api_key = os.getenv('OPENAI_API_KEY')



        self.llm = OpenAIClient(api_key=api_key)
        self.preprocessor = TextPreprocessor()
        self.THEORY_CRITERIA = THEORY_CRITERIA
        import threading
        self._lock = threading.Lock()
        # token/cost accumulators
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        
        # Rate limiting for gpt-4.1-mini: 200K TPM, 500 RPM
        self.rate_limiter = RateLimiter(
            max_tokens_per_minute=180000,  # 90% of 200K for safety
            max_requests_per_minute=450,    # 90% of 500 for safety
            tokens_per_day=1800000          # 90% of 2M for safety
        )
        
        # System prompt for LLM
        self.system_prompt = """You are a biologist with expertise in aging and senescence."""
    
    def _init_results_db(self, results_db: str):
        """Initialize results database tables if not exist."""
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        # Speed up writes for bulk processing
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA mmap_size=30000000000;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_results (
                doi TEXT PRIMARY KEY,
                pmid TEXT,
                title TEXT,
                validation_result TEXT,
                confidence_score INTEGER,
                timestamp TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS theories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                name TEXT,
                mode TEXT,
                evidence TEXT,
                confidence_is_theory TEXT,
                criteria_reasoning TEXT,
                paper_focus INTEGER,
                key_concepts TEXT,
                UNIQUE(doi, name)
            )
            """
        )
        # Table for papers that have full text but become invalid after preprocessing
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS invalid_after_preprocessing (
                doi TEXT PRIMARY KEY,
                pmid TEXT,
                title TEXT,
                had_full_text BOOLEAN,
                had_sections BOOLEAN,
                preprocessing_issue TEXT,
                timestamp TEXT
            )
            """
        )
        # Migration: ensure paper_focus column exists
        cur.execute("PRAGMA table_info(theories);")
        cols = [row[1] for row in cur.fetchall()]
        if 'paper_focus' not in cols:
            cur.execute("ALTER TABLE theories ADD COLUMN paper_focus INTEGER;")
        conn.commit()
        conn.close()
    
    def _upsert_paper_result_db(self, results_db: str, result: 'PaperTheoryExtraction'):
        """Upsert paper and theories into results DB by DOI."""
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO paper_results (doi, pmid, title, validation_result, confidence_score,  timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO UPDATE SET
                pmid=excluded.pmid,
                title=excluded.title,
                validation_result=excluded.validation_result,
                confidence_score=excluded.confidence_score,
                timestamp=excluded.timestamp
            """,
            (
                result.doi,
                result.pmid,
                result.title,
                result.validation_result,
                result.confidence_score,
                result.timestamp,
            ),
        )
        # Replace theories for DOI
        cur.execute("DELETE FROM theories WHERE doi = ?", (result.doi,))
        for t in result.theories:
            cur.execute(
                """
                INSERT OR IGNORE INTO theories (
                    doi, name, mode, evidence, confidence_is_theory, paper_focus, criteria_reasoning, key_concepts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.doi,
                    t.name,
                    t.mode,
                    t.evidence,
                    t.confidence_is_theory,
                    t.paper_focus,
                    t.criteria_reasoning,
                    json.dumps(t.key_concepts),
                ),
            )
        conn.commit()
        conn.close()
    
    def _record_invalid_preprocessing(self, results_db: str, doi: str, pmid: Optional[str], 
                                      title: str, had_full_text: bool, had_sections: bool, 
                                      issue: str):
        """Record papers that had full text but became invalid after preprocessing."""
        if not results_db:
            return
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO invalid_after_preprocessing 
            (doi, pmid, title, had_full_text, had_sections, preprocessing_issue, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doi, pmid, title, had_full_text, had_sections, issue, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def get_validated_papers(
        self,
        evaluations_db: str,
        papers_db: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get validated papers with full text.
        
        Criteria: valid OR doubted OR (not_valid AND confidence_score <= 7)
        """
        print("\n" + "="*70)
        print("LOADING VALIDATED PAPERS")
        print("="*70)
        
        # Get validated DOIs from evaluations.db
        eval_conn = sqlite3.connect(evaluations_db)
        eval_cursor = eval_conn.cursor()
        
        query = """
            SELECT doi, pmid, title, result, confidence_score
            FROM paper_evaluations
            WHERE result = 'valid'
               OR result = 'doubted'
               OR (result = 'not_valid' AND confidence_score <= 7)
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        eval_cursor.execute(query)
        validated_papers = eval_cursor.fetchall()
        eval_conn.close()
        
        print(f"\n✓ Found {len(validated_papers)} validated papers")
        
        # Get full text from papers.db
        papers_conn = sqlite3.connect(papers_db)
        papers_cursor = papers_conn.cursor()
        
        papers_with_text = []
        
        print("\n📖 Loading full text...")
        for doi, pmid, title, result, conf_score in tqdm(validated_papers):
            papers_cursor.execute(
                """
                SELECT doi, pmid, title, abstract, full_text, full_text_sections
                FROM papers
                WHERE doi = ?
                """,
                (doi,),
            )
            paper_data = papers_cursor.fetchone()
            if paper_data:
                _, pmid_db, title_db, abstract, full_text, full_text_sections = paper_data
                has_full_text = bool((full_text and str(full_text).strip()))
                has_sections = False
                if full_text_sections and str(full_text_sections).strip():
                    try:
                        sections_obj = json.loads(full_text_sections) if isinstance(full_text_sections, str) else full_text_sections
                        if isinstance(sections_obj, (list, dict)) and len(sections_obj) > 0:
                            has_sections = True
                    except Exception:
                        has_sections = True
                if not (has_full_text or has_sections):
                    continue
                processed_text = self.preprocessor.preprocess(full_text, full_text_sections, abstract)
                if processed_text:
                    papers_with_text.append({
                        'doi': doi,
                        'pmid': pmid or pmid_db,
                        'title': title or title_db,
                        'abstract': abstract,
                        'full_text': processed_text,
                        'validation_result': result,
                        'confidence_score': conf_score,
                    })
        papers_conn.close()
        print(f"✓ Loaded {len(papers_with_text)} papers with full text")
        return papers_with_text
    
    def get_validated_papers_meta(
        self,
        evaluations_db: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Return list of validated paper metadata rows (small memory footprint)."""
        eval_conn = sqlite3.connect(evaluations_db)
        eval_cursor = eval_conn.cursor()
        query = """
            SELECT doi, pmid, title, result, confidence_score
            FROM paper_evaluations
            WHERE result = 'valid'
               OR result = 'doubted'
               OR (result = 'not_valid' AND confidence_score <= 7)
        """
        if limit:
            query += f" LIMIT {limit}"
        eval_cursor.execute(query)
        rows = eval_cursor.fetchall()
        eval_conn.close()
        return rows
    
       
    async def extract_theories_stage(self, paper: Dict, max_retries: int = 3) -> Dict:
        """
        Extract theories with detailed analysis.
        Uses full text for comprehensive extraction.
        """
        full_text = paper['full_text'][:BACK_LIMIT]
        
        prompt = f"""Analyze the article and make extraction.

# TASK (Do not start with extraction, think step by step):
A. First, analyze if paper is aging theory-related and if it contains aging theories that met criteria:
{self.THEORY_CRITERIA}   
B. Extract aging theory(ies) from the paper, if any. Prioritize the extraction of main theories of the paper. To count a theory - it should be described in detail/preferably to be the main focus of the paper, not just cited/briefly mentioned, etc. If the review focus on a lot of theries, extract the most mentioned theories (up to 15 main theories, try to prioritize the most novel/not trivial theories). Make sure the theory if not too narrow.

For EACH identified theory, extract:
1. Common or proposed name. Examples: antagonistic pleiotropy theory, evolvable soma theory, etc. Try to avoid too general names and abbreviations.
2. Key Concepts: List of main concepts with detailed descriptions
- Each concept should be a core idea/mechanism of the theory
- Provide 2-3 sentence description for each
3. Mode: How paper engages with theory
- propose new theory, discuss, review, critique, test, formalize, synthesize, other (specify) (separate by comma)
4. Evidence: evidence presented (experiments, data, citations)    
5. Criteria Reasoning: does/doesn't meet criteria of aging theory
6. Confidence it's a theory: high/medium/low
7. Paper Focus. How much of the paper content is dedicated to this theory? If several theories are extracted, try to set one theory as a primary (9-10), while others should be ranked from 8-2.

If paper does not contain valid theories, retain "theories": []

#PAPER
Title: {paper['title']}
Abstract: {paper['abstract']}
Full Text (excerpt):
{full_text}

#EXAMPLE (shorten)
Input: "The wrinkling-skin theory: Aging is the loss of skin elasticity. Abstract: Wrinkling is thought to be a complex biophysical process resulting from repeated strains on a progressively, structurally and biochemistry altered aging skin..."  
Output
{{"theories": []}}

#OUTPUT FORMAT, output a valid JSON:
{{
"theories": [
    {{
    "name": "Theory name",
    "key_concepts": [
        {{"concept": "Concept name", "description": "Detailed description"}},
        {{"concept": "Another concept", "description": "Description"}}
    ],
    "mode": "propose/discuss/review/critique/test/formalize/synthesize/other(specify)",
    "evidence": "Evidence presented",
    "criteria_reasoning": "Explanation",
    "confidence_is_theory": "high/medium/low",
    "focus": "from 0 to 10, where 10 - the focus on the paper is solely on this specific theory, 2 - brief mention."
    }}
]}}

Return ONLY valid JSON:"""
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Estimate tokens for rate limiting (rough: 4 chars = 1 token)
                # estimated_tokens = (len(self.system_prompt) + len(prompt)) // 4 + MAX_TOKENS
                
                # Async rate limiting - DISABLED FOR TESTING
                # await self.rate_limiter.acquire(estimated_tokens)
                pass  # Rate limiting disabled
                
                # DEBUG: Time the API call
                api_start = time.time()
                
                if "gpt-5" in self.llm.model:
                    response = self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=MAX_TOKENS
                    )
                else:
                    response = self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )

                # DEBUG: Log API call time
                api_time = time.time() - api_start
                if api_time > 10:
                    print(f"\n⏱️  Slow API call for {paper.get('doi', 'unknown')}: {api_time:.1f}s")
                
                # Track usage for cost calculation
                try:
                    usage = response.usage
                    with self._lock:
                        self._total_prompt_tokens += usage.prompt_tokens
                        self._total_completion_tokens += usage.completion_tokens
                except Exception:
                    pass
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError as e:
                last_error = e
                time.sleep(min(2 ** attempt, 10))
            except Exception as e:
                last_error = e
                delay = min(2 ** attempt, 10)
                message = str(e).lower()
                if '429' in message or 'rate limit' in message:
                    delay = max(delay, 5)
                    print(f"⚠️  Rate limit encountered for {paper['doi']}. Retrying in {delay} seconds...")
                time.sleep(delay)
        print(f"⚠️  Extraction error for {paper['doi']}: {last_error}")
        return {
            "theories": []
        }
    
    async def process_paper(self, paper: Dict) -> PaperTheoryExtraction:
        """Process one paper through the pipeline."""
        
        # Stage 2: Extract theories
        extraction_result = await self.extract_theories_stage(paper)
        
        # Convert to AgingTheory objects
        theories = []
        for theory_data in extraction_result.get('theories', []):
            # parse paper_focus robustly
            focus_val = theory_data.get('paper_focus', theory_data.get('focus', 0))
            if isinstance(focus_val, str):
                import re
                m = re.search(r"(\d{1,2})", focus_val)
                focus_num = int(m.group(1)) if m else 0
            elif isinstance(focus_val, (int, float)):
                focus_num = int(focus_val)
            else:
                focus_num = 0
            focus_num = max(0, min(10, focus_num))
            theory = AgingTheory(
                name=theory_data.get('name', ''),
                key_concepts=theory_data.get('key_concepts', []),
                confidence_is_theory=theory_data.get('confidence_is_theory', 'low'),
                mode=theory_data.get('mode', 'other'),
                evidence=theory_data.get('evidence', ''),
                criteria_reasoning=theory_data.get('criteria_reasoning', ''),
                paper_focus=focus_num
            )
            theories.append(theory)
        
        return PaperTheoryExtraction(
            doi=paper['doi'],
            pmid=paper['pmid'],
            title=paper['title'],
            validation_result=paper['validation_result'],
            confidence_score=paper['confidence_score'],
            contains_theory=bool(theories),
            theories=theories,
            timestamp=datetime.now().isoformat()
        )
    
    def run_pipeline(
        self,
        evaluations_db: str,
        papers_db: str,
        output_file: str,
        results_db: Optional[str] = None,
        limit: Optional[int] = None,
        resume_from: Optional[str] = None,
        resume_from_db: bool = False,
        max_workers: int = 1,
        price_input_per_1M: float = 0.4,
        price_output_per_1M: float = 1.6,
        test: bool = False,
        test_dois_file: Optional[str] = None
    ):
        """Run complete extraction pipeline."""
        
        print("\n" + "="*70)
        print("AGING THEORY EXTRACTION PIPELINE")
        print("="*70)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Determine output and behavior for test mode
        if test:
            # Force a test output filename and disable DB writes
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"test_{timestamp_str}.json"
            results_db = None
            # In test mode, we still load metadata but we skip resume logic and DB init
        
        # Load paper metadata (not full texts) to keep memory low
        # In test mode with a DOI list provided, load only those DOIs (bypass validation filter)
        if test and test_dois_file and Path(test_dois_file).exists():
            with open(test_dois_file, 'r') as f:
                dois = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            dois = list(dict.fromkeys(dois))
            meta_rows = []
            if dois:
                # Try to fetch from evaluations_db first
                eval_conn = sqlite3.connect(evaluations_db)
                eval_cur = eval_conn.cursor()
                placeholders = ','.join(['?'] * len(dois))
                eval_cur.execute(
                    f"""
                    SELECT doi, pmid, title, result, confidence_score
                    FROM paper_evaluations
                    WHERE doi IN ({placeholders})
                    """,
                    tuple(dois)
                )
                found = eval_cur.fetchall()
                eval_conn.close()
                found_dois = {row[0] for row in found}
                meta_rows.extend(found)
                missing = [d for d in dois if d not in found_dois]
                if missing:
                    # Fallback to papers.db to obtain pmid/title, set default result/score
                    try:
                        pconn = sqlite3.connect(f"file:{papers_db}?mode=ro", uri=True, timeout=30)
                    except Exception:
                        pconn = sqlite3.connect(papers_db, timeout=30)
                    pcur = pconn.cursor()
                    for d in missing:
                        pcur.execute(
                            """
                            SELECT doi, pmid, title FROM papers WHERE doi = ?
                            """,
                            (d,)
                        )
                        row = pcur.fetchone()
                        if row:
                            _, pmid, title = row
                            meta_rows.append((d, pmid, title, 'unknown', 0))
                        else:
                            meta_rows.append((d, None, '', 'unknown', 0))
                    pconn.close()
        else:
            meta_rows = self.get_validated_papers_meta(evaluations_db, limit)
        print(f"\n✓ Found {len(meta_rows)} validated papers")
        
        if not meta_rows:
            print("\n❌ No papers to process")
            return
        
        # Resume from checkpoint if specified (only when not test)
        if not test:
            processed_dois = set()
            if resume_from and Path(resume_from).exists():
                print(f"\n📂 Resuming from: {resume_from}")
                with open(resume_from, 'r') as f:
                    existing_data = json.load(f)
                    processed_dois = {r['doi'] for r in existing_data['results']}
                print(f"   Already processed: {len(processed_dois)} papers")
                meta_rows = [row for row in meta_rows if row[0] not in processed_dois]
            
            # Resume from DB if requested
            if resume_from_db and results_db:
                conn = sqlite3.connect(results_db)
                cur = conn.cursor()
                # Check if paper_results table exists
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_results'")
                if cur.fetchone():
                    cur.execute("SELECT doi FROM paper_results")
                    db_dois = {row[0] for row in cur.fetchall()}
                else:
                    db_dois = set()
                conn.close()
                if db_dois:
                    print(f"   Skipping {len(db_dois)} already in DB")
                    meta_rows = [row for row in meta_rows if row[0] not in db_dois]
        
        print(f"\n🔄 Processing {len(meta_rows)} papers...")
        
        # Init results DB if provided and not in test mode
        if results_db and not test:
            self._init_results_db(results_db)
        
        # PERFORMANCE OPTIMIZATION: Pre-load all paper data into memory
        print("\n📦 Pre-loading paper data into memory...")
        papers_cache = {}
        preprocessed_cache = {}
        
        try:
            pconn = sqlite3.connect(f"file:{papers_db}?mode=ro", uri=True, timeout=30)
        except Exception:
            pconn = sqlite3.connect(papers_db, timeout=30)
        pcur = pconn.cursor()
        
        # Fetch all papers we need in one query
        dois_to_fetch = [row[0] for row in meta_rows]
        if dois_to_fetch:
            placeholders = ','.join(['?'] * len(dois_to_fetch))
            pcur.execute(
                f"""
                SELECT doi, pmid, title, abstract, full_text, full_text_sections
                FROM papers
                WHERE doi IN ({placeholders})
                """,
                dois_to_fetch
            )
            
            print("⚙️  Pre-processing texts...")
            invalid_after_preprocessing = []
            for row in tqdm(pcur.fetchall(), desc="Loading & preprocessing"):
                doi = row[0]
                papers_cache[doi] = row
                
                # Pre-process text now instead of during LLM call
                _, pmid_db, title_db, abstract, full_text, full_text_sections = row
                has_full_text = bool((full_text and str(full_text).strip()))
                has_sections = False
                if full_text_sections and str(full_text_sections).strip():
                    try:
                        sections_obj = json.loads(full_text_sections) if isinstance(full_text_sections, str) else full_text_sections
                        if isinstance(sections_obj, (list, dict)) and len(sections_obj) > 0:
                            has_sections = True
                    except Exception:
                        has_sections = True
                
                if has_full_text or has_sections:
                    processed_text = self.preprocessor.preprocess(full_text, full_text_sections, abstract)
                    if processed_text:
                        preprocessed_cache[doi] = processed_text
                    else:
                        # Track papers that had full text but became invalid after preprocessing
                        issue = "Text became empty or too short after preprocessing (< 100 chars)"
                        invalid_after_preprocessing.append({
                            'doi': doi,
                            'pmid': pmid_db,
                            'title': title_db,
                            'had_full_text': has_full_text,
                            'had_sections': has_sections,
                            'issue': issue
                        })
        
        pconn.close()
        print(f"✓ Loaded and preprocessed {len(preprocessed_cache)} papers")
        
        # Initialize stats early
        stats = {
            'total_papers': 0,  # Will be updated after filtering
            'papers_with_theories': 0,
            'papers_without_theories': 0,
            'total_theories_extracted': 0,
            'papers_invalid_after_preprocessing': len(invalid_after_preprocessing) if invalid_after_preprocessing else 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0
        }
        
        # Record invalid papers to DB if not in test mode
        if invalid_after_preprocessing:
            print(f"⚠️  Found {len(invalid_after_preprocessing)} papers with full text that became invalid after preprocessing")
            
            if results_db and not test:
                print("📝 Recording invalid papers to database...")
                for invalid_paper in invalid_after_preprocessing:
                    self._record_invalid_preprocessing(
                        results_db,
                        invalid_paper['doi'],
                        invalid_paper['pmid'],
                        invalid_paper['title'],
                        invalid_paper['had_full_text'],
                        invalid_paper['had_sections'],
                        invalid_paper['issue']
                    )
                print(f"✓ Recorded {len(invalid_after_preprocessing)} invalid papers to 'invalid_after_preprocessing' table")
        
        # Filter meta_rows to only include papers we successfully loaded
        meta_rows = [row for row in meta_rows if row[0] in preprocessed_cache]
        if not meta_rows:
            print("\n❌ No papers with valid full text found")
            return
        
        # Update total_papers count after filtering
        stats['total_papers'] = len(meta_rows)
            
        # Process papers
        results = []
        
        # Simple sequential processing (async removed for stability)
        def _process_one_paper(row):
            doi, pmid, title, result, conf_score = row
            
            try:
                # OPTIMIZED: Use pre-loaded and pre-processed data
                paper_data = papers_cache.get(doi)
                if not paper_data:
                    raise ValueError("Paper not found in cache")
                
                processed_text = preprocessed_cache.get(doi)
                if not processed_text:
                    raise ValueError("No preprocessed text available")
                
                _, pmid_db, title_db, abstract, _, _ = paper_data
                
                paper = {
                    'doi': doi,
                    'pmid': pmid or pmid_db,
                    'title': title or title_db,
                    'abstract': abstract,
                    'full_text': processed_text,
                    'validation_result': result,
                    'confidence_score': conf_score,
                }
                
                # Create event loop for this paper's async call
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(self.process_paper(paper))
            except Exception as e:
                print(f"\n⚠️  Error processing {doi}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Simple sequential processing with progress bar
        progress = tqdm(total=len(meta_rows), desc="Extracting theories")
        processed_count = 0
        
        for row in meta_rows:
            result = _process_one_paper(row)
            
            if result:
                results.append(asdict(result))
                if results_db and not test:
                    with self._lock:
                        self._upsert_paper_result_db(results_db, result)
                if result.contains_theory and result.theories:
                    stats['papers_with_theories'] += 1
                    stats['total_theories_extracted'] += len(result.theories)
                else:
                    stats['papers_without_theories'] += 1
            
            progress.update(1)
            processed_count += 1
            
            # Checkpoint every 300 papers
            if processed_count % 300 == 0:
                self._save_checkpoint(results, stats, output_file)
        
        progress.close()
        
        # Finalize token usage and estimated cost
        stats['prompt_tokens'] = self._total_prompt_tokens
        stats['completion_tokens'] = self._total_completion_tokens
        stats['total_tokens'] = self._total_prompt_tokens + self._total_completion_tokens
        stats['estimated_cost_usd'] = (
            (self._total_prompt_tokens / 1_000_000.0) * price_input_per_1M +
            (self._total_completion_tokens / 1_000_000.0) * price_output_per_1M
        )

        # Final save
        self._save_results(results, stats, output_file)
        
        # Print summary
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"\n📊 Statistics:")
        print(f"   Papers processed: {stats['total_papers']}")
        print(f"   Papers with theories: {stats['papers_with_theories']}")
        print(f"   Papers without theories: {stats['papers_without_theories']}")
        print(f"   Papers invalid after preprocessing: {stats['papers_invalid_after_preprocessing']}")
        print(f"   Total theories extracted: {stats['total_theories_extracted']}")
        print(f"   Prompt tokens: {stats['prompt_tokens']}")
        print(f"   Completion tokens: {stats['completion_tokens']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Estimated cost (USD): {stats['estimated_cost_usd']:.2f}")
        
        if stats['papers_with_theories'] > 0:
            avg_theories = stats['total_theories_extracted'] / stats['papers_with_theories']
            print(f"   Avg theories per paper: {avg_theories:.1f}")
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
    
    def _save_checkpoint(self, results: List[Dict], stats: Dict, output_file: str):
        """Save checkpoint during processing."""
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        self._save_results(results, stats, checkpoint_file)
    
    def _save_results(self, results: List[Dict], stats: Dict, output_file: str):
        """Save results to JSON file."""
        output_data = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'statistics': stats
            },
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract aging theories from validated papers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--evaluations-db',
        type=str,
        default='/home/diana.z/hack/llm_judge/data/evaluations.db',
        help='Path to evaluations.db'
    )
    parser.add_argument(
        '--papers-db',
        type=str,
        default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db',
        help='Path to papers.db'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='theories_per_paper_from_script.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--results-db',
        type=str,
        default='theories.db',
        help='Path to results SQLite DB (stores results keyed by DOI)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode: no DB writes; output to test_<timestamp>.json'
    )
    parser.add_argument(
        '--test-dois-file',
        type=str,
        help='Path to a text file with one DOI per line (used only with --test)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers (for testing)'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Resume from checkpoint file'
    )
    parser.add_argument(
        '--resume-from-db',
        action='store_true',
        help='Skip DOIs already present in results DB'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers for extraction (use with caution re: API limits)'
    )
    parser.add_argument(
        '--price-input-per-1m',
        type=float,
        default=0.40,
        help='Price in USD per 1M input tokens (e.g., 0.40 for gpt-4.1-mini)'
    )
    parser.add_argument(
        '--price-output-per-1m',
        type=float,
        default=1.60,
        help='Price in USD per 1M output tokens (e.g., 1.60 for gpt-4.1-mini)'
    )
    
    args = parser.parse_args()
    
    pipeline = TheoryExtractionPipeline()
    pipeline.run_pipeline(
        evaluations_db=args.evaluations_db,
        papers_db=args.papers_db,
        output_file=args.output,
        results_db=args.results_db,
        limit=args.limit,
        resume_from=args.resume_from,
        resume_from_db=args.resume_from_db,
        max_workers=args.max_workers,
        price_input_per_1M=args.price_input_per_1m,
        price_output_per_1M=args.price_output_per_1m,
        test=args.test,
        test_dois_file=args.test_dois_file
    )


if __name__ == "__main__":
    main()
