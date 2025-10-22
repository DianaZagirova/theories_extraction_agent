"""
Answer research questions from validated papers using LLM.

Pipeline:
1. Load validated papers from evaluations.db
2. For each paper, retrieve full text (or abstract if unavailable) from papers.db
3. Load questions from questions JSON file
4. Use LLM to answer each question based on paper content
5. Validate answers against allowed options
6. Store results in database

Similar to extract_theories_per_paper.py but for question answering.
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

# Environment-configured limits
try:
    BACK_LIMIT = int(os.getenv('BACK_LIMIT_QUESTIONS', '50000'))
except Exception:
    BACK_LIMIT = 50000 
try:
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2000'))
except Exception:
    MAX_TOKENS = 2000
#a bit increase
TEMPERATURE = 0.3

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_integration import OpenAIClient, AzureOpenAIClient
from src.core.text_preprocessor import TextPreprocessor


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
        while True:
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
                
                # Calculate current usage
                current_tokens = sum(t[1] for t in self.token_timestamps)
                current_requests = len(self.request_timestamps)
                
                # Check if we can proceed (leave 10% buffer)
                if (current_tokens + estimated_tokens) <= self.max_tpm * 0.9 and \
                   (current_requests + 1) <= self.max_rpm * 0.9:
                    # Record this request
                    self.token_timestamps.append((now, estimated_tokens))
                    self.request_timestamps.append(now)
                    self.daily_tokens += estimated_tokens
                    return
            
            # Wait before retrying
            await asyncio.sleep(0.5)


@dataclass
class PaperAnswers:
    """Results for one paper."""
    doi: str
    pmid: Optional[str]
    title: str
    validation_result: str
    confidence_score: int
    answers: Dict[str, str]  # question -> answer
    processed_text_length: int
    used_full_text: bool
    timestamp: str


class QuestionAnsweringPipeline:
    """Pipeline for answering questions from validated papers."""
    
    def __init__(self, questions_file: str, synonyms_file: str = 'data/questions_synonyms.json', max_concurrent: int = 10):
        self.llm = AzureOpenAIClient()
        self.preprocessor = TextPreprocessor(apply_reduction=False)
        import threading
        self._lock = threading.Lock()
        # token/cost accumulators
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_tokens_per_minute=180000,
            max_requests_per_minute=450,
            tokens_per_day=1800000
        )
        
        # Concurrency control
        self.max_concurrent = max_concurrent
        self.semaphore = None  # Will be created in async context
        
        # Load questions and synonyms
        self.questions = self._load_questions(questions_file)
        self.synonyms = self._load_synonyms(synonyms_file)
        
        # System prompt for LLM
        self.system_prompt = """You are a biologist with expertise in aging and senescence research."""
    
    def _load_questions(self, questions_file: str) -> Dict[str, Dict]:
        """Load questions from JSON file with new structure."""
        with open(questions_file, 'r') as f:
            questions = json.load(f)
        print(f"\n✓ Loaded {len(questions)} questions from {questions_file}")
        for q_name, q_data in questions.items():
            print(f"  - {q_name}: {q_data['question'][:60]}...")
            print(f"    Options: {q_data['answers']}")
        return questions
    
    def _load_synonyms(self, synonyms_file: str) -> Dict[str, list]:
        """Load question synonyms for filtering."""
        try:
            with open(synonyms_file, 'r') as f:
                synonyms = json.load(f)
            
            # Count how many questions have synonym filters
            filtered_count = sum(1 for q_name, syn_list in synonyms.items() if syn_list)
            if filtered_count > 0:
                print(f"\n✓ Loaded synonyms for {filtered_count} questions (will skip if keywords not found)")
                for q_name, syn_list in synonyms.items():
                    if syn_list:
                        print(f"  - {q_name}: {len(syn_list)} keywords")
            else:
                print(f"\n✓ Loaded synonyms file (no filters configured)")
            
            return synonyms
        except FileNotFoundError:
            print(f"⚠️  Synonyms file not found: {synonyms_file}, skipping synonym filtering")
            return {}
    
    def _normalize_answer(self, answer: str, allowed_options: list, question_name: str) -> Optional[str]:
        """
        Normalize LLM answer to match allowed options.
        Handles cases like 'Yes, but not shown' when only 'Yes' is allowed.
        """
        if not answer:
            return None
        
        # Clean the answer
        answer_clean = answer.strip()
        
        # Check if it's already valid
        if answer_clean in allowed_options:
            return answer_clean
        
        # Questions that have extended options (don't normalize these)
        extended_option_questions = {
            'aging_biomarker',  # Has "Yes, quantitatively shown" and "Yes, but not shown"
        }
        
        # If this question has extended options, don't normalize
        if question_name in extended_option_questions:
            return None
        
        # Try to extract base answer by splitting on common delimiters
        delimiters = [',', ';', '(', '[', '-', ':', 'but', 'however', 'although']
        
        for delimiter in delimiters:
            if delimiter in answer_clean.lower():
                # Split and take the first part
                parts = answer_clean.split(delimiter if delimiter in [',', ';', '(', '[', '-', ':'] else f' {delimiter} ')
                base_answer = parts[0].strip().rstrip(')')
                
                # Check if base answer matches any allowed option
                if base_answer in allowed_options:
                    return base_answer
                
                # Try case-insensitive match
                for option in allowed_options:
                    if base_answer.lower() == option.lower():
                        return option
        
        # Try to find if any allowed option is a substring at the start
        for option in allowed_options:
            if answer_clean.lower().startswith(option.lower()):
                return option
        
        # Try fuzzy matching for common variations
        answer_lower = answer_clean.lower()
        for option in allowed_options:
            option_lower = option.lower()
            # Check if the answer starts with the option
            if answer_lower.startswith(option_lower):
                return option
        
        return None
    
    def _should_skip_question(self, question_name: str, text: str) -> bool:
        """Check if question should be skipped based on synonym matching."""
        if question_name not in self.synonyms:
            return False
        
        synonyms_list = self.synonyms[question_name]
        if not synonyms_list:  # Empty list means don't skip
            return False
        
        text_lower = text.lower()
        # Check if any synonym appears in the text
        for synonym in synonyms_list:
            if synonym.lower() in text_lower:
                return False  # Found a match, don't skip
        
        # No synonyms found, skip this question
        return True
    
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
            CREATE TABLE IF NOT EXISTS paper_metadata (
                doi TEXT PRIMARY KEY,
                pmid TEXT,
                title TEXT,
                validation_result TEXT,
                confidence_score INTEGER,
                processed_text_length INTEGER,
                used_full_text BOOLEAN,
                timestamp TEXT
            )
            """
        )
        
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                question_name TEXT,
                question_text TEXT,
                answer TEXT,
                confidence_score REAL,
                reasoning TEXT,
                original_answer TEXT,
                UNIQUE(doi, question_name),
                FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
            )
            """
        )
        
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_texts (
                doi TEXT PRIMARY KEY,
                processed_text TEXT,
                used_full_text BOOLEAN,
                processing_timestamp TEXT,
                FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
            )
            """
        )
        
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
        
        # Migration: Add new columns to processed_texts if they don't exist
        cur.execute("PRAGMA table_info(processed_texts);")
        cols = [row[1] for row in cur.fetchall()]
        if 'used_full_text' not in cols:
            cur.execute("ALTER TABLE processed_texts ADD COLUMN used_full_text BOOLEAN DEFAULT 1;")
        if 'processing_timestamp' not in cols:
            cur.execute("ALTER TABLE processed_texts ADD COLUMN processing_timestamp TEXT;")
        
        # Migration: Add original_answer column to paper_answers if it doesn't exist
        cur.execute("PRAGMA table_info(paper_answers);")
        answer_cols = [row[1] for row in cur.fetchall()]
        if 'original_answer' not in answer_cols:
            cur.execute("ALTER TABLE paper_answers ADD COLUMN original_answer TEXT;")
            print("  ✓ Added 'original_answer' column to paper_answers table")
        
        conn.commit()
        conn.close()
    
    def _upsert_paper_result_db(self, results_db: str, result: 'PaperAnswers'):
        """Upsert paper and answers into results DB by DOI."""
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT INTO paper_metadata (doi, pmid, title, validation_result, confidence_score, 
                                       processed_text_length, used_full_text, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO UPDATE SET
                pmid=excluded.pmid,
                title=excluded.title,
                validation_result=excluded.validation_result,
                confidence_score=excluded.confidence_score,
                processed_text_length=excluded.processed_text_length,
                used_full_text=excluded.used_full_text,
                timestamp=excluded.timestamp
            """,
            (
                result.doi,
                result.pmid,
                result.title,
                result.validation_result,
                result.confidence_score,
                result.processed_text_length,
                result.used_full_text,
                result.timestamp,
            ),
        )
        
        # Replace answers for DOI
        cur.execute("DELETE FROM paper_answers WHERE doi = ?", (result.doi,))
        for q_name, answer_data in result.answers.items():
            cur.execute(
                """
                INSERT OR IGNORE INTO paper_answers (doi, question_name, question_text, answer, confidence_score, reasoning, original_answer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (result.doi, q_name, answer_data.get('question_text', ''), 
                 answer_data['answer'], answer_data['confidence'], answer_data['reasoning'],
                 answer_data.get('original_answer', None))
            )
        
        conn.commit()
        conn.close()
    
    def _store_processed_text(self, results_db: str, doi: str, processed_text: str, used_full_text: bool):
        """Store processed text in database."""
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT OR REPLACE INTO processed_texts (doi, processed_text, used_full_text, processing_timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (doi, processed_text, used_full_text, datetime.now().isoformat()),
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
    
    def get_validated_papers_meta(
        self,
        evaluations_db: str,
        limit: Optional[int] = None
    ) -> List[tuple]:
        """Return list of validated paper metadata rows."""
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
    
    async def answer_questions_stage(self, paper: Dict, max_retries: int = 3) -> Dict[str, Dict]:
        """
        Answer all questions for a paper using LLM.
        Returns dict of question_name -> {answer, confidence, reasoning, question_text}.
        """
        # Determine text to use for filtering and prompting
        if paper['used_full_text']:
            processed_text = paper['processed_text'][:BACK_LIMIT]
            abstract = paper.get('abstract', 'N/A')[:BACK_LIMIT]
            text_for_filtering = processed_text
            additional_option_note = ""
        else:
            abstract = paper['abstract']
            processed_text = None
            text_for_filtering = abstract
            additional_option_note = '\n    - If information cannot be inferred from the abstract, you may answer "Not available"'
        
        # Filter questions based on synonyms and build questions list for prompt
        questions_to_ask = {}
        skipped_questions = {}
        
        for q_name, q_data in self.questions.items():
            if self._should_skip_question(q_name, text_for_filtering):
                # Skip this question - default answer is "No"
                skipped_questions[q_name] = {
                    'answer': 'No',
                    'confidence': 1.0,
                    'reasoning': 'Skipped: no relevant keywords found in text',
                    'question_text': q_data['question']
                }
            else:
                questions_to_ask[q_name] = q_data
        
        # If all questions are skipped, return early
        if not questions_to_ask:
            print(f"  ⚡ All questions skipped for {paper['doi']} (no relevant keywords)")
            return skipped_questions
        
        # Build questions list for prompt (only for non-skipped questions)
        questions_list = []
        for q_name, q_data in questions_to_ask.items():
            questions_list.append(f'"{q_name}": {q_data["question"]}\n   Options: {q_data["answers"]}')
        
        questions_str = "\n\n".join(questions_list)
        
        if skipped_questions:
            print(f"  ⚡ Skipped {len(skipped_questions)} questions, asking {len(questions_to_ask)} questions")
        
        # Build prompt
        if paper['used_full_text']:
            prompt = f"""Deeply analyze the scientific paper and answer the following questions based on the paper content.

# INSTRUCTIONS
1. Think step by step. Analyze the given text carefully.
2. Answer each question based on what is stated in the paper.
3. Your answer MUST be one of the provided options for each question.

# PAPER INFORMATION
Title: {paper['title']}
Abstract: {abstract}
Full Text (excerpt): {processed_text}

# QUESTIONS TO ANSWER
{questions_str}

# OUTPUT FORMAT
Return ONLY a valid JSON object with question names as keys. For each question, provide:
- "answer": your selected option (must match one of the provided options for the selected queston exactly. For most of the questions except one, there are only Yes/No options )
- "confidence": confidence score from 0.0 to 1.0
- "reasoning": brief explanation (1-2 sentences)

Example:
{{
  "aging_biomarker": {{
    "answer": "Yes, quantitatively shown",
    "confidence": 0.9,
    "reasoning": "The paper presents statistical data showing correlation between the biomarker and aging rate."
  }},
  "molecular_mechanism_of_aging": {{
    "answer": "No",
    "confidence": 0.9,
    "reasoning": "The paper does not contain info on any molecular mechanisms contributing to aging."
  }}
}}

Return ONLY valid JSON:"""
        else:
            prompt = f"""Deeply analyze the abstract of the scientific paper and answer the following questions based on it.

# INSTRUCTIONS
1. Think step by step. Analyze the given text carefully.
2. Answer each question about paper as a whole based on what is stated in the abstract.
3. Your answer MUST be one of the provided options for each question.{additional_option_note}

# PAPER INFORMATION
Title: {paper['title']}
Abstract: {abstract}

# QUESTIONS TO ANSWER
{questions_str}

# OUTPUT FORMAT
Return ONLY a valid JSON object with question names as keys. For each question, provide:
- "answer": your selected option (must match one of the provided options exactly for this Question,  For most of the questions except one, there are only Yes/No options or "Not available" if unclear from abstract)
- "confidence": confidence score from 0.0 to 1.0
- "reasoning": brief explanation (1-2 sentences)

Example:
{{
  "aging_biomarker": {{
    "answer": "Not available",
    "confidence": 0.5,
    "reasoning": "The abstract does not provide enough detail to determine if a biomarker is suggested - not biomarker are mentioned."
  }},
  "molecular_mechanism_of_aging": {{
    "answer": "Yes",
    "confidence": 0.8,
    "reasoning": "The abstract mentions a specific pathway of aging but lacks detailed mechanistic information as it is not a full paper."
  }}
}}

Return ONLY valid JSON:"""
        
            
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Run synchronous OpenAI call in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                
                if "gpt-5" in self.llm.model:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.llm.client.chat.completions.create(
                            model=self.llm.model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_completion_tokens=MAX_TOKENS
                        )
                    )
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.llm.client.chat.completions.create(
                            model=self.llm.model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS
                        )
                    )
                
                # Track usage
                try:
                    usage = response.usage
                    with self._lock:
                        self._total_prompt_tokens += int(getattr(usage, 'prompt_tokens', 0))
                        self._total_completion_tokens += int(getattr(usage, 'completion_tokens', 0))
                except Exception:
                    pass
                
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                
                result = json.loads(result_text)
                
                # Strict validation: check all non-skipped questions are answered
                if len(result) != len(questions_to_ask):
                    raise ValueError(f"Expected {len(questions_to_ask)} answers, got {len(result)}")
                
                # Validate each answer (only for questions we asked)
                validated_answers = {}
                validation_failed = False
                
                for q_name, q_data in questions_to_ask.items():
                    if q_name not in result:
                        print(f"⚠️  Missing answer for question: {q_name}")
                        validation_failed = True
                        continue
                    
                    answer_obj = result[q_name]
                    
                    # Check structure
                    if not isinstance(answer_obj, dict):
                        print(f"⚠️  Invalid answer format for {q_name}: expected dict, got {type(answer_obj)}")
                        validation_failed = True
                        continue
                    
                    if 'answer' not in answer_obj:
                        print(f"⚠️  Missing 'answer' field for {q_name}")
                        validation_failed = True
                        continue
                    
                    answer = answer_obj['answer']
                    confidence = answer_obj.get('confidence', 0.0)
                    reasoning = answer_obj.get('reasoning', '')
                    
                    # Validate answer is in allowed options
                    allowed_options = q_data['answers']
                    options_list = [opt.strip() for opt in allowed_options.split('/')]
                    
                    # For papers without full text, "Not available" is always valid
                    if not paper['used_full_text']:
                        options_list.append("Not available")
                    
                    # Normalize answer if not in options list
                    original_answer = answer
                    was_normalized = False
                    if answer not in options_list:
                        normalized_answer = self._normalize_answer(answer, options_list, q_name)
                        if normalized_answer:
                            print(f"  ℹ️  Normalized '{answer}' → '{normalized_answer}' for {q_name}")
                            answer = normalized_answer
                            was_normalized = True
                        else:
                            print(f"⚠️  Invalid answer for {q_name}: '{answer}' not in {options_list}")
                            validation_failed = True
                            continue
                    
                    # Store validated answer with metadata
                    answer_data = {
                        'answer': answer,
                        'confidence': float(confidence) if confidence else 0.0,
                        'reasoning': reasoning,
                        'question_text': q_data['question']
                    }
                    
                    # Store original answer if it was normalized
                    if was_normalized:
                        answer_data['original_answer'] = original_answer
                    
                    validated_answers[q_name] = answer_data
                
                # If validation failed, raise error to retry
                if validation_failed:
                    raise ValueError("Answer validation failed - not all questions answered correctly")
                
                # Merge validated answers with skipped questions
                all_answers = {**skipped_questions, **validated_answers}
                return all_answers
                
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
        
        print(f"❌ Failed to get valid answers for {paper['doi']} after {max_retries} attempts: {last_error}")
        # Return None to indicate failure - no fallback to default answers
        return None
    
    async def process_paper(self, paper: Dict) -> Optional[PaperAnswers]:
        """Process one paper through the pipeline. Returns None if answering fails."""
        
        # Answer questions
        answers = await self.answer_questions_stage(paper)
        
        # If answering failed, return None
        if answers is None:
            return None
        
        return PaperAnswers(
            doi=paper['doi'],
            pmid=paper['pmid'],
            title=paper['title'],
            validation_result=paper['validation_result'],
            confidence_score=paper['confidence_score'],
            answers=answers,
            processed_text_length=len(paper['processed_text']),
            used_full_text=paper['used_full_text'],
            timestamp=datetime.now().isoformat()
        )
    
    async def process_paper_with_semaphore(self, paper: Dict, semaphore: asyncio.Semaphore) -> Optional[PaperAnswers]:
        """Process paper with concurrency control."""
        async with semaphore:
            return await self.process_paper(paper)
    
    def run_pipeline(
        self,
        evaluations_db: str,
        papers_db: str,
        output_file: str,
        results_db: str,
        limit: Optional[int] = None,
        resume_from_db: bool = False,
        price_input_per_1M: float = 0.04,
        price_output_per_1M: float = 1.6,
        store_processed_text: bool = True,
        dois_file: Optional[str] = None,
        reset_db: bool = False,
        only_dois_in_file: bool = False
    ):
        """Run complete question answering pipeline."""
        
        print("\n" + "="*70)
        print("QUESTION ANSWERING PIPELINE")
        print("="*70)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load paper metadata - either from DOI list or from validated papers
        if dois_file and Path(dois_file).exists():
            print(f"\n📄 Loading DOIs from file: {dois_file}")
            with open(dois_file, 'r') as f:
                dois = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            dois = list(dict.fromkeys(dois))  # Remove duplicates
            print(f"✓ Loaded {len(dois)} unique DOIs from file")
            
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
                
                # For missing DOIs, fallback to papers.db
                missing = [d for d in dois if d not in found_dois]
                if missing:
                    print(f"⚠️  {len(missing)} DOIs not found in evaluations.db, fetching from papers.db. Examples: {missing[:15]}")
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
                            print(f"⚠️  DOI not found in any database: {d}")
                    pconn.close()
        else:
            # Load from validated papers
            meta_rows = self.get_validated_papers_meta(evaluations_db, limit)
        
        print(f"\n✓ Found {len(meta_rows)} papers to process")
        
        if not meta_rows:
            print("\n❌ No papers to process")
            return
        
        # Resume from DB if requested
        db_dois = set()
        if resume_from_db and results_db:
            conn = sqlite3.connect(results_db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_metadata'")
            if cur.fetchone():
                cur.execute("SELECT doi FROM paper_metadata")
                db_dois = {row[0] for row in cur.fetchall()}
            else:
                db_dois = set()
            conn.close()
            
            if db_dois:
                print(f"\n💾 Database Status:")
                print(f"   ✅ Already processed: {len(db_dois)} papers")
                original_count = len(meta_rows)
                meta_rows = [row for row in meta_rows if row[0] not in db_dois]
                print(f"   ⏭️  Skipping: {original_count - len(meta_rows)} papers")
                print(f"   🎯 Remaining to process: {len(meta_rows)} papers")
            else:
                print(f"\n🆕 Starting fresh - no papers in database yet")
        
        print(f"\n🔄 Processing {len(meta_rows)} papers...")
        
        # Reset DB if requested
        if reset_db and results_db and Path(results_db).exists():
            print(f"\n🗑️  Resetting database: {results_db}")
            Path(results_db).unlink()
            print("✓ Database deleted, starting fresh")
        
        # Init results DB
        self._init_results_db(results_db)
        
        # Pre-load and preprocess paper data
        print("\n📦 Pre-loading paper data into memory...")
        papers_cache = {}
        preprocessed_cache = {}
        
        # First, load already processed texts from DB to avoid reprocessing
        # Skip this if only_dois_in_file is set (process only specified DOIs)
        if results_db and not only_dois_in_file:
            print("🔍 Checking for already processed texts in DB...")
            conn = sqlite3.connect(results_db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='processed_texts'")
            if cur.fetchone():
                cur.execute("SELECT doi, processed_text, used_full_text FROM processed_texts")
                cached_texts = cur.fetchall()
                for doi, text, used_ft in cached_texts:
                    if text:
                        preprocessed_cache[doi] = {
                            'text': text,
                            'used_full_text': bool(used_ft)
                        }
                print(f"✓ Loaded {len(preprocessed_cache)} already processed texts from DB")
            conn.close()
        elif only_dois_in_file:
            print("⚠️  --only-dois-in-file flag set: will process only DOIs from file, ignoring cached texts")
        
        try:
            pconn = sqlite3.connect(f"file:{papers_db}?mode=ro", uri=True, timeout=30)
        except Exception:
            pconn = sqlite3.connect(papers_db, timeout=30)
        pcur = pconn.cursor()
        
        dois_to_fetch = [row[0] for row in meta_rows]
        
        # Load paper metadata for ALL papers (even if preprocessed text is cached)
        # We need this metadata for the paper object
        print(f"📦 Loading paper metadata for {len(dois_to_fetch)} papers...")
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
            for row in pcur.fetchall():
                doi = row[0]
                papers_cache[doi] = row
            print(f"✓ Loaded metadata for {len(papers_cache)} papers")
        
        # Only preprocess papers not already in cache
        dois_to_process = [doi for doi in dois_to_fetch if doi not in preprocessed_cache]
        
        if dois_to_process:
            print(f"⚙️  Pre-processing {len(dois_to_process)} new texts...")
            invalid_after_preprocessing = []
            
            for doi in tqdm(dois_to_process, desc="Preprocessing"):
                row = papers_cache.get(doi)
                if not row:
                    continue
                
                _, pmid_db, title_db, abstract, full_text, full_text_sections = row
                
                # Check for full text
                has_full_text = bool((full_text and str(full_text).strip()))
                has_sections = False
                if full_text_sections and str(full_text_sections).strip():
                    try:
                        sections_obj = json.loads(full_text_sections) if isinstance(full_text_sections, str) else full_text_sections
                        if isinstance(sections_obj, (list, dict)) and len(sections_obj) > 0:
                            has_sections = True
                    except Exception:
                        has_sections = True
                
                # Preprocess full text if available, otherwise use abstract
                if has_full_text or has_sections:
                    processed_text = self.preprocessor.preprocess(full_text, full_text_sections, abstract)
                    if processed_text:
                        preprocessed_cache[doi] = {
                            'text': processed_text,
                            'used_full_text': True
                        }
                        # Store immediately to DB
                        if results_db and store_processed_text:
                            self._store_processed_text(results_db, doi, processed_text, True)
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
                elif abstract and str(abstract).strip():
                    # Use abstract only
                    if abstract:
                        preprocessed_cache[doi] = {
                            'text': abstract,
                            'used_full_text': False
                        }
                        # Store immediately to DB
                        if results_db and store_processed_text:
                            self._store_processed_text(results_db, doi, abstract, False)
            
            # Record invalid papers to DB
            if invalid_after_preprocessing:
                print(f"⚠️  Found {len(invalid_after_preprocessing)} papers with full text that became invalid after preprocessing")
                if results_db:
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
        
        pconn.close()
        print(f"✓ Total papers with valid processed text: {len(preprocessed_cache)}")
        
        # Filter meta_rows to only include papers we successfully loaded
        meta_rows = [row for row in meta_rows if row[0] in preprocessed_cache]
        if not meta_rows:
            print("\n❌ No papers with valid text found")
            return
        
        # Process papers
        results = []
        stats = {
            'total_papers': len(meta_rows),
            'papers_processed': 0,
            'papers_failed': 0,
            'papers_with_full_text': 0,
            'papers_with_abstract_only': 0,
            'papers_invalid_after_preprocessing': len(invalid_after_preprocessing) if 'invalid_after_preprocessing' in locals() else 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0
        }
        
        # Sequential processing
        def _process_one_paper(row):
            doi, pmid, title, result, conf_score = row
            
            try:
                paper_data = papers_cache.get(doi)
                if not paper_data:
                    raise ValueError("Paper not found in cache")
                
                preprocessed = preprocessed_cache.get(doi)
                if not preprocessed:
                    raise ValueError("No preprocessed text available")
                
                _, pmid_db, title_db, abstract, _, _ = paper_data
                
                paper = {
                    'doi': doi,
                    'pmid': pmid or pmid_db,
                    'title': title or title_db,
                    'abstract': abstract or '',
                    'processed_text': preprocessed['text'],
                    'used_full_text': preprocessed['used_full_text'],
                    'validation_result': result,
                    'confidence_score': conf_score,
                }
                
                # Create event loop for async call
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(self.process_paper(paper)), preprocessed['text']
            except Exception as e:
                print(f"\n⚠️  Error processing {doi}: {e}")
                return None, None
        
        # Parallel processing with batches
        async def process_all_papers_async(results_db_path, stats_dict, results_list, checkpoint_file):
            """Process all papers with controlled concurrency in batches."""
            print("  → Creating semaphore...")
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            print("  → Preparing paper objects...")
            all_papers = []
            skipped = []
            
            for row in meta_rows:
                doi, pmid, title, result, conf_score = row
                try:
                    paper_data = papers_cache.get(doi)
                    if not paper_data:
                        skipped.append((doi, "not in papers_cache"))
                        continue
                    
                    preprocessed = preprocessed_cache.get(doi)
                    if not preprocessed:
                        skipped.append((doi, "not in preprocessed_cache"))
                        continue
                    
                    _, pmid_db, title_db, abstract, _, _ = paper_data
                    
                    paper = {
                        'doi': doi,
                        'pmid': pmid or pmid_db,
                        'title': title or title_db,
                        'abstract': abstract or '',
                        'processed_text': preprocessed['text'],
                        'used_full_text': preprocessed['used_full_text'],
                        'validation_result': result,
                        'confidence_score': conf_score,
                    }
                    all_papers.append(paper)
                except Exception as e:
                    print(f"\n⚠️  Error preparing {doi}: {e}")
                    skipped.append((doi, str(e)))
            
            if skipped:
                print(f"\n⚠️  Skipped {len(skipped)} papers (not found in cache)")
                print(f"   Reasons:")
                reasons_count = {}
                for doi, reason in skipped:
                    reasons_count[reason] = reasons_count.get(reason, 0) + 1
                for reason, count in reasons_count.items():
                    print(f"   - {reason}: {count} papers")
                if len(skipped) <= 5:
                    print(f"   Examples:")
                    for doi, reason in skipped[:5]:
                        print(f"     - {doi}: {reason}")
            
            if not all_papers:
                print("  → No papers to process!")
                return []
            
            # Process in batches to avoid overwhelming the system
            batch_size = 50  # Process 50 papers at a time
            all_results = []
            
            print(f"  → Processing {len(all_papers)} papers in batches of {batch_size}...")
            
            # Create progress bar for batches
            from tqdm import tqdm
            total_batches = (len(all_papers)-1)//batch_size + 1
            batch_progress = tqdm(total=len(all_papers), desc="Processing papers", unit="paper")
            
            for i in range(0, len(all_papers), batch_size):
                batch = all_papers[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                tasks = [self.process_paper_with_semaphore(paper, semaphore) for paper in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(batch_results)
                
                # Save batch results to database immediately
                for result in batch_results:
                    if isinstance(result, Exception):
                        stats_dict['papers_failed'] += 1
                    elif result:
                        results_list.append(asdict(result))
                        self._upsert_paper_result_db(results_db_path, result)
                        
                        stats_dict['papers_processed'] += 1
                        if result.used_full_text:
                            stats_dict['papers_with_full_text'] += 1
                        else:
                            stats_dict['papers_with_abstract_only'] += 1
                    else:
                        stats_dict['papers_failed'] += 1
                
                # Update progress bar
                batch_progress.update(len(batch))
                
                # Checkpoint every 100 papers
                if stats_dict['papers_processed'] % 100 == 0:
                    self._save_checkpoint(results_list, stats_dict, checkpoint_file)
                
                # Small delay between batches to let rate limiter catch up
                if batch_num < total_batches:
                    await asyncio.sleep(2)  # Increased from 1 to 2 seconds
            
            batch_progress.close()
            print(f"  → All batches complete!")
            return all_results
        
        # Run async processing
        print(f"🚀 Processing {len(meta_rows)} papers with {self.max_concurrent} concurrent requests...")
        
        # Debug: Show how many papers are ready
        ready_count = sum(1 for row in meta_rows if row[0] in papers_cache and row[0] in preprocessed_cache)
        print(f"📊 Papers ready to process: {ready_count}/{len(meta_rows)}")
        
        # Start timing
        import time as time_module
        start_time = time_module.time()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print("⏳ Starting async processing...")
            all_results = loop.run_until_complete(process_all_papers_async(results_db, stats, results, output_file))
            print(f"✓ Async processing complete, got {len(all_results)} results")
            print(f"✓ All results saved to database in real-time")
        except Exception as e:
            print(f"❌ Error in async processing: {e}")
            import traceback
            traceback.print_exc()
            all_results = []
        finally:
            loop.close()
        
        # Calculate elapsed time and rate
        elapsed_time = time_module.time() - start_time
        processed_count = stats['papers_processed']
        papers_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
        papers_per_minute = papers_per_second * 60
        
        print(f"\n⏱️  Processing time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
        print(f"📈 Processing rate: {papers_per_minute:.1f} papers/minute ({papers_per_second:.2f} papers/second)")
        
        # Estimate time for full dataset if this was a test run
        if limit and limit < 1000:
            total_papers = 18175  # Approximate total
            estimated_total_time = (total_papers / processed_count) * elapsed_time if processed_count > 0 else 0
            print(f"\n🔮 Estimated time for all {total_papers} papers: {estimated_total_time/3600:.1f} hours ({estimated_total_time/60:.0f} minutes)")
        
        # Finalize stats
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
        self._print_summary(stats)
    
    def _save_checkpoint(self, results: List[Dict], stats: Dict, output_file: str):
        """Save checkpoint to JSON file."""
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        data = {
            'results': results,
            'stats': stats,
            'checkpoint_time': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_results(self, results: List[Dict], stats: Dict, output_file: str):
        """Save final results to JSON file."""
        data = {
            'results': results,
            'stats': stats,
            'completed_time': datetime.now().isoformat()
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    def _print_summary(self, stats: Dict):
        """Print pipeline summary."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"Total papers: {stats['total_papers']}")
        print(f"Successfully processed: {stats['papers_processed']}")
        print(f"Failed (validation errors): {stats['papers_failed']}")
        if stats.get('papers_invalid_after_preprocessing', 0) > 0:
            print(f"Invalid after preprocessing: {stats['papers_invalid_after_preprocessing']}")
        print(f"\nSuccessful papers breakdown:")
        print(f"  - With full text: {stats['papers_with_full_text']}")
        print(f"  - With abstract only: {stats['papers_with_abstract_only']}")
        print(f"\nToken usage:")
        print(f"  - Prompt tokens: {stats['prompt_tokens']:,}")
        print(f"  - Completion tokens: {stats['completion_tokens']:,}")
        print(f"  - Total tokens: {stats['total_tokens']:,}")
        
        # Calculate cost breakdown
        input_cost = (stats['prompt_tokens'] / 1_000_000.0) * 0.04
        output_cost = (stats['completion_tokens'] / 1_000_000.0) * 1.6
        
        print(f"\nCost breakdown:")
        print(f"  - Input cost ($0.04/1M tokens): ${input_cost:.4f}")
        print(f"  - Output cost ($1.60/1M tokens): ${output_cost:.4f}")
        print(f"  - Total estimated cost: ${stats['estimated_cost_usd']:.2f}")
        
        # Per-paper averages
        if stats['papers_processed'] > 0:
            avg_tokens = stats['total_tokens'] / stats['papers_processed']
            avg_cost = stats['estimated_cost_usd'] / stats['papers_processed']
            print(f"\nPer-paper averages:")
            print(f"  - Tokens per paper: {avg_tokens:,.0f}")
            print(f"  - Cost per paper: ${avg_cost:.4f}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Answer questions from validated papers')
    parser.add_argument('--evaluations-db', default='/home/diana.z/hack/llm_judge/data/evaluations.db', help='Path to evaluations.db')
    parser.add_argument('--papers-db',  default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db', help='Path to papers.db')
    parser.add_argument('--questions-file', default='data/questions_part2.json', help='Path to questions JSON file')
    parser.add_argument('--output-file', default='paper_answers.json', help='Output JSON file')
    parser.add_argument('--results-db', default='./qa_results/qa_results.db', help='Path to results database')
    parser.add_argument('--limit', type=int, help='Limit number of papers to process')
    parser.add_argument('--resume-from-db', action='store_true', help='Resume from existing DB')
    parser.add_argument('--no-store-text', action='store_true', help='Do not store processed text in DB')
    parser.add_argument('--dois-file', help='Path to file containing list of DOIs (one per line)')
    parser.add_argument('--reset-db', action='store_true', help='Delete and recreate the results database (start fresh)')
    parser.add_argument('--only-dois-in-file', action='store_true', help='Process only DOIs from --dois-file, ignore cached texts')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum concurrent API calls (default: 10, recommended: 5-10)')
    
    args = parser.parse_args()
    
    print(f"\n🚀 Parallel processing enabled: {args.max_concurrent} concurrent requests")
    print(f"⚡ Rate limits: 180K tokens/min, 450 requests/min")
    
    pipeline = QuestionAnsweringPipeline(args.questions_file, max_concurrent=args.max_concurrent)
    pipeline.run_pipeline(
        evaluations_db=args.evaluations_db,
        papers_db=args.papers_db,
        output_file=args.output_file,
        results_db=args.results_db,
        limit=args.limit,
        resume_from_db=args.resume_from_db,
        store_processed_text=not args.no_store_text,
        dois_file=args.dois_file,
        reset_db=args.reset_db,
        only_dois_in_file=args.only_dois_in_file
    )


if __name__ == "__main__":
    main()
