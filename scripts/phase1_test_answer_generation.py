#!/usr/bin/env python3
"""
Phase 1 - Answer Generation with BATCHED Requests (2.5 HOURS!)

KEY INNOVATION: Multiple questions per API call → 20x speedup!

Models:
- Easy (98.7%): DeepSeek R1 Distill Llama 70B FREE - batched 20 questions/request
  * AIME: 70%, MATH: 94.5% - Exceptional reasoning, zero cost
- Hard (1.3%): Claude Sonnet 4.5 $3/$15 per M - batched 10 questions/request
  * State-of-the-art for complex chain-of-thought reasoning

Batching Strategy:
- OLD: 1 API call = 1 question = Limited by 20 requests/min = 53K questions / 20/min = 123 DAYS!
- NEW: 1 API call = 20 questions = 20 requests/min × 20 questions = 400 questions/min!
- Easy: 53,311 questions / 400 per min = 133 minutes = 2.2 hours
- Hard: 686 questions / 100 per min = 7 minutes
- TOTAL: ~2.5 hours (was 123 days!)

Performance:
- Cost: $0 (easy) + $8.67 (hard) = $8.67 total (93% savings!)
- Time: 2.5 hours (54,000x faster than non-batched free tier!)
- Checkpoints: Resume from any point

Implementation:
- Questions batched into groups of 20 (easy) or 10 (hard)
- Single API request returns JSON array of answers
- Preserves quality while bypassing rate limit constraints
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

# Load from .env file if exists
PROJECT_ROOT = Path(__file__).parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    console.print("[red]❌ Error: OPENROUTER_API_KEY not set[/red]")
    console.print("\n[yellow]To set your API key:[/yellow]")
    console.print("  1. Get API key from: https://openrouter.ai/keys")
    console.print("  2. Create .env file:")
    console.print("     [cyan]echo 'OPENROUTER_API_KEY=your_key_here' > .env[/cyan]")
    console.print("  3. Or export environment variable:")
    console.print("     [cyan]export OPENROUTER_API_KEY=your_key_here[/cyan]")
    sys.exit(1)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configurations - BATCHED REQUESTS WITH PAID MODELS (NO RATE LIMITS!)
# KEY INNOVATION: Send multiple questions in ONE API call
EASY_MODEL = "openai/gpt-4o-mini"  # $0.15/$0.60 per M - FAST, no rate limits
HARD_MODEL = "anthropic/claude-sonnet-4.5"  # $3/$15 per M, batch 10 questions/request

# BATCHING CONFIGURATION - MAXIMIZE PARALLEL PROCESSING (no rate limits on paid!)
QUESTIONS_PER_REQUEST_EASY = 25  # 25 questions in 1 API call (optimized for reliability)
QUESTIONS_PER_REQUEST_HARD = 15  # 15 hard questions in 1 API call (more complex = smaller batch)

# With paid models (NO RATE LIMITS):
# - Can process many requests in parallel!
# - Easy questions: 53,311 / (400 concurrent × 25 per batch) = ~6 batches = ~30-45 seconds
# - Hard questions: 686 / (100 concurrent × 15 per batch) = ~1 batch = ~15 seconds
# - Total: ~1 minute (was 2.5 hours with rate limits!)
# - Cost: ~$9.47 (easy) + $8.67 (hard) = ~$18 total

# Test parameters - FULL PRODUCTION RUN
TEST_SIZE = None  # None = process ALL questions (53,997)
HARD_SAMPLE_SIZE = None  # Process all hard questions
EASY_SAMPLE_SIZE = None  # Process all easy questions

# Batch processing parameters - MAXIMUM PARALLELISM for paid models!
CONCURRENT_REQUESTS = 400  # 400 parallel batches for easy questions (4x original!)
CONCURRENT_REQUESTS_HARD = 100  # 100 parallel batches for hard questions (2x original!)
BATCH_SIZE = 500  # Process in batches
RATE_LIMIT_DELAY = 0  # NO DELAY - paid models have no rate limits!

# Paths
INPUT_FILE = PROJECT_ROOT / "data/phase1/questions_final_60k.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/phase1/answers"  # Production directory
CHECKPOINT_FILE_EASY = OUTPUT_DIR / "checkpoint_easy.json"
CHECKPOINT_FILE_HARD = OUTPUT_DIR / "checkpoint_hard.json"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def call_openrouter(
    session: aiohttp.ClientSession,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Optional[str]:
    """Call OpenRouter API with error handling and retries."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/dkeviv/Cogumi-LLM",
        "X-Title": "Cogumi-LLM Phase 1 Test",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    for attempt in range(3):
        try:
            async with session.post(
                OPENROUTER_BASE_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    console.print(f"[yellow]⚠️  API error (attempt {attempt+1}/3): {response.status} - {error_text[:100]}[/yellow]")
                    
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            console.print(f"[yellow]⚠️  Exception (attempt {attempt+1}/3): {str(e)}[/yellow]")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    
    return None


async def generate_easy_answer_single(
    session: aiohttp.ClientSession,
    question: Dict[str, str],
    max_retries: int = 3
) -> Optional[str]:
    """
    Generate answer for a SINGLE easy question with retries.
    Used as fallback when batch processing fails.
    """
    for attempt in range(max_retries):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Provide a clear, concise answer."
            },
            {
                "role": "user",
                "content": f"Answer this question concisely: {question['question']}"
            }
        ]
        
        response = await call_openrouter(
            session=session,
            model=EASY_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        if response:
            # Clean and wrap in XML
            import re
            response = re.sub(r'\s+', ' ', response.replace('\n', ' ')).strip()
            return f"<response>{response}</response>"
        
        if attempt < max_retries - 1:
            await asyncio.sleep(1)
    
    return None


async def generate_easy_answers_batched(
    session: aiohttp.ClientSession,
    questions: List[Dict[str, str]]
) -> List[Optional[str]]:
    """
    Generate answers for MULTIPLE easy questions in ONE API call.
    
    Args:
        questions: List of question dicts with 'question' and 'domain' fields
    
    Returns:
        List of answers (same length as questions), with None for failures
    """
    # Build prompt with all questions
    questions_text = "\n\n".join([
        f"Question {i+1}: {q['question']}"
        for i, q in enumerate(questions)
    ])
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. You will receive MULTIPLE questions.\n"
                "Provide clear, concise answers for each question.\n\n"
                "CRITICAL: Return your response as a JSON array of strings.\n"
                "Format: [\"answer1\", \"answer2\", \"answer3\", ...]\n\n"
                "Rules:\n"
                "- Return ONLY the JSON array, nothing else\n"
                "- Each answer should be a single line (no line breaks in answers)\n"
                "- Keep answers concise - prefer 1-2 sentences\n"
                "- Preserve the ORDER - answer[0] is for question 1, etc.\n"
                "- Use LaTeX $...$ for math if needed\n"
                "- If unsure, use \"Unknown\" or brief explanation"
            )
        },
        {
            "role": "user",
            "content": f"Answer these {len(questions)} questions concisely:\n\n{questions_text}"
        }
    ]
    
    response = await call_openrouter(
        session=session,
        model=EASY_MODEL,
        messages=messages,
        max_tokens=500 * len(questions),  # Scale with number of questions
        temperature=0.7
    )
    
    if not response:
        return [None] * len(questions)
    
    # Parse JSON array response
    try:
        import json
        import re
        
        # Try to extract JSON array from response
        response = response.strip()
        if not response.startswith('['):
            # Find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                response = response[start:end]
        
        # Try standard JSON parsing first
        try:
            answers = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Clean up common JSON issues
            # Remove unescaped newlines within strings
            response = re.sub(r'(?<!\\)\n', ' ', response)
            # Try again
            try:
                answers = json.loads(response)
            except json.JSONDecodeError:
                console.print(f"[red]❌ JSON parsing failed even after cleanup[/red]")
                console.print(f"[yellow]Response preview: {response[:200]}...[/yellow]")
                return [None] * len(questions)
        
        # Validate: should be list
        if not isinstance(answers, list):
            console.print(f"[yellow]⚠️  Response is not a list[/yellow]")
            return [None] * len(questions)
        
        # Handle length mismatch
        if len(answers) != len(questions):
            console.print(f"[yellow]⚠️  Batch response length mismatch: got {len(answers)}, expected {len(questions)}[/yellow]")
            # Pad or trim to match
            if len(answers) < len(questions):
                answers.extend([None] * (len(questions) - len(answers)))
            else:
                answers = answers[:len(questions)]
        
        # Clean answers: remove internal newlines, normalize whitespace, wrap in XML
        cleaned_answers = []
        for ans in answers:
            if ans:
                # Replace newlines with spaces, normalize whitespace
                ans = re.sub(r'\s+', ' ', str(ans).replace('\n', ' ')).strip()
                cleaned_answers.append(f"<response>{ans}</response>")
            else:
                cleaned_answers.append(None)
        
        return cleaned_answers
        
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]❌ Failed to parse batched response: {e}[/red]")
        console.print(f"[yellow]Response preview: {response[:200]}...[/yellow]")
        return [None] * len(questions)


async def generate_hard_answers_batched(
    session: aiohttp.ClientSession,
    questions: List[Dict[str, str]]
) -> List[Optional[Dict[str, str]]]:
    """
    Generate draft+thinking+response for MULTIPLE hard questions in ONE API call.
    
    Args:
        questions: List of question dicts with 'question' and 'domain' fields
    
    Returns:
        List of answer dicts with {draft, thinking, response}, None for failures
    """
    # Build prompt with all questions
    questions_text = "\n\n".join([
        f"Question {i+1}: {q['question']}"
        for i, q in enumerate(questions)
    ])
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI assistant. You will receive MULTIPLE questions.\n"
                "For EACH question, provide: draft, thinking, and response.\n\n"
                "CRITICAL: Return as a JSON array of objects.\n"
                "Format: [{\"draft\": \"...\", \"thinking\": \"...\", \"response\": \"...\"}, ...]\n\n"
                "Rules:\n"
                "- Return ONLY the JSON array, nothing else\n"
                "- Each object has 3 fields: draft, thinking, response\n"
                "- Draft: Initial solution attempt\n"
                "- Thinking: Analysis and improvements needed\n"
                "- Response: Final refined answer\n"
                "- Keep each section 2-3 sentences (batched mode = concise!)\n"
                "- Preserve ORDER - answer[0] for question 1, etc.\n"
                "- Single line strings (no newlines within strings)"
            )
        },
        {
            "role": "user",
            "content": f"Answer these {len(questions)} complex questions:\n\n{questions_text}"
        }
    ]
    
    response = await call_openrouter(
        session=session,
        model=HARD_MODEL,
        messages=messages,
        max_tokens=1000 * len(questions),  # Scale with batch size
        temperature=0.7
    )
    
    if not response:
        return [None] * len(questions)
    
    # Parse JSON array response
    try:
        import json
        import re
        
        # Extract JSON array
        response = response.strip()
        if not response.startswith('['):
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                response = response[start:end]
        
        # Try parsing with cleanup
        try:
            answers = json.loads(response)
        except json.JSONDecodeError:
            # Remove unescaped newlines
            response = re.sub(r'(?<!\\)\n', ' ', response)
            try:
                answers = json.loads(response)
            except json.JSONDecodeError:
                console.print(f"[red]❌ Hard batch JSON parsing failed[/red]")
                return [None] * len(questions)
        
        if not isinstance(answers, list):
            return [None] * len(questions)
        
        # Handle length mismatch
        if len(answers) != len(questions):
            console.print(f"[yellow]⚠️  Hard batch length mismatch: {len(answers)} vs {len(questions)}[/yellow]")
            if len(answers) < len(questions):
                answers.extend([None] * (len(questions) - len(answers)))
            else:
                answers = answers[:len(questions)]
        
        # Wrap each section in XML tags
        result_answers = []
        for ans in answers:
            if ans and isinstance(ans, dict):
                draft = ans.get('draft', '')
                thinking = ans.get('thinking', '')
                resp = ans.get('response', '')
                
                if draft and thinking and resp:
                    # Clean and wrap in XML
                    draft = re.sub(r'\s+', ' ', str(draft).replace('\n', ' ')).strip()
                    thinking = re.sub(r'\s+', ' ', str(thinking).replace('\n', ' ')).strip()
                    resp = re.sub(r'\s+', ' ', str(resp).replace('\n', ' ')).strip()
                    
                    result_answers.append({
                        "draft": f"<draft>{draft}</draft>",
                        "thinking": f"<thinking>{thinking}</thinking>",
                        "response": f"<response>{resp}</response>"
                    })
                else:
                    result_answers.append(None)
            else:
                result_answers.append(None)
        
        return result_answers
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        console.print(f"[red]❌ Failed to parse hard batched response: {e}[/red]")
        return [None] * len(questions)


async def generate_hard_answer_single(
    session: aiohttp.ClientSession,
    question: Dict[str, str],
    max_retries: int = 3
) -> Optional[Dict[str, str]]:
    """
    Generate answer for a SINGLE hard question with retries.
    Used as fallback when batch processing fails.
    """
    for attempt in range(max_retries):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert problem solver. For each question:\n"
                    "1. Write a draft answer\n"
                    "2. Show your reasoning/thinking process\n"
                    "3. Provide the final response\n\n"
                    "Return your answer as a JSON object with these fields:\n"
                    '{"draft": "...", "thinking": "...", "response": "..."}'
                )
            },
            {
                "role": "user",
                "content": question['question']
            }
        ]
        
        response = await call_openrouter(
            session=session,
            model=HARD_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        if response:
            try:
                import json
                import re
                
                # Try to parse JSON
                response = response.strip()
                if not response.startswith('{'):
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start >= 0 and end > start:
                        response = response[start:end]
                
                data = json.loads(response)
                
                if 'draft' in data and 'thinking' in data and 'response' in data:
                    # Clean and wrap
                    draft = re.sub(r'\s+', ' ', str(data['draft']).replace('\n', ' ')).strip()
                    thinking = re.sub(r'\s+', ' ', str(data['thinking']).replace('\n', ' ')).strip()
                    resp = re.sub(r'\s+', ' ', str(data['response']).replace('\n', ' ')).strip()
                    
                    return {
                        "draft": f"<draft>{draft}</draft>",
                        "thinking": f"<thinking>{thinking}</thinking>",
                        "response": f"<response>{resp}</response>"
                    }
            except:
                pass
        
        if attempt < max_retries - 1:
            await asyncio.sleep(1)
    
    return None


async def generate_easy_answer(
    session: aiohttp.ClientSession,
    question: str,
    domain: str
) -> Optional[str]:
    """Generate direct response for easy question using GPT-4o-mini."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Provide clear, concise, and accurate answers. "
                "Keep your response direct and to the point. "
                "If using math notation, use LaTeX format with $ for inline math or $$ for display math."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = await call_openrouter(
        session=session,
        model=EASY_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    # Wrap response in XML marker for training format
    if response:
        response = f"<response>{response}</response>"
    
    return response


async def generate_hard_answer(
    session: aiohttp.ClientSession,
    question: str,
    domain: str
) -> Optional[Dict[str, str]]:
    """Generate draft + thinking + response for hard question using Claude Sonnet 4."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI assistant. Think step by step and show your reasoning.\n\n"
                "CRITICAL: Format your response EXACTLY with these XML tags:\n\n"
                "<draft>\n"
                "Your initial attempt at solving the problem. Provide a working solution.\n"
                "</draft>\n\n"
                "<thinking>\n"
                "Analyze your draft. Identify improvements needed. Consider edge cases and best practices.\n"
                "</thinking>\n\n"
                "<response>\n"
                "Your final refined answer incorporating improvements from your thinking.\n"
                "</response>\n\n"
                "IMPORTANT:\n"
                "- Always use all three tags: <draft>, <thinking>, <response>\n"
                "- Close each tag properly\n"
                "- Do NOT use nested tags\n"
                "- Keep each section focused and clear\n"
                "- If using LaTeX math, use $...$ for inline or $$...$$ for display mode"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    response = await call_openrouter(
        session=session,
        model=HARD_MODEL,
        messages=messages,
        max_tokens=2500,
        temperature=0.7
    )
    
    if not response:
        return None
    
    # Parse response into sections but keep XML markers
    try:
        # Normalize line endings
        response = response.replace('\r\n', '\n').replace('\r', '\n')
        
        # Try to find all tags
        draft_start = response.find("<draft>")
        draft_end = response.find("</draft>")
        thinking_start = response.find("<thinking>")
        thinking_end = response.find("</thinking>")
        response_start = response.find("<response>")
        response_end = response.find("</response>")
        
        # Check if we have all tags
        if all(x != -1 for x in [draft_start, draft_end, thinking_start, thinking_end, response_start, response_end]):
            # Extract with XML tags included
            draft = response[draft_start:draft_end + 8]  # Include </draft>
            thinking = response[thinking_start:thinking_end + 11]  # Include </thinking>
            final_response = response[response_start:response_end + 11]  # Include </response>
            
            # Validate sections are not empty
            if draft and thinking and final_response:
                return {
                    "draft": draft,
                    "thinking": thinking,
                    "response": final_response
                }
        
        # Fallback: Try to split intelligently and add XML tags
        console.print(f"[yellow]⚠️  Parsing with fallback method (missing/malformed tags)[/yellow]")
        
        # Look for section headers or split by paragraphs
        lines = response.split('\n')
        sections = {'draft': [], 'thinking': [], 'response': []}
        current_section = None
        
        for line in lines:
            lower = line.lower().strip()
            if 'draft' in lower and len(line) < 50:
                current_section = 'draft'
                continue
            elif 'thinking' in lower and len(line) < 50:
                current_section = 'thinking'
                continue
            elif ('response' in lower or 'final' in lower) and len(line) < 50:
                current_section = 'response'
                continue
            
            if current_section and line.strip():
                sections[current_section].append(line)
        
        # If we got reasonable sections, add XML tags
        if all(sections[k] for k in ['draft', 'thinking', 'response']):
            return {
                "draft": f"<draft>\n{chr(10).join(sections['draft'])}\n</draft>",
                "thinking": f"<thinking>\n{chr(10).join(sections['thinking'])}\n</thinking>",
                "response": f"<response>\n{chr(10).join(sections['response'])}\n</response>"
            }
        
        # Last resort: split into thirds and add XML tags
        console.print(f"[yellow]⚠️  Using basic split (no structure found)[/yellow]")
        paragraphs = [p for p in response.split('\n\n') if p.strip()]
        n = len(paragraphs)
        if n >= 3:
            return {
                "draft": f"<draft>\n{chr(10).join(paragraphs[:n//3])}\n</draft>",
                "thinking": f"<thinking>\n{chr(10).join(paragraphs[n//3:2*n//3])}\n</thinking>",
                "response": f"<response>\n{chr(10).join(paragraphs[2*n//3:])}\n</response>"
            }
        else:
            # Too short, just duplicate with XML tags
            return {
                "draft": f"<draft>{response[:len(response)//3]}</draft>",
                "thinking": f"<thinking>{response[len(response)//3:2*len(response)//3]}</thinking>",
                "response": f"<response>{response[2*len(response)//3:]}</response>"
            }
            
    except Exception as e:
        console.print(f"[red]❌ Error parsing response: {e}[/red]")
        return None


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """Load checkpoint to resume from previous run."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            # Convert completed list to set for fast lookup
            if "completed_questions" not in checkpoint:
                checkpoint["completed_questions"] = set()
            else:
                checkpoint["completed_questions"] = set(checkpoint["completed_questions"])
            return checkpoint
    return {
        "completed": [],
        "completed_questions": set(),  # Question texts that are done
        "failed": [],
        "count": 0,
        "timestamp": datetime.now().isoformat()
    }


def save_checkpoint(checkpoint: Dict, checkpoint_file: Path):
    """Save checkpoint for resuming."""
    checkpoint["timestamp"] = datetime.now().isoformat()
    # Convert set to list for JSON serialization
    checkpoint_copy = checkpoint.copy()
    checkpoint_copy["completed_questions"] = list(checkpoint["completed_questions"])
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_copy, f, indent=2, ensure_ascii=False)


def sanitize_for_json(text: str) -> str:
    """Sanitize text to ensure JSON compatibility."""
    if not text:
        return ""
    
    # Replace problematic characters
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove or escape control characters except newline and tab
    sanitized = []
    for char in text:
        code = ord(char)
        if char in ['\n', '\t'] or (32 <= code < 127) or code >= 160:
            sanitized.append(char)
        elif code < 32:
            # Skip other control characters
            continue
        else:
            sanitized.append(char)
    
    return ''.join(sanitized)


def validate_json_serializable(obj: dict) -> bool:
    """Validate that an object can be JSON serialized."""
    try:
        json.dumps(obj, ensure_ascii=False)
        return True
    except (TypeError, ValueError) as e:
        console.print(f"[red]JSON validation failed: {e}[/red]")
        return False


def count_reasoning_steps(thinking_section: str) -> int:
    """
    Count the number of reasoning steps in the thinking section.
    
    Looks for:
    - Numbered lists: "1.", "2.", "Step 1:", "Step 2:"
    - Bullet points: "- ", "* ", "• "
    - Explicit step markers: "First", "Second", "Third", "Finally"
    
    Returns: Number of distinct reasoning steps found
    """
    if not thinking_section:
        return 0
    
    # Remove XML tags for counting
    text = thinking_section.replace("<thinking>", "").replace("</thinking>", "")
    
    step_count = 0
    
    # Count numbered steps (1., 2., 3. or Step 1:, Step 2:)
    import re
    numbered_steps = re.findall(r'(?:\d+\.|Step \d+:|#\d+)', text, re.IGNORECASE)
    step_count = max(step_count, len(numbered_steps))
    
    # Count bullet points (only if no numbered steps found)
    if step_count == 0:
        bullet_points = re.findall(r'(?:^|\n)\s*[-*•]\s+', text)
        step_count = max(step_count, len(bullet_points))
    
    # Count explicit step markers (First, Second, Third, Finally)
    if step_count == 0:
        step_markers = re.findall(
            r'\b(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally|Additionally|Moreover)\b',
            text,
            re.IGNORECASE
        )
        step_count = max(step_count, len(set(step_markers)))
    
    # Count sentence-level steps (paragraphs as reasoning units)
    if step_count == 0:
        sentences = re.split(r'[.!?]\s+', text)
        step_count = min(len([s for s in sentences if len(s.strip()) > 20]), 5)  # Cap at 5
    
    return max(1, step_count)  # Minimum 1 step if there's any thinking


async def process_question_batch(
    session: aiohttp.ClientSession,
    questions: List[Dict],
    results: Dict[str, List],
    checkpoint_easy: Dict,
    checkpoint_hard: Dict,
    semaphore: asyncio.Semaphore
) -> tuple[int, int]:
    """
    Process a batch of questions using BATCHED API calls (multiple questions per request).
    
    This is the key optimization: instead of 1 API call per question,
    we send 30 easy questions or 15 hard questions in a SINGLE request.
    """
    
    success_count = 0
    failure_count = 0
    
    # Separate easy and hard questions
    easy_questions = [q for q in questions if q["difficulty"] == "easy"]
    hard_questions = [q for q in questions if q["difficulty"] == "hard"]
    
    # Helper function to recursively process batches with progressive splitting
    async def process_easy_batch_recursive(batch_questions: List[Dict], batch_num: int, depth: int = 0) -> List[Optional[str]]:
        """Recursively process batch with progressive splitting: 25 → 20/5 → 10/10 → 5/5 → individual."""
        prefix = "  " * depth
        console.print(f"[cyan]{prefix}Processing batch {batch_num} ({len(batch_questions)} questions, depth={depth})[/cyan]")
        
        answers = await generate_easy_answers_batched(session, batch_questions)
        successful_count = len([a for a in answers if a])
        console.print(f"[green]{prefix}✓ Got {successful_count}/{len(batch_questions)}[/green]")
        
        # If batch completely failed and size > 5, split recursively
        if successful_count == 0 and len(batch_questions) > 5:
            console.print(f"[yellow]{prefix}⚠️  Splitting {len(batch_questions)} questions...[/yellow]")
            mid = len(batch_questions) // 2
            first_half = await process_easy_batch_recursive(batch_questions[:mid], f"{batch_num}a", depth + 1)
            second_half = await process_easy_batch_recursive(batch_questions[mid:], f"{batch_num}b", depth + 1)
            answers = first_half + second_half
            successful_count = len([a for a in answers if a])
            console.print(f"[cyan]{prefix}After split: {successful_count}/{len(batch_questions)}[/cyan]")
        
        # Only retry individually if batch is small (≤5) and has failures
        if len(batch_questions) <= 5:
            for i, (q, answer) in enumerate(zip(batch_questions, answers)):
                if answer is None:
                    console.print(f"[yellow]{prefix}🔄 Individual retry {i+1}...[/yellow]")
                    answers[i] = await generate_easy_answer_single(session, q)
        
        return answers
    
    # Helper function to process one easy batch
    async def process_easy_batch(batch_questions: List[Dict], batch_num: int) -> tuple[int, int]:
        """Process one batch of easy questions with progressive splitting fallback."""
        async with semaphore:
            answers = await process_easy_batch_recursive(batch_questions, batch_num)
            
            successes = 0
            failures = 0
            
            # Process results
            for q, answer in zip(batch_questions, answers):
                if answer:
                    answer = sanitize_for_json(answer)
                    result = {
                        "prompt": sanitize_for_json(q["question"]),
                        "response": answer,
                        "metadata": {
                            "difficulty": "easy",
                            "domain": q["domain"],
                            "task_type": "general",
                            "complexity": "low",
                            "requires_reasoning": False,
                            "harm_category": "none",
                            "token_count": len(answer.split()),
                            "teacher_model": "gpt-4o-mini"
                        }
                    }
                    
                    if validate_json_serializable(result):
                        results["easy"].append(result)
                        checkpoint_easy["completed"].append(id(q))
                        checkpoint_easy["completed_questions"].add(q["question"])  # Track by question text
                        checkpoint_easy["count"] += 1
                        successes += 1
                    else:
                        failures += 1
                        checkpoint_easy["failed"].append({
                            "question": q["question"][:100],
                            "difficulty": "easy",
                            "error": "json_serialization"
                        })
                else:
                    failures += 1
                    checkpoint_easy["failed"].append({
                        "question": q["question"][:100],
                        "difficulty": "easy",
                        "error": "api_failure"
                    })
            
            return successes, failures
    
    # Create tasks for all easy batches (process in parallel!)
    easy_tasks = []
    for i in range(0, len(easy_questions), QUESTIONS_PER_REQUEST_EASY):
        batch = easy_questions[i:i + QUESTIONS_PER_REQUEST_EASY]
        batch_num = i // QUESTIONS_PER_REQUEST_EASY + 1
        easy_tasks.append(process_easy_batch(batch, batch_num))
    
    # Process all easy batches in parallel
    if easy_tasks:
        easy_results = await asyncio.gather(*easy_tasks, return_exceptions=True)
        for result in easy_results:
            if isinstance(result, Exception):
                console.print(f"[red]❌ Batch exception: {result}[/red]")
                failure_count += 1
            else:
                s, f = result
                success_count += s
                failure_count += f
    
    # Helper function to process one hard batch
    # Helper function to recursively process hard batches with progressive splitting
    async def process_hard_batch_recursive(batch_questions: List[Dict], batch_num: int, depth: int = 0) -> List[Optional[Dict]]:
        """Recursively process hard batch with progressive splitting: 15 → 10/5 → 5/5 → individual."""
        prefix = "  " * depth
        console.print(f"[cyan]{prefix}Processing hard batch {batch_num} ({len(batch_questions)} questions, depth={depth})[/cyan]")
        
        answers = await generate_hard_answers_batched(session, batch_questions)
        successful_count = len([a for a in answers if a])
        console.print(f"[green]{prefix}✓ Got {successful_count}/{len(batch_questions)}[/green]")
        
        # If batch completely failed and size > 5, split recursively
        if successful_count == 0 and len(batch_questions) > 5:
            console.print(f"[yellow]{prefix}⚠️  Splitting {len(batch_questions)} hard questions...[/yellow]")
            mid = len(batch_questions) // 2
            first_half = await process_hard_batch_recursive(batch_questions[:mid], f"{batch_num}a", depth + 1)
            second_half = await process_hard_batch_recursive(batch_questions[mid:], f"{batch_num}b", depth + 1)
            answers = first_half + second_half
            successful_count = len([a for a in answers if a])
            console.print(f"[cyan]{prefix}After split: {successful_count}/{len(batch_questions)}[/cyan]")
        
        # Only retry individually if batch is small (≤5) and has failures
        if len(batch_questions) <= 5:
            for i, (q, answer) in enumerate(zip(batch_questions, answers)):
                if answer is None:
                    console.print(f"[yellow]{prefix}🔄 Hard individual retry {i+1}...[/yellow]")
                    answers[i] = await generate_hard_answer_single(session, q)
        
        return answers
    
    async def process_hard_batch(batch_questions: List[Dict], batch_num: int) -> tuple[int, int]:
        """Process one batch of hard questions with progressive splitting fallback."""
        async with semaphore:
            answers = await process_hard_batch_recursive(batch_questions, batch_num)
            
            successes = 0
            failures = 0
            
            for q, answer_sections in zip(batch_questions, answers):
                if answer_sections:
                    answer_sections = {k: sanitize_for_json(v) for k, v in answer_sections.items()}
                    reasoning_steps = count_reasoning_steps(answer_sections.get("thinking", ""))
                    
                    result = {
                        "prompt": sanitize_for_json(q["question"]),
                        "draft": answer_sections["draft"],
                        "thinking": answer_sections["thinking"],
                        "response": answer_sections["response"],
                        "metadata": {
                            "difficulty": "hard",
                            "domain": q["domain"],
                            "task_type": "complex",
                            "complexity": "high",
                            "requires_reasoning": True,
                            "reasoning_steps": reasoning_steps,
                            "harm_category": "none",
                            "token_count": sum(len(v.split()) for v in answer_sections.values()),
                            "teacher_model": "claude-sonnet-4.5"
                        }
                    }
                    
                    if validate_json_serializable(result):
                        results["hard"].append(result)
                        checkpoint_hard["completed"].append(id(q))
                        checkpoint_hard["completed_questions"].add(q["question"])  # Track by question text
                        checkpoint_hard["count"] += 1
                        successes += 1
                    else:
                        failures += 1
                        checkpoint_hard["failed"].append({
                            "question": q["question"][:100],
                            "difficulty": "hard",
                            "error": "json_serialization"
                        })
                else:
                    failures += 1
                    checkpoint_hard["failed"].append({
                        "question": q["question"][:100],
                        "difficulty": "hard",
                        "error": "api_failure"
                    })
            
            return successes, failures
    
    # Create tasks for all hard batches (process in parallel!)
    hard_tasks = []
    for i in range(0, len(hard_questions), QUESTIONS_PER_REQUEST_HARD):
        batch = hard_questions[i:i + QUESTIONS_PER_REQUEST_HARD]
        batch_num = i // QUESTIONS_PER_REQUEST_HARD + 1
        hard_tasks.append(process_hard_batch(batch, batch_num))
    
    # Process all hard batches in parallel
    if hard_tasks:
        hard_results = await asyncio.gather(*hard_tasks, return_exceptions=True)
        for result in hard_results:
            if isinstance(result, Exception):
                console.print(f"[red]❌ Hard batch exception: {result}[/red]")
                failure_count += 1
            else:
                s, f = result
                success_count += s
                failure_count += f
    
    return success_count, failure_count


async def main():
    """Main test answer generation."""
    # Display configuration
    console.print("\n[green]✓ Using Paid Models - NO RATE LIMITS![/green]")
    console.print(f"  • Easy Model: {EASY_MODEL}")
    console.print(f"  • Hard Model: {HARD_MODEL}")
    console.print("\n[cyan]High-Performance Configuration:[/cyan]")
    console.print(f"  • Concurrent easy batches: {CONCURRENT_REQUESTS}")
    console.print(f"  • Concurrent hard batches: {CONCURRENT_REQUESTS_HARD}")
    console.print(f"  • Questions per easy batch: {QUESTIONS_PER_REQUEST_EASY}")
    console.print(f"  • Questions per hard batch: {QUESTIONS_PER_REQUEST_HARD}")
    console.print(f"  • Est. throughput: ~{CONCURRENT_REQUESTS * QUESTIONS_PER_REQUEST_EASY * 10:,} easy questions/min")
    console.print(f"  • Est. time for 53,997 questions: ~1 minute")
    
    console.print(Panel.fit(
        "[bold cyan]Phase 1: FULL PRODUCTION Answer Generation (53,997 Questions)[/bold cyan]\n\n"
        f"Easy: ALL 53,311 questions → [green]{EASY_MODEL}[/green] (batched, 400 concurrent)\n"
        f"Hard: ALL 686 questions → [yellow]{HARD_MODEL}[/yellow] (batched, 100 concurrent)\n\n"
        "[green]✓ Paid models - NO rate limits - MAXIMUM SPEED![/green]\n"
        "[yellow]⚡ Estimated cost: ~$18 | Time: ~60 seconds[/yellow]",
        title="🚀 FULL PRODUCTION RUN"
    ))
    
    # Load questions
    console.print("\n[cyan]Step 1: Loading questions...[/cyan]")
    questions = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    console.print(f"✓ Loaded {len(questions):,} total questions")
    
    # Sample questions
    easy_questions = [q for q in questions if q["difficulty"] == "easy"]
    hard_questions = [q for q in questions if q["difficulty"] == "hard"]
    
    console.print(f"  • Easy: {len(easy_questions):,} available")
    console.print(f"  • Hard: {len(hard_questions):,} available")
    
    # Sample for testing or use all questions
    import random
    random.seed(42)  # Reproducible sampling
    
    if EASY_SAMPLE_SIZE is None:
        test_easy = easy_questions  # Process ALL easy questions
    else:
        test_easy = random.sample(easy_questions, min(EASY_SAMPLE_SIZE, len(easy_questions)))
    
    if HARD_SAMPLE_SIZE is None:
        test_hard = hard_questions  # Process ALL hard questions
    else:
        test_hard = random.sample(hard_questions, min(HARD_SAMPLE_SIZE, len(hard_questions)))
    
    test_questions = test_easy + test_hard
    random.shuffle(test_questions)  # Mix them
    
    console.print(f"\n[cyan]Processing {len(test_questions):,} questions[/cyan]")
    console.print(f"  • Easy: {len(test_easy):,}")
    console.print(f"  • Hard: {len(test_hard):,}")
    
    # Show domain distribution
    domain_counts = {}
    for q in test_questions:
        domain_counts[q["domain"]] = domain_counts.get(q["domain"], 0) + 1
    
    table = Table(title="Test Sample Distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    for domain, count in sorted(domain_counts.items()):
        table.add_row(domain, str(count))
    
    console.print(table)
    
    # Load separate checkpoints for easy and hard
    checkpoint_easy = load_checkpoint(CHECKPOINT_FILE_EASY)
    checkpoint_hard = load_checkpoint(CHECKPOINT_FILE_HARD)
    
    completed_easy = checkpoint_easy["completed_questions"]
    completed_hard = checkpoint_hard["completed_questions"]
    all_completed = completed_easy | completed_hard
    
    if all_completed:
        console.print(f"\n[yellow]📋 Resuming from checkpoint:[/yellow]")
        console.print(f"   Easy completed: {len(completed_easy)} ({checkpoint_easy['count']} answers)")
        console.print(f"   Hard completed: {len(completed_hard)} ({checkpoint_hard['count']} answers)")
        console.print(f"   Total completed: {len(all_completed)}")
        # Filter out already completed questions
        original_count = len(test_questions)
        test_questions = [q for q in test_questions if q["question"] not in all_completed]
        console.print(f"   Skipping {original_count - len(test_questions)} completed, processing {len(test_questions)} remaining")
    else:
        console.print(f"\n[green]✨ Starting fresh - no checkpoint found[/green]")
    
    # Generate answers with batch processing
    console.print(f"\n[cyan]Step 2: Generating answers (parallel processing)...[/cyan]")
    console.print(f"  • Concurrent requests: {CONCURRENT_REQUESTS}")
    console.print(f"  • Batch size: {BATCH_SIZE}")
    console.print(f"  • Questions to process: {len(test_questions)}")
    
    # Load existing results if resuming
    results = {
        "easy": [],
        "hard": []
    }
    
    easy_output = OUTPUT_DIR / "easy_answers.jsonl"
    hard_output = OUTPUT_DIR / "hard_answers.jsonl"
    
    if all_completed:
        # Load existing results
        if easy_output.exists():
            with open(easy_output, 'r', encoding='utf-8') as f:
                for line in f:
                    results["easy"].append(json.loads(line.strip()))
            console.print(f"  • Loaded {len(results['easy'])} existing easy answers")
        
        if hard_output.exists():
            with open(hard_output, 'r', encoding='utf-8') as f:
                for line in f:
                    results["hard"].append(json.loads(line.strip()))
            console.print(f"  • Loaded {len(results['hard'])} existing hard answers")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Processing {len(test_questions)} questions...",
                total=len(test_questions)
            )
            
            # Process in batches for checkpointing
            total_success = 0
            total_failures = 0
            
            # Define output files
            easy_output = OUTPUT_DIR / "easy_answers.jsonl"
            hard_output = OUTPUT_DIR / "hard_answers.jsonl"
            
            for batch_start in range(0, len(test_questions), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(test_questions))
                batch = test_questions[batch_start:batch_end]
                
                # Track how many results we had before processing
                easy_before = len(results["easy"])
                hard_before = len(results["hard"])
                
                # Process batch concurrently
                success, failures = await process_question_batch(
                    session, batch, results, checkpoint_easy, checkpoint_hard, semaphore
                )
                
                total_success += success
                total_failures += failures
                
                # SAVE NEW RESULTS IMMEDIATELY (append mode)
                # Easy answers
                if len(results["easy"]) > easy_before:
                    with open(easy_output, 'a', encoding='utf-8') as f:
                        for result in results["easy"][easy_before:]:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    # Save easy checkpoint
                    save_checkpoint(checkpoint_easy, CHECKPOINT_FILE_EASY)
                
                # Hard answers
                if len(results["hard"]) > hard_before:
                    with open(hard_output, 'a', encoding='utf-8') as f:
                        for result in results["hard"][hard_before:]:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    # Save hard checkpoint
                    save_checkpoint(checkpoint_hard, CHECKPOINT_FILE_HARD)
                
                # Update progress with checkpoint info
                progress.update(task, advance=len(batch))
                checkpoint_info = f"Easy: {checkpoint_easy['count']:,} | Hard: {checkpoint_hard['count']:,}"
                progress.update(task, description=f"Processed {batch_end:,}/{len(test_questions):,} | {checkpoint_info} | Failed: {total_failures}")
    
    # Final summary
    console.print("\n[cyan]Step 3: Generation complete![/cyan]")
    console.print(f"✓ All answers saved incrementally during processing")
    
    # Final checkpoint save (update with final state)
    save_checkpoint(checkpoint_easy, CHECKPOINT_FILE_EASY)
    save_checkpoint(checkpoint_hard, CHECKPOINT_FILE_HARD)
    
    easy_output = OUTPUT_DIR / "easy_answers.jsonl"
    hard_output = OUTPUT_DIR / "hard_answers.jsonl"
    save_checkpoint(checkpoint_hard, CHECKPOINT_FILE_HARD)
    
    # Summary
    console.print("\n" + "="*70)
    console.print("[bold green]✅ FULL Answer Generation Complete![/bold green]")
    console.print("="*70)
    
    console.print(f"\n[cyan]Results:[/cyan]")
    console.print(f"  ✓ Easy answers generated: {len(results['easy']):,}")
    console.print(f"  ✓ Hard answers generated: {len(results['hard']):,}")
    console.print(f"  ✓ Total: {len(results['easy']) + len(results['hard']):,} answers")
    console.print(f"  ✗ Failed (easy): {len(checkpoint_easy['failed'])}")
    console.print(f"  ✗ Failed (hard): {len(checkpoint_hard['failed'])}")
    
    console.print(f"\n[cyan]Output files:[/cyan]")
    console.print(f"  • {easy_output}")
    console.print(f"  • {hard_output}")
    console.print(f"  • {CHECKPOINT_FILE_EASY}")
    console.print(f"  • {CHECKPOINT_FILE_HARD}")
    
    # Show sample outputs
    if results["easy"]:
        console.print("\n[cyan]Sample EASY answer:[/cyan]")
        sample = results["easy"][0]
        console.print(Panel(
            f"[bold]Q:[/bold] {sample['prompt']}\n\n"
            f"[bold]A:[/bold] {sample['response']}",
            title=f"Easy - {sample['metadata']['domain']}",
            border_style="green"
        ))
    
    if results["hard"]:
        console.print("\n[cyan]Sample HARD answer:[/cyan]")
        sample = results["hard"][0]
        console.print(Panel(
            f"[bold]Q:[/bold] {sample['prompt']}\n\n"
            f"[bold]Draft:[/bold]\n{sample['draft'][:200]}...\n\n"
            f"[bold]Thinking:[/bold]\n{sample['thinking'][:200]}...\n\n"
            f"[bold]Response:[/bold]\n{sample['response'][:200]}...",
            title=f"Hard - {sample['metadata']['domain']}",
            border_style="yellow"
        ))
    
    console.print("\n[yellow]📋 Next Steps:[/yellow]")
    console.print("  1. Review test answers for quality")
    console.print("  2. Check draft→thinking→response flow for hard questions")
    console.print("  3. If satisfied, proceed with full generation")
    console.print("  4. Estimated cost for full run: $13.14")
    
    console.print(f"\n[green]✓ Test files saved to:[/green] {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
