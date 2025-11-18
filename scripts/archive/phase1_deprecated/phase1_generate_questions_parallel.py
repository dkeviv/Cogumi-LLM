#!/usr/bin/env python3
"""
Phase 1: Generate 60K Questions - PARALLEL VERSION (10-20x faster)
===================================================================

SPEED IMPROVEMENTS:
- Async/parallel API calls (20 concurrent requests)
- Smaller batches (20 questions per call instead of 50-100)
- Faster timeout handling
- Simple regex-based cleanup (no complex parsing)

Expected: 30-45 minutes total (vs 12+ hours sequential)
"""

import json
import os
import re
import asyncio
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()

# Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 20  # Run 20 API calls in parallel
BATCH_SIZE = 20  # Small batches for faster responses

MODELS = {
    "deepseek": "deepseek/deepseek-chat",
    "llama405b": "meta-llama/llama-3.3-70b-instruct"
}

DOMAIN_CONFIG = {
    "Coding": (10000, 4000, 6000, "deepseek"),
    "Math": (10000, 4000, 6000, "deepseek"),
    "Tool Use": (10000, 4000, 6000, "deepseek"),
    "Reasoning": (10000, 4000, 6000, "llama405b"),
    "Reading": (5000, 2000, 3000, "llama405b"),
    "Summarization": (5000, 2000, 3000, "llama405b"),
    "Common Sense": (5000, 2000, 3000, "llama405b"),
    "Instruction": (5000, 2000, 3000, "llama405b"),
}

def get_prompt(count: int, difficulty: str, domain: str) -> str:
    """Generate prompt dynamically."""
    diff_desc = "easy: simple, straightforward" if difficulty == "easy" else "hard: complex, multi-step, advanced"
    return f"""Generate {count} {difficulty} {domain} questions.

{difficulty} = {diff_desc}

CRITICAL: Output ONLY valid JSON array. NO LaTeX (use plain text: x^2 not \\(x^2\\)). Use Unicode: π ² ³ √ ∫

[
  {{"question": "...", "difficulty": "{difficulty}", "domain": "{domain}"}},
  {{"question": "...", "difficulty": "{difficulty}", "domain": "{domain}"}}
]"""


async def call_api_async(session: aiohttp.ClientSession, model: str, prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Async API call with semaphore limiting concurrency."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODELS[model],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 4000
        }
        
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return ""
        except Exception:
            return ""


def quick_parse_json(response: str) -> List[Dict]:
    """Fast regex-based JSON extraction."""
    if not response:
        return []
    
    # Remove markdown
    response = re.sub(r'```json\s*|\s*```', '', response)
    
    # Aggressive cleanup: remove all backslashes except valid escapes
    response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
    
    try:
        data = json.loads(response)
        return data if isinstance(data, list) else []
    except:
        return []


async def generate_batch_async(session: aiohttp.ClientSession, domain: str, difficulty: str, 
                               count: int, model: str, semaphore: asyncio.Semaphore) -> List[Dict]:
    """Generate one batch asynchronously."""
    prompt = get_prompt(count, difficulty, domain)
    response = await call_api_async(session, model, prompt, semaphore)
    questions = quick_parse_json(response)
    
    # Add metadata
    for q in questions[:count]:
        q["domain"] = domain
        q["difficulty"] = difficulty
        q["model_used"] = MODELS[model]
        q["generated_at"] = datetime.now().isoformat()
    
    return questions[:count]


async def generate_domain_parallel(domain: str, difficulty: str, count: int, model: str) -> List[Dict]:
    """Generate all questions for a domain/difficulty in parallel."""
    num_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_batches):
            batch_count = min(BATCH_SIZE, count - i * BATCH_SIZE)
            task = generate_batch_async(session, domain, difficulty, batch_count, model, semaphore)
            tasks.append(task)
        
        # Run all batches in parallel with progress
        all_questions = []
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(), 
                     TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
            task_id = progress.add_task(f"{domain} ({difficulty})", total=num_batches)
            
            for coro in asyncio.as_completed(tasks):
                batch = await coro
                all_questions.extend(batch)
                progress.advance(task_id)
        
        return all_questions


def load_existing_questions(output_dir: Path) -> Dict[str, List[Dict]]:
    """Load existing questions by domain."""
    by_domain = {}
    for cp_file in output_dir.glob("checkpoint_*.jsonl"):
        with open(cp_file, 'r') as f:
            for line in f:
                try:
                    q = json.loads(line)
                    domain = q.get('domain')
                    if domain:
                        by_domain.setdefault(domain, []).append(q)
                except:
                    pass
    return by_domain


async def main():
    """Main execution."""
    console.print(Panel.fit(
        "[bold cyan]Phase 1: Parallel Question Generation (20x faster!)[/bold cyan]\n"
        "[green]Expected: 30-45 minutes total[/green]",
        border_style="cyan"
    ))
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing
    existing_by_domain = load_existing_questions(output_dir)
    total_existing = sum(len(qs) for qs in existing_by_domain.values())
    console.print(f"\n[green]Loaded {total_existing:,} existing questions[/green]\n")
    
    start_time = time.time()
    all_questions = []
    
    # Add existing questions
    for questions in existing_by_domain.values():
        all_questions.extend(questions)
    
    # Generate missing questions for each domain
    for domain, (total, easy_count, hard_count, model) in DOMAIN_CONFIG.items():
        existing = existing_by_domain.get(domain, [])
        existing_easy = [q for q in existing if q.get('difficulty') == 'easy']
        existing_hard = [q for q in existing if q.get('difficulty') == 'hard']
        
        if len(existing_easy) >= easy_count and len(existing_hard) >= hard_count:
            console.print(f"[green]✓ {domain}: {len(existing):,} (COMPLETE)[/green]")
            continue
        
        console.print(f"\n[bold blue]━━━ {domain} ━━━[/bold blue]")
        
        # Generate easy
        if len(existing_easy) < easy_count:
            needed = easy_count - len(existing_easy)
            console.print(f"[dim]Need {needed:,} easy questions[/dim]")
            easy_qs = await generate_domain_parallel(domain, "easy", needed, model)
            all_questions.extend(easy_qs)
            console.print(f"[green]✓ Generated {len(easy_qs):,} easy[/green]")
        
        # Generate hard
        if len(existing_hard) < hard_count:
            needed = hard_count - len(existing_hard)
            console.print(f"[dim]Need {needed:,} hard questions[/dim]")
            hard_qs = await generate_domain_parallel(domain, "hard", needed, model)
            all_questions.extend(hard_qs)
            console.print(f"[yellow]✓ Generated {len(hard_qs):,} hard[/yellow]")
        
        # Save checkpoint
        checkpoint = output_dir / f"checkpoint_{domain.lower().replace(' ', '_')}.jsonl"
        with open(checkpoint, 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
        console.print(f"[dim]Checkpoint: {checkpoint}[/dim]")
    
    # Final save
    final_output = output_dir / "questions_60k.jsonl"
    with open(final_output, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[bold green]✓ Complete![/bold green]\n\n"
        f"[white]Total: {len(all_questions):,} questions[/white]\n"
        f"[white]Output: {final_output}[/white]\n"
        f"[white]Duration: {elapsed/60:.1f} minutes[/white]",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())
