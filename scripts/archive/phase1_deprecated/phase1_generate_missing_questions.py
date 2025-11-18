#!/usr/bin/env python3
"""
Phase 1: Generate Missing Questions to Reach 60K
=================================================

Generate the 15,300 missing questions to complete 60K target.
Uses parallel async generation for speed.
"""

import asyncio
import aiohttp
import json
import re
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

# Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"

MAX_CONCURRENT = 20
BATCH_SIZE = 20

# What we need to generate (calculated from dedup output)
MISSING = {
    "Coding": {"easy": 2409, "hard": 892, "model": "deepseek/deepseek-chat"},
    "Math": {"easy": 1235, "hard": 2069, "model": "deepseek/deepseek-chat"},
    "Tool Use": {"easy": 1763, "hard": 125, "model": "deepseek/deepseek-chat"},
    "Reasoning": {"easy": 1722, "hard": 2333, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Reading": {"easy": 295, "hard": 296, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Summarization": {"easy": 91, "hard": 194, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Common Sense": {"easy": 977, "hard": 157, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Instruction": {"easy": 721, "hard": 21, "model": "meta-llama/llama-3.3-70b-instruct"},
}

DOMAIN_INSTRUCTIONS = {
    "Coding": "programming, algorithms, debugging, code review, software design",
    "Math": "arithmetic, algebra, calculus, geometry, probability, statistics",
    "Tool Use": "using APIs, command-line tools, libraries, frameworks",
    "Reasoning": "logical reasoning, analytical thinking, problem-solving",
    "Reading": "reading comprehension, text analysis, understanding passages",
    "Summarization": "summarizing text, extracting key points, concise explanations",
    "Common Sense": "common sense reasoning, everyday knowledge, practical wisdom",
    "Instruction": "following instructions, task completion, procedural knowledge",
}

def create_prompt(domain: str, difficulty: str, batch_size: int) -> str:
    """Create prompt for question generation."""
    diff_desc = "simple" if difficulty == "easy" else "complex and challenging"
    
    return f"""Generate {batch_size} {diff_desc} {domain} questions.

Domain: {domain} ({DOMAIN_INSTRUCTIONS[domain]})
Difficulty: {difficulty.upper()}

Requirements:
- {"EASY: Direct questions with short answers (1-3 sentences)" if difficulty == "easy" else "HARD: Complex questions requiring detailed reasoning and multi-step solutions"}
- Diverse topics within {domain}
- Real-world applicable
- Clear and unambiguous
- NO LaTeX (use plain text: x^2 not \\(x^2\\)). Use Unicode: π ² ³ √ ∫

Return JSON array ONLY:
[
  {{"question": "...", "difficulty": "{difficulty}"}},
  ...
]"""

async def call_api_async(session, model, prompt, semaphore):
    """Call API with rate limiting."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as response:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            console.print(f"[red]API Error: {e}[/red]")
            return None

def quick_parse_json(response):
    """Fast JSON parsing with cleanup."""
    if not response:
        return []
    
    try:
        # Remove markdown code blocks
        response = re.sub(r'```json\s*|\s*```', '', response)
        # Remove LaTeX escapes
        response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
        return json.loads(response)
    except:
        return []

async def generate_batch_async(session, domain, difficulty, model, batch_size, semaphore):
    """Generate one batch of questions."""
    prompt = create_prompt(domain, difficulty, batch_size)
    response = await call_api_async(session, model, prompt, semaphore)
    questions = quick_parse_json(response)
    
    # Add metadata
    for q in questions:
        q["domain"] = domain
        q["model_used"] = model
        q["generated_at"] = datetime.utcnow().isoformat()
    
    return questions

async def generate_missing_for_domain_difficulty(domain, difficulty, count, model):
    """Generate missing questions for one domain+difficulty."""
    num_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
    
    console.print(f"[cyan]Generating {count} {difficulty} {domain} questions ({num_batches} batches)...[/cyan]")
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        tasks = [
            generate_batch_async(session, domain, difficulty, model, 
                                min(BATCH_SIZE, count - i * BATCH_SIZE), semaphore)
            for i in range(num_batches)
        ]
        
        all_questions = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(f"{domain} {difficulty}", total=num_batches)
            
            for coro in asyncio.as_completed(tasks):
                batch = await coro
                all_questions.extend(batch)
                progress.advance(task)
        
        return all_questions

async def generate_all_missing():
    """Generate all missing questions."""
    all_new_questions = []
    
    for domain, needs in MISSING.items():
        for difficulty in ["easy", "hard"]:
            count = needs[difficulty]
            if count > 0:
                model = needs["model"]
                questions = await generate_missing_for_domain_difficulty(
                    domain, difficulty, count, model
                )
                all_new_questions.extend(questions)
    
    return all_new_questions

def main():
    console.print("[bold cyan]Phase 1: Generate Missing Questions[/bold cyan]\n")
    
    # Calculate totals
    total_needed = sum(d["easy"] + d["hard"] for d in MISSING.values())
    console.print(f"[yellow]Need to generate: {total_needed:,} questions[/yellow]\n")
    
    # Generate
    new_questions = asyncio.run(generate_all_missing())
    
    # Combine with existing
    existing_file = Path("data/phase1/questions_60k_clean.jsonl")
    output_file = Path("data/phase1/questions_60k_complete.jsonl")
    
    console.print(f"\n[yellow]Merging with existing {existing_file.name}...[/yellow]")
    
    # Load existing
    existing = []
    with open(existing_file, 'r') as f:
        for line in f:
            existing.append(json.loads(line))
    
    # Combine
    all_questions = existing + new_questions
    
    # Save
    with open(output_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"\n[bold green]✓ Complete! Saved to:[/bold green]")
    console.print(f"  {output_file}")
    console.print(f"\n[bold]Stats:[/bold]")
    console.print(f"  Existing: {len(existing):,}")
    console.print(f"  New: {len(new_questions):,}")
    console.print(f"  Total: {len(all_questions):,}")

if __name__ == "__main__":
    main()
