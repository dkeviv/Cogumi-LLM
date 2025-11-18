#!/usr/bin/env python3
"""
Phase 1: Generate 60K Questions (V3 - PARALLEL & FAST)
=======================================================

MAJOR IMPROVEMENTS:
- Parallel API calls (10x faster!)
- Larger batches (200 questions per call)
- Concurrent requests (10 simultaneous)
- Better error handling
- Resume from checkpoints

Expected: 30-60 minutes total (vs 6+ hours)
"""

import json
import os
import re
import time
import asyncio
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

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

PROMPT_TEMPLATE = """Generate {count} {difficulty} {domain} questions.

{difficulty} = {"easy: straightforward, beginner" if difficulty == "easy" else "hard: complex, advanced"}

RULES:
- Plain text only (NO LaTeX: use x^2 not \\(x^2\\))
- Unicode math: π, ², ³, ∫, √, ∑
- Valid JSON strings only
- NO backslashes

Output JSON array:
[{{"question": "...", "difficulty": "{difficulty}", "domain": "{domain}"}}]"""


def call_api(model: str, prompt: str, max_retries: int = 2) -> str:
    """Single API call with timeout."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dkeviv/Cogumi-LLM"
    }
    
    payload = {
        "model": MODELS[model],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 16000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                return ""
            time.sleep(1)
    return ""


def parse_json(response: str) -> List[Dict]:
    """Parse JSON with aggressive cleanup."""
    if not response:
        return []
    
    # Extract from markdown
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()
    
    # Remove LaTeX
    response = re.sub(r'\\[\(\)\[\]]', '', response)
    response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
    
    try:
        data = json.loads(response)
        return data if isinstance(data, list) else []
    except:
        return []


def generate_batch(domain: str, difficulty: str, count: int, model: str) -> List[Dict]:
    """Generate one batch of questions."""
    prompt = PROMPT_TEMPLATE.format(count=count, difficulty=difficulty, domain=domain)
    response = call_api(model, prompt)
    questions = parse_json(response)
    
    # Add metadata
    for q in questions[:count]:
        q.update({
            "domain": domain,
            "difficulty": difficulty,
            "model_used": MODELS[model],
            "generated_at": datetime.now().isoformat()
        })
    
    return questions[:count]


def parallel_generate(domain: str, difficulty: str, total_needed: int, model: str, 
                     batch_size: int = 200, max_workers: int = 10) -> List[Dict]:
    """Generate questions in parallel with multiple concurrent API calls."""
    all_questions = []
    num_batches = (total_needed + batch_size - 1) // batch_size
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{domain} ({difficulty})", total=total_needed)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = []
            for i in range(num_batches):
                current_batch_size = min(batch_size, total_needed - len(all_questions))
                if current_batch_size <= 0:
                    break
                future = executor.submit(generate_batch, domain, difficulty, current_batch_size, model)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    questions = future.result()
                    if questions:
                        all_questions.extend(questions)
                        progress.advance(task, len(questions))
                except Exception as e:
                    console.print(f"[dim red]Batch error: {str(e)[:50]}[/dim red]")
    
    return all_questions[:total_needed]


def load_checkpoint(checkpoint_file: Path) -> List[Dict]:
    """Load existing questions."""
    if not checkpoint_file.exists():
        return []
    
    questions = []
    with open(checkpoint_file, 'r') as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except:
                pass
    return questions


def main():
    """Main execution."""
    console.print(Panel.fit(
        "[bold cyan]Phase 1: Generate 60K Questions (V3 - PARALLEL)[/bold cyan]\n"
        "[white]10x faster with concurrent API calls[/white]",
        border_style="cyan"
    ))
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing checkpoints
    existing_checkpoints = list(output_dir.glob("checkpoint_*.jsonl"))
    all_questions = []
    for cp in existing_checkpoints:
        all_questions.extend(load_checkpoint(cp))
    
    if all_questions:
        console.print(f"[green]Loaded {len(all_questions):,} existing questions[/green]\n")
    
    start_time = time.time()
    
    # Process each domain
    for domain, (total, easy_count, hard_count, model) in DOMAIN_CONFIG.items():
        checkpoint_file = output_dir / f"checkpoint_{domain.lower().replace(' ', '_')}.jsonl"
        
        # Check existing
        existing = [q for q in all_questions if q.get('domain') == domain]
        if len(existing) >= total:
            console.print(f"[green]✓ {domain}: {len(existing):,} COMPLETE[/green]")
            continue
        
        console.print(f"\n[bold blue]━━━ {domain} ({total:,}) ━━━[/bold blue]")
        
        # Easy questions
        existing_easy = [q for q in existing if q.get('difficulty') == 'easy']
        if len(existing_easy) < easy_count:
            needed = easy_count - len(existing_easy)
            console.print(f"[dim]Generating {needed:,} easy questions...[/dim]")
            easy_q = parallel_generate(domain, "easy", needed, model)
            all_questions.extend(easy_q)
            console.print(f"[green]✓ {len(easy_q):,} easy done[/green]")
        else:
            console.print(f"[green]✓ Easy complete[/green]")
        
        # Hard questions
        existing_hard = [q for q in existing if q.get('difficulty') == 'hard']
        if len(existing_hard) < hard_count:
            needed = hard_count - len(existing_hard)
            console.print(f"[dim]Generating {needed:,} hard questions...[/dim]")
            hard_q = parallel_generate(domain, "hard", needed, model)
            all_questions.extend(hard_q)
            console.print(f"[yellow]✓ {len(hard_q):,} hard done[/yellow]")
        else:
            console.print(f"[yellow]✓ Hard complete[/yellow]")
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
        console.print(f"[dim]Checkpoint saved[/dim]")
    
    # Final output
    final_output = output_dir / "questions_60k.jsonl"
    with open(final_output, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[bold green]✓ COMPLETE![/bold green]\n\n"
        f"[white]Total: {len(all_questions):,} questions[/white]\n"
        f"[white]Time: {elapsed/60:.1f} minutes[/white]\n"
        f"[white]Output: {final_output}[/white]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
