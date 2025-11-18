#!/usr/bin/env python3
"""
Phase 1: Generate 60K Synthetic Training Questions (V2 - Improved)
===================================================================

IMPROVEMENTS:
- Better JSON parsing (handles LaTeX characters)
- Plain text prompts (no LaTeX notation)
- Resume capability (skips completed domains)
- Better error handling

Generates 60K training questions across 8 domains with 40/60 easy/hard split.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configuration
MODELS = {
    "deepseek": "deepseek/deepseek-chat",
    "llama405b": "meta-llama/llama-3.3-70b-instruct"
}

# Domain configuration
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

# Question generation prompts - NO LaTeX!
EASY_PROMPT_TEMPLATE = """Generate {count} EASY {domain} questions for AI training.

EASY difficulty: Straightforward, single-step or simple multi-step, common patterns, beginner to intermediate.

{domain_specific_guidance}

CRITICAL RULES:
- Use plain text only: "x^2 + 1" NOT "\\(x^2 + 1\\)"
- Use Unicode symbols: π, ², ³, ∫, √, ∑
- NO backslashes or special escapes
- Valid JSON strings only

Generate EXACTLY {count} questions as JSON array:
[
  {{"question": "...", "difficulty": "easy", "domain": "{domain}"}},
  {{"question": "...", "difficulty": "easy", "domain": "{domain}"}}
]

Output valid JSON only."""

HARD_PROMPT_TEMPLATE = """Generate {count} HARD {domain} questions for AI training.

HARD difficulty: Complex, multi-step reasoning, edge cases, non-standard scenarios, advanced level.

{domain_specific_guidance}

CRITICAL RULES:
- Use plain text only: "x^2 + 1" NOT "\\(x^2 + 1\\)"
- Use Unicode symbols: π, ², ³, ∫, √, ∑
- NO backslashes or special escapes
- Valid JSON strings only

Generate EXACTLY {count} questions as JSON array:
[
  {{"question": "...", "difficulty": "hard", "domain": "{domain}"}},
  {{"question": "...", "difficulty": "hard", "domain": "{domain}"}}
]

Output valid JSON only."""

# Domain guidance
DOMAIN_GUIDANCE = {
    "Coding": "Examples: algorithms, data structures, system design, debugging, optimization. Use clear plain text.",
    "Math": "Examples: algebra, geometry, calculus, statistics, proofs. Use plain text: x^2, sqrt(x), integral, π, etc.",
    "Tool Use": "Examples: CLI tools, APIs, libraries, frameworks, DevOps, troubleshooting.",
    "Reasoning": "Examples: logic, deduction, induction, analogies, causal reasoning, ethics.",
    "Reading": "Examples: comprehension, inference, analysis, synthesis, interpretation.",
    "Summarization": "Examples: extraction, condensation, synthesis, prioritization, clarity.",
    "Common Sense": "Examples: physical world, social norms, human behavior, cause-effect, intuition.",
    "Instruction": "Examples: how-to, procedures, guidelines, recipes, tutorials, step-by-step."
}


def call_openrouter(model: str, prompt: str, max_retries: int = 3) -> str:
    """Call OpenRouter API with retry logic."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dkeviv/Cogumi-LLM",
        "X-Title": "Cogumi-LLM Phase 1"
    }
    
    payload = {
        "model": MODELS[model],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 8000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            console.print(f"[yellow]API error (attempt {attempt+1}/{max_retries}): {str(e)[:100]}[/yellow]")
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")


def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON response with aggressive cleanup."""
    # Extract JSON from markdown
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()
    
    # Aggressive LaTeX removal
    response = re.sub(r'\\[\(\)\[\]]', '', response)  # Remove \( \) \[ \]
    response = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', lambda m: m.group(0).replace('\\', ''), response)  # Remove \ from commands
    response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)  # Remove other backslashes
    
    # Try parsing
    try:
        data = json.loads(response)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        console.print(f"[dim red]Parse error: {str(e)[:80]}[/dim red]")
        return []


def generate_questions_for_domain(domain: str, difficulty: str, count: int, model: str, batch_size: int = 50) -> List[Dict]:
    """Generate questions for a specific domain and difficulty."""
    all_questions = []
    remaining = count
    consecutive_failures = 0
    max_failures = 10
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{domain} ({difficulty})", total=count)
        
        while remaining > 0 and consecutive_failures < max_failures:
            current_batch = min(batch_size, remaining)
            
            template = EASY_PROMPT_TEMPLATE if difficulty == "easy" else HARD_PROMPT_TEMPLATE
            prompt = template.format(
                count=current_batch,
                domain=domain,
                domain_specific_guidance=DOMAIN_GUIDANCE[domain]
            )
            
            try:
                response = call_openrouter(model, prompt)
                questions = parse_json_response(response)
                
                if questions and len(questions) > 0:
                    # Add metadata
                    for q in questions[:current_batch]:
                        q["domain"] = domain
                        q["difficulty"] = difficulty
                        q["model_used"] = MODELS[model]
                        q["generated_at"] = datetime.now().isoformat()
                    
                    all_questions.extend(questions[:current_batch])
                    progress.advance(task, len(questions[:current_batch]))
                    remaining -= len(questions[:current_batch])
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    console.print(f"[yellow]No questions parsed ({consecutive_failures}/{max_failures})[/yellow]")
                    time.sleep(2)
            
            except Exception as e:
                consecutive_failures += 1
                console.print(f"[red]Error: {str(e)[:100]} ({consecutive_failures}/{max_failures})[/red]")
                time.sleep(5)
        
        if consecutive_failures >= max_failures:
            console.print(f"[bold red]Too many failures, stopping {domain} {difficulty}[/bold red]")
    
    return all_questions


def load_existing_questions(checkpoint_file: Path) -> List[Dict]:
    """Load existing questions from checkpoint."""
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
        "[bold cyan]Phase 1: Generate 60K Training Questions (V2)[/bold cyan]\n"
        "[white]Improved JSON parsing + Resume capability[/white]",
        border_style="cyan"
    ))
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / "questions_60k.jsonl"
    
    # Check existing progress
    existing_checkpoints = list(output_dir.glob("checkpoint_*.jsonl"))
    console.print(f"\n[dim]Found {len(existing_checkpoints)} existing checkpoint(s)[/dim]\n")
    
    start_time = time.time()
    all_questions = []
    
    # Load existing questions from all checkpoints
    for cp in existing_checkpoints:
        questions = load_existing_questions(cp)
        all_questions.extend(questions)
    
    if all_questions:
        console.print(f"[green]Loaded {len(all_questions):,} existing questions from checkpoints[/green]\n")
    
    # Generate questions for each domain
    for domain, (total, easy_count, hard_count, model) in DOMAIN_CONFIG.items():
        checkpoint_file = output_dir / f"checkpoint_{domain.lower().replace(' ', '_')}.jsonl"
        
        # Check if domain already completed
        existing_for_domain = [q for q in all_questions if q.get('domain') == domain]
        if len(existing_for_domain) >= total:
            console.print(f"[green]✓ {domain}: {len(existing_for_domain):,} questions (COMPLETE, SKIPPING)[/green]\n")
            continue
        
        console.print(f"\n[bold blue]━━━ {domain} ({total:,} questions) ━━━[/bold blue]")
        
        # Easy questions
        existing_easy = [q for q in existing_for_domain if q.get('difficulty') == 'easy']
        if len(existing_easy) < easy_count:
            needed = easy_count - len(existing_easy)
            console.print(f"[dim]Need {needed:,} more easy questions[/dim]")
            easy_questions = generate_questions_for_domain(domain, "easy", needed, model)
            all_questions.extend(easy_questions)
            console.print(f"[green]✓ Generated {len(easy_questions):,} easy questions[/green]")
        else:
            console.print(f"[green]✓ Easy questions complete ({len(existing_easy):,})[/green]")
        
        # Hard questions
        existing_hard = [q for q in existing_for_domain if q.get('difficulty') == 'hard']
        if len(existing_hard) < hard_count:
            needed = hard_count - len(existing_hard)
            console.print(f"[dim]Need {needed:,} more hard questions[/dim]")
            hard_questions = generate_questions_for_domain(domain, "hard", needed, model)
            all_questions.extend(hard_questions)
            console.print(f"[yellow]✓ Generated {len(hard_questions):,} hard questions[/yellow]")
        else:
            console.print(f"[yellow]✓ Hard questions complete ({len(existing_hard):,})[/yellow]")
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
        console.print(f"[dim]Checkpoint saved: {checkpoint_file}[/dim]\n")
    
    # Save final output
    with open(final_output, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[bold green]✓ Generation Complete![/bold green]\n\n"
        f"[white]Total Questions: {len(all_questions):,}[/white]\n"
        f"[white]Output: {final_output}[/white]\n"
        f"[white]Duration: {elapsed/3600:.2f} hours[/white]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
