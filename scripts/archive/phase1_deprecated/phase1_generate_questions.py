#!/usr/bin/env python3
"""
Phase 1: Generate 60K Synthetic Training Questions
====================================================

Generates 60K training questions across 8 domains with 40/60 easy/hard split:
- Coding (10K): 4K easy, 6K hard - DeepSeek V3
- Math (10K): 4K easy, 6K hard - DeepSeek V3
- Tool Use (10K): 4K easy, 6K hard - DeepSeek V3
- Reasoning (10K): 4K easy, 6K hard - LLAMA-405B
- Reading (5K): 2K easy, 3K hard - LLAMA-405B
- Summarization (5K): 2K easy, 3K hard - LLAMA-405B
- Common Sense (5K): 2K easy, 3K hard - LLAMA-405B
- Instruction (5K): 2K easy, 3K hard - LLAMA-405B

Models used (via OpenRouter):
- DeepSeek V3: deepseek/deepseek-chat (FREE)
- LLAMA-405B: meta-llama/llama-3.3-70b-instruct (FREE)

Output: data/phase1/questions_60k.jsonl
Cost: $0 (FREE models)
Duration: 4-6 hours
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
    "deepseek": "deepseek/deepseek-chat",  # FREE
    "llama405b": "meta-llama/llama-3.3-70b-instruct"  # FREE
}

# Domain configuration: (total, easy, hard, model)
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

# Question generation prompts
EASY_PROMPT_TEMPLATE = """You are an expert AI training data generator. Generate {count} EASY {domain} questions.

EASY difficulty means:
- Straightforward, clear requirements
- Single-step or simple multi-step solutions
- Common patterns and standard approaches
- Suitable for beginners to intermediate level

{domain_specific_guidance}

CRITICAL JSON FORMATTING RULES:
- NO LaTeX notation (no \\(, \\), \\frac, etc.) - use plain text: "x^2 + 1" not "\\(x^2 + 1\\)"
- NO unescaped quotes or backslashes in questions
- Use Unicode for special characters if needed (π, ², ³, etc.)
- Keep questions as plain text strings

Generate EXACTLY {count} questions in this JSON array format:
[
  {{"question": "...", "difficulty": "easy", "domain": "{domain}"}},
  {{"question": "...", "difficulty": "easy", "domain": "{domain}"}}
]

Output ONLY valid JSON array, no markdown, no other text."""

HARD_PROMPT_TEMPLATE = """You are an expert AI training data generator. Generate {count} HARD {domain} questions.

HARD difficulty means:
- Complex, multi-step reasoning required
- Edge cases and non-standard scenarios
- Requires deep understanding and creativity
- Suitable for advanced level

{domain_specific_guidance}

CRITICAL JSON FORMATTING RULES:
- NO LaTeX notation (no \\(, \\), \\frac, etc.) - use plain text: "x^2 + 1" not "\\(x^2 + 1\\)"
- NO unescaped quotes or backslashes in questions
- Use Unicode for special characters if needed (π, ², ³, etc.)
- Keep questions as plain text strings

Generate EXACTLY {count} questions in this JSON array format:
[
  {{"question": "...", "difficulty": "hard", "domain": "{domain}"}},
  {{"question": "...", "difficulty": "hard", "domain": "{domain}"}}
]

Output ONLY valid JSON array, no markdown, no other text."""

# Domain-specific guidance
DOMAIN_GUIDANCE = {
    "Coding": """
Examples:
- EASY: "Write a function to reverse a string", "Check if a number is prime"
- HARD: "Implement a LRU cache with O(1) operations", "Design a thread-safe rate limiter"
Cover: algorithms, data structures, system design, debugging, optimization.
""",
    "Math": """
Examples:
- EASY: "Calculate compound interest over 5 years", "Find the area of a triangle"
- HARD: "Prove Fermat's Last Theorem for n=3", "Solve this differential equation system"
Cover: arithmetic, algebra, geometry, calculus, statistics, proofs.
""",
    "Tool Use": """
Examples:
- EASY: "How to install a Python package using pip", "Create a Git branch"
- HARD: "Set up CI/CD pipeline with Docker and Kubernetes", "Debug memory leak in production"
Cover: CLI tools, APIs, libraries, frameworks, DevOps, troubleshooting.
""",
    "Reasoning": """
Examples:
- EASY: "If all A are B, and C is A, is C also B?", "What comes next: 2, 4, 8, 16, ?"
- HARD: "Resolve this ethical dilemma with multiple stakeholders", "Predict outcomes of complex system"
Cover: logic, deduction, induction, analogies, causal reasoning, ethics.
""",
    "Reading": """
Examples:
- EASY: "What is the main idea of this paragraph?", "Who is the protagonist?"
- HARD: "Analyze the author's implicit assumptions", "Compare themes across these 3 texts"
Cover: comprehension, inference, analysis, synthesis, interpretation.
""",
    "Summarization": """
Examples:
- EASY: "Summarize this news article in 2 sentences", "Extract key points from this report"
- HARD: "Synthesize insights from 10 research papers", "Create executive summary with implications"
Cover: extraction, condensation, synthesis, prioritization, clarity.
""",
    "Common Sense": """
Examples:
- EASY: "Why do people use umbrellas when it rains?", "What happens if you don't water plants?"
- HARD: "Predict social dynamics in this scenario", "Explain cultural norms and their origins"
Cover: physical world, social norms, human behavior, cause-effect, intuition.
""",
    "Instruction": """
Examples:
- EASY: "List 3 ways to save energy at home", "Explain how to make a sandwich"
- HARD: "Create comprehensive guide to start a business", "Design curriculum for advanced topic"
Cover: how-to, procedures, guidelines, recipes, tutorials, step-by-step.
"""
}


def call_openrouter(model: str, prompt: str, temperature: float = 0.7, max_retries: int = 3) -> str:
    """Call OpenRouter API with retry logic."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dkeviv/Cogumi-LLM",
        "X-Title": "Cogumi-LLM Phase 1 Training"
    }
    
    payload = {
        "model": MODELS[model],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 8000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            console.print(f"[yellow]API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...[/yellow]")
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")


def parse_json_response(response: str) -> List[Dict]:
    """Parse JSON response, handling potential formatting issues."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()
    
    # Clean up common LaTeX issues that break JSON
    # Replace LaTeX delimiters with plain text markers
    response = response.replace('\\(', '').replace('\\)', '')
    response = response.replace('\\[', '').replace('\\]', '')
    # Fix common LaTeX commands - remove backslashes
    response = response.replace('\\frac', 'frac')
    response = response.replace('\\sqrt', 'sqrt')
    response = response.replace('\\sum', 'sum')
    response = response.replace('\\int', 'int')
    response = response.replace('\\lim', 'lim')
    # Remove other backslashes that aren't part of escape sequences
    import re
    response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
    
    # Try parsing again
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse JSON: {e}[/red]")
        console.print(f"[yellow]Response (first 500 chars): {response[:500]}...[/yellow]")
        return []


def generate_questions_for_domain(domain: str, difficulty: str, count: int, model: str, batch_size: int = 100) -> List[Dict]:
    """Generate questions for a specific domain and difficulty level."""
    all_questions = []
    remaining = count
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{domain} ({difficulty})", total=count)
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            # Prepare prompt
            template = EASY_PROMPT_TEMPLATE if difficulty == "easy" else HARD_PROMPT_TEMPLATE
            prompt = template.format(
                count=current_batch,
                domain=domain,
                domain_specific_guidance=DOMAIN_GUIDANCE[domain]
            )
            
            # Generate questions
            try:
                response = call_openrouter(model, prompt)
                questions = parse_json_response(response)
                
                if questions:
                    # Add metadata
                    for q in questions:
                        q["domain"] = domain
                        q["difficulty"] = difficulty
                        q["model_used"] = MODELS[model]
                        q["generated_at"] = datetime.now().isoformat()
                    
                    all_questions.extend(questions[:current_batch])  # Limit to requested count
                    progress.advance(task, len(questions[:current_batch]))
                    remaining -= len(questions[:current_batch])
                else:
                    console.print(f"[yellow]No questions parsed, retrying batch...[/yellow]")
                    time.sleep(2)
            
            except Exception as e:
                console.print(f"[red]Error generating batch: {e}[/red]")
                time.sleep(5)  # Wait before retry
    
    return all_questions


def main():
    """Main execution function."""
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Phase 1: Generate 60K Training Questions[/bold cyan]\n"
        "[white]Distribution: 40% Easy (24K), 60% Hard (36K)[/white]\n"
        "[green]Models: DeepSeek V3 (FREE) + LLAMA-405B (FREE)[/green]",
        border_style="cyan"
    ))
    
    # Create output directory
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "questions_60k.jsonl"
    
    # Summary table
    table = Table(title="Question Generation Plan", show_header=True, header_style="bold magenta")
    table.add_column("Domain", style="cyan", width=15)
    table.add_column("Total", justify="right", style="white")
    table.add_column("Easy (40%)", justify="right", style="green")
    table.add_column("Hard (60%)", justify="right", style="yellow")
    table.add_column("Model", style="blue")
    
    total_questions = 0
    for domain, (total, easy, hard, model) in DOMAIN_CONFIG.items():
        table.add_row(domain, str(total), str(easy), str(hard), MODELS[model])
        total_questions += total
    
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_questions}[/bold]", "[bold]24,000[/bold]", "[bold]36,000[/bold]", "Mixed")
    console.print(table)
    console.print()
    
    # Confirm start
    start_time = time.time()
    all_questions = []
    
    # Generate questions for each domain
    for domain, (total, easy_count, hard_count, model) in DOMAIN_CONFIG.items():
        console.print(f"\n[bold blue]━━━ {domain} ({total:,} questions) ━━━[/bold blue]")
        
        # Generate easy questions
        easy_questions = generate_questions_for_domain(domain, "easy", easy_count, model)
        all_questions.extend(easy_questions)
        console.print(f"[green]✓ Generated {len(easy_questions):,} easy questions[/green]")
        
        # Generate hard questions
        hard_questions = generate_questions_for_domain(domain, "hard", hard_count, model)
        all_questions.extend(hard_questions)
        console.print(f"[yellow]✓ Generated {len(hard_questions):,} hard questions[/yellow]")
        
        # Save checkpoint after each domain
        checkpoint_file = output_dir / f"checkpoint_{domain.lower().replace(' ', '_')}.jsonl"
        with open(checkpoint_file, 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
        console.print(f"[dim]Checkpoint saved: {checkpoint_file}[/dim]\n")
    
    # Save final output
    with open(output_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    # Summary
    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[bold green]✓ Generation Complete![/bold green]\n\n"
        f"[white]Total Questions: {len(all_questions):,}[/white]\n"
        f"[white]Output: {output_file}[/white]\n"
        f"[white]Duration: {elapsed/3600:.2f} hours[/white]\n"
        f"[white]Cost: $0 (FREE models)[/white]",
        border_style="green"
    ))
    
    # Distribution breakdown
    easy_count = sum(1 for q in all_questions if q['difficulty'] == 'easy')
    hard_count = sum(1 for q in all_questions if q['difficulty'] == 'hard')
    console.print(f"\n[cyan]Easy: {easy_count:,} ({easy_count/len(all_questions)*100:.1f}%)[/cyan]")
    console.print(f"[yellow]Hard: {hard_count:,} ({hard_count/len(all_questions)*100:.1f}%)[/yellow]")


if __name__ == "__main__":
    main()
