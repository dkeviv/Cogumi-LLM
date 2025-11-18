#!/usr/bin/env python3
"""
Phase 1: Augment with Public Datasets

Download and integrate high-quality public datasets to fill domain gaps.
All datasets are FREE and available on HuggingFace.

Target:
- Fill gaps in under-represented domains (Common Sense, Instruction, Reading, Summarization)
- Need ~1,858 questions per domain (total ~7,430)
- Reach 60K total questions after balancing
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import Counter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datasets import load_dataset
import time

console = Console()

# Only download for LACKING domains (4 domains with <6,000 questions)
# Adding buffer for duplicates (2,500 per domain to ensure ~2,000 unique)
DATASET_CONFIG = {
    "Common Sense": [
        {"name": "commonsense_qa", "split": "train", "limit": 2500, "question_field": "question", "domain": "Common Sense"},
        {"name": "piqa", "split": "train", "limit": 2500, "question_field": "goal", "domain": "Common Sense"},
    ],
    "Instruction": [
        {"name": "tatsu-lab/alpaca", "split": "train", "limit": 2500, "question_field": "instruction", "domain": "Instruction"},
    ],
    "Reading": [
        {"name": "squad_v2", "split": "train", "limit": 2500, "question_field": "question", "domain": "Reading"},
        {"name": "coqa", "split": "train", "limit": 500, "question_field": "question", "domain": "Reading"},
    ],
    "Summarization": [
        {"name": "cnn_dailymail", "split": "train", "subset": "3.0.0", "limit": 2000, "question_field": "article", "domain": "Summarization"},
        {"name": "xsum", "split": "train", "limit": 1500, "question_field": "document", "domain": "Summarization"},
    ],
}


def extract_question(item: Dict, config: Dict) -> str:
    """Extract question from dataset item based on config."""
    field = config["question_field"]
    
    # Handle nested fields
    if "." in field:
        parts = field.split(".")
        value = item
        for part in parts:
            value = value.get(part, "")
        return str(value).strip()
    
    return str(item.get(field, "")).strip()


def format_as_question(text: str, domain: str) -> str:
    """Format text as a question if needed."""
    text = text.strip()
    
    # For summarization, create a question
    if domain == "Summarization" and not text.endswith("?"):
        return f"Summarize the following: {text[:500]}..." if len(text) > 500 else f"Summarize: {text}"
    
    # For reading comprehension, might already be a question
    if text.endswith("?"):
        return text
    
    # For coding, format as task
    if domain == "Coding":
        return f"Write a function that: {text[:200]}" if len(text) > 200 else f"Write: {text}"
    
    return text


def download_and_process_dataset(config: Dict, seen_questions: set, progress_task, progress) -> List[Dict]:
    """Download and process a single dataset."""
    questions = []
    
    try:
        console.print(f"[cyan]  Loading {config['name']}...[/cyan]")
        start_time = time.time()
        
        # Load dataset
        if "subset" in config:
            dataset = load_dataset(config["name"], config["subset"], split=config["split"], trust_remote_code=True)
        else:
            dataset = load_dataset(config["name"], split=config["split"], trust_remote_code=True)
        
        load_time = time.time() - start_time
        console.print(f"[dim]  Loaded in {load_time:.1f}s, processing {len(dataset):,} samples...[/dim]")
        
        # Sample and process
        limit = min(config["limit"], len(dataset))
        samples = dataset.shuffle(seed=42).select(range(limit))
        
        duplicates = 0
        skipped = 0
        for item in samples:
            try:
                question_text = extract_question(item, config)
                if not question_text or len(question_text) < 10:
                    skipped += 1
                    continue
                
                # Format as question
                question_text = format_as_question(question_text, config["domain"])
                
                # Check for duplicates
                q_lower = question_text.lower().strip()
                if q_lower in seen_questions:
                    duplicates += 1
                    continue
                
                questions.append({
                    "question": question_text,
                    "difficulty": "easy",  # Public datasets are typically easy-medium
                    "domain": config["domain"],
                    "model_used": f"public:{config['name']}",
                    "generated_at": None
                })
                seen_questions.add(q_lower)
                
            except Exception as e:
                skipped += 1
                continue
        
        progress.advance(progress_task)
        console.print(f"[green]  ✓ {config['name']}: {len(questions):,} unique ({duplicates} duplicates, {skipped} skipped)[/green]")
        
    except Exception as e:
        console.print(f"[red]  ✗ Failed to load {config['name']}: {e}[/red]")
        progress.advance(progress_task)
    
    return questions


def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Augment with Public Datasets[/bold cyan]")
    console.print("=" * 70)
    
    # Load existing questions from UNTRIMMED file (54,852 questions)
    existing_file = Path("data/phase1/questions_all_untrimmed.jsonl")
    console.print(f"\n[cyan]Step 1: Loading existing questions...[/cyan]")
    
    existing_questions = []
    seen_questions = set()
    
    with open(existing_file) as f:
        for line in f:
            q = json.loads(line.strip())
            existing_questions.append(q)
            seen_questions.add(q['question'].lower().strip())
    
    existing_by_domain = Counter(q['domain'] for q in existing_questions)
    
    console.print(f"[green]✓ Loaded {len(existing_questions):,} existing questions[/green]")
    console.print(f"\n[cyan]Current distribution:[/cyan]")
    for domain in sorted(existing_by_domain.keys()):
        count = existing_by_domain[domain]
        target = 6857  # Target per domain for 60K total (7,500 per domain)
        gap = target - count
        status = "⚠️ LACKING" if count < 6000 else "✅"
        console.print(f"  {domain:20s}: {count:5,} (gap: {gap:+6,}) {status}")
    
    # Download and process public datasets (ONLY for lacking domains)
    console.print(f"\n[cyan]Step 2: Downloading public datasets for lacking domains...[/cyan]")
    console.print(f"[dim]Targeting: Common Sense, Instruction, Reading, Summarization[/dim]")
    
    all_new_questions = []
    total_datasets = sum(len(configs) for configs in DATASET_CONFIG.values())
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Downloading datasets", total=total_datasets)
        
        for domain, configs in DATASET_CONFIG.items():
            console.print(f"\n[bold cyan]{domain}:[/bold cyan]")
            for config in configs:
                questions = download_and_process_dataset(config, seen_questions, task, progress)
                all_new_questions.extend(questions)
    
    # Analyze new questions
    new_by_domain = Counter(q['domain'] for q in all_new_questions)
    
    console.print(f"\n[green]✓ Step 2 Complete: Downloaded {len(all_new_questions):,} new unique questions[/green]")
    console.print(f"\n[cyan]New questions by domain:[/cyan]")
    for domain in sorted(new_by_domain.keys()):
        count = new_by_domain[domain]
        console.print(f"  {domain:20s}: {count:5,}")
    
    # Combine with existing
    all_questions = existing_questions + all_new_questions
    
    # Analyze combined
    combined_by_domain = Counter(q['domain'] for q in all_questions)
    
    console.print(f"\n[cyan]Step 3: Combined distribution:[/cyan]")
    for domain in sorted(combined_by_domain.keys()):
        old = existing_by_domain.get(domain, 0)
        new = new_by_domain.get(domain, 0)
        total = combined_by_domain[domain]
        console.print(f"  {domain:20s}: {old:5,} + {new:5,} = {total:5,}")
    
    # Save augmented dataset
    output_file = Path("data/phase1/questions_augmented_with_public.jsonl")
    console.print(f"\n[cyan]Step 4: Saving augmented dataset...[/cyan]")
    
    with open(output_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓ Saved {len(all_questions):,} questions to: {output_file} ({file_size:.1f} MB)[/green]")
    
    # Calculate token balance
    easy_count = sum(1 for q in all_questions if q['difficulty'] == 'easy')
    hard_count = len(all_questions) - easy_count
    
    easy_tokens = easy_count * 15
    hard_tokens = hard_count * 750
    total_tokens = easy_tokens + hard_tokens
    
    easy_token_pct = (easy_tokens / total_tokens * 100) if total_tokens > 0 else 0
    hard_token_pct = (hard_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    console.print(f"\n[cyan]Step 5: Token Distribution Analysis:[/cyan]")
    console.print(f"  Easy: {easy_count:,} questions → {easy_tokens:,} tokens ({easy_token_pct:.1f}%)")
    console.print(f"  Hard: {hard_count:,} questions → {hard_tokens:,} tokens ({hard_token_pct:.1f}%)")
    console.print(f"  Total: {len(all_questions):,} questions → {total_tokens:,} tokens")
    
    if 58 <= easy_token_pct <= 62:
        console.print(f"[green]✓ TOKEN BALANCE MAINTAINED![/green]")
    else:
        console.print(f"[yellow]⚠ Token balance: {easy_token_pct:.1f}% / {hard_token_pct:.1f}% (target 60/40)[/yellow]")
    
    console.print(f"\n[bold green]✅ Augmentation complete![/bold green]")
    console.print(f"[green]Total dataset: {len(all_questions):,} questions (Cost: $0)[/green]")
    
    console.print(f"\n[cyan]Next step:[/cyan]")
    console.print(f"  Run: python scripts/phase1_balance_final_60k.py")
    console.print(f"  This will balance all domains to 7,500 each → 60,000 total")


if __name__ == "__main__":
    main()
