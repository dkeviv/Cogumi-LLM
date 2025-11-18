#!/usr/bin/env python3
"""
Phase 1: Deduplicate Questions and Balance to 60K Target
=========================================================

Removes duplicates and ensures correct distribution:
- Total: 60K questions (40% easy = 24K, 60% hard = 36K)
- 8 domains with proper split
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

TARGET_DISTRIBUTION = {
    "Coding": (10000, 4000, 6000),
    "Math": (10000, 4000, 6000),
    "Tool Use": (10000, 4000, 6000),
    "Reasoning": (10000, 4000, 6000),
    "Reading": (5000, 2000, 3000),
    "Summarization": (5000, 2000, 3000),
    "Common Sense": (5000, 2000, 3000),
    "Instruction": (5000, 2000, 3000),
}

def load_and_deduplicate(input_file: Path):
    """Load questions and remove duplicates."""
    seen = set()
    questions_by_domain_diff = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                q = json.loads(line)
                question_text = q.get('question', '').strip()
                domain = q.get('domain')
                difficulty = q.get('difficulty')
                
                if question_text and question_text not in seen and domain and difficulty:
                    seen.add(question_text)
                    key = (domain, difficulty)
                    questions_by_domain_diff[key].append(q)
            except:
                pass
    
    return questions_by_domain_diff

def balance_dataset(questions_by_domain_diff):
    """Balance to target distribution."""
    balanced = []
    stats = Counter()
    
    for domain, (total, easy_target, hard_target) in TARGET_DISTRIBUTION.items():
        easy_key = (domain, "easy")
        hard_key = (domain, "hard")
        
        easy_available = questions_by_domain_diff.get(easy_key, [])
        hard_available = questions_by_domain_diff.get(hard_key, [])
        
        # Take what we need (or all if not enough)
        easy_selected = easy_available[:easy_target]
        hard_selected = hard_available[:hard_target]
        
        balanced.extend(easy_selected)
        balanced.extend(hard_selected)
        
        stats[domain] = (len(easy_selected), len(hard_selected))
    
    return balanced, stats

def main():
    console.print("[bold cyan]Phase 1: Deduplicate & Balance Questions[/bold cyan]\n")
    
    input_file = Path("data/phase1/questions_60k.jsonl")
    output_file = Path("data/phase1/questions_60k_clean.jsonl")
    
    # Load and deduplicate
    console.print("[yellow]Loading and deduplicating...[/yellow]")
    questions_by_domain_diff = load_and_deduplicate(input_file)
    
    # Show what we have
    console.print("\n[bold]Available Unique Questions:[/bold]")
    table = Table(show_header=True)
    table.add_column("Domain", style="cyan")
    table.add_column("Easy", justify="right", style="green")
    table.add_column("Hard", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="white")
    
    for domain in TARGET_DISTRIBUTION.keys():
        easy_key = (domain, "easy")
        hard_key = (domain, "hard")
        easy_count = len(questions_by_domain_diff.get(easy_key, []))
        hard_count = len(questions_by_domain_diff.get(hard_key, []))
        table.add_row(domain, str(easy_count), str(hard_count), str(easy_count + hard_count))
    
    console.print(table)
    
    # Balance to target
    console.print("\n[yellow]Balancing to target distribution...[/yellow]")
    balanced, stats = balance_dataset(questions_by_domain_diff)
    
    # Show final distribution
    console.print("\n[bold]Final Balanced Dataset:[/bold]")
    table2 = Table(show_header=True)
    table2.add_column("Domain", style="cyan")
    table2.add_column("Easy", justify="right", style="green")
    table2.add_column("Hard", justify="right", style="yellow")
    table2.add_column("Total", justify="right", style="white")
    table2.add_column("Target", justify="right", style="dim")
    
    total_easy = 0
    total_hard = 0
    for domain, (target_total, target_easy, target_hard) in TARGET_DISTRIBUTION.items():
        easy_count, hard_count = stats[domain]
        total_easy += easy_count
        total_hard += hard_count
        total_count = easy_count + hard_count
        status = "✓" if total_count == target_total else "⚠"
        table2.add_row(
            f"{status} {domain}",
            str(easy_count),
            str(hard_count),
            str(total_count),
            str(target_total)
        )
    
    table2.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_easy:,}[/bold]",
        f"[bold]{total_hard:,}[/bold]",
        f"[bold]{total_easy + total_hard:,}[/bold]",
        "[bold]60,000[/bold]"
    )
    console.print(table2)
    
    # Save balanced dataset
    with open(output_file, 'w') as f:
        for q in balanced:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"\n[bold green]✓ Saved clean balanced dataset to:[/bold green]")
    console.print(f"  {output_file}")
    console.print(f"\n[bold]Stats:[/bold]")
    console.print(f"  Total questions: {len(balanced):,}")
    console.print(f"  Easy: {total_easy:,} ({total_easy/len(balanced)*100:.1f}%)")
    console.print(f"  Hard: {total_hard:,} ({total_hard/len(balanced)*100:.1f}%)")

if __name__ == "__main__":
    main()
