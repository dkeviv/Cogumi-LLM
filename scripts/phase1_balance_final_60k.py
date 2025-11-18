#!/usr/bin/env python3
"""
Phase 1: Balance to Final 60K

Balance all domains to create final 60K training dataset.
Target: 7,500 questions per domain × 8 domains = 60,000 total
"""

import json
import random
from pathlib import Path
from collections import Counter
from rich.console import Console

console = Console()


def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Balance to Final 60K[/bold cyan]")
    console.print("=" * 70)
    
    # Load augmented questions
    input_file = Path("data/phase1/questions_augmented_with_public.jsonl")
    console.print(f"\n[cyan]Step 1: Loading augmented dataset...[/cyan]")
    
    questions = []
    with open(input_file) as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    by_domain = {}
    for q in questions:
        domain = q['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(q)
    
    console.print(f"[green]✓ Loaded {len(questions):,} questions[/green]")
    console.print(f"\n[cyan]Current distribution:[/cyan]")
    for domain in sorted(by_domain.keys()):
        count = len(by_domain[domain])
        target = 7500
        gap = target - count
        status = "✅" if count >= target else f"⚠️ ({gap:+,})"
        console.print(f"  {domain:20s}: {count:5,} → {target:5,} {status}")
    
    # Balance to 7,500 per domain
    console.print(f"\n[cyan]Step 2: Balancing domains to 7,500 each...[/cyan]")
    
    balanced_questions = []
    for domain in sorted(by_domain.keys()):
        domain_questions = by_domain[domain]
        target = 7500
        
        if len(domain_questions) >= target:
            # Trim to target
            selected = random.sample(domain_questions, target)
            console.print(f"  {domain:20s}: {len(domain_questions):5,} → {target:5,} (trimmed {len(domain_questions) - target:,})")
        else:
            # Keep all
            selected = domain_questions
            console.print(f"  {domain:20s}: {len(domain_questions):5,} → {len(domain_questions):5,} (kept all, {target - len(domain_questions):,} short)")
        
        balanced_questions.extend(selected)
    
    # Shuffle
    console.print(f"\n[cyan]Step 3: Shuffling for batch mixing...[/cyan]")
    random.shuffle(balanced_questions)
    
    # Save
    output_file = Path("data/phase1/questions_final_60k.jsonl")
    console.print(f"\n[cyan]Step 4: Saving final dataset...[/cyan]")
    
    with open(output_file, 'w') as f:
        for q in balanced_questions:
            f.write(json.dumps(q) + '\n')
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓ Saved {len(balanced_questions):,} questions to: {output_file} ({file_size:.1f} MB)[/green]")
    
    # Final statistics
    console.print(f"\n[cyan]Step 5: Final Statistics:[/cyan]")
    
    final_by_domain = Counter(q['domain'] for q in balanced_questions)
    for domain in sorted(final_by_domain.keys()):
        count = final_by_domain[domain]
        console.print(f"  {domain:20s}: {count:5,}")
    
    easy_count = sum(1 for q in balanced_questions if q['difficulty'] == 'easy')
    hard_count = len(balanced_questions) - easy_count
    
    easy_tokens = easy_count * 15
    hard_tokens = hard_count * 750
    total_tokens = easy_tokens + hard_tokens
    
    easy_token_pct = (easy_tokens / total_tokens * 100) if total_tokens > 0 else 0
    hard_token_pct = (hard_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    console.print(f"\n[cyan]Token Distribution:[/cyan]")
    console.print(f"  Easy: {easy_count:,} questions → {easy_tokens:,} tokens ({easy_token_pct:.1f}%)")
    console.print(f"  Hard: {hard_count:,} questions → {hard_tokens:,} tokens ({hard_token_pct:.1f}%)")
    console.print(f"  Total: {len(balanced_questions):,} questions → {total_tokens:,} tokens")
    
    if 58 <= easy_token_pct <= 62:
        console.print(f"[green]✓ TOKEN BALANCE MAINTAINED![/green]")
    else:
        console.print(f"[yellow]⚠ Token balance: {easy_token_pct:.1f}% / {hard_token_pct:.1f}% (target 60/40)[/yellow]")
    
    # Verify batch mixing
    console.print(f"\n[cyan]Step 6: Verify batch mixing (first 3 batches):[/cyan]")
    batch_size = 32
    for i in range(3):
        batch = balanced_questions[i*batch_size:(i+1)*batch_size]
        batch_domains = set(q['domain'] for q in batch)
        batch_easy = sum(1 for q in batch if q['difficulty'] == 'easy')
        batch_hard = len(batch) - batch_easy
        console.print(f"  Batch {i+1}: {batch_easy} easy, {batch_hard} hard | Domains: {len(batch_domains)}")
    
    console.print(f"\n[bold green]✅ Final 60K dataset ready![/bold green]")
    console.print(f"[green]Total: {len(balanced_questions):,} questions[/green]")
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Validate questions (optional, $9)")
    console.print(f"  2. Generate answers ($416)")
    console.print(f"  3. Train models ($39)")


if __name__ == "__main__":
    random.seed(42)
    main()
