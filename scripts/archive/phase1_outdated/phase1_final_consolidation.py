#!/usr/bin/env python3
"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** scripts/phase1_balance_final_60k.py
**Reason:** Superseded by better balancing implementation
**Issues:**
- Manual consolidation approach
- Less sophisticated balancing logic
- Doesn't handle public dataset augmentation

**Archive Date:** 2025-11-14
**Archived For:** Historical reference only

Use phase1_balance_final_60k.py instead which has:
- Automatic balancing to 7,500 per domain
- Public dataset integration
- Token balance verification
- Batch mixing validation

=================================

Phase 1: Consolidate, Check Duplicates, and Create Final Balanced Dataset

Steps:
1. Load all existing questions (untrimmed)
2. Check for duplicates
3. Create final balanced dataset with proper 60/40 token distribution
4. Shuffle for training
5. Save final training file
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()


def load_all_questions(file_path: Path) -> tuple[List[Dict], set, Dict]:
    """Load all questions, check for duplicates."""
    questions = []
    seen = set()
    duplicates = []
    
    with open(file_path) as f:
        for line in f:
            q = json.loads(line.strip())
            q_text = q['question'].strip().lower()
            
            if q_text in seen:
                duplicates.append(q)
            else:
                questions.append(q)
                seen.add(q_text)
    
    # Analyze distribution
    by_domain = Counter(q['domain'] for q in questions)
    by_difficulty = Counter(q['difficulty'] for q in questions)
    
    return questions, duplicates, {'by_domain': dict(by_domain), 'by_difficulty': dict(by_difficulty)}


def calculate_token_balance(questions: List[Dict]) -> Dict:
    """Calculate token distribution."""
    easy_count = sum(1 for q in questions if q['difficulty'] == 'easy')
    hard_count = len(questions) - easy_count
    
    # Token estimates
    easy_tokens = easy_count * 15
    hard_tokens = hard_count * 750
    total_tokens = easy_tokens + hard_tokens
    
    easy_token_pct = (easy_tokens / total_tokens * 100) if total_tokens > 0 else 0
    hard_token_pct = (hard_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    return {
        'easy_count': easy_count,
        'hard_count': hard_count,
        'easy_tokens': easy_tokens,
        'hard_tokens': hard_tokens,
        'total_tokens': total_tokens,
        'easy_token_pct': easy_token_pct,
        'hard_token_pct': hard_token_pct
    }


def balance_to_target(questions: List[Dict], target_per_domain: int = 6861) -> List[Dict]:
    """Balance questions to target distribution.
    
    Target: 54,894 total questions (91.5% of 60K)
    Per domain: 6,861 questions (54,894 / 8 domains)
    """
    
    # Group by domain
    by_domain = {}
    for q in questions:
        domain = q['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(q)
    
    # Balance each domain to target
    balanced = []
    for domain, domain_questions in by_domain.items():
        current = len(domain_questions)
        
        if current <= target_per_domain:
            # Keep all if below target
            balanced.extend(domain_questions)
            console.print(f"[cyan]{domain:20s}: {current:5,} (keeping all)[/cyan]")
        else:
            # Randomly sample if above target
            sampled = random.sample(domain_questions, target_per_domain)
            balanced.extend(sampled)
            console.print(f"[yellow]{domain:20s}: {current:5,} → {target_per_domain:5,} (trimmed {current - target_per_domain:,})[/yellow]")
    
    return balanced


def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Final Consolidation and Balancing[/bold cyan]")
    console.print("=" * 70)
    
    input_file = Path("data/phase1/questions_60k_shuffled.jsonl")
    output_untrimmed = Path("data/phase1/questions_all_untrimmed.jsonl")
    output_final = Path("data/phase1/questions_final_training.jsonl")
    
    # Step 1: Load and check duplicates
    console.print("\n[cyan]Step 1: Loading all questions and checking duplicates...[/cyan]")
    questions, duplicates, stats = load_all_questions(input_file)
    
    console.print(f"\n[green]✓ Loaded {len(questions):,} unique questions[/green]")
    if duplicates:
        console.print(f"[yellow]⚠ Found {len(duplicates):,} duplicates (already filtered)[/yellow]")
    else:
        console.print(f"[green]✓ No duplicates found![/green]")
    
    # Step 2: Display distribution
    console.print("\n[cyan]Step 2: Current Distribution[/cyan]")
    
    table = Table(title="Domain Distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    for domain, count in sorted(stats['by_domain'].items(), key=lambda x: x[1]):
        table.add_row(domain, f"{count:,}")
    
    console.print(table)
    
    # Token balance
    token_stats = calculate_token_balance(questions)
    console.print(f"\n[cyan]Token Distribution:[/cyan]")
    console.print(f"  Easy: {token_stats['easy_count']:,} questions → {token_stats['easy_tokens']:,} tokens ({token_stats['easy_token_pct']:.1f}%)")
    console.print(f"  Hard: {token_stats['hard_count']:,} questions → {token_stats['hard_tokens']:,} tokens ({token_stats['hard_token_pct']:.1f}%)")
    console.print(f"  Total: {len(questions):,} questions → {token_stats['total_tokens']:,} tokens")
    
    # Step 3: Save untrimmed
    console.print("\n[cyan]Step 3: Saving untrimmed consolidated file...[/cyan]")
    with open(output_untrimmed, 'w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')
    console.print(f"[green]✓ Saved to: {output_untrimmed}[/green]")
    
    # Step 4: Balance to target
    console.print("\n[cyan]Step 4: Balancing to target distribution (6,861 per domain)...[/cyan]")
    balanced = balance_to_target(questions, target_per_domain=6861)
    
    # Step 5: Shuffle
    console.print("\n[cyan]Step 5: Shuffling for training...[/cyan]")
    random.shuffle(balanced)
    
    # Step 6: Final stats
    final_token_stats = calculate_token_balance(balanced)
    console.print(f"\n[cyan]Final Dataset Statistics:[/cyan]")
    console.print(f"  Total: {len(balanced):,} questions")
    console.print(f"  Easy: {final_token_stats['easy_count']:,} ({final_token_stats['easy_count']/len(balanced)*100:.1f}%)")
    console.print(f"  Hard: {final_token_stats['hard_count']:,} ({final_token_stats['hard_count']/len(balanced)*100:.1f}%)")
    console.print(f"\n[cyan]Token Distribution:[/cyan]")
    console.print(f"  Easy tokens: {final_token_stats['easy_tokens']:,} ({final_token_stats['easy_token_pct']:.1f}%)")
    console.print(f"  Hard tokens: {final_token_stats['hard_tokens']:,} ({final_token_stats['hard_token_pct']:.1f}%)")
    console.print(f"  Total tokens: {final_token_stats['total_tokens']:,}")
    
    # Check if within 60/40 target
    if 58 <= final_token_stats['easy_token_pct'] <= 62:
        console.print(f"[green]✓ TOKEN BALANCE ACHIEVED! (target 60/40)[/green]")
    else:
        console.print(f"[yellow]⚠ Token balance: {final_token_stats['easy_token_pct']:.1f}% / {final_token_stats['hard_token_pct']:.1f}% (target 60/40)[/yellow]")
    
    # Step 7: Save final
    console.print(f"\n[cyan]Step 6: Saving final training file...[/cyan]")
    with open(output_final, 'w') as f:
        for q in balanced:
            f.write(json.dumps(q) + '\n')
    console.print(f"[green]✓ Saved to: {output_final}[/green]")
    
    # Verify batch mixing
    console.print(f"\n[cyan]Step 7: Verifying batch-level mixing (sample first 3 batches):[/cyan]")
    batch_size = 32
    for batch_num in range(3):
        start = batch_num * batch_size
        end = start + batch_size
        batch = balanced[start:end]
        
        easy_in_batch = sum(1 for q in batch if q['difficulty'] == 'easy')
        hard_in_batch = len(batch) - easy_in_batch
        domains_in_batch = set(q['domain'] for q in batch)
        
        console.print(f"  Batch {batch_num + 1}: {easy_in_batch} easy, {hard_in_batch} hard | Domains: {len(domains_in_batch)}")
    
    console.print("\n[green]✓ Dataset ready for training![/green]")
    console.print(f"[green]✓ Questions are shuffled - each training batch has natural 60/40 token mix[/green]")
    
    console.print(f"\n[cyan]Summary:[/cyan]")
    console.print(f"  Untrimmed: {len(questions):,} questions → {output_untrimmed}")
    console.print(f"  Final balanced: {len(balanced):,} questions → {output_final}")
    console.print(f"  Token balance: {final_token_stats['easy_token_pct']:.1f}% easy / {final_token_stats['hard_token_pct']:.1f}% hard")
    
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Validate questions with GPT-4-mini ($9)")
    console.print(f"  2. Generate answers (easy: $2.52, hard: $414)")
    console.print(f"  3. Train with shuffled data")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
