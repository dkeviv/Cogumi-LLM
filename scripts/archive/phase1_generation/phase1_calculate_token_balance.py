#!/usr/bin/env python3
"""
Calculate Token-Weighted Distribution for Phase 1
==================================================

Best practices requirement:
- EASY examples: 5-30 tokens avg (~15 tokens)
- HARD examples: 300-1500 tokens avg (~750 tokens)
- Target: 60% EASY tokens, 40% HARD tokens in training

Current status: 44,700 questions
- Easy: 14,787 (33.1%)
- Hard: 29,913 (66.9%)

Need to calculate correct distribution to achieve token balance.
"""

from rich.console import Console
from rich.table import Table

console = Console()

# Token averages (from best practices)
EASY_AVG_TOKENS = 15  # 5-30 range
HARD_AVG_TOKENS = 750  # 300-1500 range

# Target token distribution
TARGET_EASY_TOKEN_PERCENT = 60
TARGET_HARD_TOKEN_PERCENT = 40

# Current counts
CURRENT_EASY = 14_787
CURRENT_HARD = 29_913
CURRENT_TOTAL = 44_700

# Target total
TARGET_TOTAL = 60_000

def calculate_token_distribution(easy_count, hard_count):
    """Calculate actual token distribution."""
    easy_tokens = easy_count * EASY_AVG_TOKENS
    hard_tokens = hard_count * HARD_AVG_TOKENS
    total_tokens = easy_tokens + hard_tokens
    
    easy_percent = (easy_tokens / total_tokens) * 100
    hard_percent = (hard_tokens / total_tokens) * 100
    
    return {
        "easy_tokens": easy_tokens,
        "hard_tokens": hard_tokens,
        "total_tokens": total_tokens,
        "easy_percent": easy_percent,
        "hard_percent": hard_percent,
    }

def calculate_required_counts(target_total):
    """
    Calculate required easy/hard counts to achieve 60/40 token split.
    
    Let E = easy count, H = hard count
    Constraints:
    1. E + H = target_total (60K)
    2. (E × 15) / (E × 15 + H × 750) = 0.60
    
    Solving:
    E × 15 = 0.60 × (E × 15 + H × 750)
    E × 15 = 9E + 450H
    6E = 450H
    E = 75H
    
    Substituting into E + H = 60K:
    75H + H = 60K
    76H = 60K
    H = 789
    E = 59,211
    """
    # From algebra above
    hard_count = target_total / 76
    easy_count = 75 * hard_count
    
    return int(easy_count), int(hard_count)

def main():
    console.print("[bold cyan]Token Balance Analysis for Phase 1[/bold cyan]\n")
    
    # Show current distribution
    console.print("[bold]1. CURRENT Distribution (44,700 questions):[/bold]")
    current = calculate_token_distribution(CURRENT_EASY, CURRENT_HARD)
    
    table1 = Table(show_header=True)
    table1.add_column("Metric", style="cyan")
    table1.add_column("Easy", style="green")
    table1.add_column("Hard", style="yellow")
    table1.add_column("Total", style="white")
    
    table1.add_row(
        "Sample Count",
        f"{CURRENT_EASY:,}",
        f"{CURRENT_HARD:,}",
        f"{CURRENT_TOTAL:,}"
    )
    table1.add_row(
        "Sample %",
        f"{CURRENT_EASY/CURRENT_TOTAL*100:.1f}%",
        f"{CURRENT_HARD/CURRENT_TOTAL*100:.1f}%",
        "100%"
    )
    table1.add_row(
        "Tokens (estimated)",
        f"{current['easy_tokens']:,}",
        f"{current['hard_tokens']:,}",
        f"{current['total_tokens']:,}"
    )
    table1.add_row(
        "Token %",
        f"[red]{current['easy_percent']:.1f}%[/red]",
        f"[red]{current['hard_percent']:.1f}%[/red]",
        "100%"
    )
    console.print(table1)
    
    console.print(f"\n[red]❌ Problem: Only {current['easy_percent']:.1f}% easy tokens (need 60%)![/red]")
    console.print(f"[red]   Hard examples dominate training due to length![/red]\n")
    
    # Calculate required distribution
    console.print("[bold]2. REQUIRED Distribution (60K questions):[/bold]")
    req_easy, req_hard = calculate_required_counts(TARGET_TOTAL)
    required = calculate_token_distribution(req_easy, req_hard)
    
    table2 = Table(show_header=True)
    table2.add_column("Metric", style="cyan")
    table2.add_column("Easy", style="green")
    table2.add_column("Hard", style="yellow")
    table2.add_column("Total", style="white")
    
    table2.add_row(
        "Sample Count",
        f"{req_easy:,}",
        f"{req_hard:,}",
        f"{TARGET_TOTAL:,}"
    )
    table2.add_row(
        "Sample %",
        f"[bold]{req_easy/TARGET_TOTAL*100:.1f}%[/bold]",
        f"[bold]{req_hard/TARGET_TOTAL*100:.1f}%[/bold]",
        "100%"
    )
    table2.add_row(
        "Tokens (estimated)",
        f"{required['easy_tokens']:,}",
        f"{required['hard_tokens']:,}",
        f"{required['total_tokens']:,}"
    )
    table2.add_row(
        "Token %",
        f"[green]✓ {required['easy_percent']:.1f}%[/green]",
        f"[green]✓ {required['hard_percent']:.1f}%[/green]",
        "100%"
    )
    console.print(table2)
    
    console.print(f"\n[green]✓ This achieves 60/40 token split![/green]\n")
    
    # Calculate what we need to generate
    console.print("[bold]3. WHAT TO GENERATE:[/bold]")
    
    easy_gap = req_easy - CURRENT_EASY
    hard_gap = req_hard - CURRENT_HARD
    
    # We have TOO MANY hard examples!
    if hard_gap < 0:
        console.print(f"[yellow]⚠ We have {abs(hard_gap):,} EXTRA hard examples[/yellow]")
        console.print(f"[yellow]  Current hard: {CURRENT_HARD:,}[/yellow]")
        console.print(f"[yellow]  Target hard: {req_hard:,}[/yellow]")
        console.print(f"[yellow]  → Use random sample of {req_hard:,} from {CURRENT_HARD:,}[/yellow]\n")
        hard_to_generate = 0
    else:
        console.print(f"[cyan]Need {hard_gap:,} more hard examples[/cyan]")
        hard_to_generate = hard_gap
    
    console.print(f"[cyan]Need {easy_gap:,} more easy examples[/cyan]\n")
    
    table3 = Table(show_header=True)
    table3.add_column("Action", style="cyan")
    table3.add_column("Easy", style="green")
    table3.add_column("Hard", style="yellow")
    
    table3.add_row(
        "Current (have)",
        f"{CURRENT_EASY:,}",
        f"{CURRENT_HARD:,}"
    )
    table3.add_row(
        "Target (need)",
        f"{req_easy:,}",
        f"{req_hard:,}"
    )
    table3.add_row(
        "Generate",
        f"[bold green]+{easy_gap:,}[/bold green]",
        f"[bold yellow]Use {req_hard:,} (random sample)[/bold yellow]" if hard_gap < 0 else f"[bold green]+{hard_to_generate:,}[/bold green]"
    )
    console.print(table3)
    
    # Per-domain breakdown
    console.print("\n[bold]4. PER-DOMAIN TARGETS (for 60K balanced):[/bold]")
    console.print("[dim]Using 98.7% easy / 1.3% hard split per domain to achieve 60/40 token balance[/dim]\n")
    
    domains = {
        "Coding": 10_000,
        "Math": 10_000,
        "Tool Use": 10_000,
        "Reasoning": 10_000,
        "Reading": 5_000,
        "Summarization": 5_000,
        "Common Sense": 5_000,
        "Instruction": 5_000,
    }
    
    table4 = Table(show_header=True)
    table4.add_column("Domain", style="cyan")
    table4.add_column("Total", justify="right")
    table4.add_column("Easy (98.7%)", justify="right", style="green")
    table4.add_column("Hard (1.3%)", justify="right", style="yellow")
    
    for domain, total in domains.items():
        easy = int(total * 0.987)
        hard = total - easy
        table4.add_row(domain, f"{total:,}", f"{easy:,}", f"{hard:,}")
    
    table4.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{TARGET_TOTAL:,}[/bold]",
        f"[bold]{req_easy:,}[/bold]",
        f"[bold]{req_hard:,}[/bold]"
    )
    console.print(table4)
    
    console.print("\n[bold green]✓ This distribution achieves 60% EASY tokens / 40% HARD tokens[/bold green]")
    console.print("[bold green]✓ Prevents reasoning bleed and verbosity on easy tasks[/bold green]")

if __name__ == "__main__":
    main()
