#!/usr/bin/env python3
"""
Clean Training Data - Remove XML Tags and Standardize Format

This script cleans the training data by:
1. Removing all XML tags (<response>, <draft>, <thinking>, etc.)
2. Reformatting hard examples with natural language reasoning markers
3. Keeping easy examples as direct answers
4. Validating data integrity

Author: Cogumi-LLM Team
Date: 2025-11-16
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


def remove_xml_tags(text: str) -> str:
    """Remove all XML-like tags from text."""
    if not text:
        return ""
    
    # Remove specific tags we know about
    text = re.sub(r'</?response>', '', text)
    text = re.sub(r'</?draft>', '', text)
    text = re.sub(r'</?thinking>', '', text)
    text = re.sub(r'</?answer>', '', text)
    
    # Remove any remaining XML-like tags (safety)
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()


def format_easy_example(example: Dict) -> Dict:
    """Format easy example - direct answer only."""
    cleaned = example.copy()
    
    # Clean response field
    if 'response' in cleaned:
        cleaned['response'] = remove_xml_tags(cleaned['response'])
    
    # Remove draft/thinking fields if present (shouldn't be, but safety)
    cleaned.pop('draft', None)
    cleaned.pop('thinking', None)
    
    return cleaned


def format_hard_example(example: Dict) -> Dict:
    """Format hard example - natural language reasoning structure."""
    cleaned = example.copy()
    
    # Extract and clean components
    draft = remove_xml_tags(example.get('draft', ''))
    thinking = remove_xml_tags(example.get('thinking', ''))
    response = remove_xml_tags(example.get('response', ''))
    
    # Build natural language reasoning structure
    if draft and thinking and response:
        # Full CoT with explicit structure
        full_response = f"""Let me work through this step by step:

{draft}

Checking my reasoning:
{thinking}

Final answer:
{response}"""
    elif draft and response:
        # Draft + answer (no critique)
        full_response = f"""Let me work through this:

{draft}

Final answer:
{response}"""
    elif thinking and response:
        # Thinking + answer (no draft)
        full_response = f"""Analyzing this problem:

{thinking}

Final answer:
{response}"""
    else:
        # Fallback: just clean response
        full_response = response if response else remove_xml_tags(example.get('response', ''))
    
    # Update cleaned example
    cleaned['response'] = full_response
    
    # Remove separate draft/thinking fields (now integrated into response)
    cleaned.pop('draft', None)
    cleaned.pop('thinking', None)
    
    return cleaned


def clean_example(example: Dict) -> Dict:
    """Clean a single training example based on difficulty."""
    difficulty = example.get('metadata', {}).get('difficulty', 'easy')
    
    if difficulty == 'hard':
        return format_hard_example(example)
    else:
        return format_easy_example(example)


def validate_cleaned_data(examples: List[Dict]) -> Dict:
    """Validate cleaned data and return statistics."""
    stats = {
        'total': len(examples),
        'easy': 0,
        'hard': 0,
        'missing_prompt': 0,
        'missing_response': 0,
        'has_xml_tags': 0,
        'avg_response_length': 0,
    }
    
    domain_counts = Counter()
    response_lengths = []
    
    for example in examples:
        # Check required fields
        if not example.get('prompt'):
            stats['missing_prompt'] += 1
        
        if not example.get('response'):
            stats['missing_response'] += 1
        
        # Check for remaining XML tags
        response = example.get('response', '')
        if '<' in response and '>' in response:
            # Check if it's actual XML tags (not math expressions)
            if re.search(r'</?[a-z]+>', response, re.IGNORECASE):
                stats['has_xml_tags'] += 1
        
        # Count difficulty
        difficulty = example.get('metadata', {}).get('difficulty', 'easy')
        stats[difficulty] += 1
        
        # Track domain
        domain = example.get('metadata', {}).get('domain', 'Unknown')
        domain_counts[domain] += 1
        
        # Track response length
        response_lengths.append(len(response))
    
    stats['avg_response_length'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    stats['domain_counts'] = domain_counts
    
    return stats


def display_validation_results(stats: Dict):
    """Display validation results in a nice table."""
    # Main statistics
    table = Table(title="Cleaned Data Validation", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
    table.add_row("Total Examples", f"{stats['total']:,}", "100.0%")
    table.add_row("Easy Examples", f"{stats['easy']:,}", f"{stats['easy']/stats['total']*100:.1f}%")
    table.add_row("Hard Examples", f"{stats['hard']:,}", f"{stats['hard']/stats['total']*100:.1f}%")
    table.add_row("", "", "")
    table.add_row("Missing Prompts", f"{stats['missing_prompt']:,}", 
                  f"{stats['missing_prompt']/stats['total']*100:.2f}%")
    table.add_row("Missing Responses", f"{stats['missing_response']:,}", 
                  f"{stats['missing_response']/stats['total']*100:.2f}%")
    table.add_row("Remaining XML Tags", f"{stats['has_xml_tags']:,}", 
                  f"{stats['has_xml_tags']/stats['total']*100:.2f}%")
    table.add_row("", "", "")
    table.add_row("Avg Response Length", f"{stats['avg_response_length']:.0f} chars", "")
    
    console.print()
    console.print(table)
    
    # Domain distribution
    console.print("\n[bold cyan]Domain Distribution:[/bold cyan]")
    for domain, count in sorted(stats['domain_counts'].items(), key=lambda x: x[1], reverse=True):
        console.print(f"  • {domain}: {count:,} ({count/stats['total']*100:.1f}%)")


def display_sample_comparison(original: Dict, cleaned: Dict):
    """Display before/after comparison for a sample."""
    console.print("\n[bold cyan]Sample Comparison:[/bold cyan]")
    console.print("\n[yellow]BEFORE (with XML tags):[/yellow]")
    console.print(f"Response: {original.get('response', '')[:200]}...")
    if 'draft' in original:
        console.print(f"Draft: {original.get('draft', '')[:100]}...")
    if 'thinking' in original:
        console.print(f"Thinking: {original.get('thinking', '')[:100]}...")
    
    console.print("\n[green]AFTER (cleaned):[/green]")
    console.print(f"Response: {cleaned.get('response', '')[:300]}...")


def main():
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]     Training Data Cleaning - XML Tag Removal[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    # File paths
    input_file = Path("data/phase1/answers/training_data_interleaved.jsonl")
    output_file = Path("data/phase1/answers/training_data_clean.jsonl")
    backup_file = Path("data/phase1/answers/training_data_interleaved.jsonl.backup")
    
    # Validate input exists
    if not input_file.exists():
        console.print(f"[bold red]❌ Input file not found:[/bold red] {input_file}")
        return 1
    
    # Load original data
    console.print(f"[cyan]Loading original data from:[/cyan] {input_file}")
    original_examples = []
    with open(input_file) as f:
        for line in f:
            original_examples.append(json.loads(line))
    
    console.print(f"[green]✓ Loaded {len(original_examples):,} examples[/green]\n")
    
    # Clean data with progress bar
    console.print("[cyan]Cleaning data and removing XML tags...[/cyan]")
    cleaned_examples = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing examples", total=len(original_examples))
        
        for example in original_examples:
            cleaned = clean_example(example)
            cleaned_examples.append(cleaned)
            progress.advance(task)
    
    console.print(f"[green]✓ Cleaned {len(cleaned_examples):,} examples[/green]\n")
    
    # Validate cleaned data
    console.print("[cyan]Validating cleaned data...[/cyan]")
    stats = validate_cleaned_data(cleaned_examples)
    display_validation_results(stats)
    
    # Check for issues
    issues = stats['missing_prompt'] + stats['missing_response'] + stats['has_xml_tags']
    
    if issues > 0:
        console.print(f"\n[yellow]⚠️  Found {issues} potential issues[/yellow]")
        if stats['has_xml_tags'] > 0:
            console.print(f"[yellow]  • {stats['has_xml_tags']} examples still contain XML-like tags[/yellow]")
            console.print(f"[yellow]    (May be false positives from math expressions like <, >)[/yellow]")
    else:
        console.print("\n[bold green]✓ All validation checks passed![/bold green]")
    
    # Show sample comparison
    # Find a hard example for demonstration
    hard_idx = next((i for i, ex in enumerate(original_examples) 
                     if ex.get('metadata', {}).get('difficulty') == 'hard'), 0)
    
    if hard_idx < len(original_examples):
        display_sample_comparison(original_examples[hard_idx], cleaned_examples[hard_idx])
    
    # Save cleaned data
    console.print(f"\n[cyan]Saving cleaned data to:[/cyan] {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    console.print(f"[green]✓ Saved {len(cleaned_examples):,} cleaned examples[/green]")
    
    # Create backup of original
    console.print(f"\n[cyan]Creating backup of original:[/cyan] {backup_file}")
    import shutil
    shutil.copy2(input_file, backup_file)
    console.print(f"[green]✓ Backup created[/green]")
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold green]✓ Data cleaning complete![/bold green]")
    console.print(f"\n[cyan]Summary:[/cyan]")
    console.print(f"  • Input: {input_file}")
    console.print(f"  • Output: {output_file}")
    console.print(f"  • Backup: {backup_file}")
    console.print(f"  • Total examples: {len(cleaned_examples):,}")
    console.print(f"  • Easy (direct answers): {stats['easy']:,}")
    console.print(f"  • Hard (with reasoning): {stats['hard']:,}")
    console.print(f"  • XML tags removed: ✓")
    console.print(f"  • Natural language structure: ✓")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Review sample outputs above")
    console.print(f"  2. Run validation: python scripts/validate_training_setup.py")
    console.print(f"  3. Update training script to use: {output_file}")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
