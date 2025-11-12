#!/usr/bin/env python3
"""
Deduplicate Training Data

Removes duplicate instructions from training data while preserving:
- The highest quality example per unique instruction
- Overall category distribution
- Training-ready examples only

Deduplication Strategy:
1. Group by exact instruction text (case-sensitive)
2. For each group, keep the example with:
   - Highest quality_score (if available)
   - Longest cot_response (as proxy for quality)
   - Successfully generated CoT

Usage:
    python3 deduplicate_training_data.py \
        --input Phase1C_Targeted_Distillation/data/phase1c_10k_with_cot.jsonl \
        --output Phase1C_Targeted_Distillation/data/phase1c_10k_with_cot_deduped.jsonl

Author: Cogumi-LLM Pipeline
Date: November 11, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deduplicate training data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output deduplicated JSONL file"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['quality_score', 'length', 'first'],
        default='length',
        help="Strategy for selecting best duplicate (default: length)"
    )
    parser.add_argument(
        "--keep_failed",
        action="store_true",
        help="Keep failed generations in output (default: filter out)"
    )
    
    return parser.parse_args()


def load_data(input_path: str) -> List[Dict]:
    """Load JSONL data file."""
    if not Path(input_path).exists():
        console.print(f"[red]❌ ERROR: File not found at {input_path}")
        sys.exit(1)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data
    except json.JSONDecodeError as e:
        console.print(f"[red]❌ ERROR: Invalid JSON in file: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ ERROR: Failed to load file: {e}")
        sys.exit(1)


def select_best_example(examples: List[Dict], strategy: str) -> Dict:
    """
    Select the best example from a list of duplicates.
    
    Strategies:
    - quality_score: Highest quality_score (if available)
    - length: Longest cot_response
    - first: First example in list
    """
    if len(examples) == 1:
        return examples[0]
    
    if strategy == 'quality_score':
        # Sort by quality_score (descending), fallback to length
        scored = [e for e in examples if 'quality_score' in e and e['quality_score'] is not None]
        if scored:
            return max(scored, key=lambda x: (x['quality_score'], len(x.get('cot_response', ''))))
        # Fallback to length if no quality scores
        strategy = 'length'
    
    if strategy == 'length':
        # Select example with longest cot_response
        return max(examples, key=lambda x: len(x.get('cot_response', '')))
    
    if strategy == 'first':
        # Keep first example
        return examples[0]
    
    return examples[0]


def deduplicate_data(data: List[Dict], strategy: str, keep_failed: bool) -> Tuple[List[Dict], Dict]:
    """
    Deduplicate data based on instruction text.
    
    Returns: (deduplicated_data, stats)
    """
    # Group by instruction
    instruction_groups = defaultdict(list)
    
    console.print(f"\n[yellow]Grouping by instruction...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Grouping examples", total=len(data))
        
        for item in data:
            # Filter out failed generations unless --keep_failed
            if not keep_failed and not item.get('generation_success', False):
                progress.advance(task)
                continue
            
            instruction = item.get('instruction', '').strip()
            if instruction:
                instruction_groups[instruction].append(item)
            
            progress.advance(task)
    
    # Select best example from each group
    console.print(f"\n[yellow]Selecting best examples from duplicates...")
    deduplicated = []
    duplicate_counts = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Deduplicating", total=len(instruction_groups))
        
        for instruction, examples in instruction_groups.items():
            if len(examples) > 1:
                duplicate_counts.append(len(examples))
            
            best_example = select_best_example(examples, strategy)
            deduplicated.append(best_example)
            
            progress.advance(task)
    
    # Compute statistics
    stats = {
        'original_count': len(data),
        'deduplicated_count': len(deduplicated),
        'removed_count': len(data) - len(deduplicated),
        'unique_instructions': len(instruction_groups),
        'duplicate_groups': len(duplicate_counts),
        'avg_duplicates': sum(duplicate_counts) / len(duplicate_counts) if duplicate_counts else 0,
        'max_duplicates': max(duplicate_counts) if duplicate_counts else 0,
    }
    
    return deduplicated, stats


def save_data(data: List[Dict], output_path: str):
    """Save deduplicated data to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        console.print(f"[green]✓ Saved to {output_path}")
    except Exception as e:
        console.print(f"[red]❌ ERROR: Failed to save file: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    
    # Print header
    console.print("\n" + "=" * 100)
    console.print("[bold cyan]🔄 TRAINING DATA DEDUPLICATION[/bold cyan]")
    console.print("=" * 100)
    
    console.print(f"\n[cyan]Input: {args.input}")
    console.print(f"[cyan]Output: {args.output}")
    console.print(f"[cyan]Strategy: {args.strategy}")
    console.print(f"[cyan]Keep failed: {args.keep_failed}")
    
    # Load data
    console.print(f"\n[yellow]Loading data...")
    data = load_data(args.input)
    console.print(f"[green]✓ Loaded {len(data):,} examples")
    
    # Category distribution before
    category_before = Counter(item.get('category') for item in data if item.get('generation_success', False) or args.keep_failed)
    
    # Deduplicate
    deduplicated, stats = deduplicate_data(data, args.strategy, args.keep_failed)
    
    # Category distribution after
    category_after = Counter(item.get('category') for item in deduplicated)
    
    # Print statistics
    console.print(f"\n" + "=" * 100)
    console.print(f"[bold]📊 DEDUPLICATION RESULTS[/bold]")
    console.print("=" * 100)
    
    console.print(f"\n[bold]Counts:[/bold]")
    console.print(f"   • Original examples: {stats['original_count']:,}")
    console.print(f"   • Deduplicated examples: {stats['deduplicated_count']:,}")
    console.print(f"   • Removed duplicates: {stats['removed_count']:,} ({stats['removed_count']/stats['original_count']*100:.1f}%)")
    console.print(f"   • Unique instructions: {stats['unique_instructions']:,}")
    
    console.print(f"\n[bold]Duplicate Analysis:[/bold]")
    console.print(f"   • Duplicate groups found: {stats['duplicate_groups']:,}")
    console.print(f"   • Average duplicates per group: {stats['avg_duplicates']:.1f}")
    console.print(f"   • Maximum duplicates for one instruction: {stats['max_duplicates']}")
    
    console.print(f"\n[bold]Category Distribution:[/bold]")
    console.print(f"{'Category':<12} {'Before':<12} {'After':<12} {'Change':<12}")
    console.print("-" * 50)
    
    all_categories = sorted(set(list(category_before.keys()) + list(category_after.keys())))
    for cat in all_categories:
        if cat is None:
            continue
        before_count = category_before.get(cat, 0)
        after_count = category_after.get(cat, 0)
        change = after_count - before_count
        change_pct = change / before_count * 100 if before_count > 0 else 0
        
        console.print(
            f"{cat:<12} {before_count:<12,} {after_count:<12,} "
            f"{change:+,} ({change_pct:+.1f}%)"
        )
    
    # Save deduplicated data
    console.print(f"\n[yellow]Saving deduplicated data...")
    save_data(deduplicated, args.output)
    
    # Final verdict
    console.print(f"\n" + "=" * 100)
    console.print(f"[bold]🎯 DEDUPLICATION COMPLETE[/bold]")
    console.print("=" * 100)
    
    console.print(f"\n[green]✅ Successfully deduplicated training data!")
    console.print(f"\n   • Removed {stats['removed_count']:,} duplicates ({stats['removed_count']/stats['original_count']*100:.1f}%)")
    console.print(f"   • Kept {stats['deduplicated_count']:,} unique examples")
    console.print(f"   • Output saved to: {args.output}")
    
    console.print(f"\n[cyan]📝 Next Steps:")
    console.print(f"   1. Validate deduplicated data:")
    console.print(f"      python3 scripts/validate_training_data.py --data_path {args.output}")
    console.print(f"   2. Use deduplicated data for training:")
    console.print(f"      Update --data_path in training script to {args.output}")
    
    console.print(f"\n" + "=" * 100 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
