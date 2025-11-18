#!/usr/bin/env python3
"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** scripts/convert_benchmarks_to_test.py
**Reason:** Test set contamination - model was trained on ALL training data,
           so extracting test examples from training data does not provide
           a true generalization test. Use independent benchmarks instead.

**Issue:** Extracts test set from training_data_clean.jsonl, but model was
          already trained on all 53,597 examples. This tests training accuracy,
          not generalization ability.

**Solution:** Use convert_benchmarks_to_test.py to convert independent benchmark
             datasets (DROP, GPQA, HumanEval, MATH, MMLU) that model has never
             seen during training.

**Archive Date:** November 18, 2025
**Archived For:** Historical reference only

See: docs/PHASE1D_BENCHMARK_VALIDATION.md for full explanation

=================================

Phase 1D: Create Test Set for Validation

Extracts held-out test examples from training data for validation.
Stratified sampling to ensure balanced difficulty and domain distribution.

Usage:
    python scripts/phase1_create_test_set.py \
        --train_file data/phase1/answers/training_data_clean.jsonl \
        --output_file data/phase1/test_set.jsonl \
        --test_size 200 \
        --stratify_by difficulty,domain

Output:
    - Test set with 200 examples (stratified)
    - Training set with test examples removed
    - Statistics report

Author: Cogumi-LLM
Date: November 18, 2025
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

console = Console()


def load_training_data(train_file: str) -> List[Dict]:
    """Load training data from JSONL."""
    console.print(f"\n[bold blue]Loading training data from: {train_file}[/bold blue]")
    
    examples = []
    with open(train_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    
    console.print(f"[green]✓[/green] Loaded {len(examples)} examples")
    return examples


def stratified_split(
    examples: List[Dict],
    test_size: int,
    stratify_keys: List[str] = ["difficulty"]
) -> tuple[List[Dict], List[Dict]]:
    """Perform stratified split on examples."""
    
    console.print(f"\n[bold blue]Performing stratified split (test_size={test_size})[/bold blue]")
    console.print(f"Stratifying by: {', '.join(stratify_keys)}")
    
    # Group examples by stratification keys
    strata = defaultdict(list)
    for example in examples:
        # Create stratification key
        key_parts = []
        for key in stratify_keys:
            # Check in metadata first, then top-level
            if "metadata" in example and key in example["metadata"]:
                value = example["metadata"][key]
            else:
                value = example.get(key, "unknown")
            key_parts.append(str(value))
        strata_key = "|".join(key_parts)
        strata[strata_key].append(example)
    
    # Display strata distribution
    table = Table(title="Strata Distribution")
    table.add_column("Stratum", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Percentage", style="green")
    
    for stratum, stratum_examples in sorted(strata.items()):
        pct = len(stratum_examples) / len(examples) * 100
        table.add_row(stratum, str(len(stratum_examples)), f"{pct:.1f}%")
    
    console.print(table)
    
    # Calculate samples per stratum (proportional)
    test_examples = []
    train_examples = []
    
    for stratum, stratum_examples in strata.items():
        # Proportional allocation
        stratum_test_size = max(1, int(len(stratum_examples) / len(examples) * test_size))
        
        # Ensure we don't exceed available examples
        stratum_test_size = min(stratum_test_size, len(stratum_examples))
        
        # Random sample
        random.shuffle(stratum_examples)
        test_examples.extend(stratum_examples[:stratum_test_size])
        train_examples.extend(stratum_examples[stratum_test_size:])
        
        console.print(f"  {stratum}: {stratum_test_size} test, {len(stratum_examples) - stratum_test_size} train")
    
    console.print(f"\n[green]✓[/green] Split complete:")
    console.print(f"  Test: {len(test_examples)} examples")
    console.print(f"  Train: {len(train_examples)} examples")
    
    return train_examples, test_examples


def save_split(
    train_examples: List[Dict],
    test_examples: List[Dict],
    train_file: str,
    test_file: str
):
    """Save train/test split to files."""
    
    console.print(f"\n[bold blue]Saving split to files[/bold blue]")
    
    # Save test set
    test_path = Path(test_file)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    
    console.print(f"[green]✓[/green] Test set saved: {test_file} ({len(test_examples)} examples)")
    
    # Save updated training set (without test examples)
    train_path = Path(train_file)
    backup_file = str(train_path).replace('.jsonl', '_backup.jsonl')
    
    # Backup original
    with open(train_file, 'r') as f_in:
        with open(backup_file, 'w') as f_out:
            f_out.write(f_in.read())
    
    console.print(f"[green]✓[/green] Original training data backed up: {backup_file}")
    
    # Save new training set
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    console.print(f"[green]✓[/green] Updated training set saved: {train_file} ({len(train_examples)} examples)")


def generate_statistics(
    train_examples: List[Dict],
    test_examples: List[Dict],
    output_file: str
):
    """Generate and save statistics about the split."""
    
    def compute_distribution(examples: List[Dict], key: str) -> Dict:
        dist = defaultdict(int)
        for ex in examples:
            # Check in metadata first, then top-level
            if "metadata" in ex and key in ex["metadata"]:
                value = ex["metadata"][key]
            else:
                value = ex.get(key, "unknown")
            dist[value] += 1
        return dict(dist)
    
    stats = {
        "total_examples": len(train_examples) + len(test_examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "test_percentage": len(test_examples) / (len(train_examples) + len(test_examples)) * 100,
        "train_difficulty_dist": compute_distribution(train_examples, "difficulty"),
        "test_difficulty_dist": compute_distribution(test_examples, "difficulty"),
        "train_domain_dist": compute_distribution(train_examples, "domain"),
        "test_domain_dist": compute_distribution(test_examples, "domain")
    }
    
    # Save stats
    stats_path = Path(output_file).parent / "test_split_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    console.print(f"[green]✓[/green] Statistics saved: {stats_path}")
    
    # Display key stats
    console.print("\n[bold yellow]Split Statistics:[/bold yellow]")
    console.print(f"  Total: {stats['total_examples']}")
    console.print(f"  Train: {stats['train_examples']} ({100 - stats['test_percentage']:.1f}%)")
    console.print(f"  Test: {stats['test_examples']} ({stats['test_percentage']:.1f}%)")
    
    # Difficulty distribution
    console.print("\n[bold yellow]Difficulty Distribution:[/bold yellow]")
    table = Table()
    table.add_column("Difficulty", style="cyan")
    table.add_column("Train", style="blue")
    table.add_column("Test", style="green")
    
    all_difficulties = set(stats['train_difficulty_dist'].keys()) | set(stats['test_difficulty_dist'].keys())
    for diff in sorted(all_difficulties):
        train_count = stats['train_difficulty_dist'].get(diff, 0)
        test_count = stats['test_difficulty_dist'].get(diff, 0)
        table.add_row(diff, str(train_count), str(test_count))
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Create test set for validation")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save test set JSONL"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=200,
        help="Number of examples in test set"
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="difficulty",
        help="Comma-separated keys to stratify by (e.g., 'difficulty,domain')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't modify original training file (only create test set)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Display configuration
    console.print("\n[bold]Phase 1D: Create Test Set[/bold]")
    console.print("=" * 60)
    console.print(f"Training file: {args.train_file}")
    console.print(f"Test file: {args.output_file}")
    console.print(f"Test size: {args.test_size}")
    console.print(f"Random seed: {args.seed}")
    
    # Load data
    examples = load_training_data(args.train_file)
    
    # Stratify split
    stratify_keys = [k.strip() for k in args.stratify_by.split(',')]
    train_examples, test_examples = stratified_split(examples, args.test_size, stratify_keys)
    
    # Save split
    if args.no_backup:
        # Only save test set
        test_path = Path(args.output_file)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            for example in test_examples:
                f.write(json.dumps(example) + '\n')
        console.print(f"[green]✓[/green] Test set saved: {args.output_file}")
    else:
        # Save both train and test
        save_split(train_examples, test_examples, args.train_file, args.output_file)
    
    # Generate statistics
    generate_statistics(train_examples, test_examples, args.output_file)
    
    console.print("\n[bold green]✓ Test set creation complete![/bold green]")


if __name__ == "__main__":
    main()
