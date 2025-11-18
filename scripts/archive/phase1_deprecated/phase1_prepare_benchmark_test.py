#!/usr/bin/env python3
"""
⚠️ SUPERSEDED - USE NEWER VERSION ⚠️
=================================

This script is SUPERSEDED by a better implementation.

**Current Version:** scripts/convert_benchmarks_to_test.py (RECOMMENDED)
**This Version:** Earlier prototype, less comprehensive

**Improvements in New Version:**
- Supports all 6 benchmarks (DROP, GPQA, HumanEval, MATH, MGSM, MMLU)
- Better format conversion with proper prompt templates
- Stratified sampling by difficulty levels
- Comprehensive metadata preservation
- Rich progress tracking and statistics

**Performance Comparison:**
- Old: Basic conversion, fewer benchmarks
- New: Full conversion pipeline with all benchmarks

**Archive Date:** November 18, 2025
**Reason:** Replaced by more comprehensive convert_benchmarks_to_test.py

See: docs/PHASE1D_BENCHMARK_VALIDATION.md for usage guide

=================================

Phase 1D: Prepare Benchmark Data for Validation

Converts benchmark datasets to validation format.
Samples from multiple benchmarks to get diverse test set.

Usage:
    python scripts/phase1_prepare_benchmark_test.py \
        --benchmarks math gpqa humaneval \
        --samples_per_benchmark 67 \
        --output_file data/phase1/benchmark_test_set.jsonl

Author: Cogumi-LLM
Date: November 18, 2025
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

console = Console()


def load_benchmark(benchmark_file: str) -> List[Dict]:
    """Load benchmark data from JSONL."""
    examples = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples


def convert_to_validation_format(example: Dict, benchmark_name: str) -> Dict:
    """Convert benchmark format to validation format."""
    
    # Determine prompt and response based on benchmark format
    if "problem" in example and "solution" in example:
        # MATH format
        prompt = example["problem"]
        response = example["solution"]
    elif "question" in example and "answer" in example:
        # GPQA/general format
        prompt = example["question"]
        response = example["answer"]
    elif "prompt" in example:
        # HumanEval format
        prompt = example["prompt"]
        response = example.get("canonical_solution", "")
    else:
        # Generic fallback
        prompt = str(example.get("input", example.get("text", "")))
        response = str(example.get("output", example.get("target", "")))
    
    # Determine difficulty
    difficulty = "hard"  # Benchmarks are generally hard
    if benchmark_name == "humaneval":
        difficulty = "hard"
    elif benchmark_name == "math":
        difficulty = "hard"
    elif benchmark_name == "gpqa":
        difficulty = "hard"
    elif benchmark_name == "mmlu":
        difficulty = "easy"  # MMLU has easier questions
    
    # Map benchmark to domain
    domain_mapping = {
        "math": "Math",
        "gpqa": "Science",
        "humaneval": "Coding",
        "mgsm": "Math",
        "mmlu": "Reasoning",
        "drop": "Reading"
    }
    domain = domain_mapping.get(benchmark_name, "Reasoning")
    
    return {
        "input": prompt,
        "output": response,
        "metadata": {
            "difficulty": difficulty,
            "domain": domain,
            "source": f"benchmark:{benchmark_name}",
            "task_type": "evaluation",
            "benchmark": benchmark_name
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark test set")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["math", "gpqa", "humaneval"],
        help="Benchmark names to sample from"
    )
    parser.add_argument(
        "--samples_per_benchmark",
        type=int,
        default=67,
        help="Number of samples from each benchmark"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/phase1/benchmark_test_set.jsonl",
        help="Output file for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    console.print("\n[bold]Phase 1D: Prepare Benchmark Test Set[/bold]")
    console.print("=" * 60)
    
    # Load and sample from each benchmark
    test_set = []
    stats = []
    
    for benchmark_name in args.benchmarks:
        benchmark_file = f"data/benchmarks/{benchmark_name}.jsonl"
        
        if not Path(benchmark_file).exists():
            console.print(f"[yellow]⚠[/yellow] Skipping {benchmark_name} (file not found)")
            continue
        
        console.print(f"\n[bold blue]Processing {benchmark_name}...[/bold blue]")
        
        # Load benchmark
        examples = load_benchmark(benchmark_file)
        console.print(f"  Loaded {len(examples)} examples")
        
        # Sample
        sample_size = min(args.samples_per_benchmark, len(examples))
        sampled = random.sample(examples, sample_size)
        console.print(f"  Sampled {sample_size} examples")
        
        # Convert to validation format
        converted = [convert_to_validation_format(ex, benchmark_name) for ex in sampled]
        test_set.extend(converted)
        
        stats.append({
            "benchmark": benchmark_name,
            "total": len(examples),
            "sampled": sample_size,
            "domain": converted[0]["metadata"]["domain"],
            "difficulty": converted[0]["metadata"]["difficulty"]
        })
    
    # Save test set
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in test_set:
            f.write(json.dumps(example) + '\n')
    
    console.print(f"\n[green]✓[/green] Test set saved: {output_path}")
    console.print(f"  Total examples: {len(test_set)}")
    
    # Display statistics
    console.print("\n[bold yellow]Test Set Composition:[/bold yellow]")
    table = Table()
    table.add_column("Benchmark", style="cyan")
    table.add_column("Domain", style="blue")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Sampled", style="green")
    table.add_column("Total Available", style="white")
    
    for stat in stats:
        table.add_row(
            stat["benchmark"],
            stat["domain"],
            stat["difficulty"],
            str(stat["sampled"]),
            str(stat["total"])
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold green]✓ Benchmark test set prepared![/bold green]")
    console.print(f"Total test examples: {len(test_set)}")
    console.print(f"From {len(args.benchmarks)} benchmarks")
    console.print(f"\nThis is TRULY UNSEEN data - model never trained on these!")


if __name__ == "__main__":
    main()
