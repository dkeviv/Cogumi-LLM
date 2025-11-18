#!/usr/bin/env python3
"""
Convert benchmark datasets to validation test format.

This script converts multiple benchmark datasets (DROP, GPQA, HumanEval, MATH, MGSM, MMLU)
into a unified validation format compatible with phase1_validate_maml.py.

Input: data/benchmarks/*.jsonl (6 benchmark files)
Output: data/benchmarks/validation_test.jsonl (unified test set)

Format conversion:
- Each benchmark has different schema
- Convert all to: {"prompt": str, "response": str, "metadata": {...}}
- Add difficulty estimates based on benchmark characteristics
- Extract balanced sample (~100 per benchmark, except HumanEval with only 164)

Author: AI Assistant
Date: 2025-01-XX
Phase: 1D - Validation and Merge
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def estimate_difficulty(benchmark: str, example: Dict[str, Any]) -> str:
    """
    Estimate difficulty level for an example.
    
    Args:
        benchmark: Name of benchmark (DROP, GPQA, etc.)
        example: The example data
        
    Returns:
        "easy", "medium", or "hard"
    """
    # GPQA and MATH are inherently hard (graduate-level, competition math)
    if benchmark in ["GPQA", "MATH"]:
        level = example.get("level", "").lower()
        if level in ["level 1", "level 2", "easy"]:
            return "medium"  # Even "easy" MATH is medium difficulty
        else:
            return "hard"
    
    # HumanEval is medium to hard (code generation)
    elif benchmark == "HumanEval":
        return "medium"
    
    # MMLU varies by subject
    elif benchmark == "MMLU":
        hard_subjects = [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_computer_science", "college_mathematics",
            "college_physics", "computer_security", "formal_logic",
            "high_school_physics", "machine_learning", "professional_law"
        ]
        subject = example.get("subject", "")
        return "hard" if subject in hard_subjects else "medium"
    
    # DROP and MGSM are generally easier (reading comprehension, grade-school math)
    else:
        return "easy"


def convert_drop(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DROP (reading comprehension) format.
    
    Original: {"passage": str, "question": str, "answers": [str, ...]}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    passage = example.get("passage", "").strip()
    question = example.get("question", "").strip()
    answers = example.get("answers", [])
    
    # Use first answer as response (most specific)
    response = answers[0] if answers else "Unknown"
    
    # Create prompt with passage context
    prompt = f"Read the following passage and answer the question.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
    
    return {
        "prompt": prompt,
        "response": response,
        "metadata": {
            "difficulty": "easy",
            "domain": "reading_comprehension",
            "source": "DROP",
            "num_answers": len(answers)
        }
    }


def convert_gpqa(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GPQA (graduate-level science Q&A) format.
    
    Original: {"question": str, "answer": str}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    
    prompt = f"Answer the following graduate-level science question.\n\nQuestion: {question}\n\nAnswer:"
    
    return {
        "prompt": prompt,
        "response": answer,
        "metadata": {
            "difficulty": "hard",
            "domain": "science_qa",
            "source": "GPQA"
        }
    }


def convert_humaneval(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert HumanEval (code generation) format.
    
    Original: {"task_id": str, "prompt": str, "canonical_solution": str, "test": str, "entry_point": str}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    task_id = example.get("task_id", "")
    code_prompt = example.get("prompt", "").strip()
    solution = example.get("canonical_solution", "").strip()
    entry_point = example.get("entry_point", "")
    
    # Create natural language prompt
    prompt = f"Complete the following Python function.\n\n{code_prompt}"
    
    return {
        "prompt": prompt,
        "response": solution,
        "metadata": {
            "difficulty": "medium",
            "domain": "code_generation",
            "source": "HumanEval",
            "task_id": task_id,
            "entry_point": entry_point
        }
    }


def convert_math(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MATH (mathematical reasoning) format.
    
    Original: {"problem": str, "solution": str, "level": str, "type": str}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    level = example.get("level", "").strip()
    math_type = example.get("type", "").strip()
    
    prompt = f"Solve the following math problem. Show your work.\n\nProblem: {problem}\n\nSolution:"
    
    # Determine difficulty from level
    difficulty = estimate_difficulty("MATH", example)
    
    return {
        "prompt": prompt,
        "response": solution,
        "metadata": {
            "difficulty": difficulty,
            "domain": "mathematics",
            "source": "MATH",
            "level": level,
            "type": math_type
        }
    }


def convert_mgsm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MGSM (multilingual grade-school math) format.
    
    Original: {"question": str, "answer": str, "answer_number": str}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    answer_number = example.get("answer_number", "").strip()
    
    # Skip empty examples
    if not question or not answer:
        return None
    
    prompt = f"Solve this grade-school math problem.\n\nProblem: {question}\n\nAnswer:"
    
    return {
        "prompt": prompt,
        "response": answer,
        "metadata": {
            "difficulty": "easy",
            "domain": "grade_school_math",
            "source": "MGSM",
            "answer_number": answer_number
        }
    }


def convert_mmlu(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MMLU (multitask language understanding) format.
    
    Original: {"question": str, "choices": [str, ...], "answer": int, "subject": str}
    Target: {"prompt": str, "response": str, "metadata": {...}}
    """
    question = example.get("question", "").strip()
    choices = example.get("choices", [])
    answer_idx = example.get("answer", 0)
    subject = example.get("subject", "").strip()
    
    # Format choices
    choice_letters = ["A", "B", "C", "D"]
    choices_text = "\n".join([f"{letter}. {choice}" for letter, choice in zip(choice_letters, choices)])
    
    prompt = f"Answer the following multiple choice question from {subject.replace('_', ' ')}.\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer:"
    
    # Get correct answer
    response = f"{choice_letters[answer_idx]}. {choices[answer_idx]}"
    
    # Determine difficulty
    difficulty = estimate_difficulty("MMLU", example)
    
    return {
        "prompt": prompt,
        "response": response,
        "metadata": {
            "difficulty": difficulty,
            "domain": "multitask_understanding",
            "source": "MMLU",
            "subject": subject,
            "answer_index": answer_idx
        }
    }


def load_benchmark(file_path: Path, converter_func) -> List[Dict[str, Any]]:
    """
    Load and convert a benchmark file.
    
    Args:
        file_path: Path to benchmark JSONL file
        converter_func: Function to convert each example
        
    Returns:
        List of converted examples
    """
    converted = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                example = json.loads(line)
                converted_example = converter_func(example)
                
                # Skip None results (empty examples)
                if converted_example is not None:
                    converted.append(converted_example)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to convert example: {e}[/yellow]")
                continue
    
    return converted


def sample_examples(examples: List[Dict[str, Any]], n_samples: int, benchmark_name: str) -> List[Dict[str, Any]]:
    """
    Sample examples from a benchmark, stratified by difficulty if possible.
    
    Args:
        examples: List of examples
        n_samples: Target number of samples
        benchmark_name: Name of benchmark (for logging)
        
    Returns:
        Sampled examples
    """
    # If we have fewer examples than requested, return all
    if len(examples) <= n_samples:
        console.print(f"  [cyan]{benchmark_name}[/cyan]: Using all {len(examples)} examples (< {n_samples} requested)")
        return examples
    
    # Group by difficulty for stratified sampling
    by_difficulty = defaultdict(list)
    for ex in examples:
        difficulty = ex["metadata"].get("difficulty", "unknown")
        by_difficulty[difficulty].append(ex)
    
    # Sample proportionally from each difficulty level
    sampled = []
    for difficulty, difficulty_examples in by_difficulty.items():
        # Calculate proportion for this difficulty
        proportion = len(difficulty_examples) / len(examples)
        n_from_difficulty = max(1, int(n_samples * proportion))
        
        # Don't sample more than available
        n_from_difficulty = min(n_from_difficulty, len(difficulty_examples))
        
        sampled.extend(random.sample(difficulty_examples, n_from_difficulty))
    
    # If we sampled too few (due to rounding), add random extras
    if len(sampled) < n_samples:
        remaining = [ex for ex in examples if ex not in sampled]
        n_extra = n_samples - len(sampled)
        sampled.extend(random.sample(remaining, min(n_extra, len(remaining))))
    
    # If we sampled too many, randomly remove
    if len(sampled) > n_samples:
        sampled = random.sample(sampled, n_samples)
    
    console.print(f"  [cyan]{benchmark_name}[/cyan]: Sampled {len(sampled)} from {len(examples)} examples")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark datasets to validation test format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (100 per benchmark)
  python scripts/convert_benchmarks_to_test.py
  
  # Convert with custom sample size
  python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 50
  
  # Convert all examples (no sampling)
  python scripts/convert_benchmarks_to_test.py --samples-per-benchmark -1
  
  # Custom input/output paths
  python scripts/convert_benchmarks_to_test.py \\
      --input-dir data/benchmarks \\
      --output-file data/phase1/validation_test.jsonl
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/benchmarks"),
        help="Directory containing benchmark JSONL files (default: data/benchmarks)"
    )
    
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/benchmarks/validation_test.jsonl"),
        help="Output file for unified validation test set (default: data/benchmarks/validation_test.jsonl)"
    )
    
    parser.add_argument(
        "--samples-per-benchmark",
        type=int,
        default=100,
        help="Number of samples per benchmark (default: 100, use -1 for all)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Benchmark to Validation Test Converter   [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Define benchmark configurations
    benchmarks = [
        {
            "name": "DROP",
            "file": args.input_dir / "drop.jsonl",
            "converter": convert_drop,
            "expected_count": 9535
        },
        {
            "name": "GPQA",
            "file": args.input_dir / "gpqa.jsonl",
            "converter": convert_gpqa,
            "expected_count": 1319
        },
        {
            "name": "HumanEval",
            "file": args.input_dir / "humaneval.jsonl",
            "converter": convert_humaneval,
            "expected_count": 164
        },
        {
            "name": "MATH",
            "file": args.input_dir / "math.jsonl",
            "converter": convert_math,
            "expected_count": 1319
        },
        {
            "name": "MGSM",
            "file": args.input_dir / "mgsm.jsonl",
            "converter": convert_mgsm,
            "expected_count": 249
        },
        {
            "name": "MMLU",
            "file": args.input_dir / "mmlu.jsonl",
            "converter": convert_mmlu,
            "expected_count": 14042
        }
    ]
    
    # Convert all benchmarks
    all_examples = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        for benchmark in benchmarks:
            task = progress.add_task(
                f"Converting {benchmark['name']}...",
                total=None
            )
            
            # Check if file exists
            if not benchmark["file"].exists():
                console.print(f"[yellow]Warning: {benchmark['file']} not found, skipping[/yellow]")
                progress.remove_task(task)
                continue
            
            # Load and convert
            converted = load_benchmark(benchmark["file"], benchmark["converter"])
            
            # Sample if requested
            if args.samples_per_benchmark > 0:
                sampled = sample_examples(converted, args.samples_per_benchmark, benchmark["name"])
            else:
                sampled = converted
                console.print(f"  [cyan]{benchmark['name']}[/cyan]: Using all {len(sampled)} examples")
            
            all_examples.extend(sampled)
            progress.remove_task(task)
    
    # Shuffle all examples
    random.shuffle(all_examples)
    
    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to output file
    console.print(f"\n[bold green]Writing {len(all_examples)} examples to {args.output_file}[/bold green]")
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    console.print("\n[bold cyan]Conversion Statistics:[/bold cyan]\n")
    
    # Count by source
    by_source = defaultdict(int)
    for ex in all_examples:
        source = ex["metadata"].get("source", "Unknown")
        by_source[source] += 1
    
    console.print("[cyan]Examples per benchmark:[/cyan]")
    for source, count in sorted(by_source.items()):
        console.print(f"  {source}: {count}")
    
    # Count by difficulty
    by_difficulty = defaultdict(int)
    for ex in all_examples:
        difficulty = ex["metadata"].get("difficulty", "unknown")
        by_difficulty[difficulty] += 1
    
    console.print("\n[cyan]Examples per difficulty:[/cyan]")
    for difficulty, count in sorted(by_difficulty.items()):
        console.print(f"  {difficulty}: {count}")
    
    # Count by domain
    by_domain = defaultdict(int)
    for ex in all_examples:
        domain = ex["metadata"].get("domain", "unknown")
        by_domain[domain] += 1
    
    console.print("\n[cyan]Examples per domain:[/cyan]")
    for domain, count in sorted(by_domain.items()):
        console.print(f"  {domain}: {count}")
    
    console.print(f"\n[bold green]✓ Successfully created validation test set: {args.output_file}[/bold green]")
    console.print(f"[bold green]✓ Total examples: {len(all_examples)}[/bold green]\n")


if __name__ == "__main__":
    main()
