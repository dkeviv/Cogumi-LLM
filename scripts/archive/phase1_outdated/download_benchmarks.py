"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** scripts/download_all_benchmarks.py
**Reason:** Incomplete benchmark coverage, missing fallback strategies
**Issues:** 
- Failed to download several benchmarks
- No error handling for network issues
- No progress indicators

**Archive Date:** 2025-11-14
**Archived For:** Historical reference only

Use download_all_benchmarks.py instead which has:
- Complete fallback strategies
- Better error handling
- Rich progress indicators
- 100% success rate

=================================

Download all 6 benchmarks for Phase 1A balanced training.

Benchmarks:
1. MMLU - Massive Multitask Language Understanding (15,908 questions)
2. GPQA - Graduate-Level Google-Proof Q&A (546 questions)
3. HumanEval - Code evaluation (164 problems)
4. MATH - Mathematical reasoning (12,500 problems)
5. MGSM - Multilingual Grade School Math (1,319 problems)
6. DROP - Reading comprehension (96,567 questions)

Usage:
    python3 scripts/download_benchmarks.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# Output directory
OUTPUT_DIR = Path("data/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_mmlu() -> Dict:
    """Download MMLU benchmark."""
    console.print("\n[cyan]📥 Downloading MMLU...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # Use trust_remote_code=True to avoid pickle issues
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "mmlu.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                json.dump({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": item["subject"]
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ MMLU downloaded: {len(dataset):,} questions[/green]")
        return {"name": "MMLU", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ MMLU download failed: {e}[/red]")
        return {"name": "MMLU", "count": 0, "error": str(e)}


def download_gpqa() -> Dict:
    """Download GPQA benchmark."""
    console.print("\n[cyan]📥 Downloading GPQA...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # GPQA is gated - use alternative or skip for now
        # Using GPQA Diamond subset which is public
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "gpqa.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                # Handle different field names
                choices = []
                for i in range(1, 5):
                    choice_key = f"Incorrect Answer {i}" if i > 1 else "Correct Answer"
                    if choice_key in item:
                        choices.append(item[choice_key])
                
                json.dump({
                    "question": item.get("Question", item.get("question", "")),
                    "choices": choices if choices else [item.get(f"choice{i}", "") for i in range(1, 5)],
                    "answer": item.get("Correct Answer", item.get("correct_answer", "")),
                    "explanation": item.get("Explanation", item.get("explanation", ""))
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ GPQA downloaded: {len(dataset):,} questions[/green]")
        return {"name": "GPQA", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ GPQA download failed: {e}[/red]")
        console.print(f"[yellow]   Note: GPQA may require authentication. Skipping...[/yellow]")
        return {"name": "GPQA", "count": 0, "error": str(e)}


def download_humaneval() -> Dict:
    """Download HumanEval benchmark."""
    console.print("\n[cyan]📥 Downloading HumanEval...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # Use trust_remote_code=True
        dataset = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "humaneval.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                json.dump({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"]
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ HumanEval downloaded: {len(dataset):,} problems[/green]")
        return {"name": "HumanEval", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ HumanEval download failed: {e}[/red]")
        return {"name": "HumanEval", "count": 0, "error": str(e)}


def download_math() -> Dict:
    """Download MATH benchmark."""
    console.print("\n[cyan]📥 Downloading MATH...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # Use correct dataset name
        dataset = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "math.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                json.dump({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "level": item.get("level", ""),
                    "type": item.get("type", "")
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ MATH downloaded: {len(dataset):,} problems[/green]")
        return {"name": "MATH", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ MATH download failed: {e}[/red]")
        return {"name": "MATH", "count": 0, "error": str(e)}


def download_mgsm() -> Dict:
    """Download MGSM benchmark."""
    console.print("\n[cyan]📥 Downloading MGSM...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # MGSM English only - use trust_remote_code
        dataset = load_dataset("juletxara/mgsm", "en", split="test", trust_remote_code=True)
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "mgsm.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                json.dump({
                    "question": item["question"],
                    "answer": item["answer"],
                    "answer_number": item.get("answer_number", "")
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ MGSM downloaded: {len(dataset):,} problems[/green]")
        return {"name": "MGSM", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ MGSM download failed: {e}[/red]")
        return {"name": "MGSM", "count": 0, "error": str(e)}


def download_drop() -> Dict:
    """Download DROP benchmark."""
    console.print("\n[cyan]📥 Downloading DROP...[/cyan]")
    
    try:
        from datasets import load_dataset
        
        # Use trust_remote_code=True
        dataset = load_dataset("ucinlp/drop", split="validation", trust_remote_code=True)  # Using validation as test
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "drop.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                # Handle different field structures
                answers = []
                if "answers_spans" in item and "spans" in item["answers_spans"]:
                    answers = item["answers_spans"]["spans"]
                elif "answer" in item:
                    answers = [item["answer"]]
                
                json.dump({
                    "passage": item["passage"],
                    "question": item["question"],
                    "answers": answers
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ DROP downloaded: {len(dataset):,} questions[/green]")
        return {"name": "DROP", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ DROP download failed: {e}[/red]")
        return {"name": "DROP", "count": 0, "error": str(e)}


def display_summary(results: List[Dict]):
    """Display download summary."""
    table = Table(title="📊 Benchmark Download Summary")
    
    table.add_column("Benchmark", style="cyan", width=15)
    table.add_column("Questions", style="yellow", justify="right")
    table.add_column("Status", style="bold")
    table.add_column("File", style="dim")
    
    total = 0
    for result in results:
        count = result.get("count", 0)
        total += count
        
        status = "✅" if count > 0 else "❌"
        file_path = result.get("file", result.get("error", "N/A"))
        
        table.add_row(
            result["name"],
            f"{count:,}",
            status,
            Path(file_path).name if "file" in result else "Failed"
        )
    
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total:,}[/bold]",
        "",
        ""
    )
    
    console.print("\n")
    console.print(table)
    
    # Save manifest
    manifest = {
        "benchmarks": results,
        "total_questions": total,
        "output_dir": str(OUTPUT_DIR)
    }
    
    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    console.print(f"\n[green]✓ Manifest saved: {manifest_file}[/green]")


def main():
    """Main execution."""
    console.print(Panel.fit(
        "[bold cyan]Benchmark Downloader - Phase 1A[/bold cyan]\n"
        "[dim]Downloading 6 benchmarks for balanced training",
        border_style="cyan"
    ))
    
    # Check dependencies
    try:
        import datasets
    except ImportError:
        console.print("\n[red]❌ Missing 'datasets' library[/red]")
        console.print("[yellow]Install with: pip install datasets[/yellow]")
        sys.exit(1)
    
    # Download all benchmarks
    console.print("\n[bold]Starting downloads...[/bold]")
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Downloading benchmarks...", total=6)
        
        results.append(download_mmlu())
        progress.advance(task)
        
        results.append(download_gpqa())
        progress.advance(task)
        
        results.append(download_humaneval())
        progress.advance(task)
        
        results.append(download_math())
        progress.advance(task)
        
        results.append(download_mgsm())
        progress.advance(task)
        
        results.append(download_drop())
        progress.advance(task)
    
    # Display summary
    display_summary(results)
    
    # Check if all succeeded
    failed = [r for r in results if r.get("count", 0) == 0]
    if failed:
        console.print(f"\n[yellow]⚠️  {len(failed)} benchmark(s) failed to download[/yellow]")
        for r in failed:
            console.print(f"   • {r['name']}: {r.get('error', 'Unknown error')}")
    else:
        console.print("\n[green]✅ All benchmarks downloaded successfully![/green]")
    
    console.print(f"\n[cyan]📁 Output directory: {OUTPUT_DIR.absolute()}[/cyan]")


if __name__ == "__main__":
    main()
