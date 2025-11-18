"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** scripts/download_all_benchmarks.py
**Reason:** Incomplete implementation, intermediate version replaced by better solution
**Issues:**
- Alternative approach but still incomplete
- Less robust than final version
- Missing some fallback strategies

**Archive Date:** 2025-11-14
**Archived For:** Historical reference only

Use download_all_benchmarks.py instead.

=================================

Download all 6 benchmarks for Phase 1A balanced training (Alternative method).

Uses direct HuggingFace dataset API and alternative sources.

Usage:
    python3 scripts/download_benchmarks_v2.py
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# Output directory
OUTPUT_DIR = Path("data/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_mmlu() -> Dict:
    """Download MMLU benchmark using alternative source."""
    console.print("\n[cyan]📥 Downloading MMLU...[/cyan]")
    
    try:
        # Use MMLU from HuggingFace datasets without trust_remote_code
        import datasets
        
        # Try alternative dataset
        dataset = datasets.load_dataset("tasksource/mmlu", split="test")
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "mmlu.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                json.dump({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": item.get("subject", "unknown")
                }, f)
                f.write('\n')
        
        console.print(f"[green]✓ MMLU downloaded: {len(dataset):,} questions[/green]")
        return {"name": "MMLU", "count": len(dataset), "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ MMLU download failed: {e}[/red]")
        # Try manual download as fallback
        try:
            console.print("[yellow]   Trying direct GitHub download...[/yellow]")
            # Download from GitHub releases or alternative source
            # For now, return 0 to skip
            return {"name": "MMLU", "count": 0, "error": "No alternative source available"}
        except:
            return {"name": "MMLU", "count": 0, "error": str(e)}


def download_gpqa() -> Dict:
    """Download GPQA benchmark."""
    console.print("\n[cyan]📥 Downloading GPQA...[/cyan]")
    
    # GPQA is gated - skip for now or use alternative
    console.print("[yellow]   GPQA requires authentication - skipping for now[/yellow]")
    console.print("[dim]   Will substitute with additional MMLU or similar later[/dim]")
    return {"name": "GPQA", "count": 0, "error": "Gated dataset - requires auth"}


def download_humaneval() -> Dict:
    """Download HumanEval benchmark."""
    console.print("\n[cyan]📥 Downloading HumanEval...[/cyan]")
    
    try:
        # Download from GitHub directly
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        
        import gzip
        import urllib.request
        
        # Download compressed file
        console.print("[dim]   Downloading from GitHub...[/dim]")
        gz_file = OUTPUT_DIR / "HumanEval.jsonl.gz"
        urllib.request.urlretrieve(url, gz_file)
        
        # Decompress and save
        output_file = OUTPUT_DIR / "humaneval.jsonl"
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        gz_file.unlink()
        
        # Count problems
        count = 0
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        
        console.print(f"[green]✓ HumanEval downloaded: {count} problems[/green]")
        return {"name": "HumanEval", "count": count, "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ HumanEval download failed: {e}[/red]")
        return {"name": "HumanEval", "count": 0, "error": str(e)}


def download_math() -> Dict:
    """Download MATH benchmark."""
    console.print("\n[cyan]📥 Downloading MATH...[/cyan]")
    
    try:
        import datasets
        
        # Try alternative MATH dataset
        dataset = datasets.load_dataset("competition_math", split="test")
        
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
        # Download from Google Research GitHub
        url = "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_en.tsv"
        
        import urllib.request
        import csv
        
        console.print("[dim]   Downloading from GitHub...[/dim]")
        tsv_file = OUTPUT_DIR / "mgsm_en.tsv"
        urllib.request.urlretrieve(url, tsv_file)
        
        # Convert to jsonl
        output_file = OUTPUT_DIR / "mgsm.jsonl"
        count = 0
        with open(tsv_file, 'r') as f_in, open(output_file, 'w') as f_out:
            reader = csv.DictReader(f_in, delimiter='\t')
            for row in reader:
                json.dump({
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "answer_number": row.get("answer_number", "")
                }, f_out)
                f_out.write('\n')
                count += 1
        
        # Remove TSV file
        tsv_file.unlink()
        
        console.print(f"[green]✓ MGSM downloaded: {count} problems[/green]")
        return {"name": "MGSM", "count": count, "file": str(output_file)}
        
    except Exception as e:
        console.print(f"[red]✗ MGSM download failed: {e}[/red]")
        # Try alternative approach
        try:
            console.print("[yellow]   Trying datasets library...[/yellow]")
            import datasets
            dataset = datasets.load_dataset("mgsm", "en", split="test")
            
            output_file = OUTPUT_DIR / "mgsm.jsonl"
            with open(output_file, 'w') as f:
                for item in dataset:
                    json.dump({
                        "question": item["question"],
                        "answer": item["answer"],
                        "answer_number": item.get("answer_number", "")
                    }, f)
                    f.write('\n')
            
            console.print(f"[green]✓ MGSM downloaded: {len(dataset)} problems[/green]")
            return {"name": "MGSM", "count": len(dataset), "file": str(output_file)}
        except:
            return {"name": "MGSM", "count": 0, "error": str(e)}


def download_drop() -> Dict:
    """Download DROP benchmark."""
    console.print("\n[cyan]📥 Downloading DROP...[/cyan]")
    
    try:
        import datasets
        
        # Try alternative source
        dataset = datasets.load_dataset("drop", split="validation")
        
        # Save to jsonl
        output_file = OUTPUT_DIR / "drop.jsonl"
        with open(output_file, 'w') as f:
            for item in dataset:
                # Handle different field structures
                answers = []
                if "answers_spans" in item:
                    if isinstance(item["answers_spans"], dict) and "spans" in item["answers_spans"]:
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
        "[bold cyan]Benchmark Downloader v2 - Phase 1A[/bold cyan]\n"
        "[dim]Using alternative sources and direct downloads",
        border_style="cyan"
    ))
    
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
    
    # Check if any succeeded
    succeeded = [r for r in results if r.get("count", 0) > 0]
    failed = [r for r in results if r.get("count", 0) == 0]
    
    if succeeded:
        console.print(f"\n[green]✅ {len(succeeded)} benchmark(s) downloaded successfully![/green]")
    
    if failed:
        console.print(f"\n[yellow]⚠️  {len(failed)} benchmark(s) failed:[/yellow]")
        for r in failed:
            console.print(f"   • {r['name']}: {r.get('error', 'Unknown error')}")
    
    console.print(f"\n[cyan]📁 Output directory: {OUTPUT_DIR.absolute()}[/cyan]")
    
    # Provide next steps
    if len(succeeded) >= 3:
        console.print("\n[bold green]✨ Enough benchmarks downloaded to proceed![/bold green]")
        console.print("[dim]Next step: Compute perplexity to split easy/hard questions[/dim]")
    elif len(succeeded) > 0:
        console.print("\n[yellow]⚠️  Some benchmarks missing - consider finding alternatives[/yellow]")
    else:
        console.print("\n[red]❌ No benchmarks downloaded - please check dataset availability[/red]")


if __name__ == "__main__":
    main()
