#!/usr/bin/env python3
"""
⚠️ ARCHIVED - UTILITY SCRIPT ⚠️
=================================

This utility script is ARCHIVED.

**Status:** Utility for sampling questions during generation
**Reason for Archive:** Generation phase complete, no longer needed actively
**May be useful for:** Future quality checks on new generations

**Archive Date:** 2025-11-14
**Archived For:** Utility reference, may be useful later

Can be reactivated if needed for future question generation phases.

=================================

Quality Check - Sample Generated Questions
===========================================

Check quality of generated questions without interrupting generation.
Samples from checkpoint files that are being written.
"""

import json
import random
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def sample_questions_from_checkpoint(file_path, num_samples=5):
    """Sample random questions from a checkpoint file."""
    questions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
        
        if len(questions) > num_samples:
            return random.sample(questions, num_samples)
        return questions
    except:
        return []

def check_question_quality(question):
    """Quick quality checks."""
    text = question.get('question', '')
    issues = []
    
    # Check length
    if len(text) < 10:
        issues.append("Too short")
    if len(text) > 500:
        issues.append("Too long")
    
    # Check for LaTeX
    if '\\(' in text or '\\[' in text or '\\frac' in text:
        issues.append("Contains LaTeX")
    
    # Check for incomplete sentences
    if not text.strip().endswith(('?', '.', '!', ':', '"', "'")):
        issues.append("No ending punctuation")
    
    # Check for placeholders
    placeholders = ['...', 'TODO', 'FIXME', '[placeholder]', '<placeholder>']
    if any(ph in text for ph in placeholders):
        issues.append("Contains placeholder")
    
    return issues

def main():
    console.print("[bold cyan]Phase 1: Question Quality Check[/bold cyan]")
    console.print("[dim]Sampling from checkpoint files (non-intrusive)[/dim]\n")
    
    checkpoint_dir = Path("data/phase1")
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*_easy.jsonl"))
    
    if not checkpoint_files:
        console.print("[yellow]⚠ No checkpoint files found yet. Generation may still be starting.[/yellow]")
        return
    
    console.print(f"[green]Found {len(checkpoint_files)} checkpoint files[/green]\n")
    
    total_sampled = 0
    total_issues = 0
    
    for checkpoint_file in sorted(checkpoint_files):
        domain = checkpoint_file.stem.replace('checkpoint_', '').replace('_easy', '').replace('_', ' ').title()
        
        questions = sample_questions_from_checkpoint(checkpoint_file, num_samples=3)
        
        if not questions:
            continue
        
        console.print(f"\n[bold]{domain} Domain[/bold] - Sampled {len(questions)} questions:\n")
        
        for i, q in enumerate(questions, 1):
            question_text = q.get('question', 'N/A')
            difficulty = q.get('difficulty', 'N/A')
            
            # Quality check
            issues = check_question_quality(q)
            total_sampled += 1
            if issues:
                total_issues += 1
            
            # Display
            status = "✓" if not issues else f"⚠ {', '.join(issues)}"
            
            panel = Panel(
                f"[white]{question_text}[/white]\n\n"
                f"[dim]Difficulty: {difficulty} | Status: {status}[/dim]",
                title=f"Sample {i}",
                border_style="green" if not issues else "yellow"
            )
            console.print(panel)
    
    # Summary
    console.print("\n" + "="*60)
    console.print(f"[bold]Quality Summary:[/bold]")
    console.print(f"  Total sampled: {total_sampled}")
    console.print(f"  Issues found: {total_issues} ({total_issues/total_sampled*100:.1f}%)" if total_sampled > 0 else "  No samples yet")
    console.print(f"  Pass rate: {(total_sampled-total_issues)/total_sampled*100:.1f}%" if total_sampled > 0 else "  N/A")
    
    if total_issues > 0:
        console.print("\n[yellow]⚠ Some quality issues detected. Consider reviewing prompts.[/yellow]")
    else:
        console.print("\n[green]✓ All sampled questions pass quality checks![/green]")
    
    console.print("\n[dim]Note: Generation is still running. This is just a snapshot.[/dim]")

if __name__ == "__main__":
    main()
