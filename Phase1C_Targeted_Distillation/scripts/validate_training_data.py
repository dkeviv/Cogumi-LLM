#!/usr/bin/env python3
"""
Comprehensive Data Validation Script

Validates training data before use to catch issues early.

Usage:
    python3 validate_training_data.py --data_path /workspace/data/phase1c_10k_with_cot.jsonl

Features:
- Required field checks
- Empty/null value detection
- Format validation (EOS tokens, special characters)
- Length distribution analysis
- Category distribution verification
- Duplicate detection
- Training-readiness assessment
- Detailed error reporting

Author: Cogumi-LLM Pipeline
Date: November 11, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from rich.console import Console
from rich.table import Table

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--required_fields",
        type=str,
        nargs='+',
        default=['instruction', 'cot_response', 'generation_success'],
        help="Required fields in each example"
    )
    parser.add_argument(
        "--valid_categories",
        type=str,
        nargs='+',
        default=['code', 'math', 'reasoning', 'creative'],
        help="Valid category values"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum instruction length (chars)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=5000,
        help="Maximum instruction length (chars)"
    )
    parser.add_argument(
        "--check_eos",
        action="store_true",
        default=True,
        help="Check for EOS token in responses"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit on any validation failure"
    )
    
    return parser.parse_args()


def load_data(data_path: str) -> List[Dict]:
    """Load JSONL data file."""
    if not Path(data_path).exists():
        console.print(f"[red]❌ ERROR: File not found at {data_path}")
        sys.exit(1)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data
    except json.JSONDecodeError as e:
        console.print(f"[red]❌ ERROR: Invalid JSON in file: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ ERROR: Failed to load file: {e}")
        sys.exit(1)


def validate_required_fields(data: List[Dict], required_fields: List[str]) -> Tuple[int, Dict]:
    """Check for required fields in all examples."""
    missing_counts = defaultdict(int)
    
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                missing_counts[field] += 1
    
    return len(data) - len([i for i, item in enumerate(data) if all(f in item for f in required_fields)]), dict(missing_counts)


def validate_empty_values(data: List[Dict], required_fields: List[str]) -> Tuple[int, Dict]:
    """Check for empty or null values in required fields."""
    empty_counts = defaultdict(int)
    
    for item in data:
        for field in required_fields:
            if field in item:
                if item[field] is None:
                    empty_counts[f"{field}_null"] += 1
                elif isinstance(item[field], str) and item[field].strip() == "":
                    empty_counts[f"{field}_empty"] += 1
                elif isinstance(item[field], (list, dict)) and len(item[field]) == 0:
                    empty_counts[f"{field}_empty_collection"] += 1
    
    total_empty = sum(empty_counts.values())
    return total_empty, dict(empty_counts)


def validate_instruction_length(data: List[Dict], min_len: int, max_len: int) -> Tuple[int, int, int]:
    """Check instruction length distribution."""
    too_short = 0
    too_long = 0
    
    for item in data:
        inst = item.get('instruction', '')
        if len(inst) < min_len:
            too_short += 1
        elif len(inst) > max_len:
            too_long += 1
    
    return too_short, too_long, len(data) - too_short - too_long


def validate_cot_structure(data: List[Dict], check_eos: bool) -> Dict[str, int]:
    """Validate CoT response structure."""
    issues = {
        'missing_thinking': 0,
        'missing_eos': 0,
        'missing_draft': 0,
        'missing_critique': 0,
        'missing_revised': 0,
        'too_short': 0,
        'failed_generation': 0,
    }
    
    for item in data:
        if not item.get('generation_success', False):
            issues['failed_generation'] += 1
            continue
        
        cot = item.get('cot_response', '')
        
        if check_eos and not cot.endswith('<|end_of_text|>'):
            issues['missing_eos'] += 1
        if '<thinking>' not in cot or '</thinking>' not in cot:
            issues['missing_thinking'] += 1
        if 'DRAFT:' not in cot.upper():
            issues['missing_draft'] += 1
        if 'CRITIQUE:' not in cot.upper():
            issues['missing_critique'] += 1
        if 'REVISED:' not in cot.upper():
            issues['missing_revised'] += 1
        if len(cot) < 100:
            issues['too_short'] += 1
    
    return issues


def validate_categories(data: List[Dict], valid_categories: List[str]) -> Tuple[int, Dict]:
    """Check category distribution and validity."""
    category_counts = Counter(item.get('category', 'MISSING') for item in data)
    
    invalid_count = sum(count for cat, count in category_counts.items() if cat not in valid_categories)
    
    return invalid_count, dict(category_counts)


def check_duplicates(data: List[Dict]) -> int:
    """Check for duplicate instructions."""
    instruction_hashes = defaultdict(list)
    
    for i, item in enumerate(data):
        inst = item.get('instruction', '')
        inst_hash = hash(inst)
        instruction_hashes[inst_hash].append(i)
    
    duplicates = sum(1 for indices in instruction_hashes.values() if len(indices) > 1)
    return duplicates


def assess_training_readiness(data: List[Dict], required_fields: List[str], valid_categories: List[str], check_eos: bool) -> Tuple[int, List[Tuple[int, List[str]]]]:
    """Assess if data is ready for training."""
    training_ready = []
    training_issues = []
    
    for i, item in enumerate(data):
        issues = []
        
        # Must have all required fields
        missing = [f for f in required_fields if f not in item]
        if missing:
            issues.append(f"missing fields: {', '.join(missing)}")
        
        # Must have non-empty instruction
        if not item.get('instruction') or len(item.get('instruction', '').strip()) == 0:
            issues.append("missing/empty instruction")
        
        # Must have successful generation
        if not item.get('generation_success', False):
            issues.append("generation failed")
        elif not item.get('cot_response') or len(item.get('cot_response', '').strip()) == 0:
            issues.append("missing/empty cot_response")
        
        # Must have EOS token
        cot_resp = item.get('cot_response')
        if check_eos and cot_resp and not cot_resp.endswith('<|end_of_text|>'):
            issues.append("missing EOS token")
        
        # Must have valid category
        if item.get('category') not in valid_categories:
            issues.append("invalid category")
        
        if issues:
            training_issues.append((i, issues))
        else:
            training_ready.append(i)
    
    return len(training_ready), training_issues


def main():
    args = parse_args()
    
    # Print header
    console.print("\n" + "=" * 100)
    console.print("[bold cyan]🔍 COMPREHENSIVE DATA VALIDATION[/bold cyan]")
    console.print("=" * 100)
    
    console.print(f"\n[cyan]Data file: {args.data_path}")
    console.print(f"[cyan]Required fields: {', '.join(args.required_fields)}")
    console.print(f"[cyan]Valid categories: {', '.join(args.valid_categories)}")
    
    # Load data
    console.print(f"\n[yellow]Loading data...")
    data = load_data(args.data_path)
    console.print(f"[green]✓ Loaded {len(data):,} examples")
    
    # Track validation results
    critical_failures = []
    warnings = []
    
    # 1. Required fields check
    console.print(f"\n[bold]1️⃣  REQUIRED FIELDS CHECK[/bold]")
    console.print("-" * 100)
    
    missing_count, missing_fields = validate_required_fields(data, args.required_fields)
    
    if missing_count == 0:
        console.print(f"[green]✅ All examples have required fields")
    else:
        console.print(f"[red]❌ {missing_count} examples missing required fields:")
        for field, count in missing_fields.items():
            console.print(f"   • {field}: {count} examples")
        critical_failures.append(f"{missing_count} examples missing required fields")
    
    # 2. Empty/null values check
    console.print(f"\n[bold]2️⃣  EMPTY/NULL VALUES CHECK[/bold]")
    console.print("-" * 100)
    
    empty_count, empty_fields = validate_empty_values(data, args.required_fields)
    
    if empty_count == 0:
        console.print(f"[green]✅ No empty or null values in required fields")
    else:
        console.print(f"[yellow]⚠️  {empty_count} empty/null value issues:")
        for field, count in empty_fields.items():
            console.print(f"   • {field}: {count} examples")
        warnings.append(f"{empty_count} empty/null values")
    
    # 3. Instruction length check
    console.print(f"\n[bold]3️⃣  INSTRUCTION LENGTH CHECK[/bold]")
    console.print("-" * 100)
    
    too_short, too_long, good_length = validate_instruction_length(data, args.min_length, args.max_length)
    
    console.print(f"   • Too short (<{args.min_length} chars): {too_short}")
    console.print(f"   • Too long (>{args.max_length} chars): {too_long}")
    console.print(f"   • Good length: {good_length} ({good_length/len(data)*100:.1f}%)")
    
    if too_short + too_long > len(data) * 0.05:
        warnings.append(f"{too_short + too_long} instructions with length issues (>5%)")
    
    if too_short + too_long == 0:
        console.print(f"[green]✅ All instructions have appropriate length")
    
    # 4. CoT structure check
    console.print(f"\n[bold]4️⃣  COT STRUCTURE CHECK[/bold]")
    console.print("-" * 100)
    
    cot_issues = validate_cot_structure(data, args.check_eos)
    
    console.print(f"   • Missing <thinking>: {cot_issues['missing_thinking']} ({cot_issues['missing_thinking']/len(data)*100:.1f}%)")
    console.print(f"   • Missing EOS token: {cot_issues['missing_eos']} ({cot_issues['missing_eos']/len(data)*100:.1f}%)")
    console.print(f"   • Missing DRAFT: {cot_issues['missing_draft']} ({cot_issues['missing_draft']/len(data)*100:.1f}%)")
    console.print(f"   • Missing CRITIQUE: {cot_issues['missing_critique']} ({cot_issues['missing_critique']/len(data)*100:.1f}%)")
    console.print(f"   • Missing REVISED: {cot_issues['missing_revised']} ({cot_issues['missing_revised']/len(data)*100:.1f}%)")
    console.print(f"   • Too short (<100 chars): {cot_issues['too_short']} ({cot_issues['too_short']/len(data)*100:.1f}%)")
    console.print(f"   • Failed generation: {cot_issues['failed_generation']} ({cot_issues['failed_generation']/len(data)*100:.1f}%)")
    
    critical_cot = cot_issues['missing_eos'] + cot_issues['too_short'] + cot_issues['failed_generation']
    if critical_cot > len(data) * 0.01:
        warnings.append(f"{critical_cot} examples with critical CoT issues (>1%)")
    
    if critical_cot == 0:
        console.print(f"[green]✅ No critical CoT issues")
    
    # 5. Category distribution check
    console.print(f"\n[bold]5️⃣  CATEGORY DISTRIBUTION CHECK[/bold]")
    console.print("-" * 100)
    
    invalid_cats, category_counts = validate_categories(data, args.valid_categories)
    
    console.print(f"   Categories:")
    for cat in sorted(args.valid_categories):
        count = category_counts.get(cat, 0)
        pct = count / len(data) * 100 if data else 0
        console.print(f"      • {cat:12s}: {count:5,} ({pct:5.1f}%)")
    
    if invalid_cats > 0:
        console.print(f"[red]❌ {invalid_cats} examples with invalid categories")
        for cat, count in category_counts.items():
            if cat not in args.valid_categories:
                console.print(f"      • {cat}: {count}")
        warnings.append(f"{invalid_cats} invalid categories")
    else:
        console.print(f"[green]✅ All categories valid")
    
    # 6. Duplicate check
    console.print(f"\n[bold]6️⃣  DUPLICATE CHECK[/bold]")
    console.print("-" * 100)
    
    duplicate_count = check_duplicates(data)
    
    if duplicate_count == 0:
        console.print(f"[green]✅ No duplicate instructions found")
    else:
        console.print(f"[yellow]⚠️  Found {duplicate_count} duplicate instructions")
        warnings.append(f"{duplicate_count} duplicate instructions")
    
    # 7. Training readiness assessment
    console.print(f"\n[bold]7️⃣  TRAINING READINESS ASSESSMENT[/bold]")
    console.print("-" * 100)
    
    ready_count, issues_list = assess_training_readiness(data, args.required_fields, args.valid_categories, args.check_eos)
    
    console.print(f"   • Training-ready examples: {ready_count:,} ({ready_count/len(data)*100:.1f}%)")
    console.print(f"   • Examples with issues: {len(issues_list):,} ({len(issues_list)/len(data)*100:.1f}%)")
    
    if len(issues_list) > 0 and len(issues_list) <= 10:
        console.print(f"\n   Issue details:")
        for i, issues in issues_list[:10]:
            console.print(f"      • Example {i}: {'; '.join(issues)}")
    elif len(issues_list) > 10:
        console.print(f"\n   First 10 issues:")
        for i, issues in issues_list[:10]:
            console.print(f"      • Example {i}: {'; '.join(issues)}")
    
    if ready_count < len(data) * 0.95:
        critical_failures.append(f"Only {ready_count/len(data)*100:.1f}% training-ready (<95%)")
    
    # Final verdict
    console.print(f"\n" + "=" * 100)
    console.print(f"[bold]🎯 FINAL VALIDATION VERDICT[/bold]")
    console.print("=" * 100)
    
    if critical_failures:
        console.print(f"\n[red]❌ VALIDATION FAILED - CRITICAL ISSUES:[/red]")
        for failure in critical_failures:
            console.print(f"   • {failure}")
        console.print(f"\n[red]⚠️  DATA NOT READY FOR TRAINING - FIX CRITICAL ISSUES FIRST[/red]")
        
        if args.strict:
            sys.exit(1)
    elif warnings:
        console.print(f"\n[yellow]⚠️  VALIDATION PASSED WITH WARNINGS:[/yellow]")
        for warning in warnings:
            console.print(f"   • {warning}")
        console.print(f"\n[yellow]✅ DATA IS USABLE BUT HAS MINOR ISSUES[/yellow]")
    else:
        console.print(f"\n[green]✅ VALIDATION PASSED - NO ISSUES FOUND![/green]")
        console.print(f"\n[green]🎉 DATA IS READY FOR TRAINING![/green]")
        console.print(f"\n   • {ready_count:,} training-ready examples")
        console.print(f"   • All required fields present")
        console.print(f"   • No empty/null values")
        console.print(f"   • CoT structure complete")
        console.print(f"   • Categories valid")
        console.print(f"   • Format validated")
    
    console.print(f"\n" + "=" * 100 + "\n")
    
    return 0 if not critical_failures else 1


if __name__ == "__main__":
    sys.exit(main())
