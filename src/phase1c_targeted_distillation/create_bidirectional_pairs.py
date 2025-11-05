#!/usr/bin/env python3
"""
Phase 1C/1D: Create Bidirectional Training Pairs

PURPOSE:
    Convert forward-only examples (instruction→output) into bidirectional pairs:
    - Forward: instruction → output (original)
    - Reverse: output → instruction (reversed for comprehension)
    
    Bidirectional training improves model's understanding in both directions,
    enhancing reasoning, comprehension, and task flexibility.

WHEN TO USE:
    - After Claude example generation complete
    - After self-critique examples ready
    - Before combining into unified training dataset

INPUT FORMATS:
    - Self-critique: data/phase1c/phase1c_self_critique_train.jsonl (2,389 examples)
    - Claude: data/phase1c/improved_examples.jsonl (4,942 examples)
    
OUTPUT:
    - Self-critique bidirectional: 2,389 × 2 = 4,778 examples
    - Claude bidirectional: 4,942 × 2 = 9,884 examples
    - Total: 14,662 bidirectional training examples

BENEFITS:
    - +2-3% accuracy improvement from bidirectional understanding
    - Better causal reasoning (why → answer AND answer → why)
    - Improved task flexibility and generalization
    - Stronger comprehension of instruction-response relationships

USAGE:
    # Process self-critique examples
    python Phase1A_2_0/scripts/create_bidirectional_pairs.py \\
        --input "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl" \\
        --output "data/phase1c/self_critique_bidirectional.jsonl" \\
        --source_label "self_critique"
    
    # Process Claude examples
    python Phase1A_2_0/scripts/create_bidirectional_pairs.py \\
        --input "data/phase1c/improved_examples.jsonl" \\
        --output "data/phase1c/claude_bidirectional.jsonl" \\
        --source_label "claude_generation"
    
    # Combine both into unified training file
    cat data/phase1c/self_critique_bidirectional.jsonl \\
        data/phase1c/claude_bidirectional.jsonl \\
        > data/phase1c/combined_training_bidirectional.jsonl

PIPELINE STAGE: Phase 1C/1D - Bidirectional Pair Generation
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.progress import track

console = Console()


def create_forward_pair(example: Dict, source_label: str) -> Dict:
    """
    Create forward training pair: instruction → output
    (Original direction, unchanged)
    """
    
    return {
        "instruction": example["instruction"],
        "input": example.get("input", ""),
        "output": example["output"],
        "meta": {
            **example.get("meta", {}),
            "direction": "forward",
            "source": f"{source_label}_bidirectional",
            "pair_type": "instruction_to_response"
        }
    }


def create_reverse_pair(example: Dict, source_label: str) -> Dict:
    """
    Create reverse training pair: output → instruction
    (Reversed direction for bidirectional understanding)
    
    Reverse format:
    - New instruction: "Given this response, what was the original instruction/question?"
    - Input: The model's output (response)
    - Output: The original instruction
    
    This teaches the model to infer the intent/question from the answer,
    improving comprehension and causal reasoning.
    """
    
    # Check if input field has content (some examples have instruction with separate input)
    original_input = example.get("input", "")
    
    # Create reverse instruction
    if original_input:
        reverse_instruction = (
            "Given the following response and context, reconstruct the original instruction "
            "that would have led to this answer. Be specific and capture the intent."
        )
        reverse_input_field = (
            f"**Context:** {original_input}\n\n"
            f"**Response:** {example['output']}"
        )
    else:
        reverse_instruction = (
            "Given the following response, reconstruct the original instruction/question "
            "that would have led to this answer. Be specific and capture the intent."
        )
        reverse_input_field = example['output']
    
    # Output is the original instruction
    reverse_output = example['instruction']
    
    return {
        "instruction": reverse_instruction,
        "input": reverse_input_field,
        "output": reverse_output,
        "meta": {
            **example.get("meta", {}),
            "direction": "reverse",
            "source": f"{source_label}_bidirectional",
            "pair_type": "response_to_instruction",
            "original_instruction": example['instruction'],
            "original_output": example['output']
        }
    }


def validate_example(example: Dict) -> bool:
    """Validate that example has required fields"""
    
    required_fields = ['instruction', 'output']
    
    for field in required_fields:
        if field not in example:
            console.print(f"[yellow]⚠️  Missing required field: {field}[/yellow]")
            return False
        
        if not example[field] or not example[field].strip():
            console.print(f"[yellow]⚠️  Empty field: {field}[/yellow]")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create bidirectional training pairs")
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file with forward examples')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file with bidirectional pairs')
    
    # Configuration
    parser.add_argument('--source_label', type=str, required=True,
                       choices=['self_critique', 'claude_generation', 'other'],
                       help='Label for source of examples')
    parser.add_argument('--validate', action='store_true',
                       help='Validate examples before processing')
    parser.add_argument('--preview', type=int, default=0,
                       help='Preview N examples without saving')
    
    args = parser.parse_args()
    
    # Load input data
    console.print(f"\n📂 Loading input from [cyan]{args.input}[/cyan]...")
    
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]❌ Input file not found: {args.input}[/red]")
        return
    
    with open(input_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    console.print(f"✅ Loaded {len(examples):,} examples")
    
    # Validate if requested
    if args.validate:
        console.print("\n🔍 Validating examples...")
        valid_examples = []
        invalid_count = 0
        
        for example in examples:
            if validate_example(example):
                valid_examples.append(example)
            else:
                invalid_count += 1
        
        console.print(f"✅ Valid: {len(valid_examples):,}")
        console.print(f"❌ Invalid: {invalid_count}")
        
        examples = valid_examples
    
    # Preview mode
    if args.preview > 0:
        console.print(f"\n👀 Preview mode: showing {args.preview} examples")
        console.print("="*80)
        
        for i, example in enumerate(examples[:args.preview]):
            console.print(f"\n[bold cyan]Example {i+1}:[/bold cyan]")
            
            # Forward pair
            forward = create_forward_pair(example, args.source_label)
            console.print(f"\n[green]Forward (instruction → output):[/green]")
            console.print(f"Instruction: {forward['instruction'][:100]}...")
            console.print(f"Output: {forward['output'][:100]}...")
            
            # Reverse pair
            reverse = create_reverse_pair(example, args.source_label)
            console.print(f"\n[blue]Reverse (output → instruction):[/blue]")
            console.print(f"Instruction: {reverse['instruction'][:100]}...")
            console.print(f"Input: {reverse['input'][:100]}...")
            console.print(f"Output: {reverse['output'][:100]}...")
            console.print("="*80)
        
        console.print(f"\n✅ Preview complete (use without --preview to generate)")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate bidirectional pairs
    console.print(f"\n🔄 Generating bidirectional pairs...")
    console.print(f"   Source: {args.source_label}")
    console.print(f"   Input: {len(examples):,} examples")
    console.print(f"   Output: {len(examples) * 2:,} pairs (2x)")
    console.print("="*80)
    
    forward_count = 0
    reverse_count = 0
    
    with open(output_path, 'w') as out_f:
        for example in track(examples, description="Creating pairs..."):
            # Create forward pair
            forward = create_forward_pair(example, args.source_label)
            out_f.write(json.dumps(forward) + '\n')
            forward_count += 1
            
            # Create reverse pair
            reverse = create_reverse_pair(example, args.source_label)
            out_f.write(json.dumps(reverse) + '\n')
            reverse_count += 1
    
    # Summary
    console.print("\n" + "="*80)
    console.print("📊 BIDIRECTIONAL PAIR GENERATION SUMMARY")
    console.print("="*80)
    console.print(f"Input examples: {len(examples):,}")
    console.print(f"Forward pairs: {forward_count:,}")
    console.print(f"Reverse pairs: {reverse_count:,}")
    console.print(f"Total output: {forward_count + reverse_count:,}")
    console.print(f"Expansion ratio: {(forward_count + reverse_count) / len(examples):.1f}x")
    console.print(f"Output file: {output_path}")
    console.print("="*80)
    
    console.print(f"\n✅ Bidirectional pairs generated successfully!")
    
    # Validation check
    console.print(f"\n🔍 Validating output...")
    with open(output_path, 'r') as f:
        output_examples = [json.loads(line) for line in f]
    
    forward_pairs = [ex for ex in output_examples if ex['meta'].get('direction') == 'forward']
    reverse_pairs = [ex for ex in output_examples if ex['meta'].get('direction') == 'reverse']
    
    console.print(f"✅ Verified {len(output_examples):,} total pairs")
    console.print(f"   Forward: {len(forward_pairs):,}")
    console.print(f"   Reverse: {len(reverse_pairs):,}")
    
    # Show sample
    console.print(f"\n📄 Sample output (first forward+reverse pair):")
    console.print("="*80)
    console.print("[green]Forward:[/green]")
    console.print(json.dumps(forward_pairs[0], indent=2)[:500] + "...")
    console.print("\n[blue]Reverse:[/blue]")
    console.print(json.dumps(reverse_pairs[0], indent=2)[:500] + "...")
    console.print("="*80)
    
    # Next steps
    console.print(f"\n🚀 Next Steps:")
    console.print(f"1. Verify quality: head -n 4 {output_path} | python3 -m json.tool")
    console.print(f"2. Generate pairs for other dataset (self-critique or Claude)")
    console.print(f"3. Combine all bidirectional pairs:")
    console.print(f"   cat self_critique_bidirectional.jsonl claude_bidirectional.jsonl > combined_training_bidirectional.jsonl")
    console.print(f"4. Run smart training: train_phase1c_combined_smart.py")


if __name__ == "__main__":
    main()
