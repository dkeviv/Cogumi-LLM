#!/usr/bin/env python3
"""
Phase 1B - Step 3: Evaluate Single Batch for Testing

This script evaluates a small batch (e.g., 100 examples) to:
1. Test the evaluation criteria
2. Calibrate what counts as PASS vs FAIL
3. Validate the approach before scaling to 20K examples

Usage:
    # Evaluate first 100 examples
    python "Phase 1B_2_0/step3_evaluate_batch.py" \\
        --test_dataset "./Phase 1B_2_0/data/test_dataset_20k.jsonl" \\
        --model_outputs "./Phase 1B_2_0/data/model_outputs_20k.jsonl" \\
        --output_path "./Phase 1B_2_0/data/batch_evaluation_100.jsonl" \\
        --batch_size 100 \\
        --start_index 0

Then manually review and provide judgments for the batch.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def create_compact_prompt(instruction, reference, model_output, category):
    """Create a compact evaluation prompt suitable for Copilot review."""
    
    # Truncate very long outputs for readability
    max_len = 1000
    if len(model_output) > max_len:
        model_output = model_output[:max_len] + f"\n... [truncated, {len(model_output)} total chars]"
    if len(reference) > max_len:
        reference = reference[:max_len] + f"\n... [truncated, {len(reference)} total chars]"
    
    return f"""
EVALUATE THIS OUTPUT:

Category: {category}
Instruction: {instruction}

Reference Answer:
{reference}

Model Output:
{model_output}

Criteria:
✓ CORRECTNESS: Is the core answer right?
✓ COMPLETENESS: Addresses all key points?
✓ ACCURACY: No factual errors?
✓ RELEVANCE: On topic?

Note: Verbose but correct = PASS
Note: Different phrasing = OK

Your judgment: PASS or FAIL?
Reason (1 sentence):
"""


def prepare_batch(test_dataset_path, model_outputs_path, start_index=0, batch_size=100):
    """Prepare a batch of examples for evaluation."""
    
    print(f"\n📂 Loading test dataset...")
    test_data = []
    with open(test_dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if len(test_data) >= batch_size:
                break
            
            item = json.loads(line.strip())
            item['id'] = idx
            if 'response' in item and 'reference' not in item:
                item['reference'] = item['response']
            test_data.append(item)
    
    print(f"✅ Loaded {len(test_data)} test examples (indices {start_index}-{start_index+len(test_data)-1})")
    
    print(f"\n📂 Loading model outputs...")
    model_data = {}
    with open(model_outputs_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if start_index <= item['id'] < start_index + batch_size:
                model_data[item['id']] = item
    
    print(f"✅ Loaded {len(model_data)} model outputs")
    
    # Prepare batch
    batch = []
    for test_item in test_data:
        item_id = test_item['id']
        model_item = model_data.get(item_id)
        
        if not model_item:
            batch.append({
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': "[NO OUTPUT]",
                'category': test_item['category'],
                'status': 'FAIL',
                'reason': 'Missing model output',
                'prompt': None
            })
        else:
            prompt = create_compact_prompt(
                test_item['instruction'],
                test_item['reference'],
                model_item['model_output'],
                test_item['category']
            )
            
            batch.append({
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': model_item['model_output'],
                'category': test_item['category'],
                'status': None,  # To be filled by evaluator
                'reason': None,  # To be filled by evaluator
                'prompt': prompt
            })
    
    return batch


def save_batch(batch, output_path):
    """Save batch for review."""
    with open(output_path, 'w') as f:
        for item in batch:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Saved batch to: {output_path}")


def print_batch_for_review(batch, max_examples=10):
    """Print batch in readable format for manual review."""
    print("\n" + "="*80)
    print("BATCH FOR REVIEW")
    print("="*80)
    
    for i, item in enumerate(batch[:max_examples]):
        if item['prompt']:
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i+1} / {len(batch)} (ID: {item['id']})")
            print(f"{'='*80}")
            print(item['prompt'])
            print(f"\n{'-'*80}")
            input(f"Press Enter to continue to next example...")
    
    if len(batch) > max_examples:
        print(f"\n... and {len(batch) - max_examples} more examples")


def analyze_batch_stats(batch):
    """Analyze batch statistics."""
    stats = defaultdict(lambda: {'total': 0, 'needs_review': 0})
    
    for item in batch:
        cat = item['category']
        stats[cat]['total'] += 1
        if item['status'] is None:
            stats[cat]['needs_review'] += 1
        stats['overall']['total'] += 1
        if item['status'] is None:
            stats['overall']['needs_review'] += 1
    
    print("\n📊 BATCH STATISTICS:")
    print(f"{'Category':<15} {'Total':<10} {'Needs Review':<15}")
    print("-" * 40)
    
    for cat in sorted(stats.keys()):
        if cat == 'overall':
            print("-" * 40)
        print(f"{cat:<15} {stats[cat]['total']:<10} {stats[cat]['needs_review']:<15}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a batch of examples')
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--model_outputs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--print_preview', action='store_true',
                        help='Print first 10 examples for preview')
    
    args = parser.parse_args()
    
    # Prepare batch
    batch = prepare_batch(
        args.test_dataset,
        args.model_outputs,
        start_index=args.start_index,
        batch_size=args.batch_size
    )
    
    # Save batch
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_batch(batch, args.output_path)
    
    # Analyze
    analyze_batch_stats(batch)
    
    # Print preview if requested
    if args.print_preview:
        print_batch_for_review(batch, max_examples=10)
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review the batch file: {args.output_path}")
    print(f"   2. For each example, provide: status (PASS/FAIL) and reason")
    print(f"   3. Use this to calibrate the evaluation criteria")
    print(f"   4. Once criteria are validated, scale to full 20K examples")


if __name__ == '__main__':
    main()
