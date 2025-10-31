#!/usr/bin/env python3
"""
Phase 1B - Step 3: Direct File Comparison

Compares model outputs with reference answers using automated heuristics.
No LLM API calls needed - just direct file comparison.

Usage:
    python "Phase 1B_2_0/step3_compare_files.py" \
        --test_dataset ./data/phase1b/test_dataset_20k.jsonl \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/comparison_results.jsonl
        
Output:
    - comparison_results.jsonl: All results with pass/fail
    - failures.jsonl: Only failures
    - summary.json: Statistics
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation at the end
    text = re.sub(r'[.,!?;:]+$', '', text)
    return text


def calculate_similarity(text1, text2):
    """Calculate simple word overlap similarity."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def is_pass(reference, model_output, category):
    """
    Determine if model output passes using simple heuristics.
    
    Criteria:
    - Length similarity: Model output shouldn't be too short or too long
    - Word overlap: At least 30% word overlap with reference
    - Not truncated: Doesn't end mid-sentence
    - Not repetitive: Doesn't repeat the same content
    """
    ref_norm = normalize_text(reference)
    out_norm = normalize_text(model_output)
    
    # Check if output is too short (less than 20% of reference)
    if len(out_norm) < len(ref_norm) * 0.2:
        return False, "Output too short"
    
    # Check if output is absurdly long (more than 5x reference)
    if len(out_norm) > len(ref_norm) * 5:
        return False, "Output too long/verbose"
    
    # Check if truncated (ends with incomplete sentence)
    if model_output.strip().endswith(('...', 'and', 'or', 'the', 'a', 'an', 'to')):
        return False, "Output truncated"
    
    # Check for repetition (same phrase repeated 3+ times)
    words = model_output.split()
    if len(words) > 20:
        for i in range(len(words) - 15):
            phrase = ' '.join(words[i:i+5])
            rest = ' '.join(words[i+5:])
            if rest.count(phrase) >= 2:
                return False, "Output repetitive"
    
    # Calculate word overlap similarity
    similarity = calculate_similarity(reference, model_output)
    
    # Category-specific thresholds
    if category == 'math':
        # Math requires exact answers, but check for key numbers
        ref_numbers = re.findall(r'\d+', reference)
        out_numbers = re.findall(r'\d+', model_output)
        if ref_numbers and not any(num in out_numbers for num in ref_numbers[-3:]):
            return False, "Missing key numbers"
        threshold = 0.25
    elif category == 'code':
        # Code can have different implementations but similar logic
        threshold = 0.20
    else:
        # General QA, reasoning, etc.
        threshold = 0.30
    
    if similarity < threshold:
        return False, f"Low similarity ({similarity:.2f} < {threshold})"
    
    return True, "Pass"


def compare_files(test_dataset_path, model_outputs_path, output_path):
    """Compare model outputs with test dataset."""
    
    print(f"\n📂 Loading test dataset from: {test_dataset_path}")
    test_data = []
    with open(test_dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item['id'] = idx  # Add sequential ID
            # Standardize field names
            if 'response' in item and 'reference' not in item:
                item['reference'] = item['response']
            test_data.append(item)
    
    print(f"✅ Loaded {len(test_data):,} test examples")
    
    print(f"\n📂 Loading model outputs from: {model_outputs_path}")
    model_data = {}
    with open(model_outputs_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            model_data[item['id']] = item
    
    print(f"✅ Loaded {len(model_data):,} model outputs")
    
    # Compare and classify
    results = []
    failures = []
    stats = defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0})
    
    print(f"\n🔍 Comparing outputs...")
    for test_item in tqdm(test_data, desc="Comparing"):
        item_id = test_item['id']
        model_item = model_data.get(item_id)
        
        if not model_item:
            result = {
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': None,
                'category': test_item['category'],
                'status': 'FAIL',
                'reason': 'Missing model output',
                'is_failure': True
            }
            results.append(result)
            failures.append(result)
            stats[test_item['category']]['total'] += 1
            stats[test_item['category']]['fail'] += 1
            stats['overall']['total'] += 1
            stats['overall']['fail'] += 1
            continue
        
        # Compare
        passed, reason = is_pass(
            test_item['reference'],
            model_item['model_output'],
            test_item['category']
        )
        
        result = {
            'id': item_id,
            'instruction': test_item['instruction'],
            'reference': test_item['reference'],
            'model_output': model_item['model_output'],
            'category': test_item['category'],
            'status': 'PASS' if passed else 'FAIL',
            'reason': reason,
            'is_failure': not passed
        }
        
        results.append(result)
        
        if not passed:
            failures.append(result)
        
        # Update stats
        stats[test_item['category']]['total'] += 1
        if passed:
            stats[test_item['category']]['pass'] += 1
        else:
            stats[test_item['category']]['fail'] += 1
        
        stats['overall']['total'] += 1
        if passed:
            stats['overall']['pass'] += 1
        else:
            stats['overall']['fail'] += 1
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Save failures only
    failures_path = output_path.parent / "failures.jsonl"
    print(f"💾 Saving failures to: {failures_path}")
    with open(failures_path, 'w') as f:
        for failure in failures:
            f.write(json.dumps(failure) + '\n')
    
    # Save summary
    summary = {
        'total_examples': len(results),
        'total_failures': len(failures),
        'failure_rate': len(failures) / len(results) if results else 0,
        'pass_rate': 1 - (len(failures) / len(results)) if results else 0,
        'by_category': dict(stats)
    }
    
    summary_path = output_path.parent / "summary.json"
    print(f"💾 Saving summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"📊 COMPARISON SUMMARY")
    print(f"="*80)
    print(f"\n✅ Total examples: {summary['total_examples']:,}")
    print(f"✅ Passes: {summary['total_examples'] - summary['total_failures']:,} ({summary['pass_rate']*100:.1f}%)")
    print(f"❌ Failures: {summary['total_failures']:,} ({summary['failure_rate']*100:.1f}%)")
    
    print(f"\n📈 By Category:")
    for category, cat_stats in sorted(stats.items()):
        if category == 'overall':
            continue
        pass_rate = cat_stats['pass'] / cat_stats['total'] if cat_stats['total'] > 0 else 0
        print(f"   {category:15s}: {cat_stats['pass']:5,}/{cat_stats['total']:5,} pass ({pass_rate*100:5.1f}%)")
    
    print(f"\n✨ Done! Results saved to: {output_path.parent}")
    
    return results, failures, summary


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Direct file comparison (no LLM needed)"
    )
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to test dataset from Step 1"
    )
    
    parser.add_argument(
        "--model_outputs",
        type=str,
        required=True,
        help="Path to model outputs from Step 2"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save comparison results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PHASE 1B STEP 3: DIRECT FILE COMPARISON")
    print("=" * 80)
    print(f"\nTest dataset: {args.test_dataset}")
    print(f"Model outputs: {args.model_outputs}")
    print(f"Output path: {args.output_path}")
    
    results, failures, summary = compare_files(
        args.test_dataset,
        args.model_outputs,
        args.output_path
    )


if __name__ == "__main__":
    main()
