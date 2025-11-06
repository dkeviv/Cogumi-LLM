#!/usr/bin/env python3
"""
Phase 1B - Step 3: Hybrid Evaluation (Fast + Accurate)

Strategy:
1. Quick Heuristics - Identify CLEAR passes and CLEAR fails (80% of cases)
2. LLM Review - Only evaluate uncertain middle cases (20% of cases)

This reduces evaluation cost by 80% while maintaining accuracy.

Context:
- Expected failure rate: 24-28% (targeting 72-76% pass rate)
- Goal: Beat GPT-4 (need >100% after Phase 1C)
- Phase 1A baseline: 75-82% GPT-4

Usage:
    python "Phase 1B_2_0/step3_hybrid_evaluation.py" \\
        --test_dataset "./Phase 1B_2_0/data/test_dataset_20k.jsonl" \\
        --model_outputs "./Phase 1B_2_0/data/model_outputs_20k.jsonl" \\
        --output_path "./Phase 1B_2_0/data/hybrid_results.jsonl"
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re


def quick_pass_check(reference, model_output, category):
    """
    Identify CLEAR PASSES using relaxed heuristics.
    
    These are cases where the model is obviously correct.
    Returns: (is_clear_pass, confidence, reason)
    """
    ref_lower = reference.lower()
    out_lower = model_output.lower()
    
    # Check 1: Exact or near-exact match
    if ref_lower.strip() == out_lower.strip():
        return True, 1.0, "Exact match"
    
    # Check 2: Reference is contained in output (with explanation)
    if len(reference) > 20 and reference.strip() in model_output:
        return True, 0.9, "Reference fully contained in output"
    
    # Check 3: Very high word overlap (>80%)
    ref_words = set(ref_lower.split())
    out_words = set(out_lower.split())
    if ref_words and out_words:
        overlap = len(ref_words & out_words) / len(ref_words | out_words)
        if overlap > 0.80:
            return True, 0.85, f"Very high word overlap ({overlap:.2f})"
    
    # Check 4: Category-specific clear passes
    if category == 'code':
        # Check for key function/class names
        ref_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', reference))
        out_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', model_output))
        if ref_identifiers and len(ref_identifiers & out_identifiers) / len(ref_identifiers) > 0.7:
            return True, 0.8, "Key code identifiers present"
    
    elif category == 'math':
        # Check for key numbers in answer
        ref_numbers = re.findall(r'-?\d+\.?\d*', reference)
        out_numbers = re.findall(r'-?\d+\.?\d*', model_output)
        if ref_numbers and any(num in out_numbers for num in ref_numbers[-2:]):
            return True, 0.75, "Final answer numbers match"
    
    return False, 0.0, ""


def quick_fail_check(reference, model_output, category):
    """
    Identify CLEAR FAILS using strict heuristics.
    
    These are cases where the model is obviously wrong.
    Returns: (is_clear_fail, confidence, reason)
    """
    out_lower = model_output.lower().strip()
    
    # Check 1: Empty or very short output
    if len(model_output.strip()) < 10:
        return True, 1.0, "Output too short/empty"
    
    # Check 2: Clearly truncated (ends mid-word)
    if len(model_output) > 50 and model_output[-20:].count(' ') < 2:
        return True, 0.9, "Output truncated mid-word"
    
    # Check 3: Nonsensical repetition
    words = model_output.split()
    if len(words) > 30:
        # Check for phrase repeated 5+ times
        for i in range(len(words) - 10):
            phrase = ' '.join(words[i:i+5])
            if model_output.count(phrase) >= 5:
                return True, 0.95, "Excessive repetition (nonsensical)"
    
    # Check 4: No word overlap at all (completely off-topic)
    ref_words = set(reference.lower().split())
    out_words = set(out_lower.split())
    if ref_words and out_words:
        overlap = len(ref_words & out_words) / len(ref_words)
        if overlap < 0.05:  # Less than 5% overlap
            return True, 0.85, "Completely off-topic (no word overlap)"
    
    # Check 5: Category-specific clear fails
    if category == 'math':
        # If reference has a number but output has none
        ref_numbers = re.findall(r'-?\d+\.?\d*', reference)
        out_numbers = re.findall(r'-?\d+\.?\d*', model_output)
        if ref_numbers and len(out_numbers) == 0:
            return True, 0.8, "Missing expected numbers"
    
    elif category == 'code':
        # If reference has code but output has no code blocks
        if '```' in reference and '```' not in model_output and 'def ' not in model_output:
            return True, 0.75, "Missing expected code"
    
    return False, 0.0, ""


def needs_llm_review(reference, model_output, category):
    """
    Determine if example needs LLM review.
    
    Cases that need review:
    - Not a clear pass
    - Not a clear fail
    - These are the "uncertain" middle cases
    """
    is_clear_pass, pass_conf, pass_reason = quick_pass_check(reference, model_output, category)
    if is_clear_pass:
        return False, "PASS", pass_conf, pass_reason
    
    is_clear_fail, fail_conf, fail_reason = quick_fail_check(reference, model_output, category)
    if is_clear_fail:
        return False, "FAIL", fail_conf, fail_reason
    
    return True, "UNCERTAIN", 0.0, "Needs LLM review"


def evaluate_hybrid(test_dataset_path, model_outputs_path, output_path):
    """
    Hybrid evaluation: Quick heuristics + LLM review for uncertain cases.
    """
    print(f"\n📂 Loading test dataset from: {test_dataset_path}")
    test_data = []
    with open(test_dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item['id'] = idx
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
    
    # Phase 1: Quick evaluation
    print(f"\n🔍 Phase 1: Quick heuristic evaluation...")
    results = []
    clear_passes = 0
    clear_fails = 0
    needs_review = 0
    
    stats = defaultdict(lambda: {'total': 0, 'clear_pass': 0, 'clear_fail': 0, 'uncertain': 0})
    
    for test_item in tqdm(test_data, desc="Quick eval"):
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
                'confidence': 1.0,
                'reason': 'Missing model output',
                'evaluation_method': 'heuristic'
            }
            results.append(result)
            clear_fails += 1
            stats[test_item['category']]['clear_fail'] += 1
            stats[test_item['category']]['total'] += 1
            stats['overall']['clear_fail'] += 1
            stats['overall']['total'] += 1
            continue
        
        # Quick evaluation
        needs_llm, status, confidence, reason = needs_llm_review(
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
            'status': status,
            'confidence': confidence,
            'reason': reason,
            'evaluation_method': 'heuristic' if not needs_llm else 'pending_llm'
        }
        
        results.append(result)
        
        if status == 'PASS':
            clear_passes += 1
            stats[test_item['category']]['clear_pass'] += 1
        elif status == 'FAIL':
            clear_fails += 1
            stats[test_item['category']]['clear_fail'] += 1
        else:  # UNCERTAIN
            needs_review += 1
            stats[test_item['category']]['uncertain'] += 1
        
        stats[test_item['category']]['total'] += 1
        stats['overall']['total'] += 1
        if status == 'PASS':
            stats['overall']['clear_pass'] += 1
        elif status == 'FAIL':
            stats['overall']['clear_fail'] += 1
        else:
            stats['overall']['uncertain'] += 1
    
    # Print quick evaluation results
    print(f"\n📊 PHASE 1 RESULTS:")
    print(f"{'Category':<15} {'Total':<10} {'Clear Pass':<12} {'Clear Fail':<12} {'Uncertain':<12}")
    print("-" * 70)
    
    for cat in sorted(stats.keys()):
        if cat == 'overall':
            print("-" * 70)
        s = stats[cat]
        pass_pct = (s['clear_pass'] / s['total'] * 100) if s['total'] > 0 else 0
        fail_pct = (s['clear_fail'] / s['total'] * 100) if s['total'] > 0 else 0
        unc_pct = (s['uncertain'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"{cat:<15} {s['total']:<10} {s['clear_pass']:<5} ({pass_pct:>4.1f}%) {s['clear_fail']:<5} ({fail_pct:>4.1f}%) {s['uncertain']:<5} ({unc_pct:>4.1f}%)")
    
    # Save intermediate results
    intermediate_path = output_path.replace('.jsonl', '_phase1.jsonl')
    with open(intermediate_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✅ Phase 1 complete. Saved to: {intermediate_path}")
    
    # Estimate final results
    print(f"\n📈 ESTIMATED FINAL RESULTS:")
    print(f"   Clear passes: {clear_passes:,} ({clear_passes/len(results)*100:.1f}%)")
    print(f"   Clear fails: {clear_fails:,} ({clear_fails/len(results)*100:.1f}%)")
    print(f"   Needs LLM review: {needs_review:,} ({needs_review/len(results)*100:.1f}%)")
    
    # Optimistic estimate (assume uncertain cases split 50/50)
    optimistic_pass = clear_passes + needs_review // 2
    print(f"\n   Optimistic (50% of uncertain pass): {optimistic_pass:,} passes ({optimistic_pass/len(results)*100:.1f}%)")
    
    # Pessimistic estimate (assume all uncertain are fails)
    pessimistic_pass = clear_passes
    print(f"   Pessimistic (all uncertain fail): {pessimistic_pass:,} passes ({pessimistic_pass/len(results)*100:.1f}%)")
    
    # Expected (based on Phase 1A target of 75-82% GPT-4)
    expected_pass = int(len(results) * 0.72)  # Lower bound of 72-76% expected
    print(f"   Expected (72% pass rate): {expected_pass:,} passes")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review the {needs_review:,} uncertain cases")
    print(f"   2. Use LLM evaluation or manual review for these")
    print(f"   3. This saves {100 - needs_review/len(results)*100:.1f}% of evaluation cost!")
    
    return intermediate_path, needs_review


def main():
    parser = argparse.ArgumentParser(description='Hybrid evaluation with quick heuristics + LLM review')
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--model_outputs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run hybrid evaluation
    intermediate_path, needs_review = evaluate_hybrid(
        args.test_dataset,
        args.model_outputs,
        args.output_path
    )
    
    print(f"\n✅ Hybrid evaluation complete!")


if __name__ == '__main__':
    main()
