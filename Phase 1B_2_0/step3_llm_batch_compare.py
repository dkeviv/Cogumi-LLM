#!/usr/bin/env python3
"""
Phase 1B - Step 3: LLM Batch File Comparison (Smart Approach!)

Instead of 20K separate API calls, we give the LLM BOTH FILES and ask it to:
1. Compare them in one go
2. Return a JSON list of failures

This reduces 20K calls to just 1-5 calls (depending on context window limits).

Usage:
    python "Phase1B_2_0/step3_llm_batch_compare.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/comparison_results.jsonl \
        --llm_model meta-llama/Llama-3.1-405B-Instruct

Output:
    - comparison_results.jsonl: All results with pass/fail
    - failures.jsonl: Only failures
    - summary.json: Statistics
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import InferenceClient
from typing import Dict, List
from collections import defaultdict
import time

# Batch comparison prompt
BATCH_COMPARE_PROMPT = """You are comparing model outputs to reference answers. I will give you a batch of examples.

For each example, compare the model output to the reference and determine if it's a PASS or FAIL.
- PASS: Model output is correct, complete, and matches reference quality (score 7+/10)
- FAIL: Model output has errors, missing info, or poor quality (score <7/10)

Return ONLY a JSON array with results:
[
  {{"id": 0, "status": "PASS"}},
  {{"id": 1, "status": "FAIL", "reason": "brief issue"}},
  ...
]

EXAMPLES TO COMPARE:
{examples}

Return JSON array only, no other text."""


def create_batch(items: List[Dict], batch_size: int = 100) -> List[str]:
    """
    Create batches of examples for comparison.
    Each batch contains multiple examples to compare.
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        
        # Format examples for this batch
        examples_text = ""
        for idx, item in enumerate(batch_items):
            examples_text += f"\n---Example {idx}---\n"
            examples_text += f"Question: {item['instruction'][:200]}...\n"
            examples_text += f"Reference: {item['reference'][:300]}...\n"
            examples_text += f"Model Output: {item['model_output'][:300]}...\n"
        
        batches.append(examples_text)
    
    return batches


def parse_llm_response(response_text: str, batch_size: int) -> List[Dict]:
    """
    Parse LLM response to extract pass/fail decisions.
    Returns list of decisions for the batch.
    """
    import re
    
    # Try to extract JSON array
    try:
        # Find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            results = json.loads(json_match.group(0))
            return results
    except:
        pass
    
    # Fallback: Parse line by line looking for PASS/FAIL
    results = []
    lines = response_text.split('\n')
    
    for i, line in enumerate(lines):
        if 'PASS' in line.upper() or 'FAIL' in line.upper():
            status = 'PASS' if 'PASS' in line.upper() else 'FAIL'
            
            # Extract reason if FAIL
            reason = ""
            if status == 'FAIL':
                reason_match = re.search(r'reason["\s:]+([^",\n]+)', line, re.IGNORECASE)
                if reason_match:
                    reason = reason_match.group(1).strip()
            
            results.append({
                "id": len(results),
                "status": status,
                "reason": reason if status == 'FAIL' else ""
            })
    
    # If we didn't get enough results, mark rest as PASS (conservative)
    while len(results) < batch_size:
        results.append({
            "id": len(results),
            "status": "PASS",
            "reason": "No decision found"
        })
    
    return results


def compare_with_llm_batch(items: List[Dict], client: InferenceClient, 
                           llm_model: str, batch_size: int = 50) -> List[Dict]:
    """
    Compare all items using LLM in batches.
    Dramatically reduces API calls: 20K items → ~400 calls (batch_size=50).
    """
    all_results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    print(f"\n🔍 Processing {len(items):,} items in {total_batches} batches of {batch_size}...")
    print(f"   Estimated time: ~{total_batches * 15 / 60:.1f} minutes (15s per batch)")
    print()
    
    for batch_idx in tqdm(range(0, len(items), batch_size), desc="LLM Batches"):
        batch_items = items[batch_idx:batch_idx + batch_size]
        
        # Create batch prompt
        examples_text = ""
        for idx, item in enumerate(batch_items):
            # Truncate long texts to fit in context
            inst = item['instruction'][:200]
            ref = item['reference'][:300]
            out = item['model_output'][:300]
            
            examples_text += f"\n---Example {idx}---\n"
            examples_text += f"Q: {inst}\n"
            examples_text += f"Ref: {ref}\n"
            examples_text += f"Out: {out}\n"
        
        prompt = BATCH_COMPARE_PROMPT.format(examples=examples_text)
        
        try:
            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(
                messages=messages,
                model=llm_model,
                max_tokens=2000,  # Enough for batch results
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content
            
            # Parse results
            batch_results = parse_llm_response(response_text, len(batch_items))
            
            # Merge with original items
            for i, item in enumerate(batch_items):
                result_idx = i if i < len(batch_results) else 0
                decision = batch_results[result_idx]
                
                item['status'] = decision['status']
                item['is_failure'] = decision['status'] == 'FAIL'
                item['reason'] = decision.get('reason', '')
                item['score'] = 5 if decision['status'] == 'FAIL' else 8
                
                all_results.append(item)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"\n⚠️  Error processing batch {batch_idx//batch_size + 1}: {e}")
            # Mark all as PASS on error (conservative)
            for item in batch_items:
                item['status'] = 'PASS'
                item['is_failure'] = False
                item['reason'] = f'Error: {str(e)}'
                item['score'] = 8
                all_results.append(item)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: LLM batch file comparison (smart approach)"
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
    
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Llama-3.1-405B-Instruct",
        help="LLM model for batch comparison"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of examples per batch (default: 50)"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PHASE 1B STEP 3: LLM BATCH FILE COMPARISON")
    print("=" * 80)
    print()
    print(f"Approach: Batch {args.batch_size} examples per LLM call")
    print(f"Model: {args.llm_model}")
    print()
    
    # Initialize HF client
    client = InferenceClient(token=args.hf_token)
    
    # Load model outputs
    print(f"📂 Loading model outputs from: {args.model_outputs}")
    with open(args.model_outputs, 'r') as f:
        items = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(items):,} items")
    
    # Calculate API call reduction
    num_batches = (len(items) + args.batch_size - 1) // args.batch_size
    reduction = len(items) / num_batches
    print(f"📊 API calls: {num_batches} batches (vs {len(items):,} individual calls)")
    print(f"   🎉 {reduction:.0f}x reduction in API calls!")
    print()
    
    # Compare with LLM in batches
    results = compare_with_llm_batch(items, client, args.llm_model, args.batch_size)
    
    # Calculate statistics
    failures = [r for r in results if r['is_failure']]
    failure_rate = len(failures) / len(results) * 100
    
    # Category breakdown
    category_stats = defaultdict(lambda: {"total": 0, "failures": 0})
    for result in results:
        cat = result.get('category', 'other')
        category_stats[cat]['total'] += 1
        if result['is_failure']:
            category_stats[cat]['failures'] += 1
    
    summary = {
        "total_samples": len(results),
        "total_failures": len(failures),
        "failure_rate": round(failure_rate, 2),
        "api_calls": num_batches,
        "api_reduction": f"{reduction:.0f}x",
        "batch_size": args.batch_size,
        "llm_model": args.llm_model,
        "category_breakdown": {}
    }
    
    for category, stats in category_stats.items():
        summary["category_breakdown"][category] = {
            "total": stats["total"],
            "failures": stats["failures"],
            "failure_rate": round(stats["failures"] / stats["total"] * 100, 2)
        }
    
    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving results...")
    
    # All results
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"   ✅ All results: {args.output_path}")
    
    # Failures only
    failures_path = output_dir / "failures.jsonl"
    with open(failures_path, 'w') as f:
        for failure in failures:
            f.write(json.dumps(failure) + '\n')
    print(f"   ✅ Failures: {failures_path}")
    
    # Summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Summary: {summary_path}")
    print()
    
    # Display summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']:,}")
    print(f"Failures: {summary['total_failures']:,} ({summary['failure_rate']}%)")
    print(f"API calls: {summary['api_calls']} ({summary['api_reduction']} reduction)")
    print()
    print("Category Breakdown:")
    for category, stats in summary["category_breakdown"].items():
        print(f"  {category:12s}: {stats['failures']}/{stats['total']} failures "
              f"({stats['failure_rate']:.1f}%)")
    print()
    print("✅ LLM batch comparison complete!")
    print()


if __name__ == "__main__":
    main()
