#!/usr/bin/env python3
"""
Phase 1B - Step 3B: Merge Copilot Comparison Results

Combines all JSON responses from Copilot Chat into final results.

Usage:
    python "Phase1B_2_0/step3_merge_copilot_results.py" \
        --responses_dir ./data/phase1b/responses \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/comparison_results.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_responses(responses_dir):
    """Load all Copilot JSON responses."""
    responses_dir = Path(responses_dir)
    response_files = sorted(responses_dir.glob("response_*.json"))
    
    print(f"📂 Loading {len(response_files)} response files...")
    
    all_decisions = []
    for response_file in response_files:
        with open(response_file, 'r') as f:
            decisions = json.load(f)
            all_decisions.extend(decisions)
    
    print(f"✅ Loaded {len(all_decisions)} decisions")
    return all_decisions


def merge_results(items, decisions):
    """Merge Copilot decisions with original items."""
    print(f"\n🔗 Merging {len(decisions)} decisions with {len(items)} items...")
    
    results = []
    failures = []
    category_stats = defaultdict(lambda: {"total": 0, "failures": 0})
    
    for i, item in enumerate(items):
        # Get decision for this item
        decision = decisions[i] if i < len(decisions) else {"status": "PASS", "reason": "No decision"}
        
        # Merge
        result = {
            **item,
            "status": decision.get("status", "PASS"),
            "is_failure": decision.get("status", "PASS") == "FAIL",
            "reason": decision.get("reason", ""),
            "score": 5 if decision.get("status") == "FAIL" else 8
        }
        
        results.append(result)
        
        if result["is_failure"]:
            failures.append(result)
        
        # Category stats
        cat = item.get("category", "other")
        category_stats[cat]["total"] += 1
        if result["is_failure"]:
            category_stats[cat]["failures"] += 1
    
    return results, failures, category_stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge Copilot comparison results"
    )
    
    parser.add_argument(
        "--responses_dir",
        type=str,
        required=True,
        help="Directory containing Copilot JSON responses"
    )
    
    parser.add_argument(
        "--model_outputs",
        type=str,
        required=True,
        help="Path to original model outputs"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔗 MERGING COPILOT COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Load original items
    print(f"📂 Loading model outputs from: {args.model_outputs}")
    with open(args.model_outputs, 'r') as f:
        items = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(items):,} items")
    
    # Load Copilot decisions
    decisions = load_responses(args.responses_dir)
    
    # Merge
    results, failures, category_stats = merge_results(items, decisions)
    
    # Calculate summary
    failure_rate = len(failures) / len(results) * 100
    
    summary = {
        "total_samples": len(results),
        "total_failures": len(failures),
        "failure_rate": round(failure_rate, 2),
        "comparison_method": "Copilot Chat (Claude Sonnet 4.5)",
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
    print()
    print("Category Breakdown:")
    for category, stats in summary["category_breakdown"].items():
        print(f"  {category:12s}: {stats['failures']}/{stats['total']} failures "
              f"({stats['failure_rate']:.1f}%)")
    print()
    print("✅ Copilot comparison complete!")
    print()


if __name__ == "__main__":
    main()
