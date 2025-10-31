#!/usr/bin/env python3
"""
Process ChatGPT comparison results.

After ChatGPT returns the JSON, save it as 'chatgpt_results.json' and run:
    python "Phase 1B_2_0/process_chatgpt_results.py" \\
        --chatgpt_results ./chatgpt_results.json \\
        --output_dir "./Phase 1B_2_0/data/"

This will create:
- comparison_results_chatgpt.jsonl (all results)
- failures_chatgpt.jsonl (only failures for Phase 1B Step 4)
- summary_chatgpt.json (statistics)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def process_chatgpt_results(chatgpt_json_path, output_dir):
    """Process ChatGPT comparison results into usable formats."""
    
    print(f"\n📂 Loading ChatGPT results from: {chatgpt_json_path}")
    with open(chatgpt_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract results
    all_results = data.get('all_results', [])
    summary = data.get('summary', {})
    failure_analysis = data.get('failure_analysis', {})
    
    print(f"✅ Loaded {len(all_results):,} comparison results")
    
    # Create output files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results_path = output_dir / 'comparison_results_chatgpt.jsonl'
    failures_path = output_dir / 'failures_chatgpt.jsonl'
    summary_path = output_dir / 'summary_chatgpt.json'
    
    # Write all results
    print(f"\n💾 Writing results...")
    failures_count = 0
    with open(all_results_path, 'w') as f_all, open(failures_path, 'w') as f_fail:
        for result in all_results:
            f_all.write(json.dumps(result) + '\n')
            
            if result.get('status') == 'FAIL':
                f_fail.write(json.dumps(result) + '\n')
                failures_count += 1
    
    print(f"✅ Wrote {len(all_results):,} results to: {all_results_path}")
    print(f"✅ Wrote {failures_count:,} failures to: {failures_path}")
    
    # Write summary with failure analysis
    full_summary = {
        'summary': summary,
        'failure_analysis': failure_analysis,
        'files': {
            'all_results': str(all_results_path),
            'failures': str(failures_path)
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    
    print(f"✅ Wrote summary to: {summary_path}")
    
    # Print summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   Total examples: {summary.get('total_examples', len(all_results)):,}")
    print(f"   Passes: {summary.get('passes', 0):,} ({summary.get('pass_rate', 0):.1f}%)")
    print(f"   Failures: {summary.get('failures', 0):,} ({summary.get('failure_rate', 0):.1f}%)")
    
    print(f"\n📊 BY CATEGORY:")
    by_cat = summary.get('by_category', {})
    for cat, stats in sorted(by_cat.items()):
        print(f"   {cat:<12} {stats.get('total', 0):>5} total, "
              f"{stats.get('passes', 0):>5} pass ({stats.get('pass_rate', 0):>5.1f}%), "
              f"{stats.get('failures', 0):>5} fail")
    
    if failure_analysis:
        print(f"\n🔍 FAILURE ANALYSIS:")
        patterns = failure_analysis.get('common_patterns', [])
        if patterns:
            print(f"   Common patterns:")
            for i, pattern in enumerate(patterns[:5], 1):
                print(f"   {i}. {pattern}")
        
        prob_cats = failure_analysis.get('most_problematic_categories', [])
        if prob_cats:
            print(f"\n   Most problematic categories: {', '.join(prob_cats)}")
    
    # Validation
    expected_pass_rate = 0.72
    actual_pass_rate = summary.get('pass_rate', 0) / 100
    
    print(f"\n📈 VALIDATION:")
    print(f"   Expected pass rate: {expected_pass_rate*100:.1f}%")
    print(f"   Actual pass rate: {actual_pass_rate*100:.1f}%")
    
    if actual_pass_rate >= expected_pass_rate:
        print(f"   ✅ MEETS EXPECTATIONS!")
    elif actual_pass_rate >= expected_pass_rate - 0.10:
        print(f"   ⚠️  SLIGHTLY BELOW (within 10%)")
    else:
        print(f"   ❌ BELOW EXPECTATIONS")
    
    print(f"\n✅ Processing complete!")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review failures in: {failures_path}")
    print(f"   2. Proceed to Phase 1B Step 4: Cluster {failures_count:,} failures")
    print(f"   3. Use clustering to identify 8-12 weakness categories")
    
    return all_results_path, failures_path, summary_path


def main():
    parser = argparse.ArgumentParser(description='Process ChatGPT comparison results')
    parser.add_argument('--chatgpt_results', type=str, required=True,
                        help='Path to ChatGPT JSON results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    
    args = parser.parse_args()
    
    process_chatgpt_results(args.chatgpt_results, args.output_dir)


if __name__ == '__main__':
    main()
