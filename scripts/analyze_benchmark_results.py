#!/usr/bin/env python3
"""
Analyze Phase 1B Benchmark Results
Parse the benchmark output and identify issues with responses or judging
"""

import json
from pathlib import Path
import sys
from collections import defaultdict

def analyze_category_results(results_dir, category):
    """Analyze results for a specific category."""
    results_file = Path(results_dir) / f"category_{category}.json"
    
    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        return None
    
    with open(results_file) as f:
        data = json.load(f)
    
    details = data.get('details', [])
    
    print(f"\n{'='*80}")
    print(f"  {category.upper()} ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📊 Overall Stats:")
    print(f"   Total samples: {len(details)}")
    print(f"   Wins: {data.get('wins', 0)} ({data.get('wins', 0)/len(details)*100:.1f}%)")
    print(f"   Losses: {data.get('losses', 0)} ({data.get('losses', 0)/len(details)*100:.1f}%)")
    print(f"   Ties: {data.get('ties', 0)} ({data.get('ties', 0)/len(details)*100:.1f}%)")
    print(f"   Score: {data.get('score', 0)*100:.1f}%")
    
    # Analyze wins, losses, ties
    wins = [d for d in details if d.get('judgment', {}).get('winner') == 'A']
    losses = [d for d in details if d.get('judgment', {}).get('winner') == 'B']
    ties = [d for d in details if d.get('judgment', {}).get('winner') == 'TIE']
    
    # Show examples of each
    print(f"\n{'─'*80}")
    print("EXAMPLE WINS (Local model better than GPT-4)")
    print(f"{'─'*80}")
    for i, example in enumerate(wins[:2], 1):
        print(f"\nWin #{i}:")
        print(f"Prompt: {example['prompt'][:100]}...")
        print(f"\nLocal response: {example['local_response'][:200]}...")
        print(f"\nGPT-4 response: {example['gpt4_response'][:200]}...")
        print(f"\nJudge reasoning: {example['judgment'].get('reasoning', 'N/A')}")
    
    print(f"\n{'─'*80}")
    print("EXAMPLE LOSSES (GPT-4 better than local model)")
    print(f"{'─'*80}")
    for i, example in enumerate(losses[:2], 1):
        print(f"\nLoss #{i}:")
        print(f"Prompt: {example['prompt'][:100]}...")
        print(f"\nLocal response: {example['local_response'][:200]}...")
        print(f"\nGPT-4 response: {example['gpt4_response'][:200]}...")
        print(f"\nJudge reasoning: {example['judgment'].get('reasoning', 'N/A')}")
    
    print(f"\n{'─'*80}")
    print("EXAMPLE TIES (Similar quality)")
    print(f"{'─'*80}")
    for i, example in enumerate(ties[:2], 1):
        print(f"\nTie #{i}:")
        print(f"Prompt: {example['prompt'][:100]}...")
        print(f"\nLocal response: {example['local_response'][:200]}...")
        print(f"\nGPT-4 response: {example['gpt4_response'][:200]}...")
        print(f"\nJudge reasoning: {example['judgment'].get('reasoning', 'N/A')}")
    
    # Response quality analysis
    print(f"\n{'─'*80}")
    print("RESPONSE QUALITY METRICS")
    print(f"{'─'*80}")
    
    local_lengths = [len(d['local_response']) for d in details]
    gpt4_lengths = [len(d['gpt4_response']) for d in details]
    
    print(f"\n📏 Response lengths (chars):")
    print(f"   Local model - Avg: {sum(local_lengths)/len(local_lengths):.0f}, "
          f"Min: {min(local_lengths)}, Max: {max(local_lengths)}")
    print(f"   GPT-4 - Avg: {sum(gpt4_lengths)/len(gpt4_lengths):.0f}, "
          f"Min: {min(gpt4_lengths)}, Max: {max(gpt4_lengths)}")
    
    # Check for problematic responses
    print(f"\n⚠️  Potential issues:")
    
    issues = defaultdict(int)
    for d in details:
        local_resp = d['local_response']
        
        if len(local_resp) < 20:
            issues['Too short (<20 chars)'] += 1
        if '<|' in local_resp or '|>' in local_resp:
            issues['Contains chat markers'] += 1
        if local_resp.startswith(d['prompt'][:20]):
            issues['Repeats prompt'] += 1
        if not any(c.isalpha() for c in local_resp):
            issues['No letters'] += 1
    
    if issues:
        for issue, count in issues.items():
            print(f"   - {issue}: {count} samples ({count/len(details)*100:.1f}%)")
    else:
        print(f"   ✓ No obvious issues detected")
    
    # Score analysis
    print(f"\n{'─'*80}")
    print("SCORING ANALYSIS")
    print(f"{'─'*80}")
    
    # Extract scores from judgments
    local_scores = defaultdict(list)
    gpt4_scores = defaultdict(list)
    
    for d in details:
        judgment = d.get('judgment', {})
        resp_a = judgment.get('response_a', {})
        resp_b = judgment.get('response_b', {})
        
        for criterion in ['correctness', 'completeness', 'clarity', 'relevance']:
            if criterion in resp_a:
                local_scores[criterion].append(resp_a[criterion])
            if criterion in resp_b:
                gpt4_scores[criterion].append(resp_b[criterion])
    
    print(f"\n📊 Average scores (1-10 scale):")
    print(f"{'Criterion':<15} {'Local':<10} {'GPT-4':<10} {'Diff':<10}")
    print(f"{'─'*45}")
    
    for criterion in ['correctness', 'completeness', 'clarity', 'relevance']:
        local_avg = sum(local_scores[criterion]) / len(local_scores[criterion]) if local_scores[criterion] else 0
        gpt4_avg = sum(gpt4_scores[criterion]) / len(gpt4_scores[criterion]) if gpt4_scores[criterion] else 0
        diff = local_avg - gpt4_avg
        
        print(f"{criterion:<15} {local_avg:<10.2f} {gpt4_avg:<10.2f} {diff:+.2f}")
    
    # Overall average
    all_local = [s for scores in local_scores.values() for s in scores]
    all_gpt4 = [s for scores in gpt4_scores.values() for s in scores]
    
    if all_local and all_gpt4:
        local_overall = sum(all_local) / len(all_local)
        gpt4_overall = sum(all_gpt4) / len(all_gpt4)
        print(f"{'─'*45}")
        print(f"{'OVERALL':<15} {local_overall:<10.2f} {gpt4_overall:<10.2f} {local_overall - gpt4_overall:+.2f}")
    
    return data

def main():
    """Analyze all available benchmark results."""
    print("="*80)
    print("  BENCHMARK RESULTS ANALYSIS")
    print("="*80)
    
    results_dir = Path("/workspace/data/Cogumi-LLM/checkpoints/benchmark_results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("   Make sure benchmark has run and saved results")
        sys.exit(1)
    
    print(f"\n📁 Results directory: {results_dir}")
    
    # Find all category files
    category_files = list(results_dir.glob("category_*.json"))
    print(f"   Found {len(category_files)} category result files")
    
    if not category_files:
        print("   ⚠️  No category result files found")
        print("   Checking for intermediate results...")
        
        intermediate_files = list(results_dir.glob("*_intermediate.json"))
        if intermediate_files:
            print(f"   Found {len(intermediate_files)} intermediate files:")
            for f in intermediate_files:
                print(f"      - {f.name}")
    
    # Analyze each category
    categories = ['math', 'code', 'reasoning', 'knowledge', 'instruction', 'creativity']
    
    all_results = {}
    for category in categories:
        result = analyze_category_results(results_dir, category)
        if result:
            all_results[category] = result
    
    # Overall summary
    if all_results:
        print(f"\n{'='*80}")
        print("  OVERALL SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n{'Category':<15} {'Wins':<8} {'Losses':<8} {'Ties':<8} {'Score':<8}")
        print(f"{'─'*55}")
        
        total_wins = total_losses = total_ties = 0
        
        for category, data in all_results.items():
            wins = data.get('wins', 0)
            losses = data.get('losses', 0)
            ties = data.get('ties', 0)
            score = data.get('score', 0) * 100
            
            total_wins += wins
            total_losses += losses
            total_ties += ties
            
            print(f"{category:<15} {wins:<8} {losses:<8} {ties:<8} {score:<8.1f}%")
        
        total = total_wins + total_losses + total_ties
        overall_score = (total_wins + 0.5 * total_ties) / total * 100 if total > 0 else 0
        
        print(f"{'─'*55}")
        print(f"{'TOTAL':<15} {total_wins:<8} {total_losses:<8} {total_ties:<8} {overall_score:<8.1f}%")
        
        print(f"\n📊 Key Insights:")
        print(f"   • Overall score vs GPT-4: {overall_score:.1f}%")
        print(f"   • Win rate: {total_wins/total*100:.1f}%")
        print(f"   • Loss rate: {total_losses/total*100:.1f}%")
        print(f"   • Tie rate: {total_ties/total*100:.1f}%")
        
        # Identify weakest category
        weakest = min(all_results.items(), key=lambda x: x[1].get('score', 0))
        strongest = max(all_results.items(), key=lambda x: x[1].get('score', 0))
        
        print(f"\n   • Weakest category: {weakest[0]} ({weakest[1].get('score', 0)*100:.1f}%)")
        print(f"   • Strongest category: {strongest[0]} ({strongest[1].get('score', 0)*100:.1f}%)")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("  RECOMMENDATIONS")
        print(f"{'='*80}")
        
        if overall_score >= 80:
            print("✅ PROCEED TO PHASE 2 (Compression)")
            print("   Model performance is strong enough for compression.")
        elif overall_score >= 75:
            print("⚠️  OPTIONAL PHASE 1C (Targeted Distillation)")
            print(f"   Consider generating 10-20K GPT-5 examples for weak categories:")
            weak_cats = [cat for cat, data in all_results.items() if data.get('score', 0) < 0.75]
            for cat in weak_cats:
                print(f"      - {cat}")
        else:
            print("🔴 REQUIRED PHASE 1C (Full Distillation)")
            print("   Performance below target. Need comprehensive GPT-5 distillation.")
            print(f"   Focus on these weak categories:")
            weak_cats = [cat for cat, data in all_results.items() if data.get('score', 0) < 0.70]
            for cat in weak_cats:
                print(f"      - {cat} ({all_results[cat].get('score', 0)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
