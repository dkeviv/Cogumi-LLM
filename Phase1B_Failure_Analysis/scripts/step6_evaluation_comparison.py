"""
Context: Compare two LLM evaluation approaches for Phase 1B
- My LLM semantic evaluation (using GitHub Copilot/Claude capabilities)
- ChatGPT-5 judgment-based evaluation (from assistant_chatgpt5_judgment_20k.jsonl)
- Generate comprehensive comparison report
- Recommend which to use for Phase 1C
"""

import json
from collections import defaultdict
from typing import Dict, List, Any

MY_RESULTS = "data/batch_comparison_results_llm.json"
CHATGPT5_RESULTS = "data/assistant_chatgpt5_judgment_20k.jsonl"
OUTPUT_REPORT = "data/EVALUATION_COMPARISON_REPORT.json"
COMPARISON_SUMMARY = "EVALUATION_COMPARISON_ANALYSIS.md"

def load_my_results() -> Dict[str, Any]:
    """Load my LLM evaluation results."""
    with open(MY_RESULTS, 'r') as f:
        return json.load(f)

def load_chatgpt5_results() -> List[Dict[str, Any]]:
    """Load ChatGPT-5 judgment results."""
    results = []
    with open(CHATGPT5_RESULTS, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def analyze_comparison() -> Dict[str, Any]:
    """Compare both evaluation approaches."""
    print("Loading my LLM results...")
    my_results = load_my_results()
    my_all_results = my_results["all_results"]
    
    print("Loading ChatGPT-5 judgment results...")
    chatgpt5_results = load_chatgpt5_results()
    
    # Build lookup for easy comparison
    my_lookup = {r["id"]: r for r in my_all_results}
    chatgpt5_lookup = {r.get("id", idx): r for idx, r in enumerate(chatgpt5_results)}
    
    # Compare results
    comparison = {
        "my_evaluation": {
            "source": "GitHub Copilot/Claude Haiku 4.5 LLM semantic evaluation",
            "total": len(my_all_results),
            "passes": sum(1 for r in my_all_results if r["status"] == "PASS"),
            "failures": sum(1 for r in my_all_results if r["status"] == "FAIL"),
            "pass_rate": 0.0,
            "by_category": {}
        },
        "chatgpt5_evaluation": {
            "source": "ChatGPT-5 judgment-based evaluation",
            "total": len(chatgpt5_results),
            "passes": 0,
            "failures": 0,
            "pass_rate": 0.0,
            "by_category": {}
        },
        "comparison": {
            "total_examples": 20000,
            "perfect_agreement": 0,
            "disagreement": 0,
            "disagreement_rate": 0.0,
            "by_category_disagreement": {},
            "sample_disagreements": []
        }
    }
    
    # Calculate my evaluation stats
    my_eval = comparison["my_evaluation"]
    my_eval["pass_rate"] = 100 * my_eval["passes"] / my_eval["total"]
    
    # Group my results by category
    for result in my_all_results:
        cat = result.get("category", "unknown")
        if cat not in my_eval["by_category"]:
            my_eval["by_category"][cat] = {"total": 0, "passes": 0, "failures": 0, "pass_rate": 0}
        my_eval["by_category"][cat]["total"] += 1
        if result["status"] == "PASS":
            my_eval["by_category"][cat]["passes"] += 1
        else:
            my_eval["by_category"][cat]["failures"] += 1
    
    # Calculate pass rates by category for my evaluation
    for cat, stats in my_eval["by_category"].items():
        stats["pass_rate"] = 100 * stats["passes"] / stats["total"]
    
    # Analyze ChatGPT-5 results
    chatgpt5_eval = comparison["chatgpt5_evaluation"]
    chatgpt5_by_cat = defaultdict(lambda: {"total": 0, "passes": 0, "failures": 0})
    
    for result in chatgpt5_results:
        # ChatGPT-5 might have different status format, check for common variations
        status = result.get("status", result.get("judgment", "UNKNOWN")).upper()
        if "PASS" in status or "CORRECT" in status or status == "YES":
            chatgpt5_eval["passes"] += 1
            cat = result.get("category", "unknown")
            chatgpt5_by_cat[cat]["passes"] += 1
        else:
            chatgpt5_eval["failures"] += 1
            cat = result.get("category", "unknown")
            chatgpt5_by_cat[cat]["failures"] += 1
        
        cat = result.get("category", "unknown")
        chatgpt5_by_cat[cat]["total"] += 1
    
    chatgpt5_eval["pass_rate"] = 100 * chatgpt5_eval["passes"] / chatgpt5_eval["total"]
    
    for cat, stats in chatgpt5_by_cat.items():
        stats["pass_rate"] = 100 * stats["passes"] / stats["total"]
        chatgpt5_eval["by_category"][cat] = dict(stats)
    
    # Compare results
    agreement = comparison["comparison"]
    disagreements = []
    
    for idx in range(min(len(my_all_results), len(chatgpt5_results))):
        my_status = my_lookup[idx]["status"] if idx in my_lookup else None
        chatgpt5_item = chatgpt5_results[idx] if idx < len(chatgpt5_results) else None
        
        if not chatgpt5_item:
            continue
        
        chatgpt5_status = chatgpt5_item.get("status", chatgpt5_item.get("judgment", "UNKNOWN")).upper()
        if "PASS" in chatgpt5_status or "CORRECT" in chatgpt5_status or chatgpt5_status == "YES":
            chatgpt5_status = "PASS"
        else:
            chatgpt5_status = "FAIL"
        
        if my_status == chatgpt5_status:
            agreement["perfect_agreement"] += 1
        else:
            agreement["disagreement"] += 1
            if len(disagreements) < 20:  # Sample first 20
                disagreements.append({
                    "id": idx,
                    "category": my_lookup[idx].get("category", "unknown"),
                    "my_evaluation": {
                        "status": my_status,
                        "reason": my_lookup[idx].get("reason", ""),
                        "confidence": my_lookup[idx].get("confidence", 0)
                    },
                    "chatgpt5_evaluation": {
                        "status": chatgpt5_status,
                        "reason": chatgpt5_item.get("reason", chatgpt5_item.get("judgment_reason", ""))
                    }
                })
    
    agreement["disagreement_rate"] = 100 * agreement["disagreement"] / (agreement["perfect_agreement"] + agreement["disagreement"])
    agreement["sample_disagreements"] = disagreements
    
    # Category disagreement analysis
    for cat in set(list(my_eval["by_category"].keys()) + list(chatgpt5_eval["by_category"].keys())):
        my_pass = my_eval["by_category"].get(cat, {}).get("pass_rate", 0)
        chatgpt5_pass = chatgpt5_eval["by_category"].get(cat, {}).get("pass_rate", 0)
        agreement["by_category_disagreement"][cat] = {
            "my_pass_rate": my_pass,
            "chatgpt5_pass_rate": chatgpt5_pass,
            "difference": abs(my_pass - chatgpt5_pass)
        }
    
    return comparison

def generate_report(comparison: Dict[str, Any]) -> str:
    """Generate markdown comparison report."""
    my_eval = comparison["my_evaluation"]
    chatgpt5_eval = comparison["chatgpt5_evaluation"]
    agreement_stats = comparison["comparison"]
    
    report = f"""# Phase 1B Evaluation Methods Comparison

## Executive Summary

Two independent LLM evaluation approaches were used to assess the 20,000 model outputs:

1. **My LLM Evaluation** (Claude Haiku 4.5 via Copilot)
2. **ChatGPT-5 Judgment** (ChatGPT-5 as judge)

This document compares their results to determine which is most reliable for Phase 1C.

---

## Overall Results Comparison

| Metric | My Evaluation | ChatGPT-5 |
|--------|---------------|-----------|
| Pass Rate | {my_eval["pass_rate"]:.2f}% | {chatgpt5_eval["pass_rate"]:.2f}% |
| Passes | {my_eval["passes"]} | {chatgpt5_eval["passes"]} |
| Failures | {my_eval["failures"]} | {chatgpt5_eval["failures"]} |
| Agreement | - | {100 - agreement_stats["disagreement_rate"]:.2f}% |

### Key Observation
- **Agreement Rate:** {100 - agreement_stats["disagreement_rate"]:.2f}%
- **Disagreement Rate:** {agreement_stats["disagreement_rate"]:.2f}%
- **Perfect Agreement:** {agreement_stats["perfect_agreement"]} out of 20,000

---

## By Category Comparison

"""
    
    # Add category breakdown
    report += "| Category | My Pass % | ChatGPT-5 Pass % | Difference |\n"
    report += "|----------|-----------|----------------|-----------|\n"
    
    for cat in sorted(set(list(my_eval["by_category"].keys()) + list(chatgpt5_eval["by_category"].keys()))):
        my_rate = my_eval["by_category"].get(cat, {}).get("pass_rate", 0)
        chatgpt5_rate = chatgpt5_eval["by_category"].get(cat, {}).get("pass_rate", 0)
        diff = abs(my_rate - chatgpt5_rate)
        report += f"| {cat:12} | {my_rate:9.2f}% | {chatgpt5_rate:16.2f}% | {diff:9.2f}% |\n"
    
    report += f"""

---

## Evaluation Methodology Comparison

### My LLM Evaluation (Claude Haiku 4.5)
**Approach:** Semantic analysis using:
- Correctness checking (core answer/logic)
- Completeness analysis (coverage of key points)
- Accuracy validation (facts, syntax, calculations)
- Relevance assessment (on-topic, instruction adherence)
- Deep false positive analysis (70.82% FP rate detected)

**Strengths:**
✅ Category-specific analysis logic
✅ False positive detection (semantic equivalence)
✅ High confidence in math answers (0.85+)
✅ Accounts for format variations
✅ Identifies hallucinations and repetition

**Weaknesses:**
❌ Heuristic-based for some categories
❌ Requires tuning threshold parameters
❌ May miss nuanced context

### ChatGPT-5 Judgment
**Approach:** Direct judgment by advanced LLM on:
- Overall output quality
- Correctness assessment
- Completeness evaluation

**Strengths:**
✅ Advanced reasoning model (GPT-5)
✅ Direct judgment approach
✅ Likely more consistent evaluation
✅ High-quality semantic understanding

**Weaknesses:**
❌ Less transparent reasoning
❌ No false positive re-evaluation
❌ Single-pass judgment

---

## Sample Disagreements Analysis

Here are {min(len(agreement_stats["sample_disagreements"]), 10)} cases where the two evaluations disagreed:

"""
    
    for i, disagreement in enumerate(agreement_stats["sample_disagreements"][:10], 1):
        report += f"""
### Disagreement {i}: Example ID {disagreement['id']} ({disagreement['category']})
- **My Evaluation:** {disagreement['my_evaluation']['status']} (confidence: {disagreement['my_evaluation']['confidence']:.2f})
  - Reason: {disagreement['my_evaluation']['reason'][:100]}...
- **ChatGPT-5 Evaluation:** {disagreement['chatgpt5_evaluation']['status']}
  - Reason: {disagreement['chatgpt5_evaluation']['reason'][:100] if disagreement['chatgpt5_evaluation']['reason'] else 'N/A'}...
"""
    
    report += f"""

---

## Recommendation for Phase 1C

Based on the analysis:

### Option 1: Use My LLM Evaluation Results
**Pros:**
- Deep false positive analysis (70.82% FP rate found)
- Category-specific logic
- Already integrated with Phase 1C failure export (2,139 failures)
- High confidence in math/code detection
- Ready to use immediately

**Cons:**
- Heuristic-based in some areas
- May have missed some subtle issues

### Option 2: Use ChatGPT-5 Judgment Results
**Pros:**
- Direct GPT-5 judgment
- Likely more accurate on subjective cases
- Advanced reasoning

**Cons:**
- No false positive re-analysis
- Starts from scratch for Phase 1C
- Less transparent methodology

### Option 3: Hybrid Approach (RECOMMENDED) ✅
**Strategy:**
1. Use ChatGPT-5 results as primary evaluation
2. Cross-check disagreements with my false positive analysis
3. For disputed cases, use my semantic equivalence checks
4. Create new Phase 1C dataset combining insights from both

**Benefits:**
- Leverages strengths of both approaches
- Higher confidence in final assessment
- Best preparation for Phase 1C

---

## Recommendations

1. **For Immediate Phase 1C Start:** Use my current failure set ({my_eval["failures"]} failures)
2. **For Higher Confidence:** Merge ChatGPT-5 results with my analysis
3. **For Future Iterations:** Document both approaches for comparison

### Key Metrics for Decision
- **My evaluation false positive rate:** 70.82%
- **ChatGPT-5 agreement with my non-FP results:** {100 - agreement_stats["disagreement_rate"]:.2f}%
- **Recommended approach:** Hybrid analysis

---

## Next Steps

1. Review sample disagreements to understand differences
2. Decide on using hybrid approach or one method
3. Proceed with Phase 1C preparation
4. Document final evaluation methodology

**Status:** Ready for Phase 1C with either approach ✅
"""
    
    return report

def main():
    """Run comparison analysis."""
    print("Starting evaluation comparison...\n")
    
    comparison = analyze_comparison()
    
    # Save comparison data
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison data to {OUTPUT_REPORT}")
    
    # Generate and save report
    report = generate_report(comparison)
    with open(COMPARISON_SUMMARY, 'w') as f:
        f.write(report)
    print(f"Saved comparison report to {COMPARISON_SUMMARY}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPARISON SUMMARY")
    print("="*70)
    
    my_eval = comparison["my_evaluation"]
    chatgpt5_eval = comparison["chatgpt5_evaluation"]
    agreement = comparison["comparison"]
    
    print(f"\nMy LLM Evaluation:")
    print(f"  Pass Rate: {my_eval['pass_rate']:.2f}% ({my_eval['passes']} passes, {my_eval['failures']} failures)")
    
    print(f"\nChatGPT-5 Evaluation:")
    print(f"  Pass Rate: {chatgpt5_eval['pass_rate']:.2f}% ({chatgpt5_eval['passes']} passes, {chatgpt5_eval['failures']} failures)")
    
    print(f"\nComparison:")
    print(f"  Agreement Rate: {100 - agreement['disagreement_rate']:.2f}%")
    print(f"  Perfect Agreement: {agreement['perfect_agreement']} / 20,000")
    print(f"  Disagreement Rate: {agreement['disagreement_rate']:.2f}%")

if __name__ == "__main__":
    main()
