#!/usr/bin/env python3
"""
Phase 1B - Step 3: Score Model Outputs by Comparison

Scores model outputs by comparing them to reference responses.
Reads two files (model outputs and references), asks LLM to score the comparison.
Uses the outputs from Step 2, so can be re-run with different scoring models or thresholds.

Usage:
    # Default: Use 405B for highest quality
    python "Phase1B_2_0/step3_judge_outputs.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/judged_results.jsonl
    
    # Use 70B for faster judging (optional)
    python "Phase1B_2_0/step3_judge_outputs.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/judged_results.jsonl \
        --judge_model meta-llama/Llama-3.3-70B-Instruct

Output:
    - judged_results.jsonl: All results with scores
    - failures.jsonl: Only failures (score < 7)
    - summary.json: Statistics and category breakdown
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import InferenceClient
from typing import Dict
from collections import defaultdict
import re


# Scoring prompt template - Direct comparison
JUDGE_PROMPT_TEMPLATE = """Compare these two responses and score the Model Answer quality.

Question: {instruction}

Reference Answer (Good): {reference}

Model Answer (To Evaluate): {model_output}

Compare the two responses above. Rate the Model Answer from 1-10 based on:
- Correctness vs Reference
- Completeness vs Reference  
- Quality vs Reference

9-10: Matches/exceeds reference
7-8: Good, minor differences
5-6: Acceptable but notable gaps
3-4: Poor, major issues
1-2: Very poor, mostly wrong

SCORE: [number]/10
Reason: [one sentence]"""


def score_with_judge(instruction: str, reference: str, model_output: str, 
                     client: InferenceClient, judge_model: str) -> Dict:
    """
    Score model output by comparing to reference using LLM.
    Provides both files to LLM and asks for a simple comparison score.
    
    Returns:
        dict with 'score' (1-10) and 'reason'
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        reference=reference,
        model_output=model_output
    )
    
    try:
        # Simple comparison request - just need score + brief reason
        messages = [{"role": "user", "content": judge_prompt}]
        response = client.chat_completion(
            messages=messages,
            model=judge_model,
            max_tokens=100,  # Short response: score + one sentence
            temperature=0.0  # Deterministic scoring
        )
        
        # Extract text from chat completion response
        response_text = response.choices[0].message.content
        
        # Parse score from response
        score = None
        reason = ""
        
        for line in response_text.split('\n'):
            if line.startswith('SCORE:'):
                score_str = line.replace('SCORE:', '').strip()
                # Extract number
                match = re.search(r'(\d+)', score_str)
                if match:
                    score = int(match.group(1))
            elif line.startswith('Reason:'):
                reason = line.replace('Reason:', '').strip()
        
        if score is None:
            # Fallback: look for any number 1-10
            match = re.search(r'\b([1-9]|10)\b', response_text)
            if match:
                score = int(match.group(1))
            else:
                score = 5  # Default if parsing fails
        
        return {
            "score": score,
            "reason": reason or response_text[:200],
            "raw_response": response_text
        }
        
    except Exception as e:
        print(f"⚠️  Judge error: {e}")
        return {
            "score": 5,
            "reason": f"Judge error: {str(e)}",
            "raw_response": ""
        }


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Judge model outputs using Llama judge"
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
        help="Path to save judged results"
    )
    
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/Llama-3.1-405B-Instruct",
        help="Judge model (default: 405B for highest quality, use meta-llama/Llama-3.3-70B-Instruct for speed)"
    )
    
    parser.add_argument(
        "--failure_threshold",
        type=int,
        default=7,
        help="Score threshold for failures (default: <7 is failure)"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup output paths
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    failures_path = output_path.parent / "failures.jsonl"
    summary_path = output_path.parent / "summary.json"
    
    print("=" * 80)
    print("⚖️  PHASE 1B - STEP 3: JUDGE MODEL OUTPUTS")
    print("=" * 80)
    print(f"Model outputs: {args.model_outputs}")
    print(f"Judge: {args.judge_model}")
    print(f"Failure threshold: <{args.failure_threshold}/10")
    print(f"Output: {args.output_path}")
    print("=" * 80)
    print()
    
    # Load model outputs
    print("Loading model outputs...")
    outputs = []
    with open(args.model_outputs, 'r', encoding='utf-8') as f:
        for line in f:
            outputs.append(json.loads(line))
    
    print(f"✅ Loaded {len(outputs):,} model outputs")
    print()
    
    # Initialize judge
    print(f"Initializing judge model: {args.judge_model}...")
    judge_client = InferenceClient(token=args.hf_token)
    
    if "405B" in args.judge_model:
        print("⚠️  Using 405B model - expect ~12s per example (~68 hours for 20K)")
    else:
        print("✅ Using 70B model - expect ~1-2s per example (~6-8 hours for 20K)")
    
    print("✅ Judge ready (HuggingFace Inference API - Zero Cost)")
    print()
    
    # Judge outputs
    print("Judging outputs...")
    print("=" * 80)
    
    results_file = open(output_path, 'w', encoding='utf-8')
    failures_file = open(failures_path, 'w', encoding='utf-8')
    
    scores = []
    failures = []
    category_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "scores": []})
    
    for output in tqdm(outputs, desc="Judging"):
        instruction = output["instruction"]
        reference = output["reference_response"]
        model_output = output["model_output"]
        category = output.get("category", "unknown")
        
        # Score with judge
        judgment = score_with_judge(instruction, reference, model_output, judge_client, args.judge_model)
        score = judgment['score']
        scores.append(score)
        
        # Create result record
        result = {
            **output,  # Include all original fields
            "judge_score": score,
            "judge_reason": judgment['reason'],
            "judge_raw": judgment['raw_response'],
            "passed": score >= args.failure_threshold
        }
        
        # Save result
        results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        results_file.flush()
        
        # Track failures
        if score < args.failure_threshold:
            failures.append(result)
            failures_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            failures_file.flush()
        
        # Update category statistics
        category_stats[category]["total"] += 1
        category_stats[category]["scores"].append(score)
        if score >= args.failure_threshold:
            category_stats[category]["passed"] += 1
        else:
            category_stats[category]["failed"] += 1
    
    results_file.close()
    failures_file.close()
    
    # Calculate summary statistics
    avg_score = sum(scores) / len(scores) if scores else 0
    pass_rate = sum(1 for s in scores if s >= args.failure_threshold) / len(scores) * 100
    
    # Category summaries
    category_summaries = {}
    for cat, stats in category_stats.items():
        avg_cat_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        pass_rate_cat = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        
        category_summaries[cat] = {
            "total": stats["total"],
            "passed": stats["passed"],
            "failed": stats["failed"],
            "pass_rate": pass_rate_cat,
            "average_score": avg_cat_score
        }
    
    summary = {
        "total_examples": len(outputs),
        "total_failures": len(failures),
        "failure_rate": len(failures) / len(outputs) * 100,
        "pass_rate": pass_rate,
        "average_score": avg_score,
        "failure_threshold": args.failure_threshold,
        "judge_model": args.judge_model,
        "category_breakdown": category_summaries,
        "score_distribution": {
            "1-2": sum(1 for s in scores if 1 <= s <= 2),
            "3-4": sum(1 for s in scores if 3 <= s <= 4),
            "5-6": sum(1 for s in scores if 5 <= s <= 6),
            "7-8": sum(1 for s in scores if 7 <= s <= 8),
            "9-10": sum(1 for s in scores if 9 <= s <= 10)
        }
    }
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 80)
    print("📊 JUDGING SUMMARY")
    print("=" * 80)
    print(f"Total examples: {len(outputs):,}")
    print(f"Average score: {avg_score:.2f}/10")
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"Failures (score <{args.failure_threshold}): {len(failures):,} ({len(failures)/len(outputs)*100:.1f}%)")
    print()
    print("Category breakdown:")
    for cat, stats in sorted(category_summaries.items()):
        print(f"  {cat:12s}: {stats['average_score']:.2f}/10, "
              f"{stats['pass_rate']:5.1f}% pass, {stats['failed']:,} failures")
    print()
    print("Score distribution:")
    for range_str, count in summary["score_distribution"].items():
        pct = count / len(outputs) * 100
        print(f"  {range_str}: {count:5,} ({pct:5.1f}%)")
    print("=" * 80)
    print()
    print(f"✅ Results saved to {output_path}")
    print(f"✅ Failures saved to {failures_path}")
    print(f"✅ Summary saved to {summary_path}")
    print()
    print("✅ Step 3 Complete! Ready for clustering and pattern analysis")
    print()


if __name__ == "__main__":
    main()
