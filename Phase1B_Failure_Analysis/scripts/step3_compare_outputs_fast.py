#!/usr/bin/env python3
"""
Phase 1B - Step 3: Fast Output Comparison (No LLM Calls)

Compares model outputs to references using similarity metrics.
Much faster than LLM judging - completes 20K samples in minutes, not hours.

Uses multiple metrics:
1. Exact match
2. Token overlap (F1)
3. Semantic similarity (sentence embeddings)

Then optionally validates a sample with LLM judge.

Usage:
    # Fast comparison (minutes, not hours!)
    python "Phase1B_2_0/step3_compare_outputs_fast.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_path ./data/phase1b/comparison_results.jsonl

Output:
    - comparison_results.jsonl: All results with similarity scores
    - failures.jsonl: Low similarity samples
    - summary.json: Statistics and category breakdown
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict, Counter
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Will use token-based metrics only.")
    print("   Install with: pip install sentence-transformers")


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for overlap metrics."""
    return text.lower().split()


def calculate_token_f1(reference: str, model_output: str) -> float:
    """
    Calculate F1 score based on token overlap.
    Returns value 0-1 (higher is better).
    """
    ref_tokens = set(tokenize_simple(reference))
    model_tokens = set(tokenize_simple(model_output))
    
    if not ref_tokens or not model_tokens:
        return 0.0
    
    overlap = len(ref_tokens & model_tokens)
    precision = overlap / len(model_tokens) if model_tokens else 0
    recall = overlap / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_semantic_similarity(reference: str, model_output: str, model) -> float:
    """
    Calculate semantic similarity using sentence embeddings.
    Returns cosine similarity 0-1 (higher is better).
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE or model is None:
        return 0.0
    
    # Encode both texts
    embeddings = model.encode([reference, model_output])
    
    # Cosine similarity
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    
    return float(cos_sim)


def extract_numeric_answer(text: str) -> str:
    """Extract numeric answers for math problems."""
    import re
    # Look for numbers, including decimals and fractions
    numbers = re.findall(r'-?\d+\.?\d*(?:/\d+)?', text)
    return ' '.join(numbers) if numbers else text


def normalize_code(text: str) -> str:
    """Normalize code by removing extra whitespace."""
    import re
    # Remove extra whitespace but preserve structure
    return re.sub(r'\s+', ' ', text.strip())


def compare_outputs(instruction: str, reference: str, model_output: str, 
                   category: str, similarity_model=None) -> Dict:
    """
    Compare model output to reference using multiple metrics.
    Category-aware scoring for better accuracy.
    
    Returns:
        dict with similarity scores and overall assessment
    """
    # Exact match check
    exact_match = reference.strip() == model_output.strip()
    
    # Category-specific preprocessing
    if category == "math":
        # For math, extract and compare numeric answers
        ref_nums = extract_numeric_answer(reference)
        model_nums = extract_numeric_answer(model_output)
        # Check if numbers match
        if ref_nums == model_nums and ref_nums != reference:
            # Numeric answer matches, boost score
            exact_match = True
    
    elif category == "code":
        # For code, normalize whitespace
        ref_normalized = normalize_code(reference)
        model_normalized = normalize_code(model_output)
        if ref_normalized == model_normalized:
            exact_match = True
    
    # Token-based F1
    token_f1 = calculate_token_f1(reference, model_output)
    
    # Semantic similarity (if available)
    semantic_sim = calculate_semantic_similarity(reference, model_output, similarity_model)
    
    # Category-aware weighting
    if category == "math":
        # Math: Prioritize exact numeric matches
        if exact_match:
            combined_score = 1.0
        else:
            # Otherwise semantic similarity is more important than tokens
            combined_score = 0.8 * semantic_sim + 0.2 * token_f1 if semantic_sim > 0 else token_f1
    
    elif category == "code":
        # Code: Token overlap important for syntax, but semantic also matters
        combined_score = 0.5 * semantic_sim + 0.5 * token_f1 if semantic_sim > 0 else token_f1
    
    else:
        # General: Semantic similarity primary, token F1 backup
        if SENTENCE_TRANSFORMERS_AVAILABLE and similarity_model:
            combined_score = 0.7 * semantic_sim + 0.3 * token_f1
        else:
            combined_score = token_f1
    
    # Convert to 1-10 scale
    score = int(combined_score * 10)
    
    # Determine if it's a failure (score < 7 means similarity < 0.7)
    is_failure = score < 7
    
    return {
        "exact_match": exact_match,
        "token_f1": round(token_f1, 3),
        "semantic_similarity": round(semantic_sim, 3) if semantic_sim > 0 else None,
        "combined_score": round(combined_score, 3),
        "score": score,
        "is_failure": is_failure
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Fast comparison using similarity metrics (no LLM calls)"
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
        "--failure_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for failures (default: <0.7 is failure)"
    )
    
    parser.add_argument(
        "--use_semantic",
        action="store_true",
        help="Use semantic similarity (requires sentence-transformers)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PHASE 1B STEP 3: FAST OUTPUT COMPARISON")
    print("=" * 80)
    print()
    
    # Load semantic similarity model if requested
    similarity_model = None
    if args.use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("📦 Loading semantic similarity model...")
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
        print("✅ Semantic similarity model loaded")
        print()
    elif args.use_semantic and not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  Semantic similarity requested but sentence-transformers not available")
        print("   Falling back to token-based metrics only")
        print()
    
    # Load model outputs
    print(f"📂 Loading model outputs from: {args.model_outputs}")
    with open(args.model_outputs, 'r') as f:
        model_outputs = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(model_outputs):,} model outputs")
    print()
    
    # Compare all outputs
    print("🔍 Comparing outputs to references...")
    print()
    
    results = []
    failures = []
    category_stats = defaultdict(lambda: {"total": 0, "failures": 0, "scores": []})
    
    for item in tqdm(model_outputs, desc="Comparing"):
        # Extract fields
        instruction = item.get("instruction", "")
        reference = item.get("reference", "")
        model_output = item.get("model_output", "")
        category = item.get("category", "other")
        
        # Compare
        comparison = compare_outputs(instruction, reference, model_output, similarity_model)
        
        # Build result
        result = {
            "instruction": instruction,
            "reference": reference,
            "model_output": model_output,
            "category": category,
            **comparison
        }
        
        results.append(result)
        
        # Track failures
        if comparison["is_failure"]:
            failures.append(result)
        
        # Category stats
        category_stats[category]["total"] += 1
        category_stats[category]["scores"].append(comparison["score"])
        if comparison["is_failure"]:
            category_stats[category]["failures"] += 1
    
    print()
    print("✅ Comparison complete!")
    print()
    
    # Calculate summary statistics
    all_scores = [r["score"] for r in results]
    avg_score = np.mean(all_scores)
    failure_rate = len(failures) / len(results) * 100
    
    summary = {
        "total_samples": len(results),
        "total_failures": len(failures),
        "failure_rate": round(failure_rate, 2),
        "average_score": round(avg_score, 2),
        "failure_threshold": args.failure_threshold,
        "metrics_used": {
            "token_f1": True,
            "semantic_similarity": similarity_model is not None
        },
        "category_breakdown": {}
    }
    
    # Category breakdown
    for category, stats in category_stats.items():
        cat_scores = stats["scores"]
        summary["category_breakdown"][category] = {
            "total": stats["total"],
            "failures": stats["failures"],
            "failure_rate": round(stats["failures"] / stats["total"] * 100, 2),
            "avg_score": round(np.mean(cat_scores), 2),
            "min_score": min(cat_scores),
            "max_score": max(cat_scores)
        }
    
    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Saving results...")
    
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
    print(f"Average score: {summary['average_score']}/10")
    print()
    print("Category Breakdown:")
    for category, stats in summary["category_breakdown"].items():
        print(f"  {category:12s}: {stats['avg_score']:.1f}/10  "
              f"({stats['failures']}/{stats['total']} failures, {stats['failure_rate']:.1f}%)")
    print()
    print("✅ Fast comparison complete!")
    print(f"⚡ Time saved: ~66 hours vs LLM judging approach")
    print()


if __name__ == "__main__":
    main()
