#!/usr/bin/env python3
"""
Phase 1B: Test Merged Model on Dataset and Identify Failures

Tests the Phase 1A merged model on a sample of the training dataset
and compares outputs to reference outputs using Llama-70B as judge.

**Key Innovation:** Use Llama-70B (free via HF Inference API) as judge instead of GPT-4-mini
- Zero API cost
- Consistent scoring
- Fast inference (1-2s per sample)
- GPT-4 level quality judging

**Note:** Using 70B instead of 405B for practical speed (10x faster, still highly capable)

Usage:
    # Default: 20K samples (recommended)
    python "Phase1B_2_0/phase1b_test_model.py" \
        --model_path ./Phase1A_2_0/models/phase1a_merged_10gb \
        --dataset_path ./Phase1A_2_0/data/public_500k_filtered.jsonl \
        --output_path ./data/phase1b/test_results.jsonl \
        --num_samples 20000
    
    # For comprehensive coverage: 30K samples
    python "Phase1B_2_0/phase1b_test_model.py" \
        --model_path ./Phase1A_2_0/models/phase1a_merged_10gb \
        --dataset_path ./Phase1A_2_0/data/public_500k_filtered.jsonl \
        --output_path ./data/phase1b/test_results.jsonl \
        --num_samples 30000

Expected Output:
    - test_results.jsonl: All test cases with model outputs and scores
    - failures.jsonl: Only failed cases (score < 7/10)
    - summary.json: Statistics and failure rate
    
Expected Runtime:
    - 20K samples: 4-6 hours, expect 2-4K failures (3.1% coverage)
    - 30K samples: 6-9 hours, expect 3-6K failures (4.7% coverage)
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import random
from tqdm import tqdm
from huggingface_hub import InferenceClient
import time
from typing import Dict, List

# Llama judge via HuggingFace Inference API (FREE)
# NOTE: Using chat_completion() instead of text_generation() for better provider support
# The text_generation() API routes through SambaNova which has limited task support
# NOTE: Using 70B instead of 405B for speed (405B is 10x slower at ~12s/it vs ~1-2s/it for 70B)
# 70B is still highly capable for quality judging (similar to GPT-4 level)
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # 70B model (free tier, much faster than 405B)
JUDGE_PROMPT_TEMPLATE = """You are an expert AI evaluator. Compare the reference answer to the model's answer and rate the quality.

**Question/Instruction:**
{instruction}

**Reference Answer (High Quality):**
{reference}

**Model Answer (To Evaluate):**
{model_output}

**Evaluation Criteria:**
1. Correctness: Is the answer factually correct?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-structured and easy to understand?
4. Relevance: Does it stay on topic?

**Rate the model answer from 1-10:**
- 9-10: Excellent (matches or exceeds reference)
- 7-8: Good (minor issues but acceptable)
- 5-6: Fair (significant issues but shows understanding)
- 3-4: Poor (major errors or gaps)
- 1-2: Very poor (mostly wrong or irrelevant)

Provide your rating as a single number followed by a brief explanation.

Format: SCORE: [number]/10
Reason: [brief explanation]"""


def load_model(model_path: str, device: str = "cuda"):
    """Load the merged Phase 1A model."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else "cpu",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, max_tokens: int = 512) -> str:
    """Generate response from the model."""
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract response (skip instruction)
    response = full_output[len(instruction):].strip()
    return response


def score_with_llama405b(instruction: str, reference: str, model_output: str, client: InferenceClient, judge_model: str) -> Dict:
    """
    Score model output using Llama-405B as judge via HuggingFace Inference API.
    
    Returns:
        dict with 'score' (1-10) and 'reason'
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        reference=reference,
        model_output=model_output
    )
    
    try:
        # Use chat_completion instead of text_generation for better API support
        messages = [{"role": "user", "content": judge_prompt}]
        response = client.chat_completion(
            messages=messages,
            model=judge_model,  # Explicitly specify the model
            max_tokens=256,
            temperature=0.1  # Low temperature for consistent scoring
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
                import re
                match = re.search(r'(\d+)', score_str)
                if match:
                    score = int(match.group(1))
            elif line.startswith('Reason:'):
                reason = line.replace('Reason:', '').strip()
        
        if score is None:
            # Fallback: look for any number 1-10
            import re
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
        description="Phase 1B: Test model and identify failures using Llama-405B judge"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged Phase 1A model"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset (JSONL format)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="Number of samples to test (default: 20000, use 30000 for comprehensive coverage)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save test results (JSONL)"
    )
    
    parser.add_argument(
        "--failure_threshold",
        type=float,
        default=7.0,
        help="Score threshold for failure (default: 7.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for judge model (optional, uses cached token if not provided)"
    )
    
    args = parser.parse_args()
    
    # Setup output paths
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    failures_path = output_path.parent / "failures.jsonl"
    summary_path = output_path.parent / "summary.json"
    
    print("=" * 80)
    print("🧪 PHASE 1B: MODEL TESTING & FAILURE ANALYSIS")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Judge: {JUDGE_MODEL} (Llama-405B via HF Inference API - FREE)")
    print(f"Failure threshold: <{args.failure_threshold}/10")
    print("=" * 80)
    print()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.device)
    
    # Initialize judge (Llama-405B via HF)
    print(f"Initializing judge model: {JUDGE_MODEL}...")
    judge_client = InferenceClient(token=args.hf_token)
    print("✅ Judge ready (HuggingFace Inference API - Zero Cost)")
    print()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Sample randomly
    if len(dataset) > args.num_samples:
        dataset = random.sample(dataset, args.num_samples)
    
    print(f"✅ Loaded {len(dataset)} samples")
    print()
    
    # Test model
    print("Testing model...")
    print("=" * 80)
    
    results = []
    failures = []
    scores = []
    
    output_file = open(output_path, 'w', encoding='utf-8')
    failures_file = open(failures_path, 'w', encoding='utf-8')
    
    for i, example in enumerate(tqdm(dataset, desc="Testing")):
        instruction = example.get('instruction', example.get('prompt', ''))
        reference = example.get('response', example.get('output', ''))
        
        # Generate model output
        model_output = generate_response(model, tokenizer, instruction)
        
        # Score with Llama-405B
        judgment = score_with_llama405b(instruction, reference, model_output, judge_client, JUDGE_MODEL)
        score = judgment['score']
        scores.append(score)
        
        # Save result
        result = {
            "id": i,
            "instruction": instruction,
            "reference": reference,
            "model_output": model_output,
            "score": score,
            "reason": judgment['reason'],
            "passed": score >= args.failure_threshold
        }
        
        results.append(result)
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        output_file.flush()
        
        # Track failures
        if score < args.failure_threshold:
            failures.append(result)
            failures_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            failures_file.flush()
        
        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            avg_score = sum(scores) / len(scores)
            failure_rate = len(failures) / len(results) * 100
            print(f"\nProgress: {i+1}/{len(dataset)}")
            print(f"  Avg score: {avg_score:.2f}/10")
            print(f"  Failure rate: {failure_rate:.1f}%")
        
        # Rate limiting (be nice to HF API)
        time.sleep(0.5)
    
    output_file.close()
    failures_file.close()
    
    # Summary
    print()
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    avg_score = sum(scores) / len(scores)
    failure_rate = len(failures) / len(results) * 100
    
    summary = {
        "total_samples": len(results),
        "avg_score": avg_score,
        "failure_count": len(failures),
        "failure_rate": failure_rate,
        "failure_threshold": args.failure_threshold,
        "score_distribution": {
            "9-10": sum(1 for s in scores if s >= 9),
            "7-8": sum(1 for s in scores if 7 <= s < 9),
            "5-6": sum(1 for s in scores if 5 <= s < 7),
            "3-4": sum(1 for s in scores if 3 <= s < 5),
            "1-2": sum(1 for s in scores if s < 3)
        }
    }
    
    print(f"Total samples: {summary['total_samples']}")
    print(f"Average score: {summary['avg_score']:.2f}/10")
    print(f"Failures: {summary['failure_count']} ({summary['failure_rate']:.1f}%)")
    print()
    print("Score Distribution:")
    for range_name, count in summary['score_distribution'].items():
        pct = count / len(results) * 100
        print(f"  {range_name}: {count} ({pct:.1f}%)")
    print()
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")
    print(f"✅ Failures saved to: {failures_path}")
    print(f"✅ Summary saved to: {summary_path}")
    print()
    
    print("=" * 80)
    print("🎯 NEXT STEPS")
    print("=" * 80)
    print("1. Run clustering on failures:")
    print(f"   python scripts/phase1b_cluster_failures.py \\")
    print(f"       --failures {failures_path} \\")
    print(f"       --output ./data/phase1b/clusters.json")
    print()
    print("2. Label failure patterns:")
    print(f"   python scripts/phase1b_label_patterns.py \\")
    print(f"       --clusters ./data/phase1b/clusters.json \\")
    print(f"       --output ./data/phase1b/patterns.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
