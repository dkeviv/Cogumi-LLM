#!/usr/bin/env python3
"""
Run standard benchmarks on trained model to validate quality.
Supports MMLU, HumanEval, and GSM8K benchmarks.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from typing import Dict, List
from tqdm import tqdm


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    print(f"✓ Model loaded successfully")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for deterministic answers
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def evaluate_mmlu(model, tokenizer, num_samples: int = 1000) -> Dict:
    """
    Evaluate on MMLU benchmark (Massive Multitask Language Understanding).
    Tests general knowledge across 57 subjects.
    """
    print("\n" + "="*80)
    print("MMLU Benchmark (General Knowledge)")
    print("="*80)
    
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all", split="test")
    
    # Sample for faster evaluation
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    correct = 0
    total = 0
    
    for item in tqdm(dataset, desc="Evaluating MMLU"):
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        
        # Format prompt
        prompt = f"""<|im_start|>user
Question: {question}
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D):<|im_end|>
<|im_start|>assistant
"""
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
        
        # Extract answer letter
        response_clean = response.split("<|im_start|>assistant")[-1].strip().upper()
        pred_letter = response_clean[0] if response_clean else ""
        
        # Check correctness
        correct_letter = ["A", "B", "C", "D"][answer_idx]
        if pred_letter == correct_letter:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    results = {
        "benchmark": "MMLU",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "target_range": "78-82%",
        "gpt4_baseline": "80%",
        "vs_gpt4": f"{(accuracy / 0.80) * 100:.1f}%" if accuracy > 0 else "N/A"
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Target: 78-82%")
    print(f"  GPT-4: 80%")
    print(f"  vs GPT-4: {results['vs_gpt4']}")
    
    return results


def evaluate_gsm8k(model, tokenizer, num_samples: int = 500) -> Dict:
    """
    Evaluate on GSM8K benchmark (Grade School Math 8K).
    Tests mathematical reasoning.
    """
    print("\n" + "="*80)
    print("GSM8K Benchmark (Math Reasoning)")
    print("="*80)
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Sample for faster evaluation
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    correct = 0
    total = 0
    
    for item in tqdm(dataset, desc="Evaluating GSM8K"):
        question = item["question"]
        answer = item["answer"]
        
        # Extract numeric answer
        answer_numbers = re.findall(r"#### ([\d,]+)", answer)
        if not answer_numbers:
            continue
        correct_answer = int(answer_numbers[0].replace(",", ""))
        
        # Format prompt
        prompt = f"""<|im_start|>user
Solve this math problem step by step:

{question}

Provide your final answer as a number.<|im_end|>
<|im_start|>assistant
"""
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
        
        # Extract predicted answer
        response_clean = response.split("<|im_start|>assistant")[-1].strip()
        pred_numbers = re.findall(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b", response_clean)
        
        if pred_numbers:
            try:
                pred_answer = int(pred_numbers[-1].replace(",", "").split(".")[0])
                if pred_answer == correct_answer:
                    correct += 1
            except:
                pass
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    results = {
        "benchmark": "GSM8K",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "target_range": "86-88%",
        "gpt4_baseline": "75%",
        "vs_gpt4": f"{(accuracy / 0.75) * 100:.1f}%" if accuracy > 0 else "N/A"
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Target: 86-88%")
    print(f"  GPT-4: 75%")
    print(f"  vs GPT-4: {results['vs_gpt4']}")
    
    return results


def evaluate_humaneval(model, tokenizer, num_samples: int = 164) -> Dict:
    """
    Evaluate on HumanEval benchmark (Code Generation).
    Tests Python code generation ability.
    
    Note: This is a simplified version. Full evaluation requires code execution.
    """
    print("\n" + "="*80)
    print("HumanEval Benchmark (Code Generation)")
    print("="*80)
    print("Note: This is a simplified evaluation without code execution.")
    print("For accurate results, use the official HumanEval evaluation harness.")
    
    # Load HumanEval dataset
    try:
        dataset = load_dataset("openai_humaneval", split="test")
    except:
        print("Warning: HumanEval dataset not available. Skipping...")
        return {
            "benchmark": "HumanEval",
            "accuracy": 0.0,
            "note": "Dataset not available - requires manual setup"
        }
    
    # Sample
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Simple heuristic: check if generated code has key elements
    valid_completions = 0
    total = 0
    
    for item in tqdm(dataset, desc="Evaluating HumanEval"):
        prompt_text = item["prompt"]
        
        # Format prompt
        prompt = f"""<|im_start|>user
Complete this Python function:

{prompt_text}<|im_end|>
<|im_start|>assistant
"""
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
        
        # Extract code
        code = response.split("<|im_start|>assistant")[-1].strip()
        
        # Simple heuristics (not accurate, just sanity check)
        has_return = "return" in code
        has_proper_indentation = "    " in code or "\t" in code
        has_valid_syntax = code.count("(") == code.count(")") and code.count("[") == code.count("]")
        
        if has_return and has_proper_indentation and has_valid_syntax:
            valid_completions += 1
        
        total += 1
    
    # This is NOT the real pass@1 rate, just a rough estimate
    estimated_quality = valid_completions / total if total > 0 else 0
    
    results = {
        "benchmark": "HumanEval",
        "estimated_quality": estimated_quality,
        "valid_completions": valid_completions,
        "total": total,
        "target_range": "58-62%",
        "gpt4_baseline": "65%",
        "note": "Simplified evaluation - use official harness for accurate pass@1"
    }
    
    print(f"\nResults (Estimated):")
    print(f"  Valid Completions: {estimated_quality:.1%} ({valid_completions}/{total})")
    print(f"  Target: 58-62% (pass@1)")
    print(f"  Note: Run official HumanEval harness for accurate results")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model or HuggingFace model ID"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["mmlu", "gsm8k", "humaneval", "all"],
        default=["all"],
        help="Which benchmarks to run"
    )
    parser.add_argument(
        "--num-samples-mmlu",
        type=int,
        default=1000,
        help="Number of MMLU samples (default: 1000)"
    )
    parser.add_argument(
        "--num-samples-gsm8k",
        type=int,
        default=500,
        help="Number of GSM8K samples (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Determine which benchmarks to run
    benchmarks_to_run = args.benchmarks
    if "all" in benchmarks_to_run:
        benchmarks_to_run = ["mmlu", "gsm8k", "humaneval"]
    
    # Run benchmarks
    all_results = {}
    
    if "mmlu" in benchmarks_to_run:
        all_results["mmlu"] = evaluate_mmlu(model, tokenizer, args.num_samples_mmlu)
    
    if "gsm8k" in benchmarks_to_run:
        all_results["gsm8k"] = evaluate_gsm8k(model, tokenizer, args.num_samples_gsm8k)
    
    if "humaneval" in benchmarks_to_run:
        all_results["humaneval"] = evaluate_humaneval(model, tokenizer)
    
    # Calculate overall score
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    if "mmlu" in all_results and "gsm8k" in all_results:
        # Calculate weighted average (excluding HumanEval since it's estimated)
        mmlu_acc = all_results["mmlu"]["accuracy"]
        gsm8k_acc = all_results["gsm8k"]["accuracy"]
        
        # Compare to GPT-4 baselines
        mmlu_vs_gpt4 = mmlu_acc / 0.80
        gsm8k_vs_gpt4 = gsm8k_acc / 0.75
        overall_vs_gpt4 = (mmlu_vs_gpt4 + gsm8k_vs_gpt4) / 2
        
        print(f"\nBenchmark Results:")
        print(f"  MMLU:    {mmlu_acc:.1%} (Target: 78-82%, {mmlu_vs_gpt4*100:.1f}% of GPT-4)")
        print(f"  GSM8K:   {gsm8k_acc:.1%} (Target: 86-88%, {gsm8k_vs_gpt4*100:.1f}% of GPT-4)")
        print(f"\nOverall: {overall_vs_gpt4*100:.1f}% of GPT-4 baseline")
        
        all_results["overall"] = {
            "vs_gpt4_percentage": overall_vs_gpt4 * 100,
            "target_range": "90-93%",
            "status": "✓ PASS" if overall_vs_gpt4 >= 0.87 else "⚠ REVIEW"
        }
        
        if overall_vs_gpt4 >= 0.90:
            print("\n✓ EXCELLENT - Exceeds target! Ready for Phase 2.")
        elif overall_vs_gpt4 >= 0.87:
            print("\n✓ GOOD - Meets target. Proceed to Phase 2.")
        elif overall_vs_gpt4 >= 0.80:
            print("\n⚠ ACCEPTABLE - Slightly below target but usable.")
        else:
            print("\n✗ BELOW TARGET - Review training before Phase 2.")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
