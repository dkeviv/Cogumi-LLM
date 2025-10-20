#!/usr/bin/env python3
"""
Quick quality check for Phase 1A trained model.
Tests model on 20 sample tasks across different domains.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict
import argparse

# Sample test cases across domains
TEST_CASES = [
    # Math reasoning
    {
        "domain": "math",
        "instruction": "If a train travels 120 miles in 2 hours, what is its average speed?",
        "expected_keywords": ["60", "mph", "miles per hour", "speed"]
    },
    {
        "domain": "math",
        "instruction": "What is 15% of 200?",
        "expected_keywords": ["30"]
    },
    # Code generation
    {
        "domain": "code",
        "instruction": "Write a Python function to check if a number is prime.",
        "expected_keywords": ["def", "prime", "return", "for", "range"]
    },
    {
        "domain": "code",
        "instruction": "Create a function that reverses a string.",
        "expected_keywords": ["def", "reverse", "return", "[::-1]"]
    },
    # General reasoning
    {
        "domain": "reasoning",
        "instruction": "Explain why the sky appears blue.",
        "expected_keywords": ["light", "scatter", "wavelength", "atmosphere"]
    },
    {
        "domain": "reasoning",
        "instruction": "What are the three branches of the US government?",
        "expected_keywords": ["executive", "legislative", "judicial"]
    },
    # Factual QA
    {
        "domain": "factual",
        "instruction": "What is the capital of France?",
        "expected_keywords": ["Paris"]
    },
    {
        "domain": "factual",
        "instruction": "Who wrote Romeo and Juliet?",
        "expected_keywords": ["Shakespeare", "William"]
    },
    # Problem solving
    {
        "domain": "problem_solving",
        "instruction": "If you have 3 apples and give away 1, how many do you have left?",
        "expected_keywords": ["2", "two"]
    },
    {
        "domain": "problem_solving",
        "instruction": "A rectangle has length 10 and width 5. What is its area?",
        "expected_keywords": ["50"]
    }
]


def load_model(model_path: str, device: str = "auto"):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    """Generate a response for the given instruction."""
    # Format as instruction-following prompt
    prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response


def check_quality(response: str, expected_keywords: List[str]) -> tuple[bool, float]:
    """Check if response contains expected keywords."""
    response_lower = response.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    score = matches / len(expected_keywords) if expected_keywords else 1.0
    passed = score >= 0.5  # At least 50% of keywords present
    
    return passed, score


def run_quick_test(model_path: str, num_samples: int = 10, verbose: bool = True):
    """Run quick quality check on the model."""
    
    model, tokenizer = load_model(model_path)
    
    results = []
    domain_scores = {}
    
    print(f"\n{'='*80}")
    print(f"Running Quick Quality Check ({num_samples} samples)")
    print(f"{'='*80}\n")
    
    # Test subset of cases
    test_subset = TEST_CASES[:num_samples]
    
    for i, test_case in enumerate(test_subset, 1):
        domain = test_case["domain"]
        instruction = test_case["instruction"]
        expected = test_case["expected_keywords"]
        
        if verbose:
            print(f"\n[{i}/{num_samples}] Domain: {domain}")
            print(f"Question: {instruction}")
        
        # Generate response
        response = generate_response(model, tokenizer, instruction)
        
        # Check quality
        passed, score = check_quality(response, expected)
        
        if verbose:
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"Score: {score:.1%} ({'✓ PASS' if passed else '✗ FAIL'})")
        
        # Track results
        results.append({
            "domain": domain,
            "passed": passed,
            "score": score
        })
        
        # Update domain scores
        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(score)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    total_passed = sum(1 for r in results if r["passed"])
    overall_score = sum(r["score"] for r in results) / len(results)
    
    print(f"Overall Pass Rate: {total_passed}/{len(results)} ({total_passed/len(results):.1%})")
    print(f"Overall Score: {overall_score:.1%}")
    
    print("\nBy Domain:")
    for domain, scores in domain_scores.items():
        avg_score = sum(scores) / len(scores)
        passed_count = sum(1 for s in scores if s >= 0.5)
        print(f"  {domain:15} {avg_score:.1%} ({passed_count}/{len(scores)} passed)")
    
    # Quality assessment
    print(f"\n{'='*80}")
    if overall_score >= 0.70:
        print("✓ EXCELLENT - Model quality looks good! Ready for Phase 2.")
    elif overall_score >= 0.50:
        print("⚠ ACCEPTABLE - Model is working but could be better.")
        print("  Consider checking training logs for issues.")
    else:
        print("✗ POOR - Model quality is below target.")
        print("  Review training process before proceeding to Phase 2.")
    print(f"{'='*80}\n")
    
    return results, overall_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick quality check for trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (merged or HuggingFace ID)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples (default: 10, max: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each test"
    )
    
    args = parser.parse_args()
    
    run_quick_test(args.model_path, min(args.num_samples, len(TEST_CASES)), args.verbose)
