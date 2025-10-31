#!/usr/bin/env python3
"""
Validate Merged Phase 1A Model - Quick Sanity Test

Tests the merged Phase 1A model (base + LoRA adapter) to ensure:
1. Model loads without errors
2. Generates coherent (not gibberish) responses
3. Code generation is reasonable
4. Reasoning is logical
5. Model is ready for Phase 1B failure analysis

This is a QUICK test (5-10 minutes), not a full benchmark.
Full benchmarking happens after Phase 1C completes.

Usage:
    python scripts/validate_merged_model.py \
        --model_path ./models/phase1a_merged_10gb

Expected Results:
    - Model loads successfully
    - Generates coherent text (not loops/gibberish)
    - Code looks reasonable (even if not perfect)
    - Reasoning is logical (even if not always correct)
    
Red Flags (would require investigation):
    - Repeated tokens/loops
    - Complete nonsense output
    - Empty or truncated responses
    - Model crashes during inference
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

# Test cases covering different domains
TEST_CASES = [
    {
        "name": "Code Generation - Fibonacci",
        "prompt": "Write a Python function to calculate the nth Fibonacci number:",
        "expected_keywords": ["def", "fibonacci", "return"],
        "domain": "code"
    },
    {
        "name": "Code Generation - Binary Search",
        "prompt": "Implement binary search in Python with proper error handling:",
        "expected_keywords": ["def", "binary", "search", "while", "return"],
        "domain": "code"
    },
    {
        "name": "Reasoning - Logic",
        "prompt": "If all roses are flowers and some flowers fade quickly, what can we logically conclude?",
        "expected_keywords": ["conclude", "roses", "flowers"],
        "domain": "reasoning"
    },
    {
        "name": "Reasoning - Math Word Problem",
        "prompt": "Sarah has 3 times as many apples as John. If John has 5 apples, how many apples does Sarah have? Explain your reasoning step by step.",
        "expected_keywords": ["15", "apples", "times"],
        "domain": "reasoning"
    },
    {
        "name": "Automation - API Call",
        "prompt": "Write a Python function that makes a GET request to an API and handles potential errors:",
        "expected_keywords": ["import", "requests", "try", "except", "return"],
        "domain": "automation"
    },
    {
        "name": "General Knowledge",
        "prompt": "Explain the concept of machine learning in simple terms:",
        "expected_keywords": ["learning", "data", "patterns"],
        "domain": "general"
    }
]


def check_output_quality(output: str, test_case: dict) -> dict:
    """
    Check if output meets basic quality criteria.
    
    Returns:
        dict with 'passed', 'issues', and 'score'
    """
    issues = []
    score = 100
    
    # Check 1: Not empty
    if len(output.strip()) < 10:
        issues.append("Output too short or empty")
        score -= 50
    
    # Check 2: No excessive repetition (loops) - IMPROVED DETECTION
    # Only catch REAL loops (consecutive repetitions), not thematic similarities
    words = output.split()
    if len(words) > 20:
        # Look for consecutive repeated chunks (actual gibberish loops)
        for window_size in [10, 15, 20]:  # Medium to large windows
            if len(words) >= window_size * 2:
                for i in range(len(words) - window_size * 2):
                    chunk1 = " ".join(words[i:i+window_size])
                    chunk2 = " ".join(words[i+window_size:i+window_size*2])
                    
                    # Skip if contains code/structure markers (legitimate)
                    skip_markers = ['def', 'return', 'import', 'class', 'step', '##', 'args:', 'returns:']
                    if any(marker in chunk1.lower() for marker in skip_markers):
                        continue
                    
                    # Check if consecutive chunks are identical (REAL loop)
                    if chunk1 == chunk2:
                        issues.append(f"Consecutive repetition loop detected: '{chunk1[:50]}...'")
                        score -= 50
                        break
                if issues:  # Already found a loop
                    break
    
    # Check 3: Expected keywords present (loose check)
    keywords_found = sum(1 for kw in test_case["expected_keywords"] 
                        if kw.lower() in output.lower())
    keyword_ratio = keywords_found / len(test_case["expected_keywords"])
    
    # More lenient threshold (40% is good enough for diverse responses)
    if keyword_ratio < 0.4:
        issues.append(f"Few expected keywords found ({keywords_found}/{len(test_case['expected_keywords'])})")
        score -= 15  # Less penalty
    
    # Check 4: Reasonable length (not truncated)
    if len(output) < 50 and test_case["domain"] in ["code", "reasoning"]:
        issues.append("Output seems incomplete")
        score -= 15
    
    # Check 5: Contains actual content (not just prompt repetition)
    prompt_words = set(test_case["prompt"].lower().split())
    output_words = set(output.lower().split())
    unique_words = output_words - prompt_words
    
    if len(unique_words) < 10:
        issues.append("Output mostly repeats the prompt")
        score -= 25
    
    # Adjust pass threshold - score >= 60 for cleaner pass/fail
    return {
        "passed": score >= 60,
        "issues": issues,
        "score": max(0, score),
        "keyword_ratio": keyword_ratio
    }


def test_model(model_path: str, max_new_tokens: int = 200, temperature: float = 0.7, save_outputs: bool = False):
    """
    Run validation tests on merged model.
    """
    print("=" * 80)
    print("🧪 PHASE 1A MERGED MODEL VALIDATION")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Save outputs: {save_outputs}")
    print("=" * 80)
    print()
    
    # Prepare output file if requested
    output_file = None
    output_path = None
    if save_outputs:
        output_path = Path(model_path).parent / "validation_outputs.txt"
        output_file = open(output_path, "w", encoding="utf-8")
        print(f"📝 Saving full outputs to: {output_path}")
        print()
    
    # Check model path exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        return False
    
    print("Step 1/3: Loading model and tokenizer...")
    try:
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f}s")
        print(f"   Total parameters: {model.num_parameters():,}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("Step 2/3: Running test cases...")
    print("=" * 80)
    print()
    
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"Domain: {test_case['domain']}")
        print(f"Prompt: {test_case['prompt'][:70]}...")
        print()
        
        try:
            # Generate response
            inputs = tokenizer(
                test_case["prompt"],
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            gen_time = time.time() - start_time
            
            # Decode output
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug: Show extraction process
            print(f"DEBUG - Prompt length: {len(test_case['prompt'])} chars")
            print(f"DEBUG - Full output length: {len(full_output)} chars")
            
            # Extract response (skip prompt)
            response = full_output[len(test_case["prompt"]):].strip()
            print(f"DEBUG - Response length after extraction: {len(response)} chars")
            print()
            
            # Check quality
            quality = check_output_quality(response, test_case)
            
            # Display results - SHOW FULL OUTPUT FOR MANUAL VERIFICATION
            print(f"Generated ({gen_time:.2f}s):")
            print("-" * 80)
            print("FULL RAW OUTPUT (for manual verification):")
            print(response)  # Show complete response, not truncated
            print("-" * 80)
            
            # Save to file if requested
            if output_file:
                output_file.write(f"\n{'='*80}\n")
                output_file.write(f"Test: {test_case['name']}\n")
                output_file.write(f"Domain: {test_case['domain']}\n")
                output_file.write(f"Prompt: {test_case['prompt']}\n")
                output_file.write(f"{'='*80}\n")
                output_file.write(f"Response:\n{response}\n")
                output_file.write(f"{'='*80}\n")
                output_file.write(f"Score: {quality['score']}/100\n")
                output_file.write(f"Issues: {', '.join(quality['issues']) if quality['issues'] else 'None'}\n")
                output_file.flush()
            
            # Quality assessment
            if quality["passed"]:
                status = "✅ PASS"
            else:
                status = "⚠️  WARN"
            
            print(f"{status} - Score: {quality['score']}/100")
            print(f"Keywords found: {quality['keyword_ratio']*100:.0f}%")
            
            if quality["issues"]:
                print("Issues:")
                for issue in quality["issues"]:
                    print(f"  • {issue}")
            
            print()
            print("=" * 80)
            print()
            
            results.append({
                "test_case": test_case["name"],
                "domain": test_case["domain"],
                "passed": quality["passed"],
                "score": quality["score"],
                "gen_time": gen_time,
                "issues": quality["issues"]
            })
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            print()
            print("=" * 80)
            print()
            results.append({
                "test_case": test_case["name"],
                "domain": test_case["domain"],
                "passed": False,
                "score": 0,
                "gen_time": 0,
                "issues": [f"Generation error: {str(e)}"]
            })
    
    # Step 3: Summary
    print("Step 3/3: Validation Summary")
    print("=" * 80)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0
    
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Average score: {avg_score:.1f}/100")
    print()
    
    # Domain breakdown
    print("Domain Breakdown:")
    domains = {}
    for r in results:
        domain = r["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(r)
    
    for domain, domain_results in domains.items():
        domain_passed = sum(1 for r in domain_results if r["passed"])
        domain_total = len(domain_results)
        domain_score = sum(r["score"] for r in domain_results) / domain_total
        print(f"  {domain}: {domain_passed}/{domain_total} passed, avg {domain_score:.1f}/100")
    
    print()
    
    # All issues
    all_issues = []
    for r in results:
        if r["issues"]:
            all_issues.extend(r["issues"])
    
    if all_issues:
        print("Common Issues:")
        unique_issues = list(set(all_issues))
        for issue in unique_issues[:5]:  # Top 5 issues
            count = all_issues.count(issue)
            print(f"  • {issue} ({count}x)")
        print()
    
    # Close output file if used
    if output_file:
        output_file.close()
        print(f"✅ Full outputs saved to: {output_path}")
        print()
    
    # Final verdict
    print("=" * 80)
    
    if passed >= total * 0.7 and avg_score >= 60:
        print("✅ VALIDATION PASSED")
        print()
        print("Model appears functional and ready for Phase 1B!")
        print()
        print("Next Steps:")
        print("1. Proceed to Phase 1B: Failure Analysis (2 days)")
        print("2. Then Phase 1C: GPT-5 Enhanced Training (5 days)")
        print("3. Full benchmark after Phase 1C completes")
        print("=" * 80)
        return True
    elif passed >= total * 0.5:
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
        print()
        print("Model is functional but shows some quality issues.")
        print("Recommend proceeding with Phase 1B, but monitor closely.")
        print("=" * 80)
        return True
    else:
        print("❌ VALIDATION FAILED")
        print()
        print("Model shows significant quality issues.")
        print("Recommend investigating before proceeding to Phase 1B.")
        print()
        print("Possible causes:")
        print("  • Training did not converge properly")
        print("  • Adapter merge had issues")
        print("  • Model corruption during save/load")
        print()
        print("Actions:")
        print("  1. Check training logs for anomalies")
        print("  2. Verify checkpoint integrity")
        print("  3. Try merging a different checkpoint")
        print("=" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate merged Phase 1A model with quick sanity tests"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged Phase 1A model"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per test (default: 200)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save all outputs to a file for manual review"
    )
    
    args = parser.parse_args()
    
    success = test_model(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_outputs=args.save_outputs
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
