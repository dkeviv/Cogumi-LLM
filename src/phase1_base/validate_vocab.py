#!/usr/bin/env python3
"""
Vocabulary Validation Script for LLAMA-3.2-8B
Comprehensive validation of trimmed tokenizer including coverage, perplexity, and round-trip accuracy.

This script validates:
1. Tokenization coverage (should be ≥99.5%)
2. Perplexity increase (should be <3%)
3. Round-trip accuracy (encode → decode → encode should be ≥99.9%)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_test_samples(file_path: Path, num_samples: int) -> List[str]:
    """Load test samples from JSONL file."""
    print(f"Loading {num_samples} test samples from {file_path}...")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            if 'instruction' in data and 'response' in data:
                text = f"{data['instruction']}\n{data['response']}"
            elif 'text' in data:
                text = data['text']
            else:
                text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
            samples.append(text)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def validate_coverage(base_tokenizer, trimmed_tokenizer_path: Path, 
                     samples: List[str]) -> Dict:
    """Validate tokenization coverage."""
    print("\n" + "="*60)
    print("COVERAGE VALIDATION")
    print("="*60)
    
    # Load trimmed vocabulary
    vocab_file = trimmed_tokenizer_path / 'vocab.json'
    with open(vocab_file, 'r') as f:
        trimmed_vocab = json.load(f)
    
    trimmed_token_ids = set(trimmed_vocab.values())
    
    total_tokens = 0
    covered_tokens = 0
    unk_count = 0
    
    unk_token_id = base_tokenizer.unk_token_id
    
    for sample in tqdm(samples, desc="Checking coverage"):
        tokens = base_tokenizer.encode(sample, add_special_tokens=False)
        total_tokens += len(tokens)
        
        for token_id in tokens:
            if token_id in trimmed_token_ids:
                covered_tokens += 1
            else:
                unk_count += 1
    
    coverage_percent = (covered_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    unk_rate = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0
    
    results = {
        'total_tokens': total_tokens,
        'covered_tokens': covered_tokens,
        'unk_tokens': unk_count,
        'coverage_percent': round(coverage_percent, 4),
        'unk_rate_percent': round(unk_rate, 4),
        'passes': coverage_percent >= 99.5
    }
    
    print(f"\nResults:")
    print(f"  Total tokens tested: {total_tokens:,}")
    print(f"  Covered tokens: {covered_tokens:,}")
    print(f"  UNK tokens: {unk_count:,}")
    print(f"  Coverage: {coverage_percent:.2f}%")
    print(f"  UNK rate: {unk_rate:.2f}%")
    print(f"  Target: ≥99.5% coverage")
    print(f"  Status: {'✓ PASS' if results['passes'] else '✗ FAIL'}")
    
    return results


def validate_perplexity(base_model_path: str, trimmed_tokenizer_path: Path,
                       samples: List[str], device: str = 'cpu') -> Dict:
    """
    Validate perplexity increase.
    Note: This is a simplified validation. Full perplexity calculation would require
    retraining the model with the trimmed vocabulary.
    """
    print("\n" + "="*60)
    print("PERPLEXITY VALIDATION (Simplified)")
    print("="*60)
    print("Note: Full perplexity validation requires model retraining")
    print("This provides an estimate based on token coverage")
    
    # Load trimmed vocabulary
    vocab_file = trimmed_tokenizer_path / 'vocab.json'
    with open(vocab_file, 'r') as f:
        trimmed_vocab = json.load(f)
    
    # For now, we'll use token coverage as a proxy
    # In a full implementation, you'd:
    # 1. Load the model
    # 2. Calculate perplexity with original tokenizer
    # 3. Calculate perplexity with trimmed tokenizer
    # 4. Compare the difference
    
    # Simplified approach: estimate based on UNK rate
    # Higher UNK rate typically correlates with higher perplexity
    
    print("\nSkipping full perplexity calculation (requires model loading)")
    print("Recommendation: Validate perplexity after model training completes")
    
    results = {
        'perplexity_increase_percent': None,
        'note': 'Full validation requires trained model',
        'passes': None  # Will be validated after training
    }
    
    print(f"  Status: ⚠ DEFERRED (validate after training)")
    
    return results


def validate_roundtrip(base_tokenizer, trimmed_tokenizer_path: Path,
                      samples: List[str]) -> Dict:
    """Validate round-trip accuracy (encode → decode → encode)."""
    print("\n" + "="*60)
    print("ROUND-TRIP ACCURACY VALIDATION")
    print("="*60)
    
    # Load trimmed vocabulary
    vocab_file = trimmed_tokenizer_path / 'vocab.json'
    with open(vocab_file, 'r') as f:
        trimmed_vocab = json.load(f)
    
    trimmed_token_ids = set(trimmed_vocab.values())
    
    total_samples = 0
    perfect_matches = 0
    token_level_matches = 0
    total_tokens = 0
    
    for sample in tqdm(samples[:1000], desc="Round-trip test"):  # Test subset for speed
        # Original encoding
        original_tokens = base_tokenizer.encode(sample, add_special_tokens=False)
        
        # Filter to only tokens in trimmed vocab (simulate trimmed tokenizer)
        filtered_tokens = [t if t in trimmed_token_ids else base_tokenizer.unk_token_id 
                          for t in original_tokens]
        
        # Decode
        decoded_text = base_tokenizer.decode(filtered_tokens, skip_special_tokens=True)
        
        # Re-encode
        reencoded_tokens = base_tokenizer.encode(decoded_text, add_special_tokens=False)
        refiltered_tokens = [t if t in trimmed_token_ids else base_tokenizer.unk_token_id 
                            for t in reencoded_tokens]
        
        # Compare
        total_samples += 1
        if filtered_tokens == refiltered_tokens:
            perfect_matches += 1
        
        # Token-level accuracy
        min_len = min(len(filtered_tokens), len(refiltered_tokens))
        matches = sum(1 for i in range(min_len) 
                     if filtered_tokens[i] == refiltered_tokens[i])
        token_level_matches += matches
        total_tokens += max(len(filtered_tokens), len(refiltered_tokens))
    
    sample_accuracy = (perfect_matches / total_samples) * 100 if total_samples > 0 else 0
    token_accuracy = (token_level_matches / total_tokens) * 100 if total_tokens > 0 else 0
    
    results = {
        'total_samples_tested': total_samples,
        'perfect_matches': perfect_matches,
        'sample_accuracy_percent': round(sample_accuracy, 4),
        'token_accuracy_percent': round(token_accuracy, 4),
        'passes': token_accuracy >= 99.9
    }
    
    print(f"\nResults:")
    print(f"  Samples tested: {total_samples:,}")
    print(f"  Perfect matches: {perfect_matches:,}")
    print(f"  Sample-level accuracy: {sample_accuracy:.2f}%")
    print(f"  Token-level accuracy: {token_accuracy:.2f}%")
    print(f"  Target: ≥99.9% token accuracy")
    print(f"  Status: {'✓ PASS' if results['passes'] else '✗ FAIL'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate trimmed LLAMA tokenizer')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to trimmed tokenizer directory')
    parser.add_argument('--base-model', type=str,
                       default='meta-llama/Llama-3.2-8B',
                       help='Base model for comparison')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test data JSONL file')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to test')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for model operations')
    
    args = parser.parse_args()
    
    trimmed_path = Path(args.tokenizer)
    
    # Check if trimmed tokenizer exists
    if not trimmed_path.exists():
        print(f"Error: Trimmed tokenizer not found at {trimmed_path}")
        return
    
    # Load base tokenizer
    print(f"Loading base tokenizer from {args.base_model}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Attempting to load from local models directory...")
        base_tokenizer = AutoTokenizer.from_pretrained('models/llama-3.2-8b-base')
    
    # Load test samples
    samples = load_test_samples(Path(args.test_file), args.num_samples)
    
    # Run all validations
    all_results = {}
    
    # 1. Coverage validation
    all_results['coverage'] = validate_coverage(base_tokenizer, trimmed_path, samples)
    
    # 2. Perplexity validation (deferred)
    all_results['perplexity'] = validate_perplexity(args.base_model, trimmed_path, samples, args.device)
    
    # 3. Round-trip validation
    all_results['roundtrip'] = validate_roundtrip(base_tokenizer, trimmed_path, samples)
    
    # Save results
    results_file = trimmed_path / 'comprehensive_validation.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    checks = [
        ("Coverage ≥99.5%", all_results['coverage']['passes']),
        ("Perplexity <3% increase", all_results['perplexity']['passes']),
        ("Round-trip ≥99.9%", all_results['roundtrip']['passes'])
    ]
    
    for check_name, passed in checks:
        if passed is None:
            status = "⚠ DEFERRED"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {check_name}: {status}")
    
    # Overall status
    passed_checks = [p for p in [c[1] for c in checks] if p is True]
    failed_checks = [p for p in [c[1] for c in checks] if p is False]
    deferred_checks = [p for p in [c[1] for c in checks] if p is None]
    
    print(f"\nResults saved to: {results_file}")
    
    if failed_checks:
        print("\n⚠ VALIDATION FAILED")
        print("Some checks did not pass. Review results and consider adjusting vocabulary size.")
    elif deferred_checks:
        print("\n⚠ PARTIAL VALIDATION")
        print("Some checks deferred. Complete after model training.")
    else:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("Trimmed vocabulary is ready for training!")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
