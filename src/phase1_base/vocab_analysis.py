#!/usr/bin/env python3
"""
Vocabulary Analysis Script for LLAMA-3.2-8B
Analyzes token frequency in training data to identify top 25K tokens for vocabulary trimming.

This script:
1. Loads N samples from the curated dataset
2. Tokenizes them with the LLAMA-3.2-8B tokenizer
3. Counts token frequency
4. Identifies the top 25K most frequent tokens
5. Outputs analysis including coverage statistics
"""

import json
import argparse
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_samples(file_path: Path, num_samples: int) -> List[str]:
    """Load training samples from JSONL file."""
    print(f"Loading {num_samples} samples from {file_path}...")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            # Combine instruction and response for full text analysis
            if 'instruction' in data and 'response' in data:
                text = f"{data['instruction']}\n{data['response']}"
            elif 'text' in data:
                text = data['text']
            else:
                # Fallback: use all string values
                text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
            samples.append(text)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def analyze_tokens(samples: List[str], tokenizer) -> Dict:
    """Analyze token frequency across samples."""
    print("Analyzing token frequency...")
    
    token_counter = Counter()
    total_tokens = 0
    
    for sample in tqdm(samples, desc="Tokenizing"):
        # Tokenize the sample
        tokens = tokenizer.encode(sample, add_special_tokens=False)
        token_counter.update(tokens)
        total_tokens += len(tokens)
    
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unique tokens found: {len(token_counter):,}")
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': len(token_counter),
        'token_counts': token_counter
    }


def calculate_coverage(token_counter: Counter, top_k: int) -> Dict:
    """Calculate what percentage of tokens would be covered by top K tokens."""
    total_count = sum(token_counter.values())
    top_tokens = token_counter.most_common(top_k)
    top_count = sum(count for _, count in top_tokens)
    
    coverage = (top_count / total_count) * 100
    
    return {
        'top_k': top_k,
        'coverage_percent': coverage,
        'tokens_covered': top_count,
        'total_tokens': total_count
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze token frequency in training data')
    parser.add_argument('--samples', type=str, required=True,
                       help='Path to training data JSONL file')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to analyze (default: 10000)')
    parser.add_argument('--base-model', type=str, 
                       default='meta-llama/Llama-3.2-8B',
                       help='Base model for tokenizer')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for analysis results')
    parser.add_argument('--top-k', type=int, default=25000,
                       help='Number of top tokens to identify (default: 25000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    
    # Get HuggingFace token from environment
    hf_token = os.getenv('HF_TOKEN')
    if hf_token and hf_token != 'your_huggingface_token_here':
        print("✓ Using HuggingFace token from environment")
    
    # Try multiple fallback options for LLAMA tokenizer
    tokenizer = None
    attempts = [
        args.base_model,
        'models/llama-3.2-8b-base',
        'meta-llama/Llama-2-7b-hf',  # Fallback to Llama-2
        'NousResearch/Llama-2-7b-hf',  # Another public Llama-2
        'huggyllama/llama-7b',  # Community Llama
    ]
    
    for model_path in attempts:
        try:
            print(f"Attempting to load from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token if hf_token and hf_token != 'your_huggingface_token_here' else None
            )
            print(f"✓ Successfully loaded tokenizer from: {model_path}")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}...")
            continue
    
    if tokenizer is None:
        print("\nERROR: Could not load any LLAMA tokenizer.")
        print("\nPlease either:")
        print("1. Add HF_TOKEN to .env file")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Accept LLAMA license at: https://huggingface.co/meta-llama/Llama-3.2-8B")
        print("4. Download LLAMA-3.2-8B manually to models/llama-3.2-8b-base/")
        return
    
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size:,}")
    
    # Load and analyze samples
    samples = load_samples(Path(args.samples), args.num_samples)
    analysis = analyze_tokens(samples, tokenizer)
    
    # Calculate coverage for different vocabulary sizes
    coverage_stats = {}
    for k in [1000, 5000, 10000, 15000, 20000, 25000, 30000]:
        coverage_stats[k] = calculate_coverage(analysis['token_counts'], k)
    
    # Get top K tokens
    top_tokens = analysis['token_counts'].most_common(args.top_k)
    top_token_ids = [token_id for token_id, _ in top_tokens]
    
    # Calculate target vocabulary coverage
    target_coverage = calculate_coverage(analysis['token_counts'], args.top_k)
    
    # Prepare results
    results = {
        'analysis_info': {
            'num_samples_analyzed': len(samples),
            'original_vocab_size': original_vocab_size,
            'target_vocab_size': args.top_k,
            'total_tokens_processed': analysis['total_tokens'],
            'unique_tokens_found': analysis['unique_tokens']
        },
        'coverage_by_vocab_size': {
            str(k): {
                'coverage_percent': round(stats['coverage_percent'], 2),
                'tokens_covered': stats['tokens_covered']
            }
            for k, stats in coverage_stats.items()
        },
        'target_coverage': {
            'vocab_size': args.top_k,
            'coverage_percent': round(target_coverage['coverage_percent'], 2),
            'tokens_covered': target_coverage['tokens_covered'],
            'total_tokens': target_coverage['total_tokens']
        },
        'top_25k_tokens': top_token_ids,
        'top_25k_counts': {str(token_id): count for token_id, count in top_tokens}
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("VOCABULARY ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Samples analyzed: {len(samples):,}")
    print(f"Original vocab size: {original_vocab_size:,}")
    print(f"Target vocab size: {args.top_k:,}")
    print(f"Reduction: {original_vocab_size - args.top_k:,} tokens ({((original_vocab_size - args.top_k) / original_vocab_size * 100):.1f}%)")
    print(f"\nTotal tokens processed: {analysis['total_tokens']:,}")
    print(f"Unique tokens found: {analysis['unique_tokens']:,}")
    print(f"\nTop {args.top_k} tokens cover: {target_coverage['coverage_percent']:.2f}% of all tokens")
    print(f"({target_coverage['tokens_covered']:,} / {target_coverage['total_tokens']:,})")
    print(f"\nCoverage by vocabulary size:")
    for k in [1000, 5000, 10000, 15000, 20000, 25000, 30000]:
        if k <= original_vocab_size:
            stats = coverage_stats[k]
            print(f"  {k:>6,} tokens: {stats['coverage_percent']:>6.2f}% coverage")
    print(f"\n✓ Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
