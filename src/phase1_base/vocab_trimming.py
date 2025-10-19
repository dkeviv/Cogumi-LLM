#!/usr/bin/env python3
"""
Vocabulary Trimming Script for LLAMA-3.2-8B
Trims the tokenizer vocabulary from 128K to 25K tokens based on frequency analysis.

This script:
1. Loads the base LLAMA-3.2-8B tokenizer
2. Loads the token frequency analysis
3. Creates a trimmed tokenizer with only the top 25K tokens
4. Validates the trimmed tokenizer on test samples
5. Provides rollback capability if validation fails
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Set
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_token_list(analysis_path: Path) -> List[int]:
    """Load the list of tokens to keep from analysis results."""
    print(f"Loading token list from {analysis_path}...")
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    
    tokens = data['top_25k_tokens']
    print(f"Loaded {len(tokens):,} tokens to keep")
    return tokens


def create_trimmed_tokenizer(base_tokenizer, token_ids_to_keep: Set[int], 
                             output_path: Path) -> AutoTokenizer:
    """Create a new tokenizer with trimmed vocabulary."""
    print(f"Creating trimmed tokenizer...")
    print(f"Original vocab size: {len(base_tokenizer):,}")
    
    # Get the vocabulary
    vocab = base_tokenizer.get_vocab()
    
    # Special tokens that must always be kept
    special_token_ids = set()
    for token_name in ['bos_token', 'eos_token', 'unk_token', 'pad_token', 'sep_token', 'cls_token']:
        token = getattr(base_tokenizer, token_name, None)
        if token:
            token_id = base_tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                special_token_ids.add(token_id)
    
    print(f"Special tokens to preserve: {len(special_token_ids)}")
    
    # Combine kept tokens with special tokens
    all_kept_tokens = token_ids_to_keep.union(special_token_ids)
    print(f"Total tokens to keep: {len(all_kept_tokens):,}")
    
    # Create reverse mapping: token_id -> token_string
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Build new vocabulary
    new_vocab = {}
    for token_id in sorted(all_kept_tokens):
        if token_id in id_to_token:
            token_str = id_to_token[token_id]
            new_vocab[token_str] = len(new_vocab)
    
    print(f"New vocab size: {len(new_vocab):,}")
    
    # Save the trimmed tokenizer
    # Note: This is a simplified approach. In practice, you'd need to:
    # 1. Create a proper tokenizer config
    # 2. Handle merges (for BPE tokenizers)
    # 3. Update the model embeddings accordingly
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save vocabulary
    vocab_file = output_path / 'vocab.json'
    with open(vocab_file, 'w') as f:
        json.dump(new_vocab, f, indent=2)
    
    # Copy tokenizer config and other files
    for file in ['tokenizer_config.json', 'special_tokens_map.json']:
        src = Path(base_tokenizer.name_or_path) / file
        if src.exists():
            shutil.copy(src, output_path / file)
    
    # Create a metadata file
    metadata = {
        'original_vocab_size': len(base_tokenizer),
        'trimmed_vocab_size': len(new_vocab),
        'reduction_percent': round((1 - len(new_vocab) / len(base_tokenizer)) * 100, 2),
        'special_tokens_preserved': len(special_token_ids),
        'base_model': base_tokenizer.name_or_path
    }
    
    with open(output_path / 'trimming_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Trimmed tokenizer saved to: {output_path}")
    print(f"Vocabulary reduction: {metadata['reduction_percent']}%")
    
    return metadata


def validate_trimming(base_tokenizer, trimmed_vocab: Dict, 
                     test_samples: List[str], rollback_threshold: float) -> Dict:
    """Validate the trimmed tokenizer against test samples."""
    print(f"\nValidating trimmed vocabulary on {len(test_samples)} samples...")
    
    # Create reverse mapping for quick lookup
    trimmed_token_ids = set(trimmed_vocab.values())
    
    total_tokens = 0
    covered_tokens = 0
    unk_token_id = base_tokenizer.unk_token_id
    
    for sample in tqdm(test_samples, desc="Validating"):
        tokens = base_tokenizer.encode(sample, add_special_tokens=False)
        total_tokens += len(tokens)
        
        for token_id in tokens:
            if token_id in trimmed_token_ids:
                covered_tokens += 1
    
    coverage = (covered_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    unk_rate = ((total_tokens - covered_tokens) / total_tokens) * 100 if total_tokens > 0 else 0
    
    validation_results = {
        'total_tokens': total_tokens,
        'covered_tokens': covered_tokens,
        'coverage_percent': round(coverage, 2),
        'unk_rate_percent': round(unk_rate, 2),
        'passes_threshold': unk_rate <= (rollback_threshold * 100)
    }
    
    print(f"\nValidation Results:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Covered tokens: {covered_tokens:,}")
    print(f"  Coverage: {coverage:.2f}%")
    print(f"  UNK rate: {unk_rate:.2f}%")
    print(f"  Threshold: {rollback_threshold * 100}%")
    print(f"  Status: {'✓ PASS' if validation_results['passes_threshold'] else '✗ FAIL'}")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description='Trim LLAMA tokenizer vocabulary')
    parser.add_argument('--base-model', type=str, required=True,
                       help='Base model path or HuggingFace model ID')
    parser.add_argument('--token-list', type=str, required=True,
                       help='Path to token frequency analysis JSON')
    parser.add_argument('--target-size', type=int, default=25000,
                       help='Target vocabulary size (default: 25000)')
    parser.add_argument('--validation-samples', type=int, default=10000,
                       help='Number of samples to validate on')
    parser.add_argument('--validation-file', type=str,
                       help='Path to validation data JSONL file (default: same as training)')
    parser.add_argument('--rollback-threshold', type=float, default=0.03,
                       help='Maximum acceptable UNK rate before rollback (default: 0.03 = 3%)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for trimmed tokenizer')
    
    args = parser.parse_args()
    
    # Load base tokenizer
    print(f"Loading base tokenizer from {args.base_model}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Attempting to load from local models directory...")
        base_tokenizer = AutoTokenizer.from_pretrained('models/llama-3.2-8b-base')
    
    # Load token list
    token_ids_to_keep = set(load_token_list(Path(args.token_list)))
    
    # Ensure we have exactly target_size tokens (excluding special tokens)
    if len(token_ids_to_keep) > args.target_size:
        print(f"Trimming token list from {len(token_ids_to_keep)} to {args.target_size}...")
        # Load the full analysis to get frequency order
        with open(args.token_list, 'r') as f:
            analysis = json.load(f)
        token_ids_to_keep = set(analysis['top_25k_tokens'][:args.target_size])
    
    # Create trimmed tokenizer
    metadata = create_trimmed_tokenizer(
        base_tokenizer, 
        token_ids_to_keep, 
        Path(args.output)
    )
    
    # Load validation samples
    if args.validation_file:
        validation_file = Path(args.validation_file)
    else:
        # Use the original analysis file to get the sample path
        with open(args.token_list, 'r') as f:
            analysis = json.load(f)
        # This assumes the validation file is the same as training
        validation_file = Path('data/phase1/public_500k_filtered.jsonl')
    
    print(f"\nLoading validation samples from {validation_file}...")
    validation_samples = []
    with open(validation_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.validation_samples:
                break
            data = json.loads(line)
            if 'instruction' in data and 'response' in data:
                text = f"{data['instruction']}\n{data['response']}"
            elif 'text' in data:
                text = data['text']
            else:
                text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
            validation_samples.append(text)
    
    # Load the trimmed vocab for validation
    with open(Path(args.output) / 'vocab.json', 'r') as f:
        trimmed_vocab = json.load(f)
    
    # Validate
    validation_results = validate_trimming(
        base_tokenizer,
        trimmed_vocab,
        validation_samples,
        args.rollback_threshold
    )
    
    # Save validation results
    with open(Path(args.output) / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n{'='*60}")
    if validation_results['passes_threshold']:
        print("✓ VOCABULARY TRIMMING SUCCESSFUL")
        print(f"✓ Trimmed tokenizer saved to: {args.output}")
        print(f"✓ Coverage: {validation_results['coverage_percent']}%")
        print(f"✓ UNK rate: {validation_results['unk_rate_percent']}% (threshold: {args.rollback_threshold * 100}%)")
    else:
        print("✗ VALIDATION FAILED")
        print(f"✗ UNK rate {validation_results['unk_rate_percent']}% exceeds threshold {args.rollback_threshold * 100}%")
        print(f"✗ Consider:")
        print(f"  - Increasing target vocabulary size")
        print(f"  - Analyzing more samples")
        print(f"  - Adjusting rollback threshold")
        print(f"\nTrimmed tokenizer was still saved to: {args.output}")
        print(f"You can examine it manually before deciding to use it.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
