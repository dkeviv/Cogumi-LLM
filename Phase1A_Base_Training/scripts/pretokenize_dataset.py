#!/usr/bin/env python3
"""
Pre-tokenize Dataset for Faster Training

This script tokenizes the dataset once and saves it to disk.
Eliminates tokenization overhead during training (5-10% speedup).

USAGE:
cd Phase1A_2_0/scripts
python pretokenize_dataset.py \
  --input ../data/public_500k_filtered.jsonl \
  --output ../data/tokenized/public_500k

Then use with training:
python train_phase1a_optimized_h100.py --use_pretokenized \
  --dataset_path ../data/tokenized/public_500k
"""

import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for faster training")
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input JSONL dataset')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save tokenized dataset')
    parser.add_argument('--model_name', type=str, 
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--pad_to_max_length', action='store_true',
                       help='Pad all sequences to max_length (recommended for torch.compile)')
    args = parser.parse_args()

    print("="*80)
    print("PRE-TOKENIZATION SCRIPT")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Padding: {'max_length (fixed)' if args.pad_to_max_length else 'dynamic (at training time)'}")
    if args.pad_to_max_length:
        print("  ⚙️  Fixed padding = consistent batch shapes = faster torch.compile")
    print("="*80)
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")

    print(f"\nLoading dataset from {args.input}...")
    dataset = load_dataset("json", data_files=args.input, split="train")
    print(f"✅ Dataset loaded: {len(dataset):,} examples")

    def tokenize_function(examples):
        texts = []
        for inst, resp in zip(examples["instruction"], examples["response"]):
            texts.append(f"{inst}\n\n{resp}")
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding='max_length' if args.pad_to_max_length else False,
            return_tensors=None
        )

    print("\nTokenizing dataset (this may take 10-15 minutes)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,  # Use all cores
        desc="Tokenizing"
    )
    print("✅ Tokenization complete")

    print(f"\nSaving tokenized dataset to {args.output}...")
    tokenized_dataset.save_to_disk(args.output)
    print("✅ Tokenized dataset saved")

    print("\n" + "="*80)
    print("✅ PRE-TOKENIZATION COMPLETE")
    print("="*80)
    print()
    print("USAGE IN TRAINING:")
    print(f"  python train_phase1a_optimized_h100.py --use_pretokenized")
    print()
    print("BENEFIT:")
    print("  Eliminates tokenization overhead during training")
    print("  Expected speedup: 5-10% (saves 2-4 hours)")
    print("="*80)

if __name__ == "__main__":
    main()
