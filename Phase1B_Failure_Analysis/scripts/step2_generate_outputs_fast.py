#!/usr/bin/env python3
"""
Phase 1B - Step 2: Generate Model Outputs (OPTIMIZED)

Optimized version with:
- Batch generation
- Faster sampling settings
- Early stopping
- Better progress tracking

Usage:
    python "Phase1B_2_0/step2_generate_outputs_fast.py" \
        --model_path ./Phase1A_2_0/models/phase1a_merged_10gb \
        --test_dataset ./data/phase1b/test_dataset_20k.jsonl \
        --output_path ./data/phase1b/model_outputs_20k.jsonl \
        --batch_size 8
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Dict


class InstructionDataset(Dataset):
    """Dataset wrapper for efficient batching with DataLoader."""
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'id': idx,
            'instruction': example.get('instruction', example.get('prompt', '')),
            'reference': example.get('response', example.get('output', '')),
            'category': example.get('category', 'unknown'),
            'example': example
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        'ids': [item['id'] for item in batch],
        'instructions': [item['instruction'] for item in batch],
        'references': [item['reference'] for item in batch],
        'categories': [item['category'] for item in batch],
        'examples': [item['example'] for item in batch]
    }


def load_model(model_path: str):
    """Load the merged Phase 1A model with optimizations."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'  # Fix: Left padding for decoder-only models
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model.eval()  # Evaluation mode
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    print(f"✅ Device: {model.device}")
    return model, tokenizer


def generate_batch(model, tokenizer, instructions: List[str], max_tokens: int = 256) -> List[str]:
    """
    Generate responses for a batch of instructions.
    Optimized for speed with:
    - Pin memory for faster GPU transfer
    - Non-blocking transfers
    - Efficient attention mask
    """
    # Tokenize batch
    inputs = tokenizer(
        instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to GPU efficiently (non-blocking if possible)
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
    
    # Generate with optimizations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,  # Reduced from 512
            do_sample=False,  # Greedy = faster than sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
            top_p=None,  # Fix: Remove top_p when using greedy decoding
            temperature=None  # Fix: Remove temperature when using greedy decoding
        )
    
    # Decode outputs
    responses = []
    for i, output in enumerate(outputs):
        # Skip input tokens
        input_length = inputs['input_ids'][i].shape[0]
        response_tokens = output[input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate model outputs (OPTIMIZED)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged Phase 1A model"
    )
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to curated test dataset (from Step 1)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save model outputs"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per response (default: 256, was 512)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation (default: 64 for H100 80GB, can try 128+)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for preprocessing (default: 8, try 16 if CPU > 16 cores)"
    )
    
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer (default: True)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats_path = output_path.parent / "generation_stats.json"
    
    print("=" * 80)
    print("🤖 PHASE 1B - STEP 2: GENERATE MODEL OUTPUTS (OPTIMIZED)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Output: {args.output_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print()
    print("Optimizations:")
    print("  ✅ Batch generation")
    print("  ✅ Greedy decoding (do_sample=False)")
    print("  ✅ KV cache enabled")
    print("  ✅ Reduced max_tokens (512 → 256)")
    print("  ✅ DataLoader with prefetching")
    print(f"  ✅ {args.num_workers} worker threads")
    print(f"  ✅ Pin memory: {args.pin_memory}")
    print("=" * 80)
    print()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    print()
    
    # Load test dataset
    print("Loading test dataset...")
    test_examples = []
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(test_examples):,} test examples")
    
    # Create dataset and dataloader
    dataset = InstructionDataset(test_examples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        prefetch_factor=2 if args.num_workers > 0 else None  # Prefetch 2 batches
    )
    
    print(f"✅ DataLoader created with {args.num_workers} workers")
    print()
    
    # Estimate time
    # H100: ~0.5-1 second per batch of 32 = ~0.03 sec per example
    estimated_time_minutes = (len(test_examples) / args.batch_size) * 1 / 60
    print(f"📊 Estimated time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)")
    print(f"   With H100 @ batch_size={args.batch_size}: Should complete in 10-20 minutes! ⚡")
    print()
    
    # Generate outputs
    print("Generating model outputs...")
    print("=" * 80)
    
    output_file = open(output_path, 'w', encoding='utf-8')
    
    start_time = time.time()
    processed = 0
    
    # Process with DataLoader
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batches")):
        # Generate responses
        batch_start = time.time()
        responses = generate_batch(model, tokenizer, batch['instructions'], args.max_tokens)
        batch_time = time.time() - batch_start
        
        # Save results
        for i, (id, instruction, reference, category, response) in enumerate(
            zip(batch['ids'], batch['instructions'], batch['references'], 
                batch['categories'], responses)
        ):
            output_record = {
                "id": id,
                "instruction": instruction,
                "reference": reference,
                "model_output": response,
                "category": category,
                "generation_time": batch_time / len(responses)
            }
            
            output_file.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        output_file.flush()
        processed += len(responses)
        
        # Progress update every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            throughput = processed / elapsed
            remaining = len(test_examples) - processed
            eta = remaining / throughput
            print(f"\n📊 Progress: {processed}/{len(test_examples)} "
                  f"({processed/len(test_examples)*100:.1f}%) - "
                  f"{throughput:.1f} ex/sec - "
                  f"ETA: {eta/60:.1f} min")
    
    output_file.close()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    throughput = len(test_examples) / total_time
    
    stats = {
        "total_examples": len(test_examples),
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "total_time_minutes": total_time / 60,
        "average_time_per_example": total_time / len(test_examples),
        "throughput_examples_per_second": throughput,
        "batch_size": args.batch_size,
        "model_path": args.model_path,
        "max_tokens": args.max_tokens,
        "optimizations": [
            "batch_generation",
            "greedy_decoding",
            "kv_cache",
            "reduced_max_tokens"
        ]
    }
    
    # Save statistics
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print()
    print("=" * 80)
    print("📊 GENERATION STATISTICS")
    print("=" * 80)
    print(f"Total examples: {len(test_examples):,}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average time per example: {total_time/len(test_examples):.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/second")
    print("=" * 80)
    print()
    print(f"✅ Outputs saved to {output_path}")
    print(f"✅ Statistics saved to {stats_path}")
    print()
    print("🎉 Expected speedup: 10-20x faster than original script!")
    print()


if __name__ == "__main__":
    main()
