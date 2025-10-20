# Phase 2A: Neural Magic Structured Pruning
# Prunes 60-65% of neurons using structured 2:4 sparsity patterns
# Target: 11GB → 3.85GB

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseml.transformers import oneshot
import argparse
import os

def load_calibration_data(dataset_path, num_samples=512):
    """Load calibration samples for pruning."""
    import json
    
    samples = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            text = f"{data['instruction']}\n\n{data['response']}"
            samples.append(text)
    
    return samples

def main(args):
    print("=" * 60)
    print("Phase 2A: Neural Magic Structured Pruning")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"   Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Load calibration data
    print(f"\n2. Loading calibration data ({args.num_calibration_samples} samples)...")
    calibration_texts = load_calibration_data(
        args.calibration_data,
        args.num_calibration_samples
    )
    
    # Configure pruning recipe
    print(f"\n3. Configuring structured pruning recipe...")
    print(f"   Target sparsity: {args.sparsity * 100}%")
    print(f"   Sparsity pattern: 2:4 (2 zeros per 4 weights)")
    
    recipe = f"""
    pruning_stage:
        pruning_modifiers:
            - !GlobalMagnitudePruningModifier
                start_epoch: 0.0
                end_epoch: 1.0
                init_sparsity: 0.0
                final_sparsity: {args.sparsity}
                inter_func: cubic
                mask_type: "2:4"
                leave_enabled: True
    """
    
    # Apply pruning
    print(f"\n4. Applying structured pruning...")
    print(f"   This will take 5-6 hours...")
    
    model = oneshot(
        model=model,
        dataset=calibration_texts,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=args.num_calibration_samples,
        output_dir=args.output_path
    )
    
    # Save pruned model
    print(f"\n5. Saving pruned model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    # Report results
    pruned_size = sum(p.numel() for p in model.parameters()) / 1e9
    original_size = pruned_size / (1 - args.sparsity)
    
    print("\n" + "=" * 60)
    print("✅ Pruning Complete!")
    print("=" * 60)
    print(f"Original size: {original_size:.2f}B parameters (~{original_size * 2:.2f}GB fp16)")
    print(f"Pruned size: {pruned_size:.2f}B parameters (~{pruned_size * 2:.2f}GB fp16)")
    print(f"Reduction: {(1 - pruned_size/original_size) * 100:.1f}%")
    print(f"\nNext: Run awq_quantize.py to compress to 4-bit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to Phase 1 trained model")
    parser.add_argument("--calibration-data", type=str,
                       default="data/phase1/public_500k_filtered.jsonl",
                       help="Path to calibration dataset")
    parser.add_argument("--num-calibration-samples", type=int, default=512,
                       help="Number of samples for calibration")
    parser.add_argument("--sparsity", type=float, default=0.65,
                       help="Target sparsity (0.65 = 65% pruning)")
    parser.add_argument("--output-path", type=str,
                       default="models/llama-3.1-8b-pruned",
                       help="Output directory for pruned model")
    
    args = parser.parse_args()
    main(args)
