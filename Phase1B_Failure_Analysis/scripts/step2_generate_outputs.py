#!/usr/bin/env python3
"""
Phase 1B - Step 2: Generate Model Outputs

Runs the merged Phase 1A model on the curated test dataset to generate outputs.
This can be run independently and outputs can be reused for multiple judging runs.

Usage:
    python "Phase1B_2_0/step2_generate_outputs.py" \
        --model_path ./Phase1A_2_0/models/phase1a_merged_10gb \
        --test_dataset ./data/phase1b/test_dataset_20k.jsonl \
        --output_path ./data/phase1b/model_outputs_20k.jsonl

Output:
    - model_outputs_20k.jsonl: Test examples with model-generated outputs
    - generation_stats.json: Generation statistics (time, tokens, etc.)
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict


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


def generate_response(model, tokenizer, instruction: str, max_tokens: int = 512) -> Dict:
    """Generate response from the model and return with metadata."""
    start_time = time.time()
    
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    input_length = inputs['input_ids'].shape[1]
    
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
    
    generation_time = time.time() - start_time
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    
    return {
        "response": response,
        "generation_time": generation_time,
        "input_tokens": input_length,
        "output_tokens": tokens_generated,
        "total_tokens": output_length
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate model outputs on test dataset"
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
        default=512,
        help="Maximum tokens to generate per response (default: 512)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats_path = output_path.parent / "generation_stats.json"
    
    print("=" * 80)
    print("🤖 PHASE 1B - STEP 2: GENERATE MODEL OUTPUTS")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Output: {args.output_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.device)
    print()
    
    # Load test dataset
    print("Loading test dataset...")
    test_examples = []
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(test_examples):,} test examples")
    print()
    
    # Generate outputs
    print("Generating model outputs...")
    print("=" * 80)
    
    output_file = open(output_path, 'w', encoding='utf-8')
    
    total_time = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, example in enumerate(tqdm(test_examples, desc="Generating")):
        instruction = example.get('instruction', example.get('prompt', ''))
        reference = example.get('response', example.get('output', ''))
        
        # Generate model output
        result = generate_response(model, tokenizer, instruction, args.max_tokens)
        
        # Create output record
        output_record = {
            "id": i,
            "instruction": instruction,
            "reference_response": reference,
            "model_output": result["response"],
            "category": example.get("category", "unknown"),
            "generation_time": result["generation_time"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"]
        }
        
        # Save to file
        output_file.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        output_file.flush()
        
        # Update statistics
        total_time += result["generation_time"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
    
    output_file.close()
    
    # Calculate statistics
    avg_time = total_time / len(test_examples)
    avg_input_tokens = total_input_tokens / len(test_examples)
    avg_output_tokens = total_output_tokens / len(test_examples)
    throughput = len(test_examples) / total_time
    
    stats = {
        "total_examples": len(test_examples),
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "average_time_per_example": avg_time,
        "average_input_tokens": avg_input_tokens,
        "average_output_tokens": avg_output_tokens,
        "throughput_examples_per_second": throughput,
        "model_path": args.model_path,
        "max_tokens": args.max_tokens
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
    print(f"Average time per example: {avg_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Average input tokens: {avg_input_tokens:.1f}")
    print(f"Average output tokens: {avg_output_tokens:.1f}")
    print("=" * 80)
    print()
    print(f"✅ Outputs saved to {output_path}")
    print(f"✅ Statistics saved to {stats_path}")
    print()
    print("✅ Step 2 Complete! Ready for Step 3: Judge outputs with Llama-405B")
    print()


if __name__ == "__main__":
    main()
