#!/usr/bin/env python3
"""
Phase 1B: Targeted Training on Benchmark Failures

PURPOSE:
    Trains model on specific failure examples (ties/losses from benchmarks) to improve
    consistency and accuracy on hard problems. Uses lower learning rate to avoid
    catastrophic forgetting of Phase 1A knowledge.

WHEN TO USE:
    - Phase 1B.1: Train on 73 benchmark failures (15-20 min, $0.50-1)
    - Phase 1B.2: Train on 2,000+ failures (11-16 hours, $22-30)
    - After extracting failures with extract_failures_from_benchmark.py
    - NOT for Phase 1A initial training (use train_qlora_optimized.py)

INPUT:
    - data/training_from_benchmark/*.jsonl (failures extracted from benchmarks)
    - Format: {"instruction": prompt, "output": correct_answer}

TRAINING CONFIG:
    - Learning rate: 5e-6 (lower than Phase 1A to prevent forgetting)
    - Epochs: 2-3 (fewer than Phase 1A)
    - Batch size: 4, Accumulation: 4 (Effective: 16)
    - Checkpointing: Every epoch (not step-based)

OUTPUT:
    - Model saved to checkpoints/phase1b_from_benchmark/ or specified path
    - LoRA adapter that can be merged or used directly

PIPELINE STAGE: Phase 1B.1 & 1B.2 - Targeted improvement training

Key differences from Phase 1A training:
  - Smaller datasets (targeted failures vs full 600K dataset)
  - Lower learning rate (5e-6 vs default, to avoid catastrophic forgetting)
  - Fewer epochs (2-3 vs 3+)
  - Epoch-based checkpointing (not step-based)
  - Manual JSON loading (handles schema mismatches from different benchmark sources)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
import json
import glob
import os

def load_benchmark_dataset(dataset_path: str):
    """
    Load training data from benchmark failures.
    Handles JSONL files with varying schemas (math has correct_answer, code doesn't).
    Only keeps instruction and output fields needed for training.
    """
    print(f"Loading dataset from: {dataset_path}")
    
    all_data = []
    files = glob.glob(dataset_path)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {dataset_path}")
    
    print(f"Found {len(files)} file(s):")
    for file in files:
        print(f"  - {file}")
    
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Keep only fields needed for training
                all_data.append({
                    'instruction': item['instruction'],
                    'output': item.get('output', item.get('response', ''))
                })
    
    dataset = Dataset.from_list(all_data)
    print(f"✅ Dataset loaded: {len(dataset):,} examples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Phase 1B Training on Benchmark Failures')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, 
                       default="checkpoints/phase1a_merged",
                       help='Base model name or path (should be Phase 1A merged model)')
    
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to training dataset (supports wildcards, e.g., data/*.jsonl)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--num_train_epochs', type=int, default=2,
                       help='Number of training epochs (default: 2 for small datasets)')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate (default: 5e-6, lower to avoid forgetting)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Batch size per device (default: 4 for small datasets)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    
    # Checkpointing
    parser.add_argument('--save_strategy', type=str, default="epoch",
                       choices=['epoch', 'steps', 'no'],
                       help='Save strategy (default: epoch for small datasets)')
    parser.add_argument('--logging_steps', type=int, default=5,
                       help='Log every N steps (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 PHASE 1B: TRAINING ON BENCHMARK FAILURES")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print("=" * 80)
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")
    print()
    
    # QLoRA configuration for 4-bit quantization
    print("Configuring QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("✅ QLoRA configured")
    print()
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False
    )
    print("✅ Model loaded")
    print()
    
    # Prepare model for training
    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    model.print_trainable_parameters()
    print("=" * 80)
    print()
    
    # Load dataset
    dataset = load_benchmark_dataset(args.dataset_path)
    
    # Tokenize dataset
    def tokenize_function(examples):
        """Combine instruction and output, then tokenize."""
        texts = []
        for inst, resp in zip(examples["instruction"], examples["output"]):
            texts.append(f"{inst}\n\n{resp}")
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_tensors=None
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=4
    )
    print("✅ Tokenization complete")
    print()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        
        # Batch configuration
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # 10% warmup for small datasets
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Mixed precision
        bf16=True,
        tf32=True,
        
        # Checkpointing and logging
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=3,  # Keep last 3 checkpoints
        report_to="tensorboard",
        
        # Data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Disable gradient checkpointing for faster training on small datasets
        gradient_checkpointing=False,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    print("✅ Trainer initialized")
    print()
    
    # Calculate training stats
    total_samples = len(tokenized_dataset) * args.num_train_epochs
    steps_per_epoch = len(tokenized_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Dataset size: {len(dataset):,} examples")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Total training samples: {total_samples:,}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup ratio: {training_args.warmup_ratio}")
    print("=" * 80)
    print()
    
    # Train
    print("🚀 Starting training...")
    print()
    trainer.train()
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to: {args.output_dir}")
    print()
    
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Re-run benchmarks on trained model:")
    print(f"   python scripts/run_benchmarks.py --model_path {args.output_dir}")
    print()
    print("2. Compare with Phase 1A results to measure improvement")
    print()
    print("3. If successful, proceed to Phase 1B.2 (expand training data)")
    print("=" * 80)


if __name__ == "__main__":
    main()
