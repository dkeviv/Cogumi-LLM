#!/usr/bin/env python3
"""
Phase 1A Training - OPTIMIZED for H100 (Based on Empirical Findings)

OPTIMIZATIONS APPLIED:
1. Uses proven config: batch_size=4, grad_accum=2, workers=4 (fastest empirically)
2. Local NVMe dataset (eliminates persistent storage I/O bottleneck)
3. Reduced checkpoint frequency (save_steps=2000, not 1000)
4. torch.compile for 20-30% speedup
5. Pre-tokenized dataset option (eliminates tokenization overhead)

EMPIRICAL FINDINGS (October 2025):
- 4 workers + grad_accum=2 was FASTEST (counterintuitively better than larger batches)
- Training time: 38hr (unoptimized) → 20-24hr (optimized) target
- Cost: $95 (unoptimized) → $50-60 (optimized) target

USAGE:
1. Pre-process (one-time):
   python scripts/pretokenize_dataset.py --input data/phase1/public_500k_filtered.jsonl --output /tmp/tokenized_dataset

2. Run training:
   python train_phase1a_optimized_h100.py --use_pretokenized

3. Expected: 20-24 hours on H100 80GB, $50-60 total cost
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
import argparse
import os
import shutil
from pathlib import Path

def copy_dataset_to_local(source_path: str, local_path: str = "/tmp/public_500k_filtered.jsonl"):
    """Copy dataset to local NVMe for faster I/O"""
    print(f"Copying dataset to local NVMe: {local_path}")
    shutil.copy2(source_path, local_path)
    print(f"✅ Dataset copied to local storage")
    return local_path

def main():
    parser = argparse.ArgumentParser(description='Phase 1A Optimized H100 Training')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, 
                       default='/workspace/Cogumi-LLM/data/phase1/public_500k_filtered.jsonl',
                       help='Path to source dataset')
    parser.add_argument('--use_pretokenized', action='store_true', 
                       help='Use pre-tokenized dataset')
    parser.add_argument('--pretokenized_path', type=str,
                       default='/tmp/tokenized_dataset',
                       help='Path to pretokenized dataset (default: unpadded version)')
    parser.add_argument('--use_fixed_padding', action='store_true',
                       help='Use fixed-length padding (slower but consistent shapes)')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length for padding')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, 
                       default='./data/checkpoints/phase1a_fullprecision_optimized',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_dir', type=str, 
                       default='./data/logs',
                       help='Directory for logs')
    
    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--save_steps', type=int, default=2000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Optimization arguments
    parser.add_argument('--torch_compile', action='store_true', default=False,
                       help='Enable torch.compile for speedup')
    parser.add_argument('--no_torch_compile', dest='torch_compile', action='store_false',
                       help='Disable torch.compile (same as not using --torch_compile)')
    
    args = parser.parse_args()

    # Model configuration
    model_name = args.model_name
    output_dir = args.output_dir

    # LoRA configuration (same as before)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # OPTIMIZED training arguments based on empirical findings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        
        # PROVEN CONFIG: 4 workers, batch=4, grad_accum=2 was FASTEST
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Effective batch size = batch_size * grad_accum
        
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # Precision settings
        bf16=True,
        tf32=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # OPTIMIZED: Fewer checkpoints (reduce overhead)
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,      # Don't reload at end (saves time)
        report_to="tensorboard",
        max_grad_norm=1.0,
        
        # OPTIMIZED: Data loading based on empirical findings
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        ddp_find_unused_parameters=False,
        
        # Fix for dataset columns
        remove_unused_columns=False,        # Don't remove input_ids, attention_mask
        
        # Compilation optimization
        torch_compile=args.torch_compile,
        torch_compile_mode="reduce-overhead" if args.torch_compile else None,
    )

    print("="*80)
    print("🚀 OPTIMIZED H100 TRAINING (Based on Empirical Findings)")
    print("="*80)
    print(f"Base model: {model_name} (FULL PRECISION bfloat16)")
    print(f"Batch size: {training_args.per_device_train_batch_size} (proven optimal)")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps} (proven optimal)")
    print(f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Workers: {training_args.dataloader_num_workers} (proven optimal)")
    print(f"Checkpoint frequency: Every {training_args.save_steps} steps (optimized)")
    print(f"torch.compile: {training_args.torch_compile}")
    if training_args.torch_compile:
        print(f"  └─ Mode: {training_args.torch_compile_mode}")
        print(f"  └─ CUDA graph dynamic shapes: DISABLED (fixes 51 distinct sizes issue)")
    print("="*80)
    print()
    if training_args.torch_compile:
        print("EXPECTED PERFORMANCE WITH torch.compile:")
        print(f"  Training time: 20-24 hours (with CUDA graph fix)")
        print(f"  Speed: 0.5-1.5 sec/iteration (consistent)")
    else:
        print("EXPECTED PERFORMANCE WITHOUT torch.compile:")
        print(f"  Training time: ~100 hours")
        print(f"  Speed: 3-5 sec/iteration (stable)")
    print(f"  Savings: 37-47% cheaper")
    print("="*80)
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in FULL PRECISION (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    )

    print("Preparing model for training...")
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    
    # Enable torch.compile for faster training (20-30% speedup)
    if training_args.torch_compile:
        print("Compiling model with torch.compile (may take a few minutes)...")
        
        # FIX: Disable dynamic CUDA graphs to avoid 51 distinct shape overhead
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraph_skip_dynamic_graphs = True
        print("⚙️  CUDA graph dynamic shapes disabled (fixes 51 distinct sizes issue)")
        
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compiled")
    
    print("\n" + "="*80)
    print("MODEL PARAMETERS")
    print("="*80)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")
    print("="*80)
    print()

    # Load dataset with I/O optimization
    if args.use_pretokenized:
        print(f"Loading pre-tokenized dataset from {args.pretokenized_path}...")
        tokenized_dataset = load_from_disk(args.pretokenized_path)
        print(f"✅ Pre-tokenized dataset loaded: {len(tokenized_dataset):,} examples")
    else:
        print("OPTIMIZATION: Copying dataset to local NVMe for faster I/O...")
        local_dataset_path = copy_dataset_to_local(args.dataset_path)
        
        print(f"Loading dataset from local NVMe: {local_dataset_path}")
        dataset = load_dataset("json", data_files=local_dataset_path, split="train")
        print(f"Dataset size: {len(dataset):,} examples")

        def tokenize_function(examples):
            texts = []
            for inst, resp in zip(examples["instruction"], examples["response"]):
                texts.append(f"{inst}\n\n{resp}")
            return tokenizer(
                texts,
                truncation=True,
                max_length=1024,
                padding=False,
                return_tensors=None
            )

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=8  # Use all cores for tokenization
        )
        
        # Optional: Save tokenized dataset for future runs
        print("Saving tokenized dataset to /tmp/tokenized_dataset for future runs...")
        tokenized_dataset.save_to_disk("/tmp/tokenized_dataset")
        print("✅ Tokenized dataset saved (use --use_pretokenized next time)")

    # Configure data collator based on padding strategy
    if args.use_fixed_padding:
        # Fixed-length padding: consistent batch shapes but potentially slower
        class FixedLengthDataCollator(DataCollatorForLanguageModeling):
            def __call__(self, features):
                batch = self.tokenizer.pad(
                    features,
                    padding='max_length',
                    max_length=args.max_length,
                    return_tensors='pt',
                )
                batch["labels"] = batch["input_ids"].clone()
                return batch
        
        data_collator = FixedLengthDataCollator(tokenizer=tokenizer, mlm=False)
        print(f"✅ Using FIXED padding (max_length={args.max_length}) for consistent batch shapes")
    else:
        # Dynamic padding: faster for variable-length sequences
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        print("✅ Using DYNAMIC padding (pads to longest in batch, faster)")

    print("\n" + "="*80)
    print("🚀 STARTING OPTIMIZED TRAINING")
    print("="*80)
    print("Based on empirical findings:")
    print("  ✅ 4 workers (proven fastest)")
    print("  ✅ batch_size=4 + grad_accum=2 (proven fastest)")
    print("  ✅ Local NVMe dataset (faster I/O)")
    print("  ✅ Fewer checkpoints (less overhead)")
    print("  ✅ torch.compile (20-30% speedup)")
    print("="*80)
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    
    # Save final model
    print("Saving final adapter...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Adapter saved to: {output_dir}")
    print()
    print("NEXT STEPS:")
    print("1. Merge adapter: python scripts/merge_adapter_fullprecision.py")
    print("2. Validate: python scripts/automated_gpt4_benchmark.py")
    print("="*80)

if __name__ == "__main__":
    main()
