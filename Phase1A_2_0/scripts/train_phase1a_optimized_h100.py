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
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--use_pretokenized', action='store_true', 
                       help='Use pre-tokenized dataset from /tmp/tokenized_dataset')
    parser.add_argument('--dataset_path', type=str, 
                       default='/workspace/data/Cogumi-LLM/data/phase1/public_500k_filtered.jsonl',
                       help='Path to source dataset')
    args = parser.parse_args()

    # Model configuration
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir = "./data/checkpoints/phase1a_fullprecision_optimized"

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
        num_train_epochs=3,
        
        # PROVEN CONFIG: 4 workers, batch=4, grad_accum=2 was FASTEST
        per_device_train_batch_size=4,      # Up from 2 (proven faster)
        gradient_accumulation_steps=2,      # Down from 8 (proven faster)
        # Effective batch size = 8 (faster than 16!)
        
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # Precision settings
        bf16=True,
        tf32=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # OPTIMIZED: Fewer checkpoints (reduce overhead)
        logging_steps=10,
        save_strategy="steps",
        save_steps=2000,                    # Was 1000 (save 50% less often)
        save_total_limit=3,                 # Was 5 (keep fewer checkpoints)
        load_best_model_at_end=False,      # Don't reload at end (saves time)
        report_to="tensorboard",
        max_grad_norm=1.0,
        
        # OPTIMIZED: Data loading based on empirical findings
        dataloader_num_workers=4,           # Was 8 (4 was proven fastest)
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        ddp_find_unused_parameters=False,
        
        # Compilation optimization
        torch_compile=True,                 # Enable torch.compile (20-30% speedup)
        torch_compile_mode="reduce-overhead",
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
    print(f"Torch compile: {training_args.torch_compile} (20-30% speedup)")
    print("="*80)
    print()
    print("EXPECTED PERFORMANCE:")
    print(f"  Training time: 20-24 hours (vs 38 hours unoptimized)")
    print(f"  Cost: $50-60 (vs $95 unoptimized)")
    print(f"  Speedup: 37-58% faster")
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
        print("Loading pre-tokenized dataset from /tmp/tokenized_dataset...")
        tokenized_dataset = load_from_disk("/tmp/tokenized_dataset")
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
                max_length=2048,
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
