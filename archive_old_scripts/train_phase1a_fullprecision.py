#!/usr/bin/env python3
"""
Phase 1A Training - CORRECTED: Full Precision Base

CRITICAL FIX: Original training used 4-bit quantized base (unsloth optimization)
This caused merge issues and quality degradation.

SOLUTION: Train on full precision base (bfloat16), then quantize AFTER merge.

Changes from train_qlora_optimized.py:
1. Use meta-llama/Meta-Llama-3.1-8B-Instruct (NOT unsloth 4-bit version)
2. Load in bfloat16 (NOT 4-bit quantization)
3. Train LoRA adapter on full precision weights
4. After merge, we can quantize for deployment

Expected output: 
- Adapter trained on full precision base
- Merge will work correctly (no 4-bit rounding errors)
- Can quantize final merged model for deployment
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()

    # CORRECTED: Use full precision base (NOT 4-bit quantized)
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir = "./data/checkpoints/phase1a_fullprecision"

    # LoRA configuration (same as before)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training arguments - adjusted for full precision
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Lower batch size (full precision uses more memory)
        gradient_accumulation_steps=8,  # Higher accumulation to compensate
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # Precision settings - use bfloat16
        bf16=True,
        tf32=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Logging and checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        report_to="tensorboard",
        max_grad_norm=1.0,
        
        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        ddp_find_unused_parameters=False,
    )

    print("="*80)
    print("🔧 CORRECTED TRAINING: FULL PRECISION BASE")
    print("="*80)
    print(f"Base model: {model_name} (FULL PRECISION)")
    print(f"NOT using: unsloth 4-bit quantized version")
    print(f"Precision: bfloat16")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print("="*80)
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in FULL PRECISION (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Full precision (NOT 4-bit!)
        device_map="auto",
        trust_remote_code=False
    )

    print("Preparing model for training...")
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    
    print("\n" + "="*80)
    print("MODEL PARAMETERS")
    print("="*80)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")
    print("="*80)
    print()

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="/workspace/data/Cogumi-LLM/data/phase1/public_500k_filtered.jsonl", split="train")
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
        num_proc=8
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    print("\n" + "="*80)
    print("💾 SAVING MODEL")
    print("="*80)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print()
    print("NEXT STEPS:")
    print("1. Merge adapter with base:")
    print(f"   python scripts/merge_adapter_fullprecision.py")
    print()
    print("2. Optionally quantize merged model for deployment:")
    print(f"   python scripts/quantize_merged_model.py")
    print()
    print("3. Benchmark merged model")
    print("="*80)

if __name__ == "__main__":
    main()
