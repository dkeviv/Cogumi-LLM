#!/usr/bin/env python3
"""
Optimized QLoRA Training Script for LLAMA-3.1-8B
Optimizations:
- Disabled gradient_checkpointing (faster on A100 40GB)
- Increased dataloader_num_workers to 8 (more parallel data loading)
- Increased per_device_train_batch_size to 6 (better GPU utilization)
- Adjusted gradient_accumulation_steps to 6 (maintain effective batch size ~36)
- Added dataloader_prefetch_factor=4 (prefetch batches)
- Added ddp_find_unused_parameters=False (minor speedup)

Expected speedup: 8.86s/step → 5-6s/step (40-45% faster)
Expected completion: ~90 hours (3.75 days) vs 148 hours (6 days)
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
from datasets import load_dataset
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Model configuration
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir = "./data/checkpoints/llama-3.1-8b-phase1a"

    # QLoRA configuration
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

    # OPTIMIZED Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        
        # OPTIMIZED: Increased batch size for better GPU utilization
        per_device_train_batch_size=6,        # UP from 4 → faster
        gradient_accumulation_steps=6,         # DOWN from 8 → maintain ~36 effective batch
        
        # OPTIMIZED: Disabled for speed (A100 has enough memory)
        gradient_checkpointing=False,          # CHANGED from True → 40% faster!
        
        optim="adamw_torch",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        
        # A100 optimizations
        bf16=True,
        tf32=True,
        
        # Logging and checkpointing
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        report_to="tensorboard",
        max_grad_norm=1.0,
        
        # OPTIMIZED: More parallel data loading
        dataloader_num_workers=8,              # UP from 4 → utilize all cores
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,          # ADDED → prefetch 4 batches ahead
        
        # OPTIMIZED: Minor speedup for single GPU
        ddp_find_unused_parameters=False,      # ADDED → slight optimization
    )

    print("=" * 80)
    print("🚀 OPTIMIZED TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"Dataloader workers: {training_args.dataloader_num_workers}")
    print(f"Prefetch factor: {training_args.dataloader_prefetch_factor}")
    print("=" * 80)
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False
    )

    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    model.print_trainable_parameters()
    print("=" * 80)
    print()

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/phase1/public_500k_filtered.jsonl", split="train")
    print(f"✅ Dataset loaded: {len(dataset):,} examples")

    def tokenize_function(examples):
        # Combine instruction and response
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
        desc="Tokenizing",
        num_proc=4  # Parallel tokenization
    )
    print(f"✅ Tokenization complete")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Resume from checkpoint if specified
    resume_from = args.resume_from_checkpoint
    if resume_from:
        print(f"\n🔄 Resuming training from checkpoint: {resume_from}")
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        print(f"✅ Checkpoint found")
    else:
        print("\n🆕 Starting fresh training")

    print("\n" + "=" * 80)
    print("🚀 STARTING OPTIMIZED TRAINING")
    print("=" * 80)
    print(f"Expected speed: 5-6 seconds/step (vs 8.86s/step before)")
    print(f"Expected duration: ~90 hours (3.75 days)")
    print(f"Monitor progress: TensorBoard at {output_dir}")
    print("=" * 80)
    print()

    trainer.train(resume_from_checkpoint=resume_from)

    print("\n" + "=" * 80)
    print("💾 SAVING FINAL MODEL")
    print("=" * 80)
    trainer.save_model()
    print(f"✅ Model saved to: {output_dir}")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
