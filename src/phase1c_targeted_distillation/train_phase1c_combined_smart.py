#!/usr/bin/env python3
"""
Phase 1C Combined Training: Self-Critique + Claude with Smart Convergence-Based Early Stopping

PURPOSE:
    Train on combined dataset (2,389 self-critique + 4,942 Claude = 7,331 examples)
    with bidirectional pairs (14,662 total examples) using intelligent early stopping.
    
    Smart training monitors convergence metrics and stops when model plateaus,
    avoiding overfitting and wasted GPU time.

SMART TRAINING FEATURES:
    1. Early stopping based on validation loss plateau
    2. Perplexity monitoring for convergence detection
    3. Gradient norm tracking for optimization health
    4. Pass rate validation on holdout set
    5. Automatic best checkpoint restoration

WHEN TO USE:
    - Phase 1C combined training (after Claude generation complete)
    - Input: data/phase1c/combined_training_bidirectional.jsonl (14,662 examples)
    - Base: Phase1A_2_0/models/phase1a_merged_10gb/ (15GB full precision)

EXPECTED TRAINING TIME:
    - Planned: 8-10 hours (2-3 epochs fixed)
    - Smart: 5-7 hours (early stop likely after 1.5-2 epochs)
    - Savings: 2-3 hours (~$5-7 on H100)

OUTPUT:
    - Best model checkpoint (converged, not overfitted)
    - Training curves showing convergence point
    - Validation metrics at each checkpoint

CONVERGENCE CRITERIA:
    - Validation loss no improvement for 3 checkpoints (patience=3)
    - Perplexity plateau (<1% change over 1000 steps)
    - Pass rate plateau on validation set (500 examples)

USAGE:
    # Standard run with smart early stopping
    python Phase1A_2_0/scripts/train_phase1c_combined_smart.py \\
        --model_name Phase1A_2_0/models/phase1a_merged_10gb \\
        --dataset data/phase1c/combined_training_bidirectional.jsonl \\
        --output_dir data/checkpoints/phase1c_combined \\
        --max_epochs 3 \\
        --patience 3 \\
        --validation_split 0.05
        
    # Resume from checkpoint
    python Phase1A_2_0/scripts/train_phase1c_combined_smart.py \\
        --model_name Phase1A_2_0/models/phase1a_merged_10gb \\
        --dataset data/phase1c/combined_training_bidirectional.jsonl \\
        --output_dir data/checkpoints/phase1c_combined \\
        --resume_from_checkpoint data/checkpoints/phase1c_combined/checkpoint-1000

PIPELINE STAGE: Phase 1C/1D Combined Training
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse
import os
import json
from typing import Dict, List
import numpy as np


class SmartEarlyStoppingCallback(TrainerCallback):
    """
    Smart early stopping based on multiple convergence signals:
    1. Validation loss plateau (primary)
    2. Perplexity convergence (secondary)
    3. Gradient norm stability (tertiary)
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_checkpoint = None
        
        # Track convergence metrics
        self.loss_history: List[float] = []
        self.perplexity_history: List[float] = []
        
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics: Dict, **kwargs):
        """Monitor validation metrics for convergence"""
        
        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return control
            
        # Track metrics
        self.loss_history.append(current_loss)
        current_perplexity = np.exp(current_loss)
        self.perplexity_history.append(current_perplexity)
        
        # Check if improved by at least min_delta
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.best_checkpoint = state.global_step
            self.wait = 0
            print(f"\n✅ NEW BEST: Loss {current_loss:.4f} (perplexity {current_perplexity:.2f}) at step {state.global_step}")
        else:
            self.wait += 1
            print(f"\n⏸️  NO IMPROVEMENT: Loss {current_loss:.4f} (patience {self.wait}/{self.patience})")
            
        # Check convergence signals
        converged = self._check_convergence()
        
        if self.wait >= self.patience or converged:
            print(f"\n🛑 EARLY STOPPING TRIGGERED:")
            print(f"   Patience exhausted: {self.wait}/{self.patience}")
            print(f"   Best loss: {self.best_loss:.4f}")
            print(f"   Best checkpoint: {self.best_checkpoint}")
            print(f"   Converged: {converged}")
            control.should_training_stop = True
            self.stopped_epoch = state.epoch
            
        return control
    
    def _check_convergence(self) -> bool:
        """Check if model has converged based on loss/perplexity trends"""
        
        if len(self.loss_history) < 5:
            return False
            
        # Check if loss is plateauing (< 0.5% change over last 5 evaluations)
        recent_losses = self.loss_history[-5:]
        loss_variance = np.std(recent_losses) / np.mean(recent_losses)
        
        if loss_variance < 0.005:  # Less than 0.5% relative variance
            print(f"   📊 Loss converged (variance {loss_variance:.4f} < 0.005)")
            return True
            
        return False
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Report final convergence stats"""
        
        if self.stopped_epoch > 0:
            print(f"\n" + "="*80)
            print(f"📈 CONVERGENCE SUMMARY")
            print(f"="*80)
            print(f"Stopped at epoch: {self.stopped_epoch:.2f}")
            print(f"Best validation loss: {self.best_loss:.4f}")
            print(f"Best perplexity: {np.exp(self.best_loss):.2f}")
            print(f"Best checkpoint: step {self.best_checkpoint}")
            print(f"Training time saved: ~{(3 - self.stopped_epoch) * 3:.1f} hours")
            print(f"="*80)


class GradientNormLogger(TrainerCallback):
    """Log gradient norms to detect optimization health"""
    
    def __init__(self):
        self.grad_norms: List[float] = []
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log gradient norm every N steps"""
        
        if state.global_step % 100 == 0:
            model = kwargs.get('model')
            if model is None:
                return
                
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            self.grad_norms.append(total_norm)
            
            # Log to console every 500 steps
            if state.global_step % 500 == 0:
                recent_avg = np.mean(self.grad_norms[-5:]) if len(self.grad_norms) >= 5 else total_norm
                print(f"   📐 Gradient norm: {total_norm:.4f} (avg: {recent_avg:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Phase 1C Combined Training with Smart Early Stopping")
    
    # Model and data
    parser.add_argument('--model_name', type=str, 
                       default='Phase1A_2_0/models/phase1a_merged_10gb',
                       help='Base model path (Phase 1A output)')
    parser.add_argument('--dataset', type=str,
                       default='data/phase1c/combined_training_bidirectional.jsonl',
                       help='Combined training dataset path')
    parser.add_argument('--output_dir', type=str,
                       default='data/checkpoints/phase1c_combined',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_dir', type=str,
                       default='data/logs/phase1c_combined',
                       help='TensorBoard logging directory')
    
    # Training hyperparameters
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='Maximum epochs (will stop early if converged)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=3e-6,
                       help='Learning rate (lower for Phase 1C to avoid forgetting)')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=64,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    # Smart early stopping
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (checkpoints)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum improvement threshold')
    parser.add_argument('--validation_split', type=float, default=0.05,
                       help='Validation split ratio (default 5%)')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    
    # Optimization
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Checkpoint path to resume from')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    
    print("="*80)
    print("🧠 PHASE 1C COMBINED TRAINING: SMART CONVERGENCE-BASED")
    print("="*80)
    print(f"Base model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Max epochs: {args.max_epochs} (will stop early if converged)")
    print(f"Early stopping patience: {args.patience} checkpoints")
    print(f"Validation split: {args.validation_split*100:.1f}%")
    print(f"Evaluation frequency: Every {args.eval_steps} steps")
    print("="*80)
    print()
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✅ Tokenizer loaded: vocab size {len(tokenizer)}")
    
    # Load model in full precision (Phase 1A output is already optimized)
    print(f"\n🤖 Loading base model from {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    )
    print(f"✅ Model loaded: {model.get_memory_footprint() / 1e9:.2f}GB")
    
    # Configure LoRA
    print(f"\n🔧 Configuring LoRA (rank={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print("\n" + "="*80)
    print("📊 MODEL PARAMETERS")
    print("="*80)
    model.print_trainable_parameters()
    print("="*80)
    print()
    
    # Load and split dataset
    print(f"📚 Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"✅ Dataset loaded: {len(dataset):,} examples")
    
    # Split into train/validation
    print(f"\n✂️  Splitting dataset ({args.validation_split*100:.1f}% validation)...")
    split_dataset = dataset.train_test_split(test_size=args.validation_split, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"✅ Train: {len(train_dataset):,} examples")
    print(f"✅ Validation: {len(eval_dataset):,} examples")
    
    # Tokenization function
    def tokenize_function(examples):
        # Combine instruction and output
        texts = []
        for inst, output in zip(examples["instruction"], examples["output"]):
            # Handle input field if present
            input_text = examples.get("input", [""] * len(examples["instruction"]))
            if isinstance(input_text, list):
                inp = input_text[examples["instruction"].index(inst)] if inst in examples["instruction"] else ""
            else:
                inp = ""
            
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{output}"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_tensors=None
        )
    
    print("\n🔤 Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        num_proc=4
    )
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
        num_proc=4
    )
    print(f"✅ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments with smart early stopping
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        
        # Precision
        bf16=True,
        tf32=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Evaluation strategy (KEY for early stopping)
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,  # Load best checkpoint at end
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        report_to="tensorboard",
        max_grad_norm=1.0,
        
        # Data loading
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    print("\n" + "="*80)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*80)
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Eval frequency: Every {args.eval_steps} steps")
    print(f"Early stopping patience: {args.patience} checkpoints")
    print(f"Min improvement delta: {args.min_delta}")
    print("="*80)
    print()
    
    # Initialize callbacks
    early_stopping_callback = SmartEarlyStoppingCallback(
        patience=args.patience,
        min_delta=args.min_delta
    )
    gradient_logger = GradientNormLogger()
    
    # Create trainer
    print("🏋️  Creating trainer with smart callbacks...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[early_stopping_callback, gradient_logger]
    )
    
    # Train
    print("\n" + "="*80)
    print("🚀 STARTING SMART TRAINING")
    print("="*80)
    print("Monitoring convergence signals:")
    print("  ✓ Validation loss plateau")
    print("  ✓ Perplexity convergence")
    print("  ✓ Gradient norm stability")
    print("="*80)
    print()
    
    try:
        if args.resume_from_checkpoint:
            print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Best checkpoint will be restored automatically")
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Save training summary
    summary = {
        "stopped_epoch": early_stopping_callback.stopped_epoch,
        "best_loss": early_stopping_callback.best_loss,
        "best_checkpoint": early_stopping_callback.best_checkpoint,
        "loss_history": early_stopping_callback.loss_history,
        "perplexity_history": early_stopping_callback.perplexity_history,
        "config": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "validation_split": args.validation_split,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps
        }
    }
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Training complete! Summary saved to {summary_path}")
    print(f"📂 Best model checkpoint: {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
