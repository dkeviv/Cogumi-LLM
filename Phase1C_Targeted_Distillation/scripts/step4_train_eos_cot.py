"""
Phase 1C Step 4: Retrain with EOS + CoT Focus

Purpose:
- Retrain Phase 1C model on 10K examples with EOS + CoT
- Emphasize EOS token prediction (2× loss weight)
- Learn adaptive response length
- Learn CoT reasoning for complex queries

Input:
- /data/phase1c_merged_eos_cot.jsonl (10K examples)
- /models/phase1c_current (Phase 1C current model)

Output:
- Enhanced Phase 1C model with proper stopping behavior

Expected Runtime: 5-7 hours on H100 80GB
Cost: $15-20
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - Vast.ai workspace structure
# Assumes workspace structure: /data, /models, /scripts
DATA_DIR = Path("/data")
MODEL_DIR = Path("/models")
TRAINING_DATA = DATA_DIR / "phase1c_merged_eos_cot.jsonl"

# Model configuration
BASE_MODEL_PATH = str(MODEL_DIR / "phase1c_current")
OUTPUT_DIR = str(MODEL_DIR / "phase1c_eos_cot")
CHECKPOINT_DIR = str(MODEL_DIR / "checkpoints" / "phase1c_eos_cot")

# Training hyperparameters
LEARNING_RATE = 3e-6  # Conservative to avoid catastrophic forgetting
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
MAX_SEQ_LENGTH = 4096  # Allow longer sequences for CoT
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# EOS token weight multiplier
EOS_WEIGHT_MULTIPLIER = 2.0


class EOSFocusedTrainer(SFTTrainer):
    """
    Custom trainer with 2× loss weight on EOS token prediction.
    
    This teaches the model to naturally stop generating when appropriate.
    """
    
    def __init__(self, *args, eos_token_id: int = None, eos_weight: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token_id = eos_token_id
        self.eos_weight = eos_weight
        logger.info(f"✅ EOS-focused trainer initialized (weight: {eos_weight}×)")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with 2× weight on EOS token predictions.
        """
        # Get standard loss and outputs
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Apply 2× weight to EOS token positions
        if self.eos_token_id is not None:
            eos_mask = (shift_labels.view(-1) == self.eos_token_id).float()
            weights = torch.where(
                eos_mask.bool(),
                torch.tensor(self.eos_weight, device=loss.device),
                torch.ones_like(eos_mask)
            )
            loss = loss * weights
        
        # Average loss
        loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss


def load_training_data(data_path: Path):
    """Load training dataset from JSONL."""
    logger.info(f"Loading training data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    dataset = load_dataset('json', data_files=str(data_path), split='train')
    logger.info(f"✅ Loaded {len(dataset)} training examples")
    
    return dataset


def format_example(example: dict) -> str:
    """
    Format example for training with Llama-3.1 chat template.
    
    Preserves CoT format if present, adds adaptive response behavior.
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Build prompt
    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction
    
    # Format as Llama-3.1 chat (system + user + assistant)
    formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. For complex queries, use Chain-of-Thought reasoning in <thinking> tags before giving your final <answer>. Always end your response naturally with the EOS token when complete.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}"""
    
    return formatted


def setup_model_and_tokenizer(base_model_path: str):
    """Setup model and tokenizer with LoRA for fine-tuning."""
    logger.info(f"Loading base model from: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model in bfloat16 (full precision for better quality)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info("✅ Model and tokenizer loaded")
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: str,
    checkpoint_dir: str
):
    """Train model with EOS focus and early stopping."""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # Early stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Split train/val (95/5 split)
    train_val_split = train_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    
    # Initialize trainer with EOS focus
    trainer = EOSFocusedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=format_example,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        eos_token_id=tokenizer.eos_token_id,
        eos_weight=EOS_WEIGHT_MULTIPLIER,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("🚀 Starting training...")
    logger.info(f"  - Learning rate: {LEARNING_RATE}")
    logger.info(f"  - Epochs: {EPOCHS}")
    logger.info(f"  - EOS weight: {EOS_WEIGHT_MULTIPLIER}×")
    logger.info(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
    
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds() / 3600
    logger.info(f"✅ Training complete! Duration: {duration:.2f} hours")
    
    # Save final model
    logger.info(f"Saving final model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'epochs_completed': train_result.metrics.get('epoch', EPOCHS),
            'duration_hours': duration,
            'early_stopped': train_result.metrics.get('early_stopped', False)
        }, f, indent=2)
    
    logger.info(f"✅ Training metrics saved to: {metrics_file}")
    
    return trainer


def merge_and_save(model, tokenizer, output_dir: str):
    """Merge LoRA weights and save final model."""
    logger.info("Merging LoRA weights into base model...")
    
    # Merge LoRA weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_dir = Path(output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged model to: {merged_dir}")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    logger.info(f"✅ Merged model saved to: {merged_dir}")
    return merged_dir


def main():
    """Main training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Phase 1C Step 4: Retrain with EOS + CoT Focus")
    logger.info("="*60 + "\n")
    
    # Check training data exists
    if not TRAINING_DATA.exists():
        logger.error(f"❌ Training data not found: {TRAINING_DATA}")
        logger.error("Please run step3_combine_complete_dataset.py first")
        return
    
    # Check base model exists
    if not Path(BASE_MODEL_PATH).exists():
        logger.error(f"❌ Base model not found: {BASE_MODEL_PATH}")
        logger.error("Please ensure Phase 1C current model is available")
        return
    
    # Load training data
    train_dataset = load_training_data(TRAINING_DATA)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(BASE_MODEL_PATH)
    
    # Train model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=OUTPUT_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Merge and save
    merged_dir = merge_and_save(model, tokenizer, OUTPUT_DIR)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\n📂 Outputs:")
    logger.info(f"  - LoRA adapter: {OUTPUT_DIR}")
    logger.info(f"  - Merged model: {merged_dir}")
    logger.info(f"  - Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"\n🎯 Next Steps:")
    logger.info(f"  1. Generate 10K outputs with adaptive max_tokens")
    logger.info(f"  2. Evaluate with GPT-4.1 (target: ≥90% pass rate)")
    logger.info(f"  3. Check EOS usage (target: >90%)")
    logger.info(f"  4. Verify natural length variation")
    logger.info(f"  5. If successful → Proceed to Phase 2 compression")
    logger.info("")


if __name__ == "__main__":
    main()
