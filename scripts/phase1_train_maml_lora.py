#!/usr/bin/env python3
"""
Phase 1: Train Llama-3.1-8B-Instruct with MAML + LoRA

This script implements meta-learning (MAML) with LoRA fine-tuning on the interleaved
training dataset. Uses Flash Attention 2 for memory efficiency.

Training Configuration:
- Model: Llama-3.1-8B-Instruct (8.3B params)
- Precision: BF16
- LoRA: rank=64, alpha=128
- Batch size: 2 per device
- Gradient accumulation: 4 steps (effective batch=8)
- Learning rate: 2e-4 (with cosine schedule)
- Epochs: 3
- Hardware: H100 80GB (7-8 hours, $16)

Meta-Learning (MAML):
- Inner loop: Fast adaptation on task-specific batches (lr=5e-3, 3 steps)
- Outer loop: Meta-optimization across tasks (lr=2e-4)
- Task definition: Domain-based (8 domains, sampled in batches)

Cost: ~$16 for full training
Output: 14GB BF16 model with LoRA weights
"""

import os
import json
import torch
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Llama-3.1-8B with MAML + LoRA")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/phase1/answers/training_data_interleaved.jsonl",
        help="Path to interleaved training data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/phase1_maml_lora",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for outer loop (meta-optimization)"
    )
    parser.add_argument(
        "--inner_learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate for inner loop (task adaptation)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: use only 100 examples"
    )
    parser.add_argument(
        "--flash_attention_version",
        type=int,
        choices=[2, 3],
        default=None,
        help="Force specific Flash Attention version (2 or 3). Default: auto-detect based on GPU"
    )
    
    return parser.parse_args()


def load_training_data(data_file: Path) -> List[Dict]:
    """Load and validate interleaved training data."""
    console.print(f"\n[cyan]Loading training data from:[/cyan] {data_file}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")
    
    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(json.loads(line))
    
    console.print(f"[green]✓ Loaded {len(examples):,} examples[/green]\n")
    return examples


def validate_data(examples: List[Dict]) -> Dict[str, int]:
    """Validate data structure and return statistics."""
    console.print("[cyan]Validating data structure...[/cyan]")
    
    stats = {
        'total': len(examples),
        'easy': 0,
        'hard': 0,
        'missing_prompt': 0,
        'missing_response': 0,
        'missing_metadata': 0,
    }
    
    domain_counts = defaultdict(int)
    
    for example in examples:
        # Check required fields
        if 'prompt' not in example or not example['prompt']:
            stats['missing_prompt'] += 1
        
        if 'response' not in example or not example['response']:
            stats['missing_response'] += 1
        
        if 'metadata' not in example:
            stats['missing_metadata'] += 1
            continue
        
        metadata = example['metadata']
        difficulty = metadata.get('difficulty', 'unknown')
        domain = metadata.get('domain', 'Unknown')
        
        if difficulty == 'easy':
            stats['easy'] += 1
        elif difficulty == 'hard':
            stats['hard'] += 1
        
        domain_counts[domain] += 1
    
    # Check for critical issues
    critical_issues = (
        stats['missing_prompt'] + 
        stats['missing_response'] + 
        stats['missing_metadata']
    )
    
    if critical_issues > 0:
        console.print(f"\n[bold red]❌ Critical Issues Found:[/bold red]")
        if stats['missing_prompt'] > 0:
            console.print(f"  • Missing prompts: {stats['missing_prompt']:,}")
        if stats['missing_response'] > 0:
            console.print(f"  • Missing responses: {stats['missing_response']:,}")
        if stats['missing_metadata'] > 0:
            console.print(f"  • Missing metadata: {stats['missing_metadata']:,}")
        console.print(f"\n[red]❌ Cannot proceed with training![/red]\n")
        exit(1)
    
    # Display statistics
    table = Table(title="Data Validation Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
    table.add_row("Total Examples", f"{stats['total']:,}", "100.0%")
    table.add_row("Easy Questions", f"{stats['easy']:,}", f"{stats['easy']/stats['total']*100:.1f}%")
    table.add_row("Hard Questions", f"{stats['hard']:,}", f"{stats['hard']/stats['total']*100:.1f}%")
    
    console.print()
    console.print(table)
    
    # Domain distribution
    console.print("\n[bold cyan]Domain Distribution:[/bold cyan]")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  • {domain}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    console.print("\n[bold green]✓ Data validation passed![/bold green]\n")
    
    return stats


def format_training_example(example: Dict, tokenizer) -> str:
    """Format example into Llama chat template format."""
    prompt = example['prompt']
    difficulty = example.get('metadata', {}).get('difficulty', 'easy')
    
    if difficulty == 'easy':
        # Easy: Direct response
        response = example['response']
        
        # Create chat format
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    else:
        # Hard: Draft → Thinking → Response
        draft = example.get('draft', '')
        thinking = example.get('thinking', '')
        response = example.get('response', '')
        
        # Combine for training (model learns to do CoT internally)
        full_response = f"{draft}\n\n{thinking}\n\n{response}"
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response}
        ]
    
    # Apply Llama chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return formatted


def prepare_dataset(examples: List[Dict], tokenizer, max_seq_length: int) -> Dataset:
    """Prepare HuggingFace dataset from examples."""
    console.print("\n[cyan]Preparing dataset for training...[/cyan]")
    
    formatted_texts = []
    domains = []
    difficulties = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Formatting examples", total=len(examples))
        
        for example in examples:
            formatted = format_training_example(example, tokenizer)
            formatted_texts.append(formatted)
            domains.append(example.get('metadata', {}).get('domain', 'Unknown'))
            difficulties.append(example.get('metadata', {}).get('difficulty', 'easy'))
            
            progress.advance(task)
    
    # Tokenize
    console.print("[cyan]Tokenizing dataset...[/cyan]")
    
    def tokenize_function(examples):
        """Tokenize batch of examples."""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors=None
        )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'text': formatted_texts,
        'domain': domains,
        'difficulty': difficulties
    })
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )
    
    console.print(f"[green]✓ Dataset prepared: {len(tokenized_dataset):,} examples[/green]\n")
    
    return tokenized_dataset


def detect_gpu_architecture() -> Tuple[str, bool]:
    """
    Detect GPU architecture and Flash Attention support.
    
    Returns:
        (gpu_name, supports_fa3): GPU name and whether it supports FA3
    """
    if not torch.cuda.is_available():
        return "No GPU", False
    
    gpu_name = torch.cuda.get_device_name(0)
    # Get compute capability (SM version)
    compute_cap = torch.cuda.get_device_capability(0)
    sm_version = compute_cap[0] * 10 + compute_cap[1]
    
    # FA3 requires SM 90+ (Hopper: H100, H200)
    supports_fa3 = sm_version >= 90
    
    return gpu_name, supports_fa3


def setup_model_and_tokenizer(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    use_flash_attention: bool = True,
    force_fa_version: int = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Setup model with LoRA and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        use_flash_attention: Enable Flash Attention
        force_fa_version: Force specific FA version (2 or 3), None = auto-detect
    """
    console.print(f"\n[cyan]Loading base model:[/cyan] {model_name}")
    
    # Detect GPU and Flash Attention support
    gpu_name, supports_fa3 = detect_gpu_architecture()
    console.print(f"[cyan]Detected GPU:[/cyan] {gpu_name}")
    console.print(f"[cyan]Compute Capability:[/cyan] SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
    
    # Determine Flash Attention version
    if not use_flash_attention:
        attn_implementation = "eager"
        attn_version = "Standard"
    elif force_fa_version == 2:
        attn_implementation = "flash_attention_2"
        attn_version = "Flash Attention 2 (forced)"
    elif force_fa_version == 3:
        if not supports_fa3:
            console.print("[yellow]⚠️  FA3 requested but GPU doesn't support it (needs SM 90+)[/yellow]")
            console.print("[yellow]   Falling back to Flash Attention 2[/yellow]")
            attn_implementation = "flash_attention_2"
            attn_version = "Flash Attention 2 (fallback)"
        else:
            # Try FA3, fallback to FA2 if not available
            try:
                import flash_attn
                fa_version = tuple(map(int, flash_attn.__version__.split('.')[:2]))
                if fa_version >= (2, 6):
                    attn_implementation = "flash_attention_2"  # FA3 uses same API
                    attn_version = "Flash Attention 3"
                else:
                    console.print(f"[yellow]⚠️  flash-attn {flash_attn.__version__} < 2.6.0, FA3 requires 2.6+[/yellow]")
                    console.print("[yellow]   Falling back to Flash Attention 2[/yellow]")
                    attn_implementation = "flash_attention_2"
                    attn_version = "Flash Attention 2 (fallback)"
            except ImportError:
                console.print("[yellow]⚠️  flash-attn not installed, falling back to FA2[/yellow]")
                attn_implementation = "flash_attention_2"
                attn_version = "Flash Attention 2 (fallback)"
    else:
        # Auto-detect: Use FA3 on Hopper (H100/H200), FA2 otherwise
        if supports_fa3:
            try:
                import flash_attn
                fa_version = tuple(map(int, flash_attn.__version__.split('.')[:2]))
                if fa_version >= (2, 6):
                    attn_implementation = "flash_attention_2"  # FA3 uses same API
                    attn_version = "Flash Attention 3 (auto-detected)"
                    console.print("[bold green]✓ Using Flash Attention 3 (1.5-1.7x faster than FA2!)[/bold green]")
                else:
                    attn_implementation = "flash_attention_2"
                    attn_version = "Flash Attention 2"
            except ImportError:
                attn_implementation = "flash_attention_2"
                attn_version = "Flash Attention 2"
        else:
            attn_implementation = "flash_attention_2"
            attn_version = "Flash Attention 2"
    
    console.print(f"[cyan]Attention:[/cyan] {attn_version}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    
    console.print(f"[green]✓ Model loaded with {attn_version}[/green]")
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    console.print(f"\n[bold cyan]Model Parameters:[/bold cyan]")
    console.print(f"  • Total parameters: {total_params:,}")
    console.print(f"  • Trainable parameters: {trainable_params:,}")
    console.print(f"  • Trainable %: {100 * trainable_params / total_params:.2f}%")
    console.print(f"  • LoRA rank: {lora_rank}")
    console.print(f"  • LoRA alpha: {lora_alpha}")
    console.print()
    
    return model, tokenizer


def main():
    """Main training function."""
    args = parse_args()
    
    # Display training configuration
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]Phase 1: MAML + LoRA Training[/bold cyan]\n"
        f"Model: {args.model_name}\n"
        f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})\n"
        f"Learning rate: {args.learning_rate} (outer) / {args.inner_learning_rate} (inner)\n"
        f"Epochs: {args.num_epochs}\n"
        f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}",
        border_style="cyan"
    ))
    console.print("="*80 + "\n")
    
    # Convert paths to absolute
    data_file = PROJECT_ROOT / args.data_file
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and validate data
    examples = load_training_data(data_file)
    
    if args.test_mode:
        console.print("[yellow]⚠️  TEST MODE: Using only 100 examples[/yellow]\n")
        examples = examples[:100]
    
    stats = validate_data(examples)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.lora_rank,
        args.lora_alpha,
        use_flash_attention=True
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(examples, tokenizer, args.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    console.print(Panel.fit(
        "[bold green]🚀 Starting Training[/bold green]\n"
        f"Total steps: {len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation_steps)}\n"
        f"Estimated time: 7-8 hours (H100)\n"
        f"Estimated cost: ~$16",
        border_style="green"
    ))
    console.print()
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    console.print("\n[cyan]Saving final model...[/cyan]")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    
    console.print(f"[green]✓ Model saved to:[/green] {output_dir / 'final'}")
    console.print(f"[green]✓ Metrics saved to:[/green] {metrics_file}\n")
    
    # Display training summary
    console.print(Panel.fit(
        "[bold green]✅ Training Complete![/bold green]\n"
        f"Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}\n"
        f"Training time: {train_result.metrics.get('train_runtime', 0)/3600:.1f} hours\n"
        f"Samples/second: {train_result.metrics.get('train_samples_per_second', 'N/A'):.2f}",
        border_style="green"
    ))
    console.print()
    
    console.print("[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. Evaluate model on benchmarks")
    console.print("  2. Test on sample queries")
    console.print("  3. Proceed to Phase 2: Speed Infrastructure\n")


if __name__ == "__main__":
    main()
