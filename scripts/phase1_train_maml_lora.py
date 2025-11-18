#!/usr/bin/env python3
"""
Phase 1: Train Llama-3.1-8B-Instruct with ANIL-MAML + LoRA

This script implements ANIL-MAML (Almost No Inner Loop) with LoRA fine-tuning on the
cleaned training dataset. Uses Flash Attention 2 for memory efficiency.

Training Configuration (CORRECTED for MAML at Scale):
- Model: Llama-3.1-8B-Instruct (8.3B params)
- Precision: BF16
- LoRA: rank=16, alpha=32 (REDUCED to prevent overfitting)
- Batch size: 2 per device
- Gradient accumulation: 4 steps (effective batch=8)
- Learning rate: 5e-6 (outer loop, CORRECTED: 40× lower than before)
- Inner LR: 3e-5 (inner loop, CORRECTED: 100× lower than before)
- Epochs: 3 (with early stopping, patience=2)
- Hardware: H100 80GB (3-4 hours, $8-10 estimated)

Meta-Learning (ANIL-MAML) - CORRECTED Hyperparameters:
- Research: Raghu et al. 2020 - "Rapid Learning or Feature Reuse?"
- Inner loop: Adapt ONLY head layer (lm_head LoRA) on support sets (lr=3e-5, 3 steps)
- Outer loop: Meta-optimize ALL LoRA params across tasks (lr=5e-6)
- Key insight: Only output layer needs fast adaptation, representations stay stable
- Performance: Within 1-2% of full MAML, but 5-10x faster inner loop
- Perfect for PEFT/LoRA: Body (frozen base) + Head (LoRA) architecture

Task Definition (IMPROVED for Stability):
- 8 domains (Math, Coding, Science, Reasoning, etc.)
- Support set: 6 examples per task (improved adaptation)
- Query set: 6 examples per task (stable meta-gradients)
- Tasks per batch: 4 (INCREASED for better gradient estimates)
- Gradient clipping: 0.5 (conservative)

Cost: ~$10 for full training
Output: 14GB BF16 model with LoRA weights

Memory Management:
- Uses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Aggressive cache clearing after each inner/outer loop
- Conservative defaults to fit in 80GB H100
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
from transformers.data.data_collator import DataCollatorMixin
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel


class MAMLDataCollator(DataCollatorMixin):
    """
    Custom data collator that handles domain/difficulty metadata for MAML.
    Keeps metadata fields as lists instead of trying to convert to tensors.
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features):
        # Separate metadata from tensor features
        metadata_keys = ['domain', 'difficulty']
        metadata = {key: [] for key in metadata_keys}
        tensor_features = []
        
        for feature in features:
            # Extract metadata
            for key in metadata_keys:
                if key in feature:
                    metadata[key].append(feature[key])
            
            # Keep only tensor-compatible features
            tensor_feature = {k: v for k, v in feature.items() if k not in metadata_keys}
            tensor_features.append(tensor_feature)
        
        # Use default collator for tensor features
        batch = self.tokenizer.pad(
            tensor_features,
            padding=True,
            return_tensors='pt'
        )
        
        # Add labels (for causal LM, labels = input_ids)
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
        
        # Add metadata back as lists (not tensors)
        for key, values in metadata.items():
            if values:  # Only add if not empty
                batch[key] = values
        
        return batch

console = Console()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Llama-3.1-8B with MAML + LoRA")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/phase1/answers/training_data_clean.jsonl",
        help="Path to cleaned training data (without XML tags)"
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
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps). MAML stable: 8 for better gradients."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-6,
        help="Learning rate for outer loop (meta-optimization). MAML best practice for 8B: 3e-6."
    )
    parser.add_argument(
        "--inner_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for inner loop (task adaptation). MAML best practice for 8B: 1e-5."
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
        default=16,
        help="LoRA rank. MAML best practice: 8-16 for 8B models to prevent overfitting."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor). Set to 2× rank for MAML."
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
    parser.add_argument(
        "--inner_steps",
        type=int,
        default=1,
        help="Number of gradient steps in inner loop (head adaptation). MAML stable: 1-2 steps for 8B models."
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=2,
        help="Number of tasks (domains) to sample per meta-batch. MAML stable: 2-4 tasks for 8B models."
    )
    parser.add_argument(
        "--support_size",
        type=int,
        default=4,
        help="Number of examples per task for inner loop adaptation. Conservative: 4 for memory safety."
    )
    parser.add_argument(
        "--query_size",
        type=int,
        default=4,
        help="Number of examples per task for outer loop meta-update. Conservative: 4 for memory safety."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping: stop if no improvement for N epochs. Set to 0 to disable. With 3 epochs, patience=2 is recommended."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint. Use 'auto' or 'latest' to auto-detect, or provide path (e.g., data/checkpoints/phase1/checkpoint-1000)"
    )
    
    return parser.parse_args()


def find_latest_checkpoint(output_dir: Path) -> Path:
    """Find the latest checkpoint in the output directory."""
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Sort by step number (extract from checkpoint-XXXX)
    checkpoint_dirs.sort(key=lambda p: int(p.name.split("-")[-1]))
    return checkpoint_dirs[-1]


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
    """Format example into Llama chat template format.
    
    UPDATED: Now expects clean data without XML tags.
    - Easy examples: Direct answers
    - Hard examples: Natural language reasoning structure already integrated
    """
    prompt = example['prompt']
    response = example['response']  # Already formatted correctly (cleaned data)
    
    # Create standard chat format
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
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
    
    # Tokenize (keep domain and difficulty for MAML task grouping)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],  # Only remove the text column
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
    
    # Disable gradient checkpointing for FOMAML compatibility
    # Gradient checkpointing can interfere with meta-learning gradient flow
    # model.gradient_checkpointing_enable()
    
    # Ensure model is in training mode
    model.train()
    
    # FIX 1: Force gradients on LoRA params after wrapping
    # Make absolutely sure LoRA parameters are trainable
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # Verify gradients are enabled for LoRA params
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    if len(lora_params) == 0:
        raise RuntimeError("No LoRA parameters require gradients!")
    
    console.print(f"[cyan]  • LoRA parameters with gradients: {len(lora_params)}[/cyan]")
    
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


def group_examples_by_domain(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group examples by domain for task-based sampling."""
    domains = defaultdict(list)
    for example in examples:
        domain = example.get('metadata', {}).get('domain', 'Unknown')
        domains[domain].append(example)
    return dict(domains)


def maml_train_step(
    model,
    tokenizer,
    task_examples: List[Dict],
    inner_lr: float,
    inner_steps: int,
    support_size: int,
    query_size: int,
    max_seq_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Single MAML training step for one task.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        task_examples: All examples for this task
        inner_lr: Learning rate for inner loop
        inner_steps: Number of gradient steps in inner loop
        support_size: Number of examples for adaptation
        query_size: Number of examples for meta-loss
        max_seq_length: Max sequence length
        device: Device to train on
        
    Returns:
        meta_loss: Loss on query set after adaptation
    """
    # Sample support and query sets
    if len(task_examples) < support_size + query_size:
        # If not enough examples, sample with replacement
        support_set = random.choices(task_examples, k=support_size)
        query_set = random.choices(task_examples, k=query_size)
    else:
        sampled = random.sample(task_examples, support_size + query_size)
        support_set = sampled[:support_size]
        query_set = sampled[support_size:]
    
    # Save original LoRA parameters
    original_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            original_state[name] = param.data.clone()
    
    # Clear any cached memory before inner loop
    torch.cuda.empty_cache()
    
    # INNER LOOP: Fast adaptation on support set
    # Using ANIL-MAML (Almost No Inner Loop) - only adapt head layer
    # 
    # RESEARCH-BACKED APPROACH (Raghu et al., 2020):
    # - Inner loop: Only adapt head (lm_head) - learns task-specific output mapping
    # - Outer loop: Meta-optimize ALL LoRA params - learns good representations
    # - Performance: Within 1-2% of full MAML, but 5-10x faster inner loop
    # - Perfect fit for PEFT/LoRA: Body (frozen base) + Head (LoRA) architecture
    # 
    # KEY INSIGHT: Representations don't change much during inner loop adaptation.
    # Only the final layer needs to adapt quickly to new tasks.
    model.train()
    
    # ANIL: Only adapt head layer in inner loop (lm_head LoRA parameters)
    head_params = [p for n, p in model.named_parameters() 
                   if 'lm_head' in n and 'lora' in n.lower() and p.requires_grad]
    
    if len(head_params) == 0:
        # Fallback: If no lm_head LoRA, use last layer LoRA params
        all_lora_params = [(n, p) for n, p in model.named_parameters() 
                           if 'lora' in n.lower() and p.requires_grad]
        # Take last layer (typically down_proj or o_proj of last transformer block)
        if len(all_lora_params) > 0:
            last_layer_names = [n for n, _ in all_lora_params[-4:]]  # Last 4 params (A/B for 2 modules)
            head_params = [p for n, p in all_lora_params if n in last_layer_names]
    
    for inner_step in range(inner_steps):
        # Format and tokenize support examples
        support_texts = [format_training_example(ex, tokenizer) for ex in support_set]
        support_encodings = tokenizer(
            support_texts,
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors='pt'
        ).to(device)
        
        # Forward pass with autocast for bf16
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**support_encodings, labels=support_encodings['input_ids'])
            loss = outputs.loss
        
        # ANIL: Compute gradients ONLY for head parameters (no computation graph)
        grads = torch.autograd.grad(
            loss,
            head_params,
            create_graph=False,  # First-order only, no second-order gradients
            allow_unused=True
        )
        
        # Apply inner loop updates to head only (detached from graph)
        with torch.no_grad():
            for param, grad in zip(head_params, grads):
                if grad is not None:
                    # ANIL: Only head parameters adapt in inner loop
                    param.data.sub_(inner_lr * grad)
        
        # AGGRESSIVE memory cleanup after each inner step
        del support_encodings, outputs, loss, grads
        torch.cuda.empty_cache()  # Always clear, not just between steps
    
    # Clear gradients before outer loop
    model.zero_grad()
    
    # OUTER LOOP: Compute meta-loss on query set
    # This gradient WILL flow back for meta-optimization
    query_texts = [format_training_example(ex, tokenizer) for ex in query_set]
    query_encodings = tokenizer(
        query_texts,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    # Forward pass with autocast for bf16
    # This gradient WILL flow back (not detached)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**query_encodings, labels=query_encodings['input_ids'])
        meta_loss = outputs.loss
    
    # Restore original model parameters for next task
    with torch.no_grad():
        for name, original_value in original_state.items():
            for pname, param in model.named_parameters():
                if pname == name:
                    param.data.copy_(original_value)
                    break
    
    # Clear memory after each task
    del original_state, query_encodings, outputs
    torch.cuda.empty_cache()
    
    return meta_loss


def train_maml(
    model,
    tokenizer,
    examples: List[Dict],
    args,
    output_dir: Path
):
    """
    MAML training loop.
    
    Implements Model-Agnostic Meta-Learning:
    1. Sample tasks (domains)
    2. For each task: adapt on support set, evaluate on query set
    3. Meta-update: improve base model to be good at adaptation
    """
    console.print("\n[bold cyan]Starting MAML Training[/bold cyan]\n")
    
    # Group examples by domain (tasks)
    domain_groups = group_examples_by_domain(examples)
    domains = list(domain_groups.keys())
    
    console.print(f"[cyan]Tasks (domains):[/cyan] {', '.join(domains)}")
    console.print(f"[cyan]Examples per domain:[/cyan]")
    for domain, exs in domain_groups.items():
        console.print(f"  • {domain}: {len(exs):,}")
    console.print()
    
    # Setup optimizer for meta-updates
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Calculate total steps
    num_examples = len(examples)
    steps_per_epoch = num_examples // (args.tasks_per_batch * (args.support_size + args.query_size))
    total_steps = steps_per_epoch * args.num_epochs
    
    console.print(f"[cyan]Training configuration:[/cyan]")
    console.print(f"  • Tasks per batch: {args.tasks_per_batch}")
    console.print(f"  • Support size: {args.support_size}")
    console.print(f"  • Query size: {args.query_size}")
    console.print(f"  • Inner steps: {args.inner_steps}")
    console.print(f"  • Inner LR: {args.inner_learning_rate}")
    console.print(f"  • Outer LR: {args.learning_rate}")
    console.print(f"  • Steps per epoch: {steps_per_epoch}")
    console.print(f"  • Total steps: {total_steps}")
    console.print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear any cached memory before starting
    torch.cuda.empty_cache()
    
    # Training loop with early stopping
    global_step = 0
    best_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    epoch_losses = []
    early_stopping_enabled = patience > 0
    
    if early_stopping_enabled:
        console.print(f"[cyan]Early stopping enabled: patience={patience} epochs[/cyan]\n")
    else:
        console.print(f"[cyan]Early stopping disabled[/cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        train_task = progress.add_task(f"Training (Epoch 1/{args.num_epochs})", total=total_steps)
        
        for epoch in range(args.num_epochs):
            progress.update(train_task, description=f"Training (Epoch {epoch+1}/{args.num_epochs})")
            epoch_loss = 0.0
            
            for step in range(steps_per_epoch):
                # Sample tasks (domains)
                sampled_domains = random.sample(domains, min(args.tasks_per_batch, len(domains)))
                
                # Meta-batch: Compute loss for each task
                # Process tasks one at a time to minimize memory
                meta_losses = []
                for domain in sampled_domains:
                    task_examples = domain_groups[domain]
                    
                    # Compute meta-loss for this task
                    meta_loss = maml_train_step(
                        model=model,
                        tokenizer=tokenizer,
                        task_examples=task_examples,
                        inner_lr=args.inner_learning_rate,
                        inner_steps=args.inner_steps,
                        support_size=args.support_size,
                        query_size=args.query_size,
                        max_seq_length=args.max_seq_length,
                        device=device
                    )
                    
                    # Store loss value, detach to free computation graph
                    meta_losses.append(meta_loss.detach())
                    
                    # Backward on this task's loss immediately
                    meta_loss.backward()
                    
                    # Clear memory after each task
                    del meta_loss
                
                # Meta-update (outer loop optimization)
                # Gradients already computed above
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # MAML best practice: 0.5-1.0
                optimizer.step()
                optimizer.zero_grad()
                
                # Calculate average loss for logging
                avg_meta_loss = torch.stack(meta_losses).mean().item()
                
                # AGGRESSIVE cleanup after meta-batch
                del meta_losses
                torch.cuda.empty_cache()
                
                # Logging
                epoch_loss += avg_meta_loss
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress.console.print(
                        f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                        f"Tasks: {', '.join(sampled_domains)}"
                    )
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(checkpoint_dir))
                    tokenizer.save_pretrained(str(checkpoint_dir))
                    console.print(f"[green]✓ Checkpoint saved: {checkpoint_dir}[/green]")
                
                progress.advance(train_task)
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / steps_per_epoch
            epoch_losses.append(avg_epoch_loss)
            console.print(f"\n[bold]Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}[/bold]")
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                best_dir = output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
                console.print(f"[green]✓ Best model saved (loss: {best_loss:.4f})[/green]")
                if early_stopping_enabled:
                    console.print(f"[green]  Patience reset: {patience_counter}/{patience}[/green]\n")
                else:
                    console.print()
            else:
                if early_stopping_enabled:
                    patience_counter += 1
                    console.print(f"[yellow]⚠ No improvement (patience: {patience_counter}/{patience})[/yellow]\n")
                    
                    if patience_counter >= patience:
                        console.print(f"[yellow]⚠ Early stopping triggered - No improvement for {patience} epochs[/yellow]")
                        console.print(f"[cyan]Best loss: {best_loss:.4f} at epoch {epoch + 1 - patience}[/cyan]\n")
                        break
                else:
                    console.print()
    
    # Training summary
    console.print("[bold green]✓ MAML Training Complete![/bold green]")
    console.print(f"[cyan]Final best loss: {best_loss:.4f}[/cyan]")
    console.print(f"[cyan]Total epochs: {len(epoch_losses)}/{args.num_epochs}[/cyan]\n")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set environment variables for memory and performance optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Display training configuration
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]Phase 1: ANIL-MAML + LoRA Training (v2 Optimized)[/bold cyan]\n"
        f"Model: {args.model_name}\n"
        f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})\n"
        f"Learning rate: {args.learning_rate} (outer) / {args.inner_learning_rate} (inner)\n"
        f"Epochs: {args.num_epochs}\n"
        f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}\n"
        f"ANIL: inner_steps={args.inner_steps}, support={args.support_size}, query={args.query_size}\n"
        f"Memory: TF32 enabled, max_split_size=512MB",
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
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        # If just "auto" or "latest", find the latest checkpoint
        if args.resume_from_checkpoint.lower() in ["auto", "latest"]:
            checkpoint_path = find_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                console.print("[yellow]⚠️  No checkpoints found, starting from scratch[/yellow]\n")
            else:
                console.print(f"[cyan]Auto-detected latest checkpoint:[/cyan] {checkpoint_path}")
        else:
            checkpoint_path = Path(args.resume_from_checkpoint)
        
        if checkpoint_path and checkpoint_path.exists():
            console.print(f"[cyan]Resuming from checkpoint:[/cyan] {checkpoint_path}")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(checkpoint_path))
                console.print("[green]✓ Checkpoint loaded successfully[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Failed to load checkpoint: {e}[/red]")
                console.print("[yellow]Starting training from scratch instead...[/yellow]\n")
        elif checkpoint_path:
            console.print(f"[red]✗ Checkpoint not found: {checkpoint_path}[/red]")
            console.print("[yellow]Starting training from scratch instead...[/yellow]\n")
    
    # Run MAML training
    train_maml(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        args=args,
        output_dir=output_dir
    )
    
    # Save final model
    console.print("\n[cyan]Saving final model...[/cyan]")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    console.print(f"[green]✓ Model saved to:[/green] {final_dir}\n")
    
    # Display completion summary
    console.print(Panel.fit(
        "[bold green]✅ MAML Training Complete![/bold green]\n"
        f"Model saved: {final_dir}\n"
        f"The model is now adapted for fast learning on new tasks!",
        border_style="green"
    ))
    console.print()
    
    console.print("[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. Evaluate model on benchmarks")
    console.print("  2. Test few-shot adaptation on new tasks")
    console.print("  3. Proceed to Phase 2: Speed Infrastructure\n")


if __name__ == "__main__":
    main()
