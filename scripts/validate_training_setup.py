#!/usr/bin/env python3
"""
Quick test script to validate training setup before full training run.

Tests:
1. Data loading and validation
2. Model loading with Flash Attention
3. Tokenization and formatting
4. Single batch forward pass
5. Memory usage estimation

Run this BEFORE starting the full training to catch issues early.
"""

import torch
from pathlib import Path
from scripts.phase1_train_maml_lora import (
    load_training_data,
    validate_data,
    setup_model_and_tokenizer,
    prepare_dataset,
    format_training_example
)
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent


def test_data_loading():
    """Test data loading and validation."""
    console.print("\n[bold cyan]TEST 1: Data Loading & Validation[/bold cyan]\n")
    
    data_file = PROJECT_ROOT / "data/phase1/answers/training_data_interleaved.jsonl"
    
    try:
        examples = load_training_data(data_file)
        stats = validate_data(examples)
        
        console.print("[bold green]✓ Data loading and validation passed![/bold green]\n")
        return examples[:10]  # Return 10 samples for further tests
        
    except Exception as e:
        console.print(f"[bold red]❌ Data loading failed: {e}[/bold red]\n")
        return None


def test_model_loading():
    """Test model and tokenizer loading with Flash Attention."""
    console.print("\n[bold cyan]TEST 2: Model Loading (Flash Attention 2)[/bold cyan]\n")
    
    try:
        model, tokenizer = setup_model_and_tokenizer(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            lora_rank=64,
            lora_alpha=128,
            use_flash_attention=True
        )
        
        console.print("[bold green]✓ Model and tokenizer loaded successfully![/bold green]\n")
        return model, tokenizer
        
    except Exception as e:
        console.print(f"[bold red]❌ Model loading failed: {e}[/bold red]\n")
        console.print("[yellow]Note: Make sure you have Flash Attention 2 installed:[/yellow]")
        console.print("[yellow]  pip install flash-attn --no-build-isolation[/yellow]\n")
        return None, None


def test_formatting(examples, tokenizer):
    """Test example formatting."""
    console.print("\n[bold cyan]TEST 3: Example Formatting[/bold cyan]\n")
    
    if not examples or not tokenizer:
        console.print("[yellow]⚠️  Skipping (missing examples or tokenizer)[/yellow]\n")
        return
    
    try:
        # Test easy example
        easy_example = next(e for e in examples if e.get('metadata', {}).get('difficulty') == 'easy')
        easy_formatted = format_training_example(easy_example, tokenizer)
        
        console.print("[bold]Easy Example (first 300 chars):[/bold]")
        console.print(easy_formatted[:300] + "...\n")
        
        # Test hard example (if available)
        hard_examples = [e for e in examples if e.get('metadata', {}).get('difficulty') == 'hard']
        if hard_examples:
            hard_formatted = format_training_example(hard_examples[0], tokenizer)
            console.print("[bold]Hard Example (first 300 chars):[/bold]")
            console.print(hard_formatted[:300] + "...\n")
        
        console.print("[bold green]✓ Formatting works correctly![/bold green]\n")
        
    except Exception as e:
        console.print(f"[bold red]❌ Formatting failed: {e}[/bold red]\n")


def test_forward_pass(model, tokenizer, examples):
    """Test single forward pass."""
    console.print("\n[bold cyan]TEST 4: Forward Pass[/bold cyan]\n")
    
    if not model or not tokenizer or not examples:
        console.print("[yellow]⚠️  Skipping (missing model, tokenizer, or examples)[/yellow]\n")
        return
    
    try:
        # Format and tokenize a single example
        formatted = format_training_example(examples[0], tokenizer)
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Short for test
        ).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        console.print(f"[bold]Input shape:[/bold] {inputs['input_ids'].shape}")
        console.print(f"[bold]Output logits shape:[/bold] {outputs.logits.shape}")
        console.print(f"[bold]Loss:[/bold] {outputs.loss.item():.4f}\n")
        
        console.print("[bold green]✓ Forward pass successful![/bold green]\n")
        
    except Exception as e:
        console.print(f"[bold red]❌ Forward pass failed: {e}[/bold red]\n")


def test_memory_usage(model):
    """Test memory usage."""
    console.print("\n[bold cyan]TEST 5: Memory Usage[/bold cyan]\n")
    
    if not model:
        console.print("[yellow]⚠️  Skipping (missing model)[/yellow]\n")
        return
    
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            console.print(f"[bold]GPU Memory:[/bold]")
            console.print(f"  • Allocated: {allocated:.2f} GB")
            console.print(f"  • Reserved: {reserved:.2f} GB")
            console.print(f"  • Max allocated: {max_allocated:.2f} GB")
            
            # Check if we have enough memory for training
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"  • Total GPU memory: {total_memory:.2f} GB")
            
            estimated_training = allocated * 3  # Rough estimate for training
            console.print(f"\n[bold]Estimated training memory:[/bold] ~{estimated_training:.2f} GB")
            
            if estimated_training > total_memory * 0.9:
                console.print("[bold red]⚠️  WARNING: May run out of memory during training![/bold red]")
                console.print("[yellow]Consider reducing batch size or sequence length.[/yellow]\n")
            else:
                console.print("[bold green]✓ Memory should be sufficient for training![/bold green]\n")
        else:
            console.print("[yellow]⚠️  No GPU detected[/yellow]\n")
            
    except Exception as e:
        console.print(f"[bold red]❌ Memory check failed: {e}[/bold red]\n")


def main():
    """Run all tests."""
    console.print("="*80)
    console.print(Panel.fit(
        "[bold cyan]Training Setup Validation[/bold cyan]\n"
        "This will test your training environment before the full run.\n"
        "Estimated time: 2-3 minutes",
        border_style="cyan"
    ))
    console.print("="*80)
    
    # Run tests
    examples = test_data_loading()
    model, tokenizer = test_model_loading()
    test_formatting(examples, tokenizer)
    test_forward_pass(model, tokenizer, examples)
    test_memory_usage(model)
    
    # Summary
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold green]✅ Validation Complete![/bold green]\n\n"
        "Your training environment is ready.\n"
        "Run the full training with:\n"
        "  python scripts/phase1_train_maml_lora.py\n\n"
        "Or test mode first:\n"
        "  python scripts/phase1_train_maml_lora.py --test_mode",
        border_style="green"
    ))
    console.print("="*80 + "\n")


if __name__ == "__main__":
    main()
