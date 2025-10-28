#!/usr/bin/env python3
"""
Download LLAMA-3.2-8B Base Model

PURPOSE:
    Downloads LLAMA-3.2-8B model and tokenizer from HuggingFace to local cache.
    Ensures model is available offline for training without re-downloading.

WHEN TO USE:
    - Before starting Phase 1A training on new machine
    - To pre-cache model for offline training environments
    - After fresh environment setup

OUTPUT:
    - Model cached in: ~/.cache/huggingface/hub/
    - ~16GB download (FP16 weights)

PIPELINE STAGE: Setup utility - run once per machine

NOTE: Requires HuggingFace token for gated models (LLAMA requires acceptance of license)
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def download_qwen():
    """Download Qwen 2.5 7B model and tokenizer."""
    
    model_name = "Qwen/Qwen2.5-7B"
    
    console.print(f"\n[bold cyan]Downloading Qwen 2.5 7B Model[/bold cyan]")
    console.print(f"Model: {model_name}")
    console.print(f"Size: ~14GB (full precision)")
    console.print(f"This may take 10-30 minutes depending on your connection...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Download tokenizer
        task1 = progress.add_task("[cyan]Downloading tokenizer...", total=None)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            progress.update(task1, description="[green]✓ Tokenizer downloaded")
            console.print(f"[green]✓ Tokenizer vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            console.print(f"[red]✗ Tokenizer download failed: {e}")
            return False
        
        # Download model (this is the large download)
        task2 = progress.add_task("[cyan]Downloading model weights (14GB)...", total=None)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=None  # Don't load to GPU yet, just download
            )
            progress.update(task2, description="[green]✓ Model weights downloaded")
            
            # Model info
            num_params = sum(p.numel() for p in model.parameters())
            console.print(f"[green]✓ Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")
            
            # Clear memory
            del model
            
        except Exception as e:
            console.print(f"[red]✗ Model download failed: {e}")
            return False
    
    console.print("\n[bold green]✓ Qwen 2.5 7B successfully downloaded and cached!")
    console.print(f"[dim]Cache location: ~/.cache/huggingface/hub/[/dim]\n")
    
    return True

if __name__ == "__main__":
    success = download_qwen()
    if not success:
        console.print("[red]Download failed. Please check your internet connection and try again.")
        exit(1)
