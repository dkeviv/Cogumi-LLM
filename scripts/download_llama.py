"""
Download Qwen 2.5 7B base model from HuggingFace.

This script downloads the model and tokenizer to the local cache.
The model will be available for training without re-downloading.
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
