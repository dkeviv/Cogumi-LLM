#!/usr/bin/env python3
"""
Phase 1D: Merge LoRA Weights into Base Model

Merges trained LoRA adapter weights into base model and saves merged version.
This creates a standalone model without requiring PEFT library at inference.

Usage:
    python scripts/phase1_merge_lora.py \
        --lora_path models/phase1_maml_lora_v2/best \
        --output_path models/phase1_maml_lora_v2/merged

Process:
    1. Load base Llama-3.1-8B-Instruct model
    2. Load LoRA adapter weights
    3. Merge LoRA into base model weights
    4. Save merged model (14GB FP16 or 7GB BF16)

Benefits:
    - Simpler deployment (no PEFT dependency)
    - Faster inference (no adapter overhead)
    - Easier quantization (standard model format)
    - Identical results to LoRA (within numerical precision)

Author: Cogumi-LLM
Date: November 18, 2025
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_model_name: str,
    lora_path: str,
    output_path: str,
    precision: str = "bfloat16"
):
    """Merge LoRA adapter weights into base model."""
    
    console.print("\n[bold blue]Starting LoRA Weight Merge[/bold blue]")
    console.print(f"Base model: {base_model_name}")
    console.print(f"LoRA adapter: {lora_path}")
    console.print(f"Output path: {output_path}")
    console.print(f"Precision: {precision}")
    
    # Determine dtype
    if precision == "bfloat16":
        dtype = torch.bfloat16
    elif precision == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Step 1: Load base model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task1 = progress.add_task("Loading base model...", total=None)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        progress.update(task1, completed=True)
        console.print("[green]✓[/green] Base model loaded")
    
    # Step 2: Load LoRA adapter
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task2 = progress.add_task("Loading LoRA adapter...", total=None)
        
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=dtype
        )
        
        progress.update(task2, completed=True)
        console.print("[green]✓[/green] LoRA adapter loaded")
    
    # Step 3: Merge LoRA weights into base model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task3 = progress.add_task("Merging LoRA weights...", total=None)
        
        # Merge and unload LoRA (creates standalone model)
        merged_model = model_with_lora.merge_and_unload()
        
        progress.update(task3, completed=True)
        console.print("[green]✓[/green] LoRA weights merged successfully")
    
    # Step 4: Save merged model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task4 = progress.add_task("Saving merged model...", total=None)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True
        )
        
        progress.update(task4, completed=True)
        console.print(f"[green]✓[/green] Model saved to: {output_path}")
    
    # Step 5: Save tokenizer
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task5 = progress.add_task("Saving tokenizer...", total=None)
        
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        tokenizer.save_pretrained(output_path)
        
        progress.update(task5, completed=True)
        console.print(f"[green]✓[/green] Tokenizer saved")
    
    # Display size info
    model_size = sum(
        os.path.getsize(output_dir / f)
        for f in os.listdir(output_dir)
        if f.endswith('.safetensors') or f.endswith('.bin')
    ) / (1024 ** 3)  # Convert to GB
    
    console.print(f"\n[bold green]✓ Merge Complete![/bold green]")
    console.print(f"Merged model size: {model_size:.2f} GB")
    console.print(f"Location: {output_path}")
    
    return merged_model


def verify_merge(lora_path: str, merged_path: str):
    """Quick verification that merge was successful."""
    
    console.print("\n[bold yellow]Verifying Merge...[/bold yellow]")
    
    # Load both models
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    console.print("Loading LoRA model...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_model, lora_path)
    
    console.print("Loading merged model...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Compare a few random weights
    console.print("Comparing model weights...")
    
    # Get state dicts
    lora_model.eval()
    merged_model.eval()
    
    # Merge LoRA weights for comparison
    lora_merged = lora_model.merge_and_unload()
    
    lora_state = lora_merged.state_dict()
    merged_state = merged_model.state_dict()
    
    # Compare a sample of parameters
    sample_params = list(lora_state.keys())[:10]
    max_diff = 0.0
    
    for param_name in sample_params:
        if param_name in merged_state:
            lora_param = lora_state[param_name]
            merged_param = merged_state[param_name]
            
            diff = torch.max(torch.abs(lora_param - merged_param)).item()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-4:
                console.print(f"[yellow]Warning: {param_name} differs by {diff:.6f}[/yellow]")
    
    if max_diff < 1e-4:
        console.print(f"[green]✓[/green] Verification passed (max diff: {max_diff:.2e})")
    else:
        console.print(f"[yellow]⚠[/yellow] Some differences detected (max diff: {max_diff:.2e})")
        console.print("  This may be due to numerical precision differences")
    
    return max_diff < 1e-4


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint (e.g., models/phase1_maml_lora_v2/best)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model (e.g., models/phase1_maml_lora_v2/merged)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision for merged model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after merge (compares weights)"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    console.print("\n[bold]Phase 1D: LoRA Weight Merge[/bold]")
    console.print("=" * 60)
    
    # Perform merge
    merged_model = merge_lora_weights(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        precision=args.precision
    )
    
    # Clean up
    del merged_model
    torch.cuda.empty_cache()
    
    # Verify if requested
    if args.verify:
        verify_merge(args.lora_path, args.output_path)
    
    console.print("\n[bold green]✓ All operations complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("1. Run validation: python scripts/phase1_validate_maml.py")
    console.print("2. Test inference with merged model")
    console.print("3. Proceed to quantization (Phase 2)")


if __name__ == "__main__":
    main()
