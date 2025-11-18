#!/usr/bin/env python3
"""
Phase 1D: MAML Model Validation Script

Validates ANIL-MAML trained model on held-out test set.
Tests both LoRA adapter and merged model versions.

Usage:
    python scripts/phase1_validate_maml.py \
        --model_path models/phase1_maml_lora_v2/best \
        --test_file data/phase1/test_set.jsonl \
        --output_dir results/phase1_validation

Expected:
    - Loss on unseen examples < 0.10 (generalization)
    - Perplexity < 1.5 on easy examples
    - Perplexity < 3.0 on hard examples
    - Identical results pre/post merge (within 1e-4)

Author: Cogumi-LLM
Date: November 18, 2025
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel, get_peft_model
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str, use_merged: bool = False):
    """Load base model with LoRA adapter or merged model."""
    console.print(f"\n[bold blue]Loading model from: {model_path}[/bold blue]")
    
    # Determine paths
    if use_merged:
        model_dir = model_path.replace("/best", "/merged").replace("/final", "/merged")
    else:
        model_dir = model_path
    
    # Check if this is a base model (no adapter) or LoRA model
    # Base models: HuggingFace model IDs like "meta-llama/..."
    # LoRA models: Local paths with adapter_config.json
    is_base_model = "/" in model_dir and not model_dir.startswith(".")
    has_adapter = False
    
    if not is_base_model:
        # Check if adapter_config.json exists for local paths
        adapter_config_path = Path(model_dir) / "adapter_config.json"
        has_adapter = adapter_config_path.exists()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except:
        # If loading from LoRA checkpoint fails, try base model
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if use_merged or (is_base_model and not has_adapter):
        # Load model directly (merged or base model without adapter)
        console.print(f"[cyan]Loading as {'merged' if use_merged else 'base'} model (no adapter)[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load base model + LoRA adapter
        console.print(f"[cyan]Loading base model + LoRA adapter[/cyan]")
        base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_dir)
    
    model.eval()
    console.print(f"[green]✓[/green] Model loaded successfully")
    return model, tokenizer


def load_test_set(test_file: str, max_samples: int = None) -> List[Dict]:
    """Load test set from JSONL file."""
    console.print(f"\n[bold blue]Loading test set from: {test_file}[/bold blue]")
    
    test_examples = []
    with open(test_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            test_examples.append(example)
            if max_samples and len(test_examples) >= max_samples:
                break
    
    console.print(f"[green]✓[/green] Loaded {len(test_examples)} test examples")
    return test_examples


def compute_loss_and_perplexity(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int = 4,
    max_length: int = 2048
) -> Tuple[float, float, Dict]:
    """Compute average loss and perplexity on examples."""
    
    total_loss = 0.0
    total_tokens = 0
    results_by_difficulty = {"easy": [], "medium": [], "hard": []}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Computing loss on {len(examples)} examples...",
            total=len(examples)
        )
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            # Prepare inputs
            texts = []
            for ex in batch:
                # Support both formats: input/output (old) and prompt/response (benchmark)
                if 'input' in ex and 'output' in ex:
                    text = f"{ex['input']}\n\n{ex['output']}"
                elif 'prompt' in ex and 'response' in ex:
                    text = f"{ex['prompt']}\n\n{ex['response']}"
                else:
                    raise ValueError(f"Example must have either 'input'/'output' or 'prompt'/'response' fields. Got: {ex.keys()}")
                texts.append(text)
            
            # Tokenize
            encodings = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(model.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                
                # Accumulate
                batch_loss = loss.item() * len(batch)
                total_loss += batch_loss
                
                # Track by difficulty
                for ex in batch:
                    # Check both top-level and metadata for difficulty
                    if "metadata" in ex and "difficulty" in ex["metadata"]:
                        difficulty = ex["metadata"]["difficulty"]
                    else:
                        difficulty = ex.get("difficulty", "easy")
                    
                    # Normalize difficulty values
                    if difficulty not in results_by_difficulty:
                        results_by_difficulty[difficulty] = []
                    results_by_difficulty[difficulty].append(loss.item())
            
            total_tokens += encodings["input_ids"].numel()
            progress.advance(task, len(batch))
    
    # Compute metrics
    avg_loss = total_loss / len(examples)
    perplexity = np.exp(avg_loss)
    
    # Compute per-difficulty metrics
    difficulty_metrics = {}
    for diff, losses in results_by_difficulty.items():
        if losses:
            difficulty_metrics[diff] = {
                "avg_loss": np.mean(losses),
                "perplexity": np.exp(np.mean(losses)),
                "count": len(losses)
            }
    
    return avg_loss, perplexity, difficulty_metrics


def validate_model(
    model,
    tokenizer,
    test_examples: List[Dict],
    model_type: str = "LoRA"
) -> Dict:
    """Run validation on model and return metrics."""
    
    console.print(f"\n[bold yellow]Validating {model_type} Model[/bold yellow]")
    
    # Compute overall metrics
    avg_loss, perplexity, difficulty_metrics = compute_loss_and_perplexity(
        model, tokenizer, test_examples
    )
    
    # Build results
    results = {
        "model_type": model_type,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "num_examples": len(test_examples),
        "difficulty_metrics": difficulty_metrics
    }
    
    # Display results
    table = Table(title=f"{model_type} Model Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Average Loss", f"{avg_loss:.4f}")
    table.add_row("Perplexity", f"{perplexity:.4f}")
    table.add_row("Examples", str(len(test_examples)))
    
    for diff, metrics in difficulty_metrics.items():
        table.add_row(f"{diff.capitalize()} Loss", f"{metrics['avg_loss']:.4f}")
        table.add_row(f"{diff.capitalize()} Perplexity", f"{metrics['perplexity']:.4f}")
        table.add_row(f"{diff.capitalize()} Count", str(metrics['count']))
    
    console.print(table)
    
    return results


def compare_results(lora_results: Dict, merged_results: Dict) -> bool:
    """Compare LoRA vs merged results and check if identical."""
    
    console.print("\n[bold yellow]Comparing LoRA vs Merged Results[/bold yellow]")
    
    # Compute differences
    loss_diff = abs(lora_results["avg_loss"] - merged_results["avg_loss"])
    ppl_diff = abs(lora_results["perplexity"] - merged_results["perplexity"])
    
    # Display comparison
    table = Table(title="LoRA vs Merged Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("LoRA", style="blue")
    table.add_column("Merged", style="green")
    table.add_column("Difference", style="yellow")
    
    table.add_row(
        "Average Loss",
        f"{lora_results['avg_loss']:.6f}",
        f"{merged_results['avg_loss']:.6f}",
        f"{loss_diff:.6f}"
    )
    table.add_row(
        "Perplexity",
        f"{lora_results['perplexity']:.6f}",
        f"{merged_results['perplexity']:.6f}",
        f"{ppl_diff:.6f}"
    )
    
    console.print(table)
    
    # Check if acceptable (within 1e-4 tolerance for loss)
    acceptable = loss_diff < 1e-4
    
    if acceptable:
        console.print("\n[green]✓ Results are IDENTICAL (within tolerance)[/green]")
    else:
        console.print(f"\n[red]✗ Results differ by {loss_diff:.6f} (expected < 1e-4)[/red]")
    
    return acceptable


def save_results(results: Dict, output_path: str):
    """Save validation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate MAML trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., models/phase1_maml_lora_v2/best)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test set JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phase1_validation",
        help="Directory to save validation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (for quick testing)"
    )
    parser.add_argument(
        "--skip_merged",
        action="store_true",
        help="Skip merged model validation (only validate LoRA)"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    console.print("\n[bold]Phase 1D: MAML Model Validation[/bold]")
    console.print(f"Model path: {args.model_path}")
    console.print(f"Test file: {args.test_file}")
    console.print(f"Output dir: {args.output_dir}")
    
    # Load test set
    test_examples = load_test_set(args.test_file, args.max_samples)
    
    # Validate LoRA model
    lora_model, lora_tokenizer = load_model_and_tokenizer(args.model_path, use_merged=False)
    lora_results = validate_model(lora_model, lora_tokenizer, test_examples, "LoRA")
    
    # Save LoRA results
    lora_output = Path(args.output_dir) / "lora_validation.json"
    save_results(lora_results, lora_output)
    
    # Clean up
    del lora_model
    torch.cuda.empty_cache()
    
    # Validate merged model (if not skipped)
    if not args.skip_merged:
        merged_model, merged_tokenizer = load_model_and_tokenizer(args.model_path, use_merged=True)
        merged_results = validate_model(merged_model, merged_tokenizer, test_examples, "Merged")
        
        # Save merged results
        merged_output = Path(args.output_dir) / "merged_validation.json"
        save_results(merged_results, merged_output)
        
        # Compare results
        identical = compare_results(lora_results, merged_results)
        
        # Save comparison
        comparison = {
            "lora": lora_results,
            "merged": merged_results,
            "identical": identical,
            "loss_difference": abs(lora_results["avg_loss"] - merged_results["avg_loss"]),
            "perplexity_difference": abs(lora_results["perplexity"] - merged_results["perplexity"])
        }
        comparison_output = Path(args.output_dir) / "comparison.json"
        save_results(comparison, comparison_output)
        
        # Clean up
        del merged_model
        torch.cuda.empty_cache()
    
    console.print("\n[bold green]✓ Validation Complete![/bold green]")


if __name__ == "__main__":
    main()
