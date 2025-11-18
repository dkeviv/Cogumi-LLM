#!/usr/bin/env python3
"""
Phase 1E: Generate Teacher Outputs for Draft Model Training (Single-Task Approach)

Generates responses from the trained 8B MAML model (merged) on ALL 53,597 training examples.
These outputs will be used to train the Qwen2.5-0.5B draft model using standard language modeling.

Key Points:
- Generates for ALL examples (easy + hard) - no special handling
- Draft will be trained on all base model responses using single-task LM
- Confidence routing emerges naturally at inference (not learned during training)
- OPTIMIZED for H200 with batch_size=32 (15× speedup vs sequential)

Process:
1. Load merged 8B MAML model
2. For each batch of training examples:
   - Generate responses in parallel on GPU (batch_size=32 on H200)
   - Save input, base response, metadata (difficulty, domain)
3. Output format ready for standard causal LM training

Usage (H200 - RECOMMENDED):
    python scripts/phase1e_generate_teacher_outputs.py \
        --model_path models/phase1_maml_lora_v2/merged \
        --input_file data/phase1/answers/training_data_clean.jsonl \
        --output_file data/phase1e/teacher_outputs_53k.jsonl \
        --batch_size 32 \
        --max_new_tokens 2048

Usage (H100):
    python scripts/phase1e_generate_teacher_outputs.py \
        --model_path models/phase1_maml_lora_v2/merged \
        --input_file data/phase1/answers/training_data_clean.jsonl \
        --output_file data/phase1e/teacher_outputs_53k.jsonl \
        --batch_size 16 \
        --max_new_tokens 2048

Note: Use max_new_tokens=2048 to allow full responses (model trained with EOS token).
      Hard questions may require longer outputs. Model will stop at EOS naturally.

Expected Runtime (H200, batch_size=32): ~2-3 hours for 53,597 examples (vs 38 hours sequential)
Expected Cost: ~$1-2 on Vast.ai (inference only, 2-3 hours @ $0.50/hr)

Author: Cogumi-LLM
Date: November 18, 2025 (Updated for batched generation on H200)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str):
    """Load merged 8B MAML model."""
    console.print(f"\n[bold blue]Loading merged model from: {model_path}[/bold blue]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # CRITICAL: Set padding_side='left' for decoder-only models
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    console.print(f"[green]✓[/green] Model loaded successfully")
    return model, tokenizer


def load_training_data(input_file: str) -> List[Dict]:
    """Load training data."""
    console.print(f"\n[bold blue]Loading training data from: {input_file}[/bold blue]")
    
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    console.print(f"[green]✓[/green] Loaded {len(examples)} training examples")
    return examples


def generate_teacher_output(
    model,
    tokenizer,
    example: Dict,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Generate response from teacher model.
    
    Note: Model trained with EOS token, will naturally stop when complete.
          max_new_tokens is safety limit for hard questions with long outputs.
    """
    
    # Extract input/prompt
    if 'input' in example:
        prompt = example['input']
    elif 'prompt' in example:
        prompt = example['prompt']
    else:
        raise ValueError(f"Example must have 'input' or 'prompt' field. Got: {example.keys()}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate (greedy decoding for speed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (faster)
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (skip input prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response


def generate_batch_outputs(
    model,
    tokenizer,
    examples: List[Dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[str]:
    """Generate responses for a batch of examples (parallel GPU processing).
    
    Uses batched generation for better GPU utilization.
    """
    
    # Extract prompts
    prompts = []
    for example in examples:
        if 'input' in example:
            prompts.append(example['input'])
        elif 'prompt' in example:
            prompts.append(example['prompt'])
        else:
            raise ValueError(f"Example must have 'input' or 'prompt' field. Got: {example.keys()}")
    
    # Tokenize batch with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,  # Pad to same length
        truncation=True, 
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate batch (greedy decoding for speed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (faster than sampling)
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode each output (skip input prompt for each)
    responses = []
    for i, output in enumerate(outputs):
        # Find where the input ends (skip padding and input tokens)
        input_length = inputs['input_ids'][i].ne(tokenizer.pad_token_id).sum()
        generated_tokens = output[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def generate_all_outputs(
    model,
    tokenizer,
    examples: List[Dict],
    output_file: str,
    batch_size: int = 32,  # H200 optimized: 32 examples in parallel
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate teacher outputs for all examples using batched processing.
    
    Optimized for H200 (141GB VRAM):
    - Batch size 32 for maximum GPU utilization
    - Parallel generation reduces 38 hours → ~2-3 hours
    
    Note: max_new_tokens=2048 allows hard questions to generate full responses.
          Model will stop at EOS token naturally.
    """
    
    console.print(f"\n[bold blue]Generating teacher outputs for {len(examples)} examples[/bold blue]")
    console.print(f"Output: {output_file}")
    console.print(f"Batch size: {batch_size} (H200 optimized)")
    console.print(f"Max new tokens: {max_new_tokens}")
    console.print(f"Generation mode: Greedy decoding (do_sample=False)")
    console.print(f"[yellow]Expected speedup: ~12-15× (batched + greedy)[/yellow]")
    
    console.print(f"[yellow]Expected speedup: ~12-15× (batched generation)[/yellow]")
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        "total": len(examples),
        "generated": 0,
        "failed": 0,
        "by_difficulty": {},
        "by_domain": {}
    }
    
    # Thread-safe file writing
    write_lock = Lock()
    
    # Split into batches
    num_batches = (len(examples) + batch_size - 1) // batch_size
    
    console.print(f"[cyan]Processing {num_batches} batches of size {batch_size}[/cyan]\n")
    
    # Generate with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=10  # Update 10 times per second for responsive updates
    ) as progress:
        
        task = progress.add_task(
            f"Generating responses...",
            total=len(examples)
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process in batches
            for batch_idx in range(0, len(examples), batch_size):
                batch_end = min(batch_idx + batch_size, len(examples))
                batch_examples = examples[batch_idx:batch_end]
                current_batch_num = batch_idx // batch_size + 1
                
                # Update progress to show batch is starting (BEFORE generation)
                progress.update(
                    task,
                    completed=batch_idx,  # Show progress up to current position
                    description=f"⏳ Generating batch {current_batch_num}/{num_batches} ({len(batch_examples)} examples)..."
                )
                
                try:
                    # Generate entire batch in parallel on GPU
                    batch_responses = generate_batch_outputs(
                        model, tokenizer, batch_examples,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    # Process and write results
                    for i, (example, teacher_response) in enumerate(zip(batch_examples, batch_responses)):
                        example_idx = batch_idx + i
                        
                        # Extract metadata
                        metadata = example.get('metadata', {})
                        difficulty = metadata.get('difficulty', 'unknown')
                        domain = metadata.get('domain', 'unknown')
                        
                        # Create output format aligned with single-task LM training
                        output = {
                            'input': example.get('input') or example.get('prompt'),
                            'output': teacher_response,  # Base model response (for draft training)
                            'difficulty': difficulty,
                            'domain': domain,
                            'metadata': {
                                'original_output': example.get('output') or example.get('response'),  # GPT-4o-mini
                                'example_id': example_idx,
                                'teacher_model': 'llama-3.1-8b-maml-merged',
                                'generation_params': {
                                    'temperature': 0.0,  # Greedy decoding
                                    'do_sample': False,
                                    'max_new_tokens': max_new_tokens
                                }
                            }
                        }
                        
                        # Write to file
                        f.write(json.dumps(output, ensure_ascii=False) + '\n')
                        
                        # Update stats
                        stats['generated'] += 1
                        stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
                        stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1
                    
                    # Flush after each batch
                    f.flush()
                    
                    # Update progress bar with completion (AFTER generation)
                    progress.update(
                        task,
                        completed=batch_end,
                        description=f"✅ Completed batch {current_batch_num}/{num_batches} - [{stats['generated']} done, {stats['failed']} failed]"
                    )
                    
                    # Periodic logging every 5 batches
                    if current_batch_num % 5 == 0:
                        throughput = stats['generated'] / ((current_batch_num * batch_size) / len(examples)) if batch_end > 0 else 0
                        logger.info(f"Batch {current_batch_num}/{num_batches}: {batch_end}/{len(examples)} ({batch_end/len(examples)*100:.1f}%) - Generated: {stats['generated']}, Failed: {stats['failed']}")
                
                except Exception as e:
                    logger.error(f"Failed to generate batch {current_batch_num} (examples {batch_idx}-{batch_end}): {e}")
                    stats['failed'] += len(batch_examples)
                    
                    # Still update progress bar
                    progress.update(
                        task,
                        completed=batch_end,
                        description=f"Batch {current_batch_num}/{num_batches} FAILED - [{stats['generated']} done, {stats['failed']} failed]"
                    )
    
    # Print final statistics
    console.print("\n[bold green]Generation Complete![/bold green]")
    console.print(f"\n[cyan]Statistics:[/cyan]")
    console.print(f"  Total examples: {stats['total']}")
    console.print(f"  Generated: {stats['generated']}")
    console.print(f"  Failed: {stats['failed']}")
    
    console.print(f"\n[cyan]By Difficulty:[/cyan]")
    for difficulty, count in sorted(stats['by_difficulty'].items()):
        console.print(f"  {difficulty}: {count}")
    
    console.print(f"\n[cyan]By Domain:[/cyan]")
    for domain, count in sorted(stats['by_domain'].items()):
        console.print(f"  {domain}: {count}")
    
    console.print(f"\n[green]✓[/green] Teacher outputs saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher outputs from merged 8B MAML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate teacher outputs for all 53,597 examples (H200 OPTIMIZED - RECOMMENDED)
  python scripts/phase1e_generate_teacher_outputs.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_53k.jsonl \\
      --batch_size 32 \\
      --max_new_tokens 2048
  
  # For H100 (80GB VRAM)
  python scripts/phase1e_generate_teacher_outputs.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_53k.jsonl \\
      --batch_size 16 \\
      --max_new_tokens 2048
  
  # Test on small sample first (100 examples)
  python scripts/phase1e_generate_teacher_outputs.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_test.jsonl \\
      --max_examples 100 \\
      --batch_size 32 \\
      --max_new_tokens 2048
  
  # Conservative batch size for A100 (40GB)
  python scripts/phase1e_generate_teacher_outputs.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_53k.jsonl \\
      --batch_size 8 \\
      --max_new_tokens 2048

Performance:
  H200 (141GB): batch_size=32 → ~2-3 hours for 53,597 examples (15× speedup)
  H100 (80GB):  batch_size=16 → ~3-4 hours for 53,597 examples (10× speedup)
  A100 (40GB):  batch_size=8  → ~5-6 hours for 53,597 examples (6× speedup)
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged 8B MAML model"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input training data file (JSONL)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for teacher outputs (JSONL)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel generation (default: 32 for H200, use 16 for H100, 8 for A100)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048, allows full responses with EOS)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)"
    )
    
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples to process (for testing, default: all)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load training data
    examples = load_training_data(args.input_file)
    
    # Limit examples if specified
    if args.max_examples:
        console.print(f"[yellow]Limiting to {args.max_examples} examples for testing[/yellow]")
        examples = examples[:args.max_examples]
    
    # Generate teacher outputs
    generate_all_outputs(
        model,
        tokenizer,
        examples,
        args.output_file,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    console.print("\n[bold green]✓ Teacher output generation complete![/bold green]")


if __name__ == "__main__":
    main()
