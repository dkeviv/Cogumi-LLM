#!/usr/bin/env python3
"""
Phase 1E: Generate Teacher Outputs using vLLM (HIGH PERFORMANCE)

Uses vLLM for 5-10× faster inference compared to standard transformers.
Optimized for H200 with high-throughput batch processing.

Expected Runtime: ~30-60 minutes for 53,597 examples (vs 3-4 hours with transformers)

Install vLLM first:
    pip install vllm

Usage:
    python scripts/phase1e_generate_teacher_outputs_vllm.py \
        --model_path models/phase1_maml_lora_v2/merged \
        --input_file data/phase1/answers/training_data_clean.jsonl \
        --output_file data/phase1e/teacher_outputs_53k.jsonl \
        --max_tokens 1024

Author: Cogumi-LLM
Date: November 18, 2025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import time

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)
from vllm import LLM, SamplingParams

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(input_file: str, max_examples: int = None) -> List[Dict]:
    """Load training data."""
    console.print(f"\n[bold blue]Loading training data from: {input_file}[/bold blue]")
    
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
                if max_examples and len(examples) >= max_examples:
                    break
    
    console.print(f"[green]✓[/green] Loaded {len(examples)} training examples")
    return examples


def generate_with_vllm(
    model_path: str,
    examples: List[Dict],
    output_file: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1000  # vLLM can handle very large batches
):
    """Generate teacher outputs using vLLM for high throughput."""
    
    console.print(f"\n[bold blue]Initializing vLLM engine...[/bold blue]")
    console.print(f"Model: {model_path}")
    console.print(f"Max tokens: 512 (capped to prevent verbose output)")
    console.print(f"Generation mode: Low temperature sampling (temp=0.3, repetition_penalty=1.2)")
    console.print(f"[yellow]Multiple stopping mechanisms enabled to prevent repetitive output[/yellow]")
    console.print(f"[yellow]vLLM will automatically optimize batch processing[/yellow]")
    
    # Initialize vLLM engine
    # vLLM automatically uses all available GPU memory efficiently
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # Single GPU
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,  # Max sequence length
        gpu_memory_utilization=0.9  # Use 90% of GPU memory
    )
    
    console.print(f"[green]✓[/green] vLLM engine initialized\n")
    
    # Sampling parameters with multiple stopping mechanisms
    # FIX: Prevent verbose/repetitive generation (84% were hitting max_tokens)
    sampling_params = SamplingParams(
        temperature=0.3,  # Low temperature (not fully greedy) to allow EOS generation
        max_tokens=512,   # Reduced from 1024 to encourage concise answers
        repetition_penalty=1.2,  # Penalize repetitive content
        stop_token_ids=[llm.get_tokenizer().eos_token_id],  # Stop on EOS token
        stop=["<|eot_id|>", "\n\n\n", "Question:", "Answer:", "If a car", "If "]  # Additional stop strings
    )
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        "total": len(examples),
        "generated": 0,
        "failed": 0,
        "by_difficulty": {},
        "by_domain": {},
        "total_tokens": 0,
        "total_time": 0
    }
    
    # Extract prompts
    console.print(f"[cyan]Extracting prompts...[/cyan]")
    prompts = []
    for example in examples:
        prompt = example.get('input') or example.get('prompt')
        if not prompt:
            raise ValueError(f"Example must have 'input' or 'prompt' field")
        prompts.append(prompt)
    
    console.print(f"[green]✓[/green] Extracted {len(prompts)} prompts\n")
    
    # Process in batches (vLLM handles internal batching, but we batch for memory)
    num_batches = (len(examples) + batch_size - 1) // batch_size
    console.print(f"[cyan]Processing {num_batches} batches of up to {batch_size} examples[/cyan]\n")
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=10
    ) as progress:
        
        task = progress.add_task(
            f"Generating responses...",
            total=len(examples)
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for batch_idx in range(0, len(examples), batch_size):
                batch_end = min(batch_idx + batch_size, len(examples))
                batch_examples = examples[batch_idx:batch_end]
                batch_prompts = prompts[batch_idx:batch_end]
                current_batch_num = batch_idx // batch_size + 1
                
                progress.update(
                    task,
                    completed=batch_idx,
                    description=f"⏳ Generating batch {current_batch_num}/{num_batches} ({len(batch_examples)} examples)..."
                )
                
                try:
                    batch_start = time.time()
                    
                    # Generate with vLLM (handles internal optimization)
                    outputs = llm.generate(batch_prompts, sampling_params)
                    
                    batch_time = time.time() - batch_start
                    stats['total_time'] += batch_time
                    
                    # Process outputs
                    for i, (example, output) in enumerate(zip(batch_examples, outputs)):
                        example_idx = batch_idx + i
                        
                        # Extract generated text
                        teacher_response = output.outputs[0].text
                        
                        # Extract metadata
                        metadata = example.get('metadata', {})
                        difficulty = metadata.get('difficulty', 'unknown')
                        domain = metadata.get('domain', 'unknown')
                        
                        # Count tokens
                        num_tokens = len(output.outputs[0].token_ids)
                        stats['total_tokens'] += num_tokens
                        
                        # Create output format
                        output_data = {
                            'input': example.get('input') or example.get('prompt'),
                            'output': teacher_response,
                            'difficulty': difficulty,
                            'domain': domain,
                            'metadata': {
                                'original_output': example.get('output') or example.get('response'),
                                'example_id': example_idx,
                                'teacher_model': 'llama-3.1-8b-maml-merged',
                                'generation_params': {
                                    'temperature': 0.3,  # Low temperature sampling
                                    'do_sample': True,
                                    'repetition_penalty': 1.2,
                                    'max_tokens': 512  # Capped at 512
                                },
                                'num_tokens_generated': num_tokens,
                                'generation_time': batch_time / len(batch_examples)
                            }
                        }
                        
                        # Write to file
                        f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        
                        # Update stats
                        stats['generated'] += 1
                        stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
                        stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1
                    
                    # Flush after each batch
                    f.flush()
                    
                    # Calculate throughput
                    avg_time_per_example = batch_time / len(batch_examples)
                    throughput = len(batch_examples) / batch_time
                    
                    progress.update(
                        task,
                        completed=batch_end,
                        description=f"✅ Batch {current_batch_num}/{num_batches} - [{stats['generated']} done, {throughput:.1f} ex/s]"
                    )
                    
                    # Periodic logging
                    if current_batch_num % 5 == 0 or current_batch_num == num_batches:
                        logger.info(
                            f"Batch {current_batch_num}/{num_batches}: "
                            f"{batch_end}/{len(examples)} ({batch_end/len(examples)*100:.1f}%) - "
                            f"Throughput: {throughput:.1f} ex/s, "
                            f"Avg time: {avg_time_per_example:.2f}s/ex"
                        )
                
                except Exception as e:
                    logger.error(f"Failed batch {current_batch_num}: {e}")
                    stats['failed'] += len(batch_examples)
                    
                    progress.update(
                        task,
                        completed=batch_end,
                        description=f"❌ Batch {current_batch_num}/{num_batches} FAILED"
                    )
    
    total_time = time.time() - start_time
    
    # Print final statistics
    console.print("\n[bold green]Generation Complete![/bold green]")
    console.print(f"\n[cyan]Statistics:[/cyan]")
    console.print(f"  Total examples: {stats['total']}")
    console.print(f"  Generated: {stats['generated']}")
    console.print(f"  Failed: {stats['failed']}")
    console.print(f"  Total time: {total_time/60:.1f} minutes")
    console.print(f"  Average throughput: {stats['generated']/total_time:.1f} examples/second")
    console.print(f"  Average time per example: {total_time/stats['generated']:.2f} seconds")
    console.print(f"  Total tokens generated: {stats['total_tokens']:,}")
    console.print(f"  Average tokens per example: {stats['total_tokens']/stats['generated']:.0f}")
    
    console.print(f"\n[cyan]By Difficulty:[/cyan]")
    for difficulty, count in sorted(stats['by_difficulty'].items()):
        console.print(f"  {difficulty}: {count}")
    
    console.print(f"\n[cyan]By Domain:[/cyan]")
    for domain, count in sorted(stats['by_domain'].items()):
        console.print(f"  {domain}: {count}")
    
    console.print(f"\n[green]✓[/green] Teacher outputs saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher outputs using vLLM (HIGH PERFORMANCE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full generation with vLLM (H200 - FASTEST)
  python scripts/phase1e_generate_teacher_outputs_vllm.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_53k.jsonl \\
      --max_tokens 1024
  
  # Test with small sample first
  python scripts/phase1e_generate_teacher_outputs_vllm.py \\
      --model_path models/phase1_maml_lora_v2/merged \\
      --input_file data/phase1/answers/training_data_clean.jsonl \\
      --output_file data/phase1e/teacher_outputs_test.jsonl \\
      --max_examples 1000 \\
      --max_tokens 1024

Performance (vLLM vs transformers):
  vLLM on H200:      ~30-60 minutes (50-100 examples/sec)
  transformers:      ~3-4 hours (4-5 examples/sec)
  Speedup:           5-10× faster with vLLM!
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
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
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
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000, vLLM optimizes internally)"
    )
    
    args = parser.parse_args()
    
    # Load training data
    examples = load_training_data(args.input_file, args.max_examples)
    
    # Generate with vLLM
    generate_with_vllm(
        model_path=args.model_path,
        examples=examples,
        output_file=args.output_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size
    )
    
    console.print("\n[bold green]✨ All done! Ready for Phase 1F (draft model training)[/bold green]")


if __name__ == "__main__":
    main()
