#!/usr/bin/env python3
"""
Phase 1C/1D: Generate Improved Examples Using Claude Sonnet 4.5 via GitHub Copilot

PURPOSE:
    Generate high-quality improved examples for 4,942 hard failures using Claude Sonnet 4.5
    through GitHub Copilot API. Shows the model's previous wrong answer and reference answer,
    then asks Claude to generate a significantly better response with Chain-of-Thought reasoning.

WHEN TO USE:
    - After Phase 1C self-critique evaluation complete
    - Input: Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl (4,942 examples)
    - When ready to generate teacher-quality examples for hard cases

COST ESTIMATE:
    - Claude Sonnet 4.5: ~$0.003 per example (input) + ~$0.015 per example (output)
    - Total: ~$18 per example × 4,942 = ~$89,000 (WAY TOO EXPENSIVE!)
    - WAIT - GitHub Copilot uses Claude but pricing is different
    - Need to check actual Copilot API costs or use alternative

ALTERNATIVE APPROACHES:
    1. Use Claude API directly (more transparent pricing)
    2. Use Anthropic API with batching
    3. Use local Llama-405B-Instruct (FREE via Together AI/Fireworks)
    4. Use GPT-4o-mini (cheaper: $0.15/1M input, $0.60/1M output)

OUTPUT:
    - data/phase1c/claude_improved_examples.jsonl
    - Format: {instruction, input, output, meta:{source, category, teacher, quality}}
    - Ready for bidirectional pair generation

USAGE:
    # Using Claude API directly (recommended)
    export ANTHROPIC_API_KEY="your-key"
    python Phase1A_2_0/scripts/generate_claude_examples.py \\
        --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \\
        --output "data/phase1c/claude_improved_examples.jsonl" \\
        --api_provider claude \\
        --model claude-sonnet-4.5 \\
        --batch_size 10 \\
        --max_examples 100  # Test first!
    
    # Using GPT-4o-mini (cheaper alternative)
    export OPENAI_API_KEY="your-key"
    python Phase1A_2_0/scripts/generate_claude_examples.py \\
        --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \\
        --output "data/phase1c/gpt4o_mini_improved_examples.jsonl" \\
        --api_provider openai \\
        --model gpt-4o-mini \\
        --batch_size 50 \\
        --max_examples 4942

PIPELINE STAGE: Phase 1C/1D - Claude Example Generation
"""

import json
import os
import argparse
from typing import Dict, List, Optional
from pathlib import Path
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
import anthropic
from openai import OpenAI

console = Console()


def create_improvement_prompt(
    instruction: str,
    reference_answer: str,
    previous_output: str,
    category: str
) -> str:
    """
    Create a prompt that asks the teacher model to generate an improved answer.
    Emphasizes Chain-of-Thought reasoning and learning from the previous mistake.
    """
    
    prompt = f"""You are an expert AI teacher helping to improve a student model's responses.

**Task Category:** {category}

**Original Instruction:**
{instruction}

**Student's Previous Wrong Answer:**
{previous_output}

**Reference Correct Answer:**
{reference_answer}

**Your Task:**
Generate a significantly BETTER response than both the student's wrong answer and the reference answer. Your response should:

1. **Start with Chain-of-Thought reasoning:** Explain the thought process step-by-step
2. **Identify the student's mistake:** What went wrong in the previous answer?
3. **Provide the correct approach:** How to properly answer this question?
4. **Give a complete, detailed answer:** Be thorough and educational
5. **Use clear structure:** Break down complex concepts into digestible parts

**Category-Specific Guidelines:**

- **Code:** Include well-commented code, explain edge cases, show test cases
- **Math:** Show step-by-step calculations, explain formulas, verify the result
- **Reasoning:** Use logical deduction, consider multiple perspectives, validate conclusions
- **QA:** Provide accurate facts, cite reasoning, be comprehensive
- **Creative:** Balance creativity with coherence, maintain context
- **Other:** Adapt to the specific task requirements

**Output Format:**
Provide ONLY your improved response. Make it significantly better than the reference by adding depth, clarity, and educational value.

**Your Improved Response:**"""
    
    return prompt


def generate_with_claude(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",  # Claude Sonnet 4.5
    max_tokens: int = 4096
) -> str:
    """Generate improved response using Claude API"""
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.7,  # Balanced between creativity and consistency
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        console.print(f"[red]Claude API error: {e}[/red]")
        return None


def generate_with_openai(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 4096
) -> str:
    """Generate improved response using OpenAI API"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert AI teacher helping to improve student model responses with clear, detailed explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        console.print(f"[red]OpenAI API error: {e}[/red]")
        return None


def estimate_cost(
    num_examples: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 1000,
    api_provider: str = "claude"
) -> Dict[str, float]:
    """Estimate API costs"""
    
    if api_provider == "claude":
        # Claude Sonnet 4.5 pricing (as of Nov 2025)
        input_cost_per_1m = 3.0  # $3 per 1M input tokens
        output_cost_per_1m = 15.0  # $15 per 1M output tokens
    elif api_provider == "openai":
        # GPT-4o-mini pricing
        input_cost_per_1m = 0.15  # $0.15 per 1M input tokens
        output_cost_per_1m = 0.60  # $0.60 per 1M output tokens
    else:
        return {"error": "Unknown API provider"}
    
    total_input_tokens = num_examples * avg_input_tokens
    total_output_tokens = num_examples * avg_output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return {
        "num_examples": num_examples,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "cost_per_example": total_cost / num_examples
    }


def main():
    parser = argparse.ArgumentParser(description="Generate improved examples using Claude/GPT-4o-mini")
    
    # Input/Output
    parser.add_argument('--input', type=str,
                       default='./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl',
                       help='Input hard failures file')
    parser.add_argument('--output', type=str,
                       default='data/phase1c/improved_examples.jsonl',
                       help='Output improved examples file')
    
    # API Configuration
    parser.add_argument('--api_provider', type=str, default='openai',
                       choices=['claude', 'openai'],
                       help='API provider (claude=expensive, openai=cheaper)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name (claude-sonnet-4-20250514, gpt-4o-mini, gpt-4o)')
    parser.add_argument('--max_tokens', type=int, default=4096,
                       help='Maximum tokens per response')
    
    # Processing
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Process N examples before saving checkpoint')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Limit number of examples (for testing)')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start from Nth example (for resuming)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls (seconds)')
    
    # Cost estimation
    parser.add_argument('--estimate_only', action='store_true',
                       help='Only estimate costs, do not generate')
    
    args = parser.parse_args()
    
    # Load input data
    console.print(f"\n📂 Loading input from [cyan]{args.input}[/cyan]...")
    
    with open(args.input, 'r') as f:
        failures = [json.loads(line) for line in f]
    
    total_examples = len(failures)
    console.print(f"✅ Loaded {total_examples:,} hard failures")
    
    # Apply max_examples limit
    if args.max_examples:
        failures = failures[args.start_index:args.start_index + args.max_examples]
        console.print(f"📊 Processing {len(failures):,} examples (start={args.start_index}, max={args.max_examples})")
    else:
        failures = failures[args.start_index:]
        console.print(f"📊 Processing {len(failures):,} examples (start={args.start_index})")
    
    # Cost estimation
    console.print(f"\n💰 Cost Estimation ({args.api_provider} - {args.model}):")
    cost_info = estimate_cost(len(failures), api_provider=args.api_provider)
    console.print(f"   Examples: {cost_info['num_examples']:,}")
    console.print(f"   Input tokens: ~{cost_info['total_input_tokens']:,}")
    console.print(f"   Output tokens: ~{cost_info['total_output_tokens']:,}")
    console.print(f"   Input cost: ${cost_info['input_cost']:.2f}")
    console.print(f"   Output cost: ${cost_info['output_cost']:.2f}")
    console.print(f"   [bold]Total cost: ${cost_info['total_cost']:.2f}[/bold]")
    console.print(f"   Per example: ${cost_info['cost_per_example']:.4f}")
    
    if args.estimate_only:
        console.print("\n✅ Estimation complete (--estimate_only flag set)")
        return
    
    # Confirm before proceeding
    if cost_info['total_cost'] > 50:
        console.print(f"\n[yellow]⚠️  Warning: Estimated cost ${cost_info['total_cost']:.2f} is high![/yellow]")
        confirm = input("Type 'yes' to proceed: ")
        if confirm.lower() != 'yes':
            console.print("❌ Aborted by user")
            return
    
    # Initialize API client
    console.print(f"\n🔌 Initializing {args.api_provider} API client...")
    
    if args.api_provider == 'claude':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            console.print("[red]❌ ANTHROPIC_API_KEY not set![/red]")
            return
        client = anthropic.Anthropic(api_key=api_key)
        generate_fn = lambda prompt: generate_with_claude(client, prompt, args.model, args.max_tokens)
        
    elif args.api_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            console.print("[red]❌ OPENAI_API_KEY not set![/red]")
            return
        client = OpenAI(api_key=api_key)
        generate_fn = lambda prompt: generate_with_openai(client, prompt, args.model, args.max_tokens)
    
    console.print(f"✅ API client initialized")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for existing output (resume capability)
    existing_ids = set()
    if output_path.exists():
        console.print(f"\n📄 Found existing output file, will append...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    existing_ids.add(item['meta']['id'])
                except:
                    pass
        console.print(f"   Already processed: {len(existing_ids)} examples")
    
    # Process examples
    console.print(f"\n🚀 Starting generation...")
    console.print(f"   Model: {args.model}")
    console.print(f"   Batch size: {args.batch_size}")
    console.print(f"   Delay: {args.delay}s between calls")
    console.print("="*80)
    
    generated_count = 0
    skipped_count = 0
    error_count = 0
    total_cost_actual = 0.0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(
            f"Generating with {args.model}...",
            total=len(failures)
        )
        
        with open(output_path, 'a') as out_f:
            for i, failure in enumerate(failures):
                # Skip if already processed
                if failure['id'] in existing_ids:
                    skipped_count += 1
                    progress.advance(task)
                    continue
                
                # Create prompt
                prompt = create_improvement_prompt(
                    instruction=failure['instruction'],
                    reference_answer=failure['reference_answer'],
                    previous_output=failure['previous_output'],
                    category=failure['category']
                )
                
                # Generate improved response
                improved_output = generate_fn(prompt)
                
                if improved_output is None:
                    error_count += 1
                    progress.advance(task)
                    continue
                
                # Create training example
                example = {
                    "instruction": failure['instruction'],
                    "input": "",
                    "output": improved_output,
                    "meta": {
                        "source": "phase1c_claude_generation",
                        "category": failure['category'],
                        "id": failure['id'],
                        "teacher": args.model,
                        "previous_output": failure['previous_output'],
                        "reference_answer": failure['reference_answer'],
                        "quality": "teacher_generated"
                    }
                }
                
                # Save immediately (checkpoint)
                out_f.write(json.dumps(example) + '\n')
                out_f.flush()
                
                generated_count += 1
                
                # Update progress with stats
                progress.update(
                    task,
                    description=f"Generated: {generated_count}, Skipped: {skipped_count}, Errors: {error_count}"
                )
                progress.advance(task)
                
                # Rate limiting
                time.sleep(args.delay)
    
    # Final summary
    console.print("\n" + "="*80)
    console.print("📊 GENERATION SUMMARY")
    console.print("="*80)
    console.print(f"Total processed: {len(failures):,}")
    console.print(f"Generated: {generated_count:,}")
    console.print(f"Skipped: {skipped_count:,}")
    console.print(f"Errors: {error_count:,}")
    console.print(f"Output: {output_path}")
    console.print(f"Estimated cost: ${cost_info['total_cost']:.2f}")
    console.print("="*80)
    
    console.print(f"\n✅ Generation complete!")
    console.print(f"📂 Output saved to: [cyan]{output_path}[/cyan]")
    
    # Next steps
    console.print(f"\n🚀 Next Steps:")
    console.print(f"1. Verify output quality: head -n 2 {output_path} | python3 -m json.tool")
    console.print(f"2. Create bidirectional pairs script")
    console.print(f"3. Combine with self-critique examples")
    console.print(f"4. Run smart training with early stopping")


if __name__ == "__main__":
    main()
