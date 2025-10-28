#!/usr/bin/env python3
"""
Optimized Phase 1B.1 Validation - Reuses Phase 1A GPT-4 Responses

PURPOSE:
    Validates Phase 1B.1 improvements by comparing with Phase 1A baseline.
    OPTIMIZATION: Reuses GPT-4 responses from Phase 1A benchmarks instead of
    regenerating them, saving ~50% API cost.

WHEN TO USE:
    - After Phase 1B.1 training completes
    - Requires: Phase 1A benchmark results with GPT-4 responses

WHAT IT DOES:
    1. Loads Phase 1A benchmark results (prompts + GPT-4 responses)
    2. Generates Phase 1B.1 model responses for SAME prompts
    3. Uses GPT-4 to judge: Phase 1B.1 vs Phase 1A's GPT-4 (no regeneration!)
    4. Compares results: Phase 1B.1 wins vs Phase 1A wins

COST SAVINGS:
    - Original approach: 100 prompts × (1 GPT-4 gen + 1 judge) = 200 API calls
    - Optimized approach: 100 prompts × (1 judge only) = 100 API calls
    - Savings: 50% cost reduction (~$0.75 instead of $1.50)

TIME SAVINGS:
    - Original: 30-40 minutes
    - Optimized: 15-20 minutes (no GPT-4 generation)

INPUT:
    - checkpoints/benchmark_results_full/*_intermediate.json (Phase 1A results)
    - checkpoints/phase1b_from_benchmark/ (Phase 1B.1 model)

OUTPUT:
    - checkpoints/benchmark_results_phase1b1/*_intermediate.json
    - validation_summary.txt with comparison

PIPELINE STAGE: Phase 1B.4 - Validation
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI
from tqdm import tqdm
import time
import argparse
import sys


class OptimizedPhase1BValidator:
    """Validates Phase 1B.1 by reusing Phase 1A GPT-4 responses."""
    
    def __init__(
        self,
        phase1a_results_dir: str,
        phase1b_model_path: str,
        openai_key: str,
        output_dir: str = "checkpoints/benchmark_results_phase1b1"
    ):
        self.phase1a_results_dir = Path(phase1a_results_dir)
        self.phase1b_model_path = phase1b_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_key)
        
        # Load Phase 1B.1 model
        print("📥 Loading Phase 1B.1 model...")
        
        # Check if this is a LoRA adapter or full model
        adapter_config = Path(phase1b_model_path) / "adapter_config.json"
        
        if adapter_config.exists():
            # This is a LoRA adapter - need to load base model + adapter
            print("   Detected LoRA adapter, loading base model + adapter...")
            with open(adapter_config) as f:
                config = json.load(f)
                base_model_path = config.get("base_model_name_or_path", "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
            
            print(f"   Loading base model: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            print(f"   Loading adapter: {phase1b_model_path}")
            self.model = PeftModel.from_pretrained(base_model, phase1b_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(phase1b_model_path)
        else:
            # This is a full merged model
            print("   Loading full model...")
            self.tokenizer = AutoTokenizer.from_pretrained(phase1b_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                phase1b_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        self.model.eval()
        
        # Check if model is on GPU
        print(f"✅ Phase 1B.1 model loaded on: {next(self.model.parameters()).device}")
        print(f"✅ Model dtype: {next(self.model.parameters()).dtype}")
    
    def generate_phase1b_response(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response from Phase 1B.1 model (optimized for speed)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # Reduced from 512 to 256 for speed
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,  # No beam search for speed
                use_cache=True  # Enable KV cache for faster generation
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after prompt)
        response = response[len(prompt):].strip()
        return response
    
    def judge_responses(self, prompt: str, phase1b_response: str, gpt4_response: str) -> Dict:
        """Use GPT-4 to judge which response is better."""
        judge_prompt = f"""You are an expert judge evaluating two AI responses.

Original Question/Prompt:
{prompt}

Response A (New Model):
{phase1b_response}

Response B (GPT-4 Baseline):
{gpt4_response}

Rate each response on these criteria (1-10 scale):
1. Correctness: Is the answer factually accurate?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-explained and easy to understand?
4. Relevance: Does it stay on topic?

Provide your rating in this JSON format:
{{
    "response_a": {{"correctness": X, "completeness": X, "clarity": X, "relevance": X}},
    "response_b": {{"correctness": X, "completeness": X, "clarity": X, "relevance": X}},
    "winner": "A" or "B" or "TIE",
    "reasoning": "Brief explanation"
}}
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            judgment_text = response.choices[0].message.content
            if not judgment_text:
                return None
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', judgment_text, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
                return judgment
            else:
                return None
        except Exception as e:
            print(f"⚠️ Judgment error: {e}")
            return None
    
    def validate_category(self, category: str) -> Dict:
        """Validate one category by reusing Phase 1A GPT-4 responses."""
        print(f"\n{'='*60}")
        print(f"Validating {category.upper()}")
        print(f"{'='*60}")
        
        # Load Phase 1A results (has prompts + GPT-4 responses)
        phase1a_file = self.phase1a_results_dir / f"{category}_intermediate.json"
        if not phase1a_file.exists():
            print(f"⚠️ Warning: Phase 1A results not found: {phase1a_file}")
            return None
        
        with open(phase1a_file, 'r') as f:
            phase1a_data = json.load(f)
        
        phase1a_results = phase1a_data['results']
        print(f"📋 Loaded {len(phase1a_results)} problems from Phase 1A benchmark")
        print(f"✅ Reusing GPT-4 responses (saves 50% API cost!)")
        
        # Validate Phase 1B.1 on same problems
        phase1b_results = []
        wins = losses = ties = 0
        
        for i, phase1a_result in enumerate(tqdm(phase1a_results, desc=f"Validating {category}")):
            prompt = phase1a_result['prompt']
            gpt4_response = phase1a_result['gpt4_response']  # REUSE from Phase 1A!
            
            # Generate Phase 1B.1 response
            phase1b_response = self.generate_phase1b_response(prompt)
            time.sleep(0.5)  # Rate limiting
            
            # Judge: Phase 1B.1 vs GPT-4 (from Phase 1A)
            judgment = self.judge_responses(prompt, phase1b_response, gpt4_response)
            time.sleep(1.0)  # Rate limiting for API
            
            if judgment:
                if judgment['winner'] == 'A':
                    wins += 1
                elif judgment['winner'] == 'B':
                    losses += 1
                else:
                    ties += 1
                
                phase1b_results.append({
                    "prompt": prompt,
                    "phase1b_response": phase1b_response,
                    "gpt4_response": gpt4_response,  # Same as Phase 1A
                    "judgment": judgment,
                    "ground_truth": phase1a_result.get('ground_truth')
                })
            
            # Save intermediate results every 10 examples
            if (i + 1) % 10 == 0:
                self._save_intermediate(category, phase1b_results, wins, losses, ties)
        
        # Save final results
        self._save_intermediate(category, phase1b_results, wins, losses, ties)
        
        print(f"\n✅ {category.upper()} Results:")
        print(f"   Wins: {wins} ({wins/len(phase1b_results)*100:.1f}%)")
        print(f"   Losses: {losses} ({losses/len(phase1b_results)*100:.1f}%)")
        print(f"   Ties: {ties} ({ties/len(phase1b_results)*100:.1f}%)")
        
        return {
            "category": category,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total": len(phase1b_results)
        }
    
    def _save_intermediate(self, category: str, results: List, wins: int, losses: int, ties: int):
        """Save intermediate results."""
        output_file = self.output_dir / f"{category}_intermediate.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "results": results,
                "wins": wins,
                "losses": losses,
                "ties": ties
            }, f, indent=2)
    
    def compare_with_phase1a(self, category: str, phase1b_stats: Dict):
        """Compare Phase 1B.1 results with Phase 1A baseline."""
        phase1a_file = self.phase1a_results_dir / f"{category}_intermediate.json"
        if not phase1a_file.exists():
            return
        
        with open(phase1a_file, 'r') as f:
            phase1a_data = json.load(f)
        
        phase1a_wins = phase1a_data['wins']
        phase1a_losses = phase1a_data['losses']
        phase1a_ties = phase1a_data['ties']
        phase1a_total = phase1a_wins + phase1a_losses + phase1a_ties
        
        phase1b_wins = phase1b_stats['wins']
        phase1b_total = phase1b_stats['total']
        
        phase1a_win_pct = (phase1a_wins / phase1a_total * 100) if phase1a_total > 0 else 0
        phase1a_tie_pct = (phase1a_ties / phase1a_total * 100) if phase1a_total > 0 else 0
        
        phase1b_win_pct = (phase1b_wins / phase1b_total * 100) if phase1b_total > 0 else 0
        phase1b_ties = phase1b_stats['ties']
        phase1b_tie_pct = (phase1b_ties / phase1b_total * 100) if phase1b_total > 0 else 0
        
        improvement = phase1b_win_pct - phase1a_win_pct
        
        print(f"\n{category.upper()} Comparison:")
        print(f"  Phase 1A: {phase1a_wins}/{phase1a_total} wins ({phase1a_win_pct:.1f}%), {phase1a_ties} ties ({phase1a_tie_pct:.1f}%)")
        print(f"  Phase 1B.1: {phase1b_wins}/{phase1b_total} wins ({phase1b_win_pct:.1f}%), {phase1b_ties} ties ({phase1b_tie_pct:.1f}%)")
        print(f"  Improvement: {improvement:+.1f} percentage points")


def main():
    parser = argparse.ArgumentParser(description="Optimized Phase 1B.1 Validation")
    parser.add_argument("--phase1a_results", default="checkpoints/benchmark_results_full",
                      help="Phase 1A benchmark results directory")
    parser.add_argument("--phase1b_model", default="checkpoints/phase1b_from_benchmark",
                      help="Phase 1B.1 model path")
    parser.add_argument("--openai_key", required=True, help="OpenAI API key")
    parser.add_argument("--output_dir", default="checkpoints/benchmark_results_phase1b1",
                      help="Output directory for Phase 1B.1 results")
    parser.add_argument("--categories", nargs="+", default=["math", "code"],
                      help="Categories to validate")
    
    args = parser.parse_args()
    
    # Validate Phase 1A results exist
    phase1a_dir = Path(args.phase1a_results)
    if not phase1a_dir.exists():
        print(f"❌ Error: Phase 1A results not found: {phase1a_dir}")
        print("Run Phase 1A benchmarks first with automated_gpt4_benchmark.py")
        sys.exit(1)
    
    # Validate Phase 1B.1 model exists
    phase1b_path = Path(args.phase1b_model)
    if not phase1b_path.exists():
        print(f"❌ Error: Phase 1B.1 model not found: {phase1b_path}")
        print("Run Phase 1B.1 training first")
        sys.exit(1)
    
    print("🔍 Phase 1B.1 Validation - OPTIMIZED (Reuses Phase 1A GPT-4 responses)")
    print("="*70)
    print(f"💰 Cost savings: ~50% (only judging, no GPT-4 generation)")
    print(f"⏱️  Time savings: ~15 minutes (was 30-40 min)")
    print("")
    
    # Create validator
    validator = OptimizedPhase1BValidator(
        phase1a_results_dir=args.phase1a_results,
        phase1b_model_path=args.phase1b_model,
        openai_key=args.openai_key,
        output_dir=args.output_dir
    )
    
    # Validate each category
    all_stats = {}
    for category in args.categories:
        stats = validator.validate_category(category)
        if stats:
            all_stats[category] = stats
            validator.compare_with_phase1a(category, stats)
    
    # Save summary
    summary_file = Path(args.output_dir) / "validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Phase 1B.1 Validation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Optimization: Reused Phase 1A GPT-4 responses\n")
        f.write("Cost savings: ~50% (only judging, no generation)\n\n")
        for category, stats in all_stats.items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Wins: {stats['wins']}/{stats['total']} ({stats['wins']/stats['total']*100:.1f}%)\n")
            f.write(f"  Losses: {stats['losses']}/{stats['total']} ({stats['losses']/stats['total']*100:.1f}%)\n")
            f.write(f"  Ties: {stats['ties']}/{stats['total']} ({stats['ties']/stats['total']*100:.1f}%)\n\n")
    
    print(f"\n📄 Summary saved to: {summary_file}")
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
