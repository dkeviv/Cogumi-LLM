#!/usr/bin/env python3
"""
Automated Benchmarking Against GPT-4 Baseline
Compares trained model performance against ChatGPT (GPT-4) on diverse tasks.

Usage:
    python automated_gpt4_benchmark.py \
        --model_path /data/Cogumi-LLM/checkpoints/final \
        --output_dir ./benchmark_results \
        --openai_key YOUR_KEY \
        --num_samples 500
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from collections import defaultdict
from peft import PeftModel, PeftConfig


class BenchmarkSuite:
    """Comprehensive benchmark suite comparing model to GPT-4."""
    
    def __init__(
        self,
        model_path: str,
        openai_key: str,
        output_dir: str = "./benchmark_results",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client (new SDK v1.0+)
        self.openai_client = OpenAI(api_key=openai_key)
        
        # Load local model
        print("📥 Loading local model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # CRITICAL: Set pad token for Llama-3.1 (it doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        # Check if this is a LoRA adapter or full model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        
        if adapter_config_path.exists():
            print("🔧 Detected LoRA adapter - loading base model + adapter...")
            # Load the adapter config to get base model name
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            print(f"📥 Loading base model: {base_model_name}")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map=device,
                load_in_4bit=True
            )
            
            print(f"🔧 Applying LoRA adapter from: {model_path}")
            # Load and apply adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # CRITICAL: Merge adapter for proper inference
            print("🔧 Merging LoRA adapter with base model...")
            self.model = self.model.merge_and_unload()
            print("✅ Base model + LoRA adapter merged successfully!")
        else:
            print("📥 Loading full model...")
            # Load as regular model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                load_in_4bit=True
            )
            print("✅ Full model loaded")
        
        # CRITICAL: Set to evaluation mode
        self.model.eval()
        print("✅ Model set to evaluation mode")
        
        # Benchmark categories
        self.categories = {
            "math": "Mathematical reasoning and calculations",
            "code": "Code generation and debugging",
            "reasoning": "Logical reasoning and problem solving",
            "knowledge": "Factual knowledge and comprehension",
            "instruction": "Instruction following",
            "creativity": "Creative writing and generation"
        }
        
        self.results = defaultdict(lambda: {"local": [], "gpt4": [], "wins": 0, "losses": 0, "ties": 0})
    
    def generate_local(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from local model with proper Llama-3.1-Instruct formatting."""
        # Format prompt for Llama-3.1-Instruct chat format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,  # No temperature scaling for deterministic output
                do_sample=False,  # Greedy decoding - always pick highest probability token
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # First decode WITH special tokens to find the assistant response
        response_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response using the special token marker
        if "<|start_header_id|>assistant<|end_header_id|>" in response_with_tokens:
            # Split on the assistant header and take everything after it
            assistant_part = response_with_tokens.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            # Remove the end-of-text token if present
            if "<|eot_id|>" in assistant_part:
                assistant_part = assistant_part.split("<|eot_id|>")[0]
            response = assistant_part.strip()
        else:
            # Fallback: decode without special tokens and remove the formatted prompt
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        
        return response
    
    def generate_gpt4(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from GPT-4."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            content = response.choices[0].message.content
            return content.strip() if content else "[ERROR: Empty response]"
        except Exception as e:
            print(f"⚠️ GPT-4 API error: {e}")
            return "[ERROR]"
    
    def judge_response(self, prompt: str, local_response: str, gpt4_response: str) -> Dict:
        """Use GPT-4 to judge which response is better."""
        judge_prompt = f"""You are an expert judge evaluating two AI responses.

Original Question/Prompt:
{prompt}

Response A:
{local_response}

Response B:
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
                temperature=0.1  # Low temperature for consistent judging
            )
            
            judgment_text = response.choices[0].message.content
            if judgment_text:
                judgment_text = judgment_text.strip()
            else:
                print("⚠️ Empty judgment response")
                return None
                
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', judgment_text, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
                return judgment
            else:
                print("⚠️ Could not parse judgment")
                return None
        except Exception as e:
            print(f"⚠️ Judgment error: {e}")
            return None
    
    def load_test_dataset(self, category: str, num_samples: int = 100) -> List[Dict]:
        """Load test examples for a category."""
        test_examples = []
        
        if category == "math":
            # GSM8K math problems
            dataset = load_dataset("gsm8k", "main", split="test")
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                test_examples.append({
                    "prompt": f"Solve this math problem step by step:\n\n{example['question']}",
                    "ground_truth": example['answer']
                })
        
        elif category == "code":
            # HumanEval coding tasks
            dataset = load_dataset("openai_humaneval", split="test")
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                test_examples.append({
                    "prompt": f"Complete this Python function:\n\n{example['prompt']}",
                    "ground_truth": example['canonical_solution']
                })
        
        elif category == "reasoning":
            # ARC-Challenge reasoning
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                choices = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(example['choices']['text'])])
                test_examples.append({
                    "prompt": f"Question: {example['question']}\n\nChoices:\n{choices}\n\nAnswer:",
                    "ground_truth": example['answerKey']
                })
        
        elif category == "knowledge":
            # MMLU general knowledge
            dataset = load_dataset("cais/mmlu", "all", split="test")
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                choices = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(example['choices'])])
                test_examples.append({
                    "prompt": f"Question: {example['question']}\n\nChoices:\n{choices}\n\nAnswer:",
                    "ground_truth": chr(65 + example['answer'])
                })
        
        elif category == "instruction":
            # Alpaca instruction following
            dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                prompt = example['instruction']
                if example.get('input'):
                    prompt += f"\n\nInput: {example['input']}"
                test_examples.append({
                    "prompt": prompt,
                    "ground_truth": example['output']
                })
        
        elif category == "creativity":
            # Creative writing prompts
            creative_prompts = [
                "Write a short story about a robot learning to paint.",
                "Compose a haiku about artificial intelligence.",
                "Create a dialogue between Socrates and a modern AI.",
                "Write a product description for an invisible umbrella.",
                "Create a recipe for happiness (metaphorically)."
            ]
            test_examples = [{"prompt": p, "ground_truth": None} for p in creative_prompts[:num_samples]]
        
        return test_examples
    
    def run_category_benchmark(self, category: str, num_samples: int = 100):
        """Run benchmark for a specific category."""
        print(f"\n{'='*60}")
        print(f"📊 Benchmarking: {category.upper()}")
        print(f"{'='*60}")
        
        # Load test examples
        test_examples = self.load_test_dataset(category, num_samples)
        print(f"Loaded {len(test_examples)} test examples")
        
        results = []
        wins = losses = ties = 0
        
        for i, example in enumerate(tqdm(test_examples, desc=f"Testing {category}")):
            prompt = example['prompt']
            
            # Generate responses
            local_response = self.generate_local(prompt)
            time.sleep(0.5)  # Rate limiting
            
            gpt4_response = self.generate_gpt4(prompt)
            time.sleep(1.0)  # Rate limiting for API
            
            # Judge responses
            judgment = self.judge_response(prompt, local_response, gpt4_response)
            
            if judgment:
                if judgment['winner'] == 'A':
                    wins += 1
                elif judgment['winner'] == 'B':
                    losses += 1
                else:
                    ties += 1
                
                results.append({
                    "prompt": prompt,
                    "local_response": local_response,
                    "gpt4_response": gpt4_response,
                    "judgment": judgment,
                    "ground_truth": example.get('ground_truth')
                })
            
            # Save intermediate results every 10 examples
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(category, results, wins, losses, ties)
        
        # Store final results
        self.results[category] = {
            "details": results,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / len(results) if results else 0,
            "score": (wins + 0.5 * ties) / len(results) if results else 0
        }
        
        print(f"\n✅ {category.upper()} Results:")
        print(f"   Wins: {wins} ({wins/len(results)*100:.1f}%)")
        print(f"   Losses: {losses} ({losses/len(results)*100:.1f}%)")
        print(f"   Ties: {ties} ({ties/len(results)*100:.1f}%)")
        print(f"   Score vs GPT-4: {self.results[category]['score']*100:.1f}%")
    
    def _save_intermediate_results(self, category: str, results: List, wins: int, losses: int, ties: int):
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
    
    def run_full_benchmark(self, categories: List[str] = None, samples_per_category: int = 100):
        """Run full benchmark suite."""
        if categories is None:
            categories = list(self.categories.keys())
        
        print(f"\n🚀 Starting Full Benchmark Suite")
        print(f"Categories: {', '.join(categories)}")
        print(f"Samples per category: {samples_per_category}")
        print(f"Model: {self.model_path}")
        
        start_time = time.time()
        
        for category in categories:
            self.run_category_benchmark(category, samples_per_category)
        
        elapsed_time = time.time() - start_time
        
        # Generate final report
        self._generate_report(elapsed_time)
    
    def _generate_report(self, elapsed_time: float):
        """Generate comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        
        # Calculate overall statistics
        total_wins = sum(r['wins'] for r in self.results.values())
        total_losses = sum(r['losses'] for r in self.results.values())
        total_ties = sum(r['ties'] for r in self.results.values())
        total_tests = total_wins + total_losses + total_ties
        
        overall_score = (total_wins + 0.5 * total_ties) / total_tests if total_tests > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "elapsed_time_minutes": elapsed_time / 60,
            "overall": {
                "total_tests": total_tests,
                "wins": total_wins,
                "losses": total_losses,
                "ties": total_ties,
                "win_rate": total_wins / total_tests if total_tests > 0 else 0,
                "score_vs_gpt4": overall_score * 100,
                "performance_rating": self._get_performance_rating(overall_score)
            },
            "by_category": {}
        }
        
        # Add per-category results
        for category, results in self.results.items():
            report["by_category"][category] = {
                "wins": results['wins'],
                "losses": results['losses'],
                "ties": results['ties'],
                "win_rate": results.get('win_rate', 0) * 100,
                "score": results.get('score', 0) * 100,
                "description": self.categories.get(category, "")
            }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(report)
        
        print(f"\n📄 Full report saved to: {report_file}")
    
    def _get_performance_rating(self, score: float) -> str:
        """Get performance rating based on score."""
        if score >= 1.0:
            return "🏆 EXCEEDS GPT-4"
        elif score >= 0.95:
            return "⭐ MATCHES GPT-4"
        elif score >= 0.90:
            return "✅ EXCELLENT (90-95% of GPT-4)"
        elif score >= 0.85:
            return "👍 GOOD (85-90% of GPT-4)"
        elif score >= 0.80:
            return "📊 ACCEPTABLE (80-85% of GPT-4)"
        else:
            return "⚠️ NEEDS IMPROVEMENT (<80% of GPT-4)"
    
    def _print_summary(self, report: Dict):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print(f"🎯 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"\n📊 Overall Performance vs GPT-4:")
        print(f"   Score: {report['overall']['score_vs_gpt4']:.1f}%")
        print(f"   Rating: {report['overall']['performance_rating']}")
        print(f"   Win Rate: {report['overall']['win_rate']*100:.1f}%")
        print(f"\n📈 Breakdown by Category:")
        
        for category, results in sorted(report['by_category'].items(), key=lambda x: x[1]['score'], reverse=True):
            print(f"\n   {category.upper()}:")
            print(f"      Score: {results['score']:.1f}%")
            print(f"      W/L/T: {results['wins']}/{results['losses']}/{results['ties']}")
        
        print(f"\n⏱️ Total Time: {report['elapsed_time_minutes']:.1f} minutes")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Automated GPT-4 Benchmark")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--openai_key", required=True, help="OpenAI API key")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to test (default: all)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples per category (default: 100)")
    
    args = parser.parse_args()
    
    # Run benchmark
    suite = BenchmarkSuite(
        model_path=args.model_path,
        openai_key=args.openai_key,
        output_dir=args.output_dir
    )
    
    suite.run_full_benchmark(
        categories=args.categories,
        samples_per_category=args.num_samples
    )


if __name__ == "__main__":
    main()
