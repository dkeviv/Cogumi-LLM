#!/usr/bin/env python3
"""
Deterministic Distillation Strategy
====================================

Strategy: Generate training data with greedy decoding (temp=0, do_sample=False)
         Then train model to be deterministic even at higher temperatures.

Goal: Model gives consistent answers even with temp=0.7
Cost: ~$50 (self-distillation, no GPT-5)
Time: 2-3 days
Expected: 47% → 65-75% math accuracy (more consistent)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import re


class DeterministicDistillationTrainer:
    """
    Generate deterministic training data from model's best knowledge.
    Train model to be consistent even with sampling.
    """
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError("No base model found in config")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        self.model = peft_model.merge_and_unload()
        self.model.eval()
        print("✓ Model loaded")
    
    def generate_deterministic(self, prompt: str, max_tokens: int = 400) -> str:
        """
        Generate with MAXIMUM determinism (greedy decoding).
        This captures the model's "best" knowledge.
        """
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,     # Ignored when do_sample=False
                do_sample=False,     # ← GREEDY: Always highest probability
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract response
        response_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response_with_tokens:
            assistant_part = response_with_tokens.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            if "<|eot_id|>" in assistant_part:
                assistant_part = assistant_part.split("<|eot_id|>")[0]
            response = assistant_part.strip()
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        
        return response
    
    def generate_with_sampling(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate with sampling (for comparison/testing).
        """
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,  # ← SAMPLING: Probabilistic
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract response
        response_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response_with_tokens:
            assistant_part = response_with_tokens.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            if "<|eot_id|>" in assistant_part:
                assistant_part = assistant_part.split("<|eot_id|>")[0]
            response = assistant_part.strip()
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt[:50]):
                response = response[len(prompt):].strip()
        
        return response
    
    def extract_answer(self, response: str) -> str:
        """Extract numerical answer from response."""
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Look for "final answer" pattern
        final_match = re.search(r'final answer is:?\s*\$?([0-9,]+)', response, re.IGNORECASE)
        if final_match:
            return final_match.group(1).replace(',', '')
        
        # Last number in response
        numbers = re.findall(r'\$?([0-9,]+)', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def verify_determinism(self, prompt: str, n_trials: int = 5) -> Dict:
        """
        Test if greedy decoding is truly deterministic.
        Generate n_trials times and check consistency.
        """
        greedy_outputs = []
        sampling_outputs = []
        
        # Test greedy
        for _ in range(n_trials):
            output = self.generate_deterministic(prompt)
            greedy_outputs.append(output)
        
        # Test sampling
        for _ in range(n_trials):
            output = self.generate_with_sampling(prompt)
            sampling_outputs.append(output)
        
        # Check consistency
        greedy_answers = [self.extract_answer(o) for o in greedy_outputs]
        sampling_answers = [self.extract_answer(o) for o in sampling_outputs]
        
        greedy_consistent = len(set(greedy_answers)) == 1
        sampling_consistent = len(set(sampling_answers)) == 1
        
        return {
            'greedy_outputs': greedy_outputs,
            'sampling_outputs': sampling_outputs,
            'greedy_answers': greedy_answers,
            'sampling_answers': sampling_answers,
            'greedy_consistent': greedy_consistent,
            'sampling_consistent': sampling_consistent,
            'greedy_unique_answers': len(set(greedy_answers)),
            'sampling_unique_answers': len(set(sampling_answers))
        }
    
    def generate_deterministic_dataset(
        self, 
        problems: List[Dict], 
        verify_correctness: bool = True
    ) -> List[Dict]:
        """
        Generate training dataset using greedy decoding.
        Optionally filter for correct answers only.
        """
        print(f"\nGenerating deterministic solutions for {len(problems)} problems...")
        
        training_examples = []
        
        for problem in tqdm(problems):
            prompt = problem['prompt']
            ground_truth = problem.get('ground_truth', '')
            
            # Generate with greedy decoding
            solution = self.generate_deterministic(prompt)
            answer = self.extract_answer(solution)
            
            # Verify correctness if ground truth available
            is_correct = None
            if ground_truth and verify_correctness:
                # Check if answer matches ground truth
                is_correct = answer in ground_truth
            
            # Include all examples if not verifying, or only correct ones if verifying
            if not verify_correctness or is_correct:
                training_examples.append({
                    'prompt': prompt,
                    'solution': solution,
                    'answer': answer,
                    'ground_truth': ground_truth,
                    'is_correct': is_correct
                })
        
        return training_examples
    
    def load_math_problems(self, num_samples: int = 1000) -> List[Dict]:
        """Load math problems from GSM8K."""
        print(f"Loading {num_samples} math problems...")
        dataset = load_dataset("gsm8k", "main", split="train")
        
        problems = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            problems.append({
                'prompt': f"Solve this math problem step by step:\n\n{example['question']}",
                'ground_truth': example['answer']
            })
        
        return problems
    
    def save_training_data(self, examples: List[Dict], filename: str):
        """Save in training format."""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            for ex in examples:
                training_item = {
                    'instruction': ex['prompt'],
                    'output': ex['solution'],
                    'metadata': {
                        'answer': ex['answer'],
                        'is_correct': ex['is_correct'],
                        'ground_truth': ex['ground_truth']
                    }
                }
                f.write(json.dumps(training_item) + '\n')
        
        print(f"✓ Saved {len(examples)} examples to {output_file}")
        
        # Save statistics
        stats_file = self.output_dir / filename.replace('.jsonl', '_stats.json')
        stats = {
            'total_examples': len(examples),
            'correct_examples': sum(1 for ex in examples if ex.get('is_correct') is True),
            'incorrect_examples': sum(1 for ex in examples if ex.get('is_correct') is False),
            'unknown_correctness': sum(1 for ex in examples if ex.get('is_correct') is None),
            'accuracy': sum(1 for ex in examples if ex.get('is_correct') is True) / len(examples) if examples else 0
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved statistics to {stats_file}")
        print(f"\nStatistics:")
        print(f"  Total: {stats['total_examples']}")
        print(f"  Correct: {stats['correct_examples']} ({stats['accuracy']*100:.1f}%)")
        print(f"  Incorrect: {stats['incorrect_examples']}")


def main():
    """Run deterministic distillation pipeline."""
    
    MODEL_PATH = "/workspace/data/Cogumi-LLM/checkpoints/final"
    OUTPUT_DIR = "/workspace/data/Cogumi-LLM/deterministic_distillation"
    NUM_PROBLEMS = 1000  # Start with 1K, scale to 5K later
    
    print("="*80)
    print("DETERMINISTIC DISTILLATION STRATEGY")
    print("="*80)
    print("\nStrategy:")
    print("  1. Generate solutions with greedy decoding (temp=0, do_sample=False)")
    print("  2. Filter for correct answers only")
    print("  3. Train model on these deterministic examples")
    print("  4. Model learns to be consistent even at higher temperatures")
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Problems: {NUM_PROBLEMS}")
    print(f"  Verification: Enabled (filter for correct answers)")
    
    # Initialize trainer
    trainer = DeterministicDistillationTrainer(MODEL_PATH, OUTPUT_DIR)
    
    # Optional: Test determinism first
    print("\n" + "="*80)
    print("TESTING DETERMINISM")
    print("="*80)
    
    test_problem = "Solve this math problem step by step:\n\nJanet has 3 apples. She buys 7 more apples. How many apples does she have?"
    determinism_test = trainer.verify_determinism(test_problem, n_trials=3)
    
    print(f"\nGreedy decoding (temp=0, do_sample=False):")
    print(f"  Consistent: {'YES ✓' if determinism_test['greedy_consistent'] else 'NO ✗'}")
    print(f"  Unique answers: {determinism_test['greedy_unique_answers']}")
    print(f"  Answers: {set(determinism_test['greedy_answers'])}")
    
    print(f"\nSampling (temp=0.7, do_sample=True):")
    print(f"  Consistent: {'YES ✓' if determinism_test['sampling_consistent'] else 'NO ✗'}")
    print(f"  Unique answers: {determinism_test['sampling_unique_answers']}")
    print(f"  Answers: {set(determinism_test['sampling_answers'])}")
    
    # Load problems
    problems = trainer.load_math_problems(NUM_PROBLEMS)
    
    # Generate deterministic dataset
    training_examples = trainer.generate_deterministic_dataset(
        problems,
        verify_correctness=True  # Only keep correct answers
    )
    
    # Save
    trainer.save_training_data(training_examples, "math_deterministic.jsonl")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review generated examples for quality")
    print("2. Fine-tune model on deterministic data:")
    print(f"   python train_qlora.py --data {OUTPUT_DIR}/math_deterministic.jsonl")
    print("   Use: epochs=2, lr=1e-6 (conservative)")
    print("3. Re-benchmark with BOTH settings:")
    print("   a) Greedy (temp=0, do_sample=False) - Should be high")
    print("   b) Sampling (temp=0.7, do_sample=True) - Should improve")
    print("4. Model should become more deterministic inherently")
    print("\nExpected improvement:")
    print("  Current @ temp=0.7: 47% (inconsistent)")
    print("  After training @ temp=0.7: 60-70% (more consistent)")
    print("  With greedy @ temp=0: 70-80% (optimal)")


if __name__ == "__main__":
    main()
