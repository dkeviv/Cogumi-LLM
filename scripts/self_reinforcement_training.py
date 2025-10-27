#!/usr/bin/env python3
"""
Self-Reinforcement Training for Phase 1C
Improves model by filtering and retraining on its own correct solutions.

Cost: ~$50 (vs $280 GPT-5 distillation)
Time: 2-3 days
Expected: 47% → 60-65% math accuracy
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from collections import Counter
import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

class SelfReinforcementTrainer:
    """
    Self-reinforcement training via self-consistency filtering.
    """
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        print("✓ Model loaded")
    
    def generate_solution(self, prompt: str, temperature: float = 0.8) -> str:
        """Generate a single solution with diversity."""
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
                do_sample=True,  # Add diversity for self-consistency
                pad_token_id=self.tokenizer.eos_token_id
            )
        
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
    
    def extract_answer(self, response: str) -> str:
        """Extract numerical answer from response."""
        # Look for boxed answer first
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Look for "final answer is" pattern
        final_match = re.search(r'final answer is:?\s*\$?([0-9,]+)', response, re.IGNORECASE)
        if final_match:
            return final_match.group(1).replace(',', '')
        
        # Look for last number in response
        numbers = re.findall(r'\$?([0-9,]+)', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def self_consistency_filter(
        self, 
        problems: List[Dict], 
        n_samples: int = 10,
        agreement_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Generate multiple solutions per problem.
        Keep only problems where ≥60% of solutions agree.
        """
        print(f"\nGenerating {n_samples} solutions per problem...")
        high_quality_examples = []
        
        for problem in tqdm(problems):
            prompt = problem['prompt']
            ground_truth = problem.get('ground_truth', '')
            
            # Generate multiple solutions
            solutions = []
            for _ in range(n_samples):
                solution = self.generate_solution(prompt)
                answer = self.extract_answer(solution)
                solutions.append({
                    'text': solution,
                    'answer': answer
                })
            
            # Count answer frequency
            answers = [s['answer'] for s in solutions if s['answer']]
            if not answers:
                continue
            
            answer_counts = Counter(answers)
            most_common_answer, count = answer_counts.most_common(1)[0]
            
            # Check agreement threshold
            agreement_rate = count / len(answers)
            if agreement_rate >= agreement_threshold:
                # Keep the longest (most detailed) solution with this answer
                matching_solutions = [s for s in solutions if s['answer'] == most_common_answer]
                best_solution = max(matching_solutions, key=lambda x: len(x['text']))
                
                # Check if answer is correct (if ground truth available)
                is_correct = most_common_answer in ground_truth if ground_truth else None
                
                high_quality_examples.append({
                    'prompt': prompt,
                    'solution': best_solution['text'],
                    'answer': most_common_answer,
                    'agreement_rate': agreement_rate,
                    'is_correct': is_correct,
                    'ground_truth': ground_truth
                })
        
        return high_quality_examples
    
    def load_math_problems(self, num_samples: int = 500) -> List[Dict]:
        """Load math problems from GSM8K dataset."""
        print(f"Loading {num_samples} math problems from GSM8K...")
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
        """Save filtered examples in training format."""
        output_file = self.output_dir / filename
        
        training_data = []
        for ex in examples:
            training_data.append({
                'instruction': ex['prompt'],
                'output': ex['solution'],
                'metadata': {
                    'agreement_rate': ex['agreement_rate'],
                    'is_correct': ex['is_correct'],
                    'answer': ex['answer']
                }
            })
        
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✓ Saved {len(training_data)} examples to {output_file}")
    
    def analyze_filtered_data(self, examples: List[Dict]):
        """Print analysis of filtered examples."""
        print("\n" + "="*80)
        print("SELF-CONSISTENCY ANALYSIS")
        print("="*80)
        
        total = len(examples)
        if total == 0:
            print("No high-quality examples found!")
            return
        
        # Agreement rate distribution
        agreement_rates = [ex['agreement_rate'] for ex in examples]
        avg_agreement = sum(agreement_rates) / len(agreement_rates)
        
        # Correctness (if ground truth available)
        correct_examples = [ex for ex in examples if ex.get('is_correct') is True]
        incorrect_examples = [ex for ex in examples if ex.get('is_correct') is False]
        unknown_examples = [ex for ex in examples if ex.get('is_correct') is None]
        
        print(f"\nTotal filtered examples: {total}")
        print(f"Average agreement rate: {avg_agreement*100:.1f}%")
        print(f"\nCorrectness:")
        print(f"  Correct: {len(correct_examples)} ({len(correct_examples)/total*100:.1f}%)")
        print(f"  Incorrect: {len(incorrect_examples)} ({len(incorrect_examples)/total*100:.1f}%)")
        print(f"  Unknown: {len(unknown_examples)} ({len(unknown_examples)/total*100:.1f}%)")
        
        # Show sample
        print("\n" + "="*80)
        print("SAMPLE HIGH-QUALITY EXAMPLE")
        print("="*80)
        sample = examples[0]
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"\nSolution: {sample['solution'][:300]}...")
        print(f"\nAnswer: {sample['answer']}")
        print(f"Agreement: {sample['agreement_rate']*100:.0f}%")
        print(f"Correct: {sample['is_correct']}")


def main():
    """Run self-reinforcement training pipeline."""
    
    # Configuration
    MODEL_PATH = "/workspace/data/Cogumi-LLM/checkpoints/final"
    OUTPUT_DIR = "/workspace/data/Cogumi-LLM/self_reinforcement"
    NUM_PROBLEMS = 500  # Start with 500, scale to 5K later
    N_SAMPLES = 10  # Generate 10 solutions per problem
    AGREEMENT_THRESHOLD = 0.6  # 60% must agree
    
    print("="*80)
    print("SELF-REINFORCEMENT TRAINING - PHASE 1C ALTERNATIVE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Problems: {NUM_PROBLEMS}")
    print(f"  Solutions per problem: {N_SAMPLES}")
    print(f"  Agreement threshold: {AGREEMENT_THRESHOLD*100:.0f}%")
    
    # Initialize trainer
    trainer = SelfReinforcementTrainer(MODEL_PATH, OUTPUT_DIR)
    
    # Load math problems
    problems = trainer.load_math_problems(NUM_PROBLEMS)
    
    # Self-consistency filtering
    high_quality_examples = trainer.self_consistency_filter(
        problems,
        n_samples=N_SAMPLES,
        agreement_threshold=AGREEMENT_THRESHOLD
    )
    
    # Analyze results
    trainer.analyze_filtered_data(high_quality_examples)
    
    # Save training data
    trainer.save_training_data(high_quality_examples, "math_self_reinforcement.jsonl")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review generated examples for quality")
    print("2. Fine-tune model on filtered data:")
    print(f"   python train_qlora.py --data {OUTPUT_DIR}/math_self_reinforcement.jsonl")
    print("3. Re-benchmark to measure improvement")
    print("4. If needed, run second round or add GPT-5 for remaining failures")
    print("\nExpected improvement: 47% → 60-65% math accuracy")


if __name__ == "__main__":
    main()
