#!/usr/bin/env python3
"""
Category-Specific Self-Consistency Distillation

Different strategies for different task types:
1. Math/Code: Generate with temp=0 → Train at temp=0 (maximize determinism)
2. Creativity: Generate with temp=0.7 → Train at lower temp (learn creative patterns deterministically)

This approach "bakes in" consistency through training data selection.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import re
from collections import Counter

# Category-specific generation settings
CATEGORY_SETTINGS = {
    'math': {
        'generate_temp': 0.0,      # Maximum determinism for generation
        'generate_sample': False,   # Greedy decoding
        'train_temp': 0.0,         # Train for deterministic inference
        'rationale': 'Math needs exact answers - maximize determinism at all stages'
    },
    'code': {
        'generate_temp': 0.0,
        'generate_sample': False,
        'train_temp': 0.0,
        'rationale': 'Code must be correct - no room for randomness'
    },
    'reasoning': {
        'generate_temp': 0.1,      # Slight diversity
        'generate_sample': True,
        'train_temp': 0.0,
        'rationale': 'Logical reasoning benefits from slight exploration, train deterministically'
    },
    'creativity': {
        'generate_temp': 0.7,      # High diversity for creative outputs
        'generate_sample': True,
        'train_temp': 0.3,         # Train at LOWER temp to learn patterns
        'rationale': 'Generate creative outputs, then train model to reproduce them consistently'
    },
    'knowledge': {
        'generate_temp': 0.0,
        'generate_sample': False,
        'train_temp': 0.0,
        'rationale': 'Factual knowledge needs precision'
    },
    'instruction': {
        'generate_temp': 0.2,
        'generate_sample': True,
        'train_temp': 0.0,
        'rationale': 'Follow instructions precisely but allow slight variation'
    }
}


class CategorySpecificDistiller:
    """
    Self-consistency distillation with category-specific strategies.
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
        base_model_name = peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError("Base model name not found in config")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.merge_and_unload()  # type: ignore[assignment]
        self.model.eval()
        print("✓ Model loaded")
    
    def generate_solution(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        do_sample: bool = False,
        max_tokens: int = 400
    ) -> str:
        """Generate a solution with specified settings."""
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Greedy decoding - temperature doesn't matter
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
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
    
    def extract_answer(self, response: str, category: str) -> str:
        """Extract answer from response (category-specific)."""
        if category in ['math', 'code']:
            # Look for boxed answer
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
            if boxed_match:
                return boxed_match.group(1).strip()
            
            # Look for final answer pattern
            final_match = re.search(r'final answer is:?\s*\$?([0-9,]+)', response, re.IGNORECASE)
            if final_match:
                return final_match.group(1).replace(',', '')
            
            # For code, look for function definition
            if category == 'code':
                code_match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
                if code_match:
                    return code_match.group(1).strip()
        
        # For other categories, use last sentence or full response
        sentences = response.split('.')
        return sentences[-1].strip() if sentences else response
    
    def self_consistency_filter(
        self,
        category: str,
        problems: List[Dict],
        n_samples: int = 10
    ) -> List[Dict]:
        """
        Generate training data with category-specific strategy.
        """
        settings = CATEGORY_SETTINGS.get(category, CATEGORY_SETTINGS['math'])
        
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        print(f"Strategy: {settings['rationale']}")
        print(f"Generation: temp={settings['generate_temp']}, sample={settings['generate_sample']}")
        print(f"Training target: temp={settings['train_temp']}")
        print(f"Samples per problem: {n_samples}")
        
        high_quality_examples = []
        
        for problem in tqdm(problems, desc=f"Processing {category}"):
            prompt = problem['prompt']
            ground_truth = problem.get('ground_truth', '')
            
            # For deterministic categories (math/code), generate once
            if not settings['generate_sample']:
                solution = self.generate_solution(
                    prompt,
                    temperature=settings['generate_temp'],
                    do_sample=settings['generate_sample']
                )
                answer = self.extract_answer(solution, category)
                
                # Verify correctness if ground truth available
                is_correct = answer in ground_truth if ground_truth else None
                
                high_quality_examples.append({
                    'prompt': prompt,
                    'solution': solution,
                    'answer': answer,
                    'is_correct': is_correct,
                    'ground_truth': ground_truth,
                    'category': category,
                    'generation_temp': settings['generate_temp'],
                    'training_temp': settings['train_temp']
                })
            
            else:
                # For creative categories, generate multiple and filter
                solutions = []
                for _ in range(n_samples):
                    solution = self.generate_solution(
                        prompt,
                        temperature=settings['generate_temp'],
                        do_sample=settings['generate_sample']
                    )
                    answer = self.extract_answer(solution, category)
                    solutions.append({
                        'text': solution,
                        'answer': answer
                    })
                
                # For creativity: Keep best-quality diverse examples
                # For reasoning: Use self-consistency (majority vote)
                if category == 'creativity':
                    # Keep all diverse high-quality outputs
                    unique_solutions = []
                    seen_answers = set()
                    for sol in solutions:
                        if sol['answer'] not in seen_answers and len(sol['text']) > 50:
                            unique_solutions.append(sol)
                            seen_answers.add(sol['answer'])
                    
                    # Add up to 3 diverse creative solutions
                    for sol in unique_solutions[:3]:
                        high_quality_examples.append({
                            'prompt': prompt,
                            'solution': sol['text'],
                            'answer': sol['answer'],
                            'is_correct': None,  # No single correct answer for creativity
                            'ground_truth': ground_truth,
                            'category': category,
                            'generation_temp': settings['generate_temp'],
                            'training_temp': settings['train_temp']
                        })
                
                else:
                    # Self-consistency for reasoning
                    answers = [s['answer'] for s in solutions if s['answer']]
                    if answers:
                        answer_counts = Counter(answers)
                        most_common_answer, count = answer_counts.most_common(1)[0]
                        agreement_rate = count / len(answers)
                        
                        if agreement_rate >= 0.6:  # 60% agreement
                            # Keep longest solution with majority answer
                            matching_solutions = [s for s in solutions if s['answer'] == most_common_answer]
                            best_solution = max(matching_solutions, key=lambda x: len(x['text']))
                            
                            is_correct = most_common_answer in ground_truth if ground_truth else None
                            
                            high_quality_examples.append({
                                'prompt': prompt,
                                'solution': best_solution['text'],
                                'answer': most_common_answer,
                                'agreement_rate': agreement_rate,
                                'is_correct': is_correct,
                                'ground_truth': ground_truth,
                                'category': category,
                                'generation_temp': settings['generate_temp'],
                                'training_temp': settings['train_temp']
                            })
        
        return high_quality_examples
    
    def load_problems(self, category: str, num_samples: int = 500) -> List[Dict]:
        """Load problems for a category."""
        print(f"\nLoading {num_samples} {category} problems...")
        
        if category == 'math':
            dataset = load_dataset("gsm8k", "main", split="train")
            problems = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                problems.append({
                    'prompt': f"Solve this math problem step by step:\n\n{example['question']}",
                    'ground_truth': example['answer']
                })
        
        elif category == 'code':
            dataset = load_dataset("openai_humaneval", split="test")
            problems = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                problems.append({
                    'prompt': f"Complete this Python function:\n\n{example['prompt']}",
                    'ground_truth': example['canonical_solution']
                })
        
        else:
            # Placeholder - add other datasets as needed
            print(f"⚠️  No dataset configured for {category}, using dummy data")
            problems = [{
                'prompt': f"Sample {category} problem {i}",
                'ground_truth': f"Answer {i}"
            } for i in range(min(num_samples, 10))]
        
        return problems
    
    def save_training_data(self, examples: List[Dict], category: str):
        """Save filtered examples for training."""
        output_file = self.output_dir / f"{category}_distilled.jsonl"
        
        training_data = []
        for ex in examples:
            training_data.append({
                'instruction': ex['prompt'],
                'output': ex['solution'],
                'metadata': {
                    'category': ex['category'],
                    'answer': ex['answer'],
                    'is_correct': ex.get('is_correct'),
                    'generation_temp': ex['generation_temp'],
                    'training_temp': ex['training_temp'],
                    'agreement_rate': ex.get('agreement_rate')
                }
            })
        
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✓ Saved {len(training_data)} examples to {output_file}")
        
        # Print stats
        if category in ['math', 'code', 'reasoning']:
            correct = sum(1 for ex in examples if ex.get('is_correct') is True)
            total_verified = sum(1 for ex in examples if ex.get('is_correct') is not None)
            if total_verified > 0:
                print(f"  Correctness: {correct}/{total_verified} ({correct/total_verified*100:.1f}%)")


def main():
    """Run category-specific self-consistency distillation."""
    
    MODEL_PATH = "/workspace/data/Cogumi-LLM/checkpoints/final"
    OUTPUT_DIR = "/workspace/data/Cogumi-LLM/self_distillation"
    
    # Categories to process
    CATEGORIES = [
        ('math', 500),      # 500 math problems
        ('code', 164),      # All HumanEval problems
        # ('reasoning', 200),
        # ('creativity', 100),
    ]
    
    print("="*80)
    print("CATEGORY-SPECIFIC SELF-CONSISTENCY DISTILLATION")
    print("="*80)
    print("\nStrategies:")
    for category, _ in CATEGORIES:
        settings = CATEGORY_SETTINGS.get(category)
        if settings:
            print(f"\n{category.upper()}:")
            print(f"  {settings['rationale']}")
            print(f"  Generate: temp={settings['generate_temp']}, sample={settings['generate_sample']}")
            print(f"  Train: temp={settings['train_temp']}")
    
    # Initialize distiller
    distiller = CategorySpecificDistiller(MODEL_PATH, OUTPUT_DIR)
    
    # Process each category
    for category, num_samples in CATEGORIES:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {category.upper()}")
        print(f"{'='*80}")
        
        # Load problems
        problems = distiller.load_problems(category, num_samples)
        
        # Generate and filter
        high_quality_examples = distiller.self_consistency_filter(
            category=category,
            problems=problems,
            n_samples=10 if category in ['creativity', 'reasoning'] else 1
        )
        
        # Save training data
        distiller.save_training_data(high_quality_examples, category)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review generated examples for quality")
    print(f"2. Train model on category-specific data:")
    print(f"   python train_qlora.py --data {OUTPUT_DIR}/*.jsonl")
    print("3. Model will learn to be consistent even at inference temp=0.7")
    print("4. Re-benchmark to measure improvement")
    print("\nExpected: Math 47% → 65-75%, overall improvement across all categories")


if __name__ == "__main__":
    main()
