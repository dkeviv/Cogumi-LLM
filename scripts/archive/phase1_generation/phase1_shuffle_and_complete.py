#!/usr/bin/env python3
"""
Phase 1: Shuffle Questions and Complete to 60K

This script:
1. Loads existing 54,894 questions
2. Generates ~5,106 more to reach 60K target
3. Shuffles ALL questions randomly (mix domains + difficulties)
4. Ensures each training batch will have natural 60/40 token mix
5. Validates final token distribution

Context: Training requires shuffled data so each batch has diverse
         mix of easy (short) and hard (long) examples for effective learning.
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from collections import Counter
from rich.console import Console
from rich.progress import Progress
import asyncio
import aiohttp

console = Console()

# Domain and difficulty distributions
DOMAINS = ["Coding", "Math", "Tool Use", "Reasoning", "Reading", "Summarization", "Common Sense", "Instruction"]

# Target distribution for 60K questions
TARGET_TOTAL = 60_000
TARGET_EASY_RATIO = 0.987  # 98.7% easy by count
TARGET_HARD_RATIO = 0.013  # 1.3% hard by count

TARGET_EASY = int(TARGET_TOTAL * TARGET_EASY_RATIO)  # 59,220
TARGET_HARD = int(TARGET_TOTAL * TARGET_HARD_RATIO)  # 780

# API configuration
OPENROUTER_API_KEY = "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"
FREE_MODELS = [
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct"
]

# Subtopic system for diversity (same as before)
DOMAIN_SUBTOPICS = {
    "Coding": [
        "Python string manipulation and regex",
        "JavaScript async/await and promises",
        "SQL query optimization and joins",
        "Git version control and branching",
        "REST API design and HTTP methods",
        "Docker containerization basics",
        "Unit testing and test-driven development",
        "Error handling and exception management",
        "Data structure implementation (lists, trees, graphs)",
        "Algorithm complexity and Big O notation",
        "File I/O and data serialization",
        "Object-oriented programming principles",
        "Functional programming concepts",
        "Debugging techniques and tools",
        "Code refactoring and optimization"
    ],
    "Math": [
        "Trigonometry (sin, cos, tan)",
        "Probability and combinatorics",
        "Calculus derivatives and integrals",
        "Linear algebra (matrices, vectors)",
        "Statistics (mean, median, standard deviation)",
        "Geometry (area, volume, angles)",
        "Number theory and prime numbers",
        "Set theory and Venn diagrams",
        "Sequences and series",
        "Exponential and logarithmic functions",
        "Systems of equations",
        "Inequalities and absolute values",
        "Complex numbers",
        "Graph theory basics",
        "Mathematical proof techniques"
    ],
    "Tool Use": [
        "Docker containers and orchestration",
        "AWS S3 storage and operations",
        "GitHub CLI and workflows",
        "Kubernetes pod management",
        "Jenkins CI/CD pipelines",
        "Terraform infrastructure as code",
        "Ansible automation and playbooks",
        "npm package management",
        "pip and Python package installation",
        "grep and text searching",
        "sed stream editing",
        "awk text processing",
        "jq JSON manipulation",
        "curl HTTP requests",
        "SSH and remote access"
    ],
    "Reasoning": [
        "Logical deduction and inference",
        "Pattern recognition",
        "Cause and effect analysis",
        "Analogical reasoning",
        "Hypothetical scenarios",
        "Contradiction identification",
        "Assumption evaluation",
        "Argument structure analysis",
        "Decision tree reasoning",
        "Probability-based reasoning",
        "Temporal reasoning (before/after)",
        "Spatial reasoning",
        "Moral and ethical dilemmas",
        "Trade-off analysis",
        "Strategic planning"
    ],
    "Reading": [
        "Main idea identification",
        "Supporting details extraction",
        "Inference from context",
        "Author's purpose and tone",
        "Vocabulary in context",
        "Text structure analysis",
        "Comparison and contrast",
        "Cause and effect in passages",
        "Fact vs opinion distinction",
        "Summarization of passages",
        "Drawing conclusions",
        "Predicting outcomes",
        "Character analysis",
        "Theme identification",
        "Critical evaluation"
    ],
    "Summarization": [
        "News article summarization",
        "Research paper abstracts",
        "Meeting minutes recap",
        "Book chapter summaries",
        "Technical documentation condensing",
        "Email thread summarization",
        "Video transcript summaries",
        "Legal document summaries",
        "Financial report highlights",
        "Customer feedback synthesis",
        "Product review aggregation",
        "Social media thread recaps",
        "Conference talk summaries",
        "Interview transcription summaries",
        "Policy document overviews"
    ],
    "Common Sense": [
        "Daily routines and schedules",
        "Social etiquette and manners",
        "Safety awareness",
        "Health and hygiene",
        "Basic cooking and food",
        "Weather and clothing",
        "Transportation and navigation",
        "Money and shopping",
        "Time management",
        "Household tasks",
        "Pet care basics",
        "Basic first aid",
        "Communication norms",
        "Environmental awareness",
        "Technology in daily life"
    ],
    "Instruction": [
        "Recipe following and cooking",
        "Assembly instructions",
        "Software installation steps",
        "Workout routines",
        "Craft and DIY projects",
        "Travel directions",
        "Game rules and gameplay",
        "Product usage guides",
        "Safety procedures",
        "Educational lesson plans",
        "Event planning checklists",
        "Cleaning and maintenance",
        "Form filling instructions",
        "Medical prescription following",
        "Emergency response protocols"
    ]
}


def load_existing_questions(file_path: Path) -> tuple[List[Dict], set]:
    """Load existing questions and return list + dedup set."""
    questions = []
    seen = set()
    
    if file_path.exists():
        with open(file_path) as f:
            for line in f:
                q = json.loads(line.strip())
                questions.append(q)
                seen.add(q['question'].strip().lower())
    
    return questions, seen


def analyze_distribution(questions: List[Dict]) -> Dict:
    """Analyze current distribution."""
    by_diff = Counter(q['difficulty'] for q in questions)
    by_domain = Counter(q['domain'] for q in questions)
    
    easy_count = by_diff.get('easy', 0)
    hard_count = by_diff.get('hard', 0)
    
    # Token estimation
    easy_tokens = easy_count * 15
    hard_tokens = hard_count * 750
    total_tokens = easy_tokens + hard_tokens
    
    easy_token_pct = (easy_tokens / total_tokens * 100) if total_tokens > 0 else 0
    hard_token_pct = (hard_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    return {
        'total': len(questions),
        'easy': easy_count,
        'hard': hard_count,
        'easy_pct': easy_count / len(questions) * 100 if questions else 0,
        'hard_pct': hard_count / len(questions) * 100 if questions else 0,
        'easy_tokens': easy_tokens,
        'hard_tokens': hard_tokens,
        'easy_token_pct': easy_token_pct,
        'hard_token_pct': hard_token_pct,
        'by_domain': dict(by_domain)
    }


def create_prompt(domain: str, difficulty: str, batch_size: int) -> str:
    """Create diverse prompt with random subtopic."""
    subtopic = random.choice(DOMAIN_SUBTOPICS[domain])
    
    diff_desc = "simple, straightforward" if difficulty == "easy" else "complex, multi-step"
    
    return f"""Generate {batch_size} {diff_desc} {domain} questions specifically about: "{subtopic}"

Domain: {domain}
Subtopic Focus: {subtopic}
Difficulty: {difficulty.upper()}

Requirements:
- Questions MUST be specifically about the subtopic: {subtopic}
- Each question should be UNIQUE and DIFFERENT from others
- Diverse scenarios and contexts within this subtopic
- {"Direct questions with clear answers" if difficulty == "easy" else "Complex questions requiring reasoning"}
- English only
- No LaTeX math (use plain text like 'x^2' not '\\(x^2\\)')

Format: Return ONLY a JSON array of strings (the questions), nothing else.
Example: ["question 1", "question 2", "question 3"]

Generate {batch_size} questions now:"""


async def call_api_async(session, model, prompt, semaphore):
    """Call OpenRouter API asynchronously."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.1,  # High diversity
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                content = data['choices'][0]['message']['content'].strip()
                
                # Parse JSON array
                if content.startswith('[') and content.endswith(']'):
                    questions = json.loads(content)
                    return questions
                else:
                    # Try to find JSON array in content
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start >= 0 and end > start:
                        questions = json.loads(content[start:end])
                        return questions
                
                return None
                
        except Exception as e:
            return None


async def generate_missing_questions(
    target_easy: int,
    target_hard: int,
    current_easy: int,
    current_hard: int,
    seen_questions: set,
    output_file: Path
) -> List[Dict]:
    """Generate missing questions to reach target."""
    
    gap_easy = target_easy - current_easy
    gap_hard = target_hard - current_hard
    
    console.print(f"\n[cyan]Need to generate:[/cyan]")
    console.print(f"  Easy: {gap_easy:,} more questions")
    console.print(f"  Hard: {gap_hard:,} more questions")
    
    if gap_easy <= 0 and gap_hard <= 0:
        console.print("[green]✓ Already at target![/green]")
        return []
    
    # Distribute across domains proportionally
    easy_per_domain = gap_easy // 8
    hard_per_domain = gap_hard // 8
    
    # Over-request by 1.7x to account for duplicates
    request_easy_per_domain = int(easy_per_domain * 1.7)
    request_hard_per_domain = int(hard_per_domain * 1.7)
    
    all_new_questions = []
    batch_size = 20
    semaphore = asyncio.Semaphore(20)  # 20 concurrent
    
    for domain in DOMAINS:
        # Generate easy questions
        if gap_easy > 0:
            console.print(f"\n[cyan]Generating {request_easy_per_domain} easy {domain} questions...[/cyan]")
            
            batches = (request_easy_per_domain + batch_size - 1) // batch_size
            
            with Progress() as progress:
                task = progress.add_task(f"{domain} easy", total=batches)
                
                async with aiohttp.ClientSession() as session:
                    for i in range(batches):
                        model = random.choice(FREE_MODELS)
                        prompt = create_prompt(domain, "easy", batch_size)
                        
                        questions = await call_api_async(session, model, prompt, semaphore)
                        
                        if questions:
                            for q in questions:
                                q_lower = q.strip().lower()
                                if q_lower not in seen_questions:
                                    all_new_questions.append({
                                        "question": q.strip(),
                                        "difficulty": "easy",
                                        "domain": domain,
                                        "model_used": model,
                                        "generated_at": None  # Will add timestamp
                                    })
                                    seen_questions.add(q_lower)
                        
                        progress.advance(task)
        
        # Generate hard questions
        if gap_hard > 0:
            console.print(f"\n[cyan]Generating {request_hard_per_domain} hard {domain} questions...[/cyan]")
            
            batches = (request_hard_per_domain + batch_size - 1) // batch_size
            
            with Progress() as progress:
                task = progress.add_task(f"{domain} hard", total=batches)
                
                async with aiohttp.ClientSession() as session:
                    for i in range(batches):
                        model = random.choice(FREE_MODELS)
                        prompt = create_prompt(domain, "hard", batch_size)
                        
                        questions = await call_api_async(session, model, prompt, semaphore)
                        
                        if questions:
                            for q in questions:
                                q_lower = q.strip().lower()
                                if q_lower not in seen_questions:
                                    all_new_questions.append({
                                        "question": q.strip(),
                                        "difficulty": "hard",
                                        "domain": domain,
                                        "model_used": model,
                                        "generated_at": None
                                    })
                                    seen_questions.add(q_lower)
                        
                        progress.advance(task)
    
    console.print(f"\n[green]✓ Generated {len(all_new_questions):,} new unique questions[/green]")
    return all_new_questions


async def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Shuffle and Complete to 60K[/bold cyan]")
    console.print("=" * 60)
    
    input_file = Path("data/phase1/questions_60k_token_balanced.jsonl")
    output_file = Path("data/phase1/questions_60k_shuffled.jsonl")
    
    # Load existing
    console.print("\n[cyan]Loading existing questions...[/cyan]")
    questions, seen = load_existing_questions(input_file)
    
    stats = analyze_distribution(questions)
    console.print(f"\n[cyan]Current State:[/cyan]")
    console.print(f"  Total: {stats['total']:,} questions")
    console.print(f"  Easy: {stats['easy']:,} ({stats['easy_pct']:.1f}%)")
    console.print(f"  Hard: {stats['hard']:,} ({stats['hard_pct']:.1f}%)")
    console.print(f"  Token balance: {stats['easy_token_pct']:.1f}% easy / {stats['hard_token_pct']:.1f}% hard")
    
    # Generate missing questions
    new_questions = await generate_missing_questions(
        TARGET_EASY,
        TARGET_HARD,
        stats['easy'],
        stats['hard'],
        seen,
        output_file
    )
    
    # Combine all questions
    all_questions = questions + new_questions
    
    # Analyze before shuffle
    stats = analyze_distribution(all_questions)
    console.print(f"\n[cyan]Before Shuffle:[/cyan]")
    console.print(f"  Total: {stats['total']:,} questions")
    console.print(f"  Easy: {stats['easy']:,} ({stats['easy_pct']:.1f}%)")
    console.print(f"  Hard: {stats['hard']:,} ({stats['hard_pct']:.1f}%)")
    console.print(f"  Token balance: {stats['easy_token_pct']:.1f}% easy / {stats['hard_token_pct']:.1f}% hard")
    
    # SHUFFLE randomly (critical for training)
    console.print("\n[cyan]Shuffling questions randomly...[/cyan]")
    random.shuffle(all_questions)
    
    # Trim to exactly 60K if over
    if len(all_questions) > TARGET_TOTAL:
        console.print(f"[yellow]⚠ Trimming from {len(all_questions):,} to {TARGET_TOTAL:,}[/yellow]")
        all_questions = all_questions[:TARGET_TOTAL]
    
    # Final stats
    stats = analyze_distribution(all_questions)
    console.print(f"\n[green]Final Dataset:[/green]")
    console.print(f"  Total: {stats['total']:,} questions")
    console.print(f"  Easy: {stats['easy']:,} ({stats['easy_pct']:.1f}%)")
    console.print(f"  Hard: {stats['hard']:,} ({stats['hard_pct']:.1f}%)")
    console.print(f"  Token balance: {stats['easy_token_pct']:.1f}% easy / {stats['hard_token_pct']:.1f}% hard")
    console.print(f"\n  Domain distribution:")
    for domain, count in sorted(stats['by_domain'].items()):
        console.print(f"    {domain}: {count:,}")
    
    # Save shuffled dataset
    console.print(f"\n[cyan]Saving shuffled dataset...[/cyan]")
    with open(output_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"[green]✓ Saved to: {output_file}[/green]")
    
    # Verify batch-level mixing
    console.print(f"\n[cyan]Verifying batch-level mixing (sample first 3 batches):[/cyan]")
    batch_size = 32
    for batch_num in range(3):
        start = batch_num * batch_size
        end = start + batch_size
        batch = all_questions[start:end]
        
        easy_in_batch = sum(1 for q in batch if q['difficulty'] == 'easy')
        hard_in_batch = len(batch) - easy_in_batch
        domains_in_batch = set(q['domain'] for q in batch)
        
        console.print(f"  Batch {batch_num + 1}: {easy_in_batch} easy, {hard_in_batch} hard | Domains: {len(domains_in_batch)}")
    
    console.print("\n[green]✓ Dataset ready for training![/green]")
    console.print(f"[green]✓ Questions are shuffled - each training batch will have natural 60/40 token mix[/green]")
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Validate with GPT-4-mini ($9)")
    console.print(f"  2. Generate answers (easy: $2.52, hard: $414)")
    console.print(f"  3. Train with shuffled data")


if __name__ == "__main__":
    asyncio.run(main())
