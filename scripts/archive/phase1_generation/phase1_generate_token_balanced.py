#!/usr/bin/env python3
"""
Phase 1: Generate Token-Balanced 60K Dataset
==============================================

Generate 60K questions with 60% EASY tokens / 40% HARD tokens.

Key insight: Easy = 15 tokens avg, Hard = 750 tokens avg
→ Need 98.7% easy samples / 1.3% hard samples for token balance!

Current: 14,787 easy, 29,913 hard
Target:  59,210 easy, 789 hard
Action:  Generate 44,423 easy, randomly sample 789 from existing hard
"""

import asyncio
import aiohttp
import json
import re
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

# Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"

MAX_CONCURRENT = 20
BATCH_SIZE = 20

# Target distribution (98.7% easy / 1.3% hard per domain)
TARGET_DISTRIBUTION = {
    "Coding": {"total": 10000, "easy": 9870, "hard": 130, "model": "deepseek/deepseek-chat"},
    "Math": {"total": 10000, "easy": 9870, "hard": 130, "model": "deepseek/deepseek-chat"},
    "Tool Use": {"total": 10000, "easy": 9870, "hard": 130, "model": "deepseek/deepseek-chat"},
    "Reasoning": {"total": 10000, "easy": 9870, "hard": 130, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Reading": {"total": 5000, "easy": 4935, "hard": 65, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Summarization": {"total": 5000, "easy": 4935, "hard": 65, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Common Sense": {"total": 5000, "easy": 4935, "hard": 65, "model": "meta-llama/llama-3.3-70b-instruct"},
    "Instruction": {"total": 5000, "easy": 4935, "hard": 65, "model": "meta-llama/llama-3.3-70b-instruct"},
}

# Diverse subtopics for each domain (for prompt variation)
DOMAIN_SUBTOPICS = {
    "Coding": [
        "Python string manipulation and regex",
        "JavaScript async/await and promises",
        "SQL query optimization and joins",
        "Git version control and branching",
        "REST API design and HTTP methods",
        "Debugging runtime errors and exceptions",
        "Code refactoring and design patterns",
        "Unit testing and test-driven development",
        "Data structures (arrays, lists, trees, graphs)",
        "Algorithm complexity and Big O notation",
        "Object-oriented programming concepts",
        "Functional programming and lambda functions",
        "Package management (pip, npm, yarn)",
        "Environment variables and configuration",
        "File I/O and data serialization",
    ],
    "Math": [
        "Basic arithmetic and order of operations",
        "Fractions, decimals, and percentages",
        "Linear equations and inequalities",
        "Quadratic equations and factoring",
        "Trigonometry (sin, cos, tan)",
        "Calculus derivatives and integrals",
        "Probability and combinatorics",
        "Statistics (mean, median, mode, standard deviation)",
        "Set theory and Venn diagrams",
        "Number theory and prime numbers",
        "Geometry (area, perimeter, volume)",
        "Matrix operations and linear algebra",
        "Logarithms and exponential functions",
        "Series and sequences",
        "Graph theory basics",
    ],
    "Tool Use": [
        "Command-line navigation (cd, ls, pwd)",
        "File manipulation (cp, mv, rm, find)",
        "Text processing (grep, sed, awk)",
        "Package managers (apt, brew, pip)",
        "Docker containers and images",
        "Kubernetes pods and deployments",
        "AWS S3 bucket operations",
        "GitHub CLI and Actions",
        "Jupyter Notebook shortcuts",
        "VS Code extensions and settings",
        "Postman API testing",
        "Database CLI tools (psql, mysql)",
        "SSH and remote server access",
        "Cron jobs and scheduling",
        "Environment management (conda, virtualenv)",
    ],
    "Reasoning": [
        "Logical deduction and inference",
        "Cause and effect relationships",
        "Pattern recognition and analogies",
        "Hypothesis testing and validation",
        "Argument evaluation and fallacies",
        "Decision trees and flow charts",
        "Risk assessment and mitigation",
        "Root cause analysis",
        "Constraint satisfaction problems",
        "Strategic planning and optimization",
        "Counterfactual thinking",
        "Comparative analysis",
        "Temporal reasoning (before/after)",
        "Spatial reasoning (left/right/above)",
        "Ethical dilemmas and trade-offs",
    ],
    "Reading": [
        "Main idea identification",
        "Supporting details extraction",
        "Author's intent and tone",
        "Inference from context clues",
        "Vocabulary in context",
        "Comparison and contrast",
        "Chronological order",
        "Fact vs opinion distinction",
        "Text structure analysis",
        "Theme identification",
        "Character motivation",
        "Figurative language",
        "Point of view analysis",
        "Predicting outcomes",
        "Drawing conclusions",
    ],
    "Summarization": [
        "News article summarization",
        "Research paper abstracts",
        "Meeting notes condensation",
        "Technical documentation overview",
        "Book chapter summaries",
        "Product review synthesis",
        "Email thread summarization",
        "Legal document key points",
        "Financial report highlights",
        "Scientific findings summary",
        "Policy document essentials",
        "Tutorial step extraction",
        "Bug report triage",
        "Customer feedback themes",
        "Event recap generation",
    ],
    "Common Sense": [
        "Daily routines and habits",
        "Social norms and etiquette",
        "Safety and risk awareness",
        "Time management basics",
        "Weather and clothing choices",
        "Food preparation and storage",
        "Health and hygiene practices",
        "Money and budgeting basics",
        "Transportation and directions",
        "Communication conventions",
        "Home maintenance tasks",
        "Emergency response",
        "Relationship dynamics",
        "Workplace behavior",
        "Physical properties of objects",
    ],
    "Instruction": [
        "Recipe following",
        "Assembly instructions",
        "Installation procedures",
        "Tutorial step execution",
        "Form filling guidelines",
        "Game rule interpretation",
        "Safety protocol adherence",
        "Experiment procedure",
        "Workout routine following",
        "Medication instructions",
        "Software setup steps",
        "Travel itinerary planning",
        "Event organization checklist",
        "Troubleshooting flowcharts",
        "Compliance requirements",
    ],
}

def create_prompt(domain: str, difficulty: str, batch_size: int) -> str:
    """Create prompt for question generation with specific subtopic."""
    # Select a random subtopic for diversity
    subtopic = random.choice(DOMAIN_SUBTOPICS[domain])
    
    diff_desc = "simple, quick" if difficulty == "easy" else "complex and challenging"
    
    return f"""Generate {batch_size} {diff_desc} {domain} questions specifically about: "{subtopic}"

Domain: {domain}
Subtopic Focus: {subtopic}
Difficulty: {difficulty.upper()}

Requirements:
- {"EASY: Direct questions with SHORT answers (1-2 sentences, <30 tokens). Focus on basic concepts." if difficulty == "easy" else "HARD: Complex questions requiring detailed reasoning and multi-step solutions (300+ tokens). Focus on advanced concepts."}
- Questions MUST be specifically about the subtopic: {subtopic}
- Diverse scenarios and contexts within this subtopic
- Real-world applicable examples
- Clear and unambiguous wording
- NO LaTeX (use plain text: x^2 not \\(x^2\\)). Use Unicode: π ² ³ √ ∫
- Each question should be UNIQUE and DIFFERENT from others

Return JSON array ONLY:
[
  {{"question": "...", "difficulty": "{difficulty}"}},
  ...
]"""

async def call_api_async(session, model, prompt, semaphore):
    """Call API with rate limiting and higher temperature for diversity."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.1,  # Increased from 0.9 for more diversity
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as response:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            console.print(f"[red]API Error: {e}[/red]")
            return None

def quick_parse_json(response):
    """Fast JSON parsing with cleanup."""
    if not response:
        return []
    
    try:
        response = re.sub(r'```json\s*|\s*```', '', response)
        response = re.sub(r'\\(?!["\\/bfnrtu])', '', response)
        return json.loads(response)
    except:
        return []

async def generate_batch_async(session, domain, difficulty, model, batch_size, semaphore):
    """Generate one batch of questions."""
    prompt = create_prompt(domain, difficulty, batch_size)
    response = await call_api_async(session, model, prompt, semaphore)
    questions = quick_parse_json(response)
    
    # CRITICAL: Normalize tags and ensure consistency
    for q in questions:
        q["domain"] = domain
        q["difficulty"] = difficulty  # Force to lowercase for consistency
        q["model_used"] = model
        q["generated_at"] = datetime.utcnow().isoformat()
    
    return questions

async def generate_for_domain_difficulty(domain, difficulty, count, model, output_dir, seen_questions):
    """Generate questions for one domain+difficulty with real-time deduplication."""
    if count == 0:
        return []
    
    num_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
    checkpoint_file = output_dir / f"checkpoint_{domain.lower().replace(' ', '_')}_{difficulty}.jsonl"
    
    console.print(f"[cyan]Generating {count:,} {difficulty} {domain} questions ({num_batches} batches)...[/cyan]")
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        tasks = [
            generate_batch_async(session, domain, difficulty, model, 
                                min(BATCH_SIZE, count - i * BATCH_SIZE), semaphore)
            for i in range(num_batches)
        ]
        
        all_questions = []
        duplicates_found = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(f"{domain} {difficulty}", total=num_batches)
            
            # CRITICAL: Save incrementally with deduplication check
            with open(checkpoint_file, 'w') as f:
                for coro in asyncio.as_completed(tasks):
                    batch = await coro
                    
                    # Deduplicate in real-time
                    for q in batch:
                        question_text = q.get('question', '').strip().lower()
                        if question_text and question_text not in seen_questions:
                            all_questions.append(q)
                            seen_questions.add(question_text)  # Track globally
                            f.write(json.dumps(q) + '\n')
                        else:
                            duplicates_found += 1
                    
                    f.flush()
                    progress.advance(task)
        
        if duplicates_found > 0:
            console.print(f"[yellow]  ⚠ Filtered {duplicates_found} duplicates[/yellow]")
        console.print(f"[dim]  → Saved {len(all_questions)} unique to {checkpoint_file.name}[/dim]")
        
        return all_questions

def load_existing_questions():
    """Load existing clean questions and build deduplication set."""
    existing_file = Path("data/phase1/questions_60k_clean.jsonl")
    
    if not existing_file.exists():
        console.print(f"[yellow]⚠ {existing_file} not found, starting fresh[/yellow]")
        return defaultdict(list), set()
    
    questions_by_domain_diff = defaultdict(list)
    seen_questions = set()  # For deduplication
    
    with open(existing_file, 'r') as f:
        for line in f:
            q = json.loads(line)
            # CRITICAL: Normalize difficulty tag
            difficulty = q.get('difficulty', '').lower()
            domain = q.get('domain')
            question_text = q.get('question', '').strip().lower()
            
            if domain and difficulty in ['easy', 'hard'] and question_text:
                key = (domain, difficulty)
                questions_by_domain_diff[key].append(q)
                seen_questions.add(question_text)  # Track for deduplication
    
    console.print(f"[dim]  Loaded {len(seen_questions):,} unique questions for dedup check[/dim]")
    return questions_by_domain_diff, seen_questions

async def generate_all_needed(output_dir):
    """Generate all needed questions to reach token-balanced 60K."""
    console.print("[yellow]Loading existing questions...[/yellow]\n")
    existing, seen_questions = load_existing_questions()
    
    # Calculate what we need
    to_generate = []
    
    console.print("[bold]Generation Plan (with 60% expected unique rate):[/bold]")
    for domain, targets in TARGET_DISTRIBUTION.items():
        model = targets["model"]
        for difficulty in ["easy", "hard"]:
            current = len(existing.get((domain, difficulty), []))
            needed = targets[difficulty]
            gap = needed - current
            
            if gap > 0:
                # REQUEST MORE to account for duplicates (assume 60% unique rate)
                request_count = int(gap * 1.7)  # Request 1.7x to account for 40% duplicates
                to_generate.append({
                    "domain": domain,
                    "difficulty": difficulty,
                    "count": request_count,
                    "target": gap,
                    "model": model
                })
                console.print(f"  {domain} {difficulty}: need {gap:,}, requesting {request_count:,} (have {current:,}/{needed:,})")
    
    if not to_generate:
        console.print("[green]✓ No generation needed![/green]")
        return [], existing, seen_questions
    
    total_request = sum(t['count'] for t in to_generate)
    total_target = sum(t['target'] for t in to_generate)
    console.print(f"\n[bold]Total target: {total_target:,} unique questions[/bold]")
    console.print(f"[bold]Total requesting: {total_request:,} questions (assuming ~60% unique rate)[/bold]")
    console.print(f"[bold]Deduplication enabled: Checking against {len(seen_questions):,} existing questions[/bold]\n")
    
    # Generate in parallel (all domains at once for speed)
    all_new = []
    tasks = [
        generate_for_domain_difficulty(
            item["domain"], item["difficulty"], item["count"], item["model"], output_dir, seen_questions
        )
        for item in to_generate
    ]
    
    results = await asyncio.gather(*tasks)
    for questions in results:
        all_new.extend(questions)
    
    return all_new, existing, seen_questions

def balance_to_target(new_questions, existing_by_domain_diff):
    """Combine new + sampled existing to reach exact target."""
    final_dataset = []
    
    for domain, targets in TARGET_DISTRIBUTION.items():
        for difficulty in ["easy", "hard"]:
            target_count = targets[difficulty]
            
            # Get all available (existing + new)
            key = (domain, difficulty)
            available = existing_by_domain_diff.get(key, []).copy()
            available.extend([q for q in new_questions 
                             if q['domain'] == domain and q['difficulty'] == difficulty])
            
            # Random sample to target (handles both under and over)
            if len(available) >= target_count:
                selected = random.sample(available, target_count)
            else:
                selected = available
                console.print(f"[yellow]⚠ {domain} {difficulty}: only {len(available)}/{target_count}[/yellow]")
            
            final_dataset.extend(selected)
    
    return final_dataset

def main():
    console.print("[bold cyan]Phase 1: Generate Token-Balanced 60K Dataset[/bold cyan]")
    console.print("[dim]Target: 98.7% easy / 1.3% hard for 60% easy tokens / 40% hard tokens[/dim]\n")
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate with real-time deduplication
    new_questions, existing, seen_questions = asyncio.run(generate_all_needed(output_dir))
    
    console.print(f"\n[green]✓ Generated {len(new_questions):,} new unique questions[/green]")
    console.print(f"[dim]  Total in dedup set: {len(seen_questions):,}[/dim]")
    
    # Balance
    console.print("\n[yellow]Balancing to target distribution...[/yellow]")
    final_dataset = balance_to_target(new_questions, existing)
    
    # Shuffle for training (prevent domain clustering)
    random.shuffle(final_dataset)
    
    # Save
    output_file = output_dir / "questions_60k_token_balanced.jsonl"
    with open(output_file, 'w') as f:
        for q in final_dataset:
            f.write(json.dumps(q) + '\n')
    
    # Stats
    easy_count = sum(1 for q in final_dataset if q['difficulty'] == 'easy')
    hard_count = sum(1 for q in final_dataset if q['difficulty'] == 'hard')
    
    easy_tokens = easy_count * 15
    hard_tokens = hard_count * 750
    total_tokens = easy_tokens + hard_tokens
    
    # Per-domain stats
    from collections import Counter
    domain_counts = Counter(q['domain'] for q in final_dataset)
    
    console.print(f"\n[bold green]✓ Saved token-balanced dataset to:[/bold green]")
    console.print(f"  {output_file}\n")
    
    console.print("[bold]Final Stats:[/bold]")
    console.print(f"  Total: {len(final_dataset):,} questions")
    console.print(f"  Easy: {easy_count:,} ({easy_count/len(final_dataset)*100:.1f}%)")
    console.print(f"  Hard: {hard_count:,} ({hard_count/len(final_dataset)*100:.1f}%)")
    console.print(f"\n[bold]Per-Domain Distribution:[/bold]")
    for domain in sorted(domain_counts.keys()):
        console.print(f"  {domain}: {domain_counts[domain]:,}")
    
    console.print(f"\n[bold]Token Distribution (estimated):[/bold]")
    console.print(f"  Easy tokens: {easy_tokens:,} ({easy_tokens/total_tokens*100:.1f}%)")
    console.print(f"  Hard tokens: {hard_tokens:,} ({hard_tokens/total_tokens*100:.1f}%)")
    console.print(f"  Total tokens: {total_tokens:,}")
    
    if 58 <= easy_tokens/total_tokens*100 <= 62:
        console.print("\n[bold green]✓ TOKEN BALANCE ACHIEVED! 60/40 split ✓[/bold green]")
    else:
        console.print("\n[yellow]⚠ Token balance needs adjustment[/yellow]")
    
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Validate questions with GPT-4-mini ($9)")
    console.print("  2. Generate EASY answers with GPT-4o-mini ($2.52)")
    console.print("  3. Generate HARD answers with Claude Sonnet 4 ($414)")
    console.print("  4. Train with proper loss masking (EASY: <final> only, HARD: <draft><thinking><final>)")

if __name__ == "__main__":
    main()
