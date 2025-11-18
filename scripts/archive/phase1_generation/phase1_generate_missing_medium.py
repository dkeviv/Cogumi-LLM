#!/usr/bin/env python3
"""
Phase 1: Generate Missing Questions - MEDIUM Difficulty

Fallback strategy: Generate MEDIUM difficulty questions instead of EASY
to avoid duplicate space exhaustion.
"""

import json
import random
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.progress import Progress
from datetime import datetime

console = Console()

# Try models in sequence: DeepSeek V3 -> Llama 405B -> Qwen -> Mistral
ALL_MODELS = [
    ("deepseek/deepseek-chat", "DeepSeek V3"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
    ("qwen/qwen-2.5-72b-instruct:free", "Qwen 2.5 72B"),
    ("mistralai/mistral-large-2411:free", "Mistral Large 2 128B"),
]

CURRENT_MODEL_INDEX = 0  # Start with DeepSeek V3

OPENROUTER_API_KEY = "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"

# Target gaps for lacking domains
DOMAIN_GAPS = {
    "Common Sense": 2500,
    "Instruction": 2500,
    "Reading": 2500,
    "Summarization": 2500
}

# Subtopics for diversity
DOMAIN_SUBTOPICS = {
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
    ]
}


def load_existing_questions(file_path: Path) -> set:
    """Load existing questions for deduplication."""
    seen = set()
    if file_path.exists():
        with open(file_path) as f:
            for line in f:
                q = json.loads(line.strip())
                seen.add(q['question'].strip().lower())
    return seen


def create_prompt(domain: str, batch_size: int) -> str:
    """Create diverse prompt with MEDIUM difficulty."""
    subtopic = random.choice(DOMAIN_SUBTOPICS[domain])
    
    return f"""Generate {batch_size} MEDIUM difficulty {domain} questions specifically about: "{subtopic}"

Domain: {domain}
Subtopic Focus: {subtopic}
Difficulty: MEDIUM (not too easy, not too hard)

Requirements:
- Questions MUST be specifically about the subtopic: {subtopic}
- Each question should be UNIQUE and DIFFERENT from others
- MEDIUM complexity - more nuanced than basic questions
- Diverse scenarios and contexts within this subtopic
- Questions that require some thought but not expert knowledge
- English only
- No LaTeX math (use plain text like 'x^2' not '\\(x^2\\)')

Format: Return ONLY a JSON array of strings (the questions), nothing else.
Example: ["question 1", "question 2", "question 3"]

Generate {batch_size} MEDIUM difficulty questions now:"""


async def call_api_async(session, model, prompt, semaphore):
    """Call OpenRouter API asynchronously."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.3,  # Even higher for diversity
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


async def generate_for_domain(
    domain: str,
    target_count: int,
    seen_questions: set,
    output_file: Path,
    model_index: int
) -> tuple[int, int]:
    """Generate questions for a single domain.
    
    Returns: (generated_count, new_model_index)
    """
    
    current_model, model_name = ALL_MODELS[model_index]
    
    # Over-request by 3x for MEDIUM difficulty (more unique space)
    request_count = int(target_count * 3.0)
    
    console.print(f"\n[cyan]Generating {request_count:,} MEDIUM {domain} questions (target {target_count:,} unique)...[/cyan]")
    console.print(f"[cyan]Using: {model_name}[/cyan]")
    
    batch_size = 20
    batches = (request_count + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(20)
    
    all_questions = []
    duplicates_found = 0
    
    with Progress() as progress:
        task = progress.add_task(f"{domain}", total=batches)
        
        async with aiohttp.ClientSession() as session:
            for i in range(batches):
                prompt = create_prompt(domain, batch_size)
                
                questions = await call_api_async(session, current_model, prompt, semaphore)
                
                if questions:
                    for q in questions:
                        q_lower = q.strip().lower()
                        if q_lower not in seen_questions:
                            all_questions.append({
                                "question": q.strip(),
                                "difficulty": "easy",  # Still mark as easy for training
                                "domain": domain,
                                "model_used": current_model,
                                "generated_at": datetime.now().isoformat()
                            })
                            seen_questions.add(q_lower)
                        else:
                            duplicates_found += 1
                
                progress.advance(task)
                
                # Stop early if we hit target
                if len(all_questions) >= target_count:
                    progress.update(task, completed=batches)
                    break
    
    # Save to checkpoint
    checkpoint_file = output_file.parent / f"checkpoint_{domain.lower().replace(' ', '_')}_medium.jsonl"
    with open(checkpoint_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"[green]✓ Generated {len(all_questions):,} unique questions for {domain}[/green]")
    if duplicates_found > 0:
        dup_rate = duplicates_found/(len(all_questions)+duplicates_found)*100
        console.print(f"[yellow]⚠ Filtered {duplicates_found:,} duplicates ({dup_rate:.1f}%)[/yellow]")
    
    # Check if we should switch models
    new_model_index = model_index
    if len(all_questions) < target_count * 0.3:  # Less than 30% success rate
        if model_index < len(ALL_MODELS) - 1:
            new_model_index = model_index + 1
            console.print(f"[yellow]⚠ Low success rate, switching to {ALL_MODELS[new_model_index][1]}[/yellow]")
    
    return len(all_questions), new_model_index


async def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Generate Missing Questions - MEDIUM Difficulty[/bold cyan]")
    console.print("=" * 70)
    console.print("\n[cyan]Strategy: Using MEDIUM difficulty to avoid EASY duplicate space[/cyan]")
    console.print("\n[cyan]Model sequence (will auto-switch if low success):[/cyan]")
    for i, (_, name) in enumerate(ALL_MODELS, 1):
        console.print(f"  {i}. {name}")
    console.print(f"\n[cyan]Target domains:[/cyan]")
    for domain, gap in DOMAIN_GAPS.items():
        console.print(f"  • {domain}: +{gap:,} questions")
    
    # Load existing for deduplication
    existing_file = Path("data/phase1/questions_60k_shuffled.jsonl")
    console.print(f"\n[cyan]Loading existing questions for deduplication...[/cyan]")
    seen = load_existing_questions(existing_file)
    console.print(f"[green]✓ Loaded {len(seen):,} existing questions[/green]")
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate for each lacking domain
    total_generated = 0
    current_model_idx = 0
    
    for domain, target in DOMAIN_GAPS.items():
        generated, current_model_idx = await generate_for_domain(
            domain, target, seen, output_dir / "temp.jsonl", current_model_idx
        )
        total_generated += generated
        
        # Report progress
        console.print(f"[cyan]Progress: {total_generated:,} / 10,000 total ({total_generated/10000*100:.1f}%)[/cyan]")
    
    console.print(f"\n[green]✓ Total generated: {total_generated:,} new unique questions[/green]")
    console.print(f"[green]✓ Total in dedup set: {len(seen):,}[/green]")
    
    final_model = ALL_MODELS[current_model_idx][1]
    console.print(f"[cyan]Final model used: {final_model}[/cyan]")
    
    console.print("\n[cyan]Next step: Consolidate all files[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
