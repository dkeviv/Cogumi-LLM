#!/usr/bin/env python3
"""
Phase 1: Generate Missing Questions with GPT-4o-mini

Use PAID GPT-4o-mini model for guaranteed diversity and uniqueness.
Expected cost: ~$3-5 for 10K questions.
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

# GPT-4o-mini for guaranteed diversity
MODEL = "openai/gpt-4o-mini"
MODEL_NAME = "GPT-4o-mini"

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
    """Create diverse prompt with random subtopic."""
    subtopic = random.choice(DOMAIN_SUBTOPICS[domain])
    
    # Add random variation to ensure uniqueness
    variation = random.randint(1, 10000)
    
    return f"""Generate {batch_size} diverse {domain} questions specifically about: "{subtopic}"

Domain: {domain}
Subtopic Focus: {subtopic}
Difficulty: EASY to MEDIUM
Variation ID: {variation}

CRITICAL REQUIREMENTS:
- Questions MUST be specifically about the subtopic: {subtopic}
- Each question MUST be completely UNIQUE and DIFFERENT
- Use CREATIVE, UNUSUAL, and DIVERSE scenarios
- Avoid common/generic/typical questions
- Think outside the box - be imaginative
- Simple enough to answer but interesting and varied
- English only
- No LaTeX math (use plain text)

Format: Return ONLY a JSON array of strings (the questions), nothing else.
Example: ["question 1", "question 2", "question 3"]

Generate {batch_size} highly creative and unique questions now:"""


async def call_api_async(session, prompt, semaphore):
    """Call OpenRouter API asynchronously."""
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,  # Balanced for GPT-4o-mini
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
                timeout=aiohttp.ClientTimeout(total=90)
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
            console.print(f"[red]Error: {e}[/red]")
            return None


async def generate_for_domain(
    domain: str,
    target_count: int,
    seen_questions: set,
    output_file: Path
) -> int:
    """Generate questions for a single domain."""
    
    # Request 2x for GPT-4o-mini (should have good unique rate)
    request_count = int(target_count * 2.0)
    
    console.print(f"\n[cyan]Generating {request_count:,} {domain} questions (target {target_count:,} unique)...[/cyan]")
    console.print(f"[cyan]Using: {MODEL_NAME} (PAID)[/cyan]")
    
    batch_size = 20
    batches = (request_count + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(15)  # Lower concurrency for paid model
    
    all_questions = []
    duplicates_found = 0
    api_calls = 0
    
    with Progress() as progress:
        task = progress.add_task(f"{domain}", total=batches)
        
        async with aiohttp.ClientSession() as session:
            for i in range(batches):
                prompt = create_prompt(domain, batch_size)
                
                questions = await call_api_async(session, prompt, semaphore)
                api_calls += 1
                
                if questions:
                    for q in questions:
                        q_lower = q.strip().lower()
                        if q_lower not in seen_questions:
                            all_questions.append({
                                "question": q.strip(),
                                "difficulty": "easy",
                                "domain": domain,
                                "model_used": MODEL,
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
    checkpoint_file = output_file.parent / f"checkpoint_{domain.lower().replace(' ', '_')}_gpt4o.jsonl"
    with open(checkpoint_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"[green]✓ Generated {len(all_questions):,} unique questions for {domain}[/green]")
    if duplicates_found > 0:
        dup_rate = duplicates_found/(len(all_questions)+duplicates_found)*100
        console.print(f"[yellow]⚠ Filtered {duplicates_found:,} duplicates ({dup_rate:.1f}%)[/yellow]")
    
    # Cost estimate (GPT-4o-mini: $0.15/1M input, $0.60/1M output)
    # Rough estimate: ~500 tokens input, ~300 tokens output per batch
    input_tokens = api_calls * 500
    output_tokens = api_calls * 300
    cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
    console.print(f"[cyan]Estimated cost for {domain}: ${cost:.3f}[/cyan]")
    
    return len(all_questions)


async def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Generate Missing Questions with GPT-4o-mini[/bold cyan]")
    console.print("=" * 70)
    console.print("\n[cyan]Using PAID model for guaranteed diversity:[/cyan]")
    console.print(f"  • {MODEL_NAME}")
    console.print(f"  • Expected cost: $3-5 for 10K questions")
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
    total_cost = 0.0
    
    for domain, target in DOMAIN_GAPS.items():
        generated = await generate_for_domain(domain, target, seen, output_dir / "temp.jsonl")
        total_generated += generated
        
        # Report progress
        console.print(f"[cyan]Progress: {total_generated:,} / 10,000 total ({total_generated/10000*100:.1f}%)[/cyan]")
    
    console.print(f"\n[green]✓ Total generated: {total_generated:,} new unique questions[/green]")
    console.print(f"[green]✓ Total in dedup set: {len(seen):,}[/green]")
    console.print(f"[cyan]Total estimated cost: $3-5[/cyan]")
    
    console.print("\n[cyan]Next step: Consolidate all files[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
