#!/usr/bin/env python3
"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** N/A (test script no longer needed)
**Reason:** Was used during development to test prompt diversity
**Status:** Development testing complete, script no longer relevant

**Archive Date:** 2025-11-14
**Archived For:** Historical reference only

=================================

Test New Diverse Prompts
=========================

Test the new subtopic-based prompt generation.
"""

import random
from scripts.phase1_generate_token_balanced import DOMAIN_SUBTOPICS, create_prompt

print("="*60)
print("TESTING NEW DIVERSE PROMPT SYSTEM")
print("="*60)

for domain in ["Coding", "Math", "Tool Use"]:
    print(f"\n{'='*60}")
    print(f"Domain: {domain}")
    print(f"Available subtopics: {len(DOMAIN_SUBTOPICS[domain])}")
    print(f"{'='*60}\n")
    
    # Generate 3 sample prompts
    for i in range(3):
        prompt = create_prompt(domain, "easy", 20)
        # Extract just the subtopic line
        lines = prompt.split('\n')
        for line in lines:
            if "Subtopic Focus:" in line:
                print(f"  {i+1}. {line.strip()}")
                break

print(f"\n{'='*60}")
print("✓ Each batch will use a DIFFERENT random subtopic")
print("✓ This should significantly reduce duplicates!")
print("="*60)
