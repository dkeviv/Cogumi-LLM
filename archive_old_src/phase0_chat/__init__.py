"""
⚠️ ARCHIVED - DO NOT USE ⚠️
================================

**Status**: DEPRECATED
**Reason**: Old implementation superseded by current src/ directory
**Use Instead**: See archive_old_src/README_ARCHIVE.md for replacements

---

ORIGINAL DOCSTRING (for historical reference):


Phase 0: Chat Interface & Infrastructure
========================================

This module provides the chat interface and infrastructure for the LLM distillation pipeline.
It includes API clients for teacher models, cascading router, session management, and cost tracking.

Teacher Models:
- Groq Llama 405B (primary, 89% of data, batch API 50% discount)
- Together.ai Llama 405B (fallback)
- ChatGPT-4 (critical patterns, 8% of data)
- ChatGPT-5 (elite patterns, 3% of data)
- Qwen3-Coder-480B (coding foundation, 69% of coding data)

Components:
- chat_interface.py: Rich CLI chat interface
- api_clients.py: Teacher model API clients
- router.py: Cascading provider selection
- session_manager.py: SQLite session persistence
- token_counter.py: Cost tracking with batch API optimization
"""

from .chat_interface import ChatInterface
from .api_clients import GroqClient, TogetherClient, OpenAIClient
from .router import CascadingRouter
from .session_manager import SessionManager
from .token_counter import TokenCounter

__all__ = [
    "ChatInterface",
    "GroqClient", 
    "TogetherClient",
    "OpenAIClient",
    "CascadingRouter",
    "SessionManager",
    "TokenCounter"
]