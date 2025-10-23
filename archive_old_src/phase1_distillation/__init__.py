"""
⚠️ ARCHIVED - DO NOT USE ⚠️
================================

**Status**: DEPRECATED
**Reason**: Old implementation superseded by current src/ directory
**Use Instead**: See archive_old_src/README_ARCHIVE.md for replacements

---

ORIGINAL DOCSTRING (for historical reference):


Phase 1: Data Distillation
==========================

This module implements the data distillation phase of the LLM pipeline.
Generates high-quality training data using cascaded teacher models.

Context:
- Uses cascaded teacher selection (Llama 405B → GPT-4 → GPT-5)
- Implements batch API for 50% cost savings
- Generates 100K samples with quality filtering
- Provides real-time cost tracking and progress monitoring
"""

from .data_generator import DataGenerator, GenerationConfig
from .cascading_selector import CascadingSelector, CascadeConfig  
from .batch_processor import BatchProcessor, BatchConfig
from .quality_filter import QualityFilter, FilterConfig
from .prompt_engineer import PromptEngineer, PromptConfig

__version__ = "1.0.0"
__all__ = [
    "DataGenerator",
    "GenerationConfig", 
    "CascadingSelector",
    "CascadeConfig",
    "BatchProcessor", 
    "BatchConfig",
    "QualityFilter",
    "FilterConfig",
    "PromptEngineer",
    "PromptConfig"
]