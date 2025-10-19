"""
Utilities Package for LLM Distillation Pipeline
==============================================

This package provides utility modules for the LLM distillation pipeline:
- batch_api.py: Batch API management for 50% cost savings
- cost_tracker.py: Real-time cost tracking and budgeting
- deduplication.py: MinHash LSH for data deduplication
- validation.py: JSON schema validation for data
- logging.py: Structured logging for the pipeline
"""

from .batch_api import BatchAPIManager
from .cost_tracker import CostTracker
from .deduplication import MinHashLSH, DeduplicationConfig, DataDeduplicator
from .validation import InputDataValidator, ModelOutputValidator, ValidationResult
from .logging import setup_logging

__all__ = [
    "BatchAPIManager",
    "CostTracker", 
    "MinHashLSH",
    "DeduplicationConfig",
    "DataDeduplicator",
    "InputDataValidator",
    "ModelOutputValidator",
    "ValidationResult",
    "setup_logging"
]