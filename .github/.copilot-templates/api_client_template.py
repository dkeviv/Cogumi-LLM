"""
Template for API client wrappers.
Use this pattern for: Groq, OpenAI, Together.ai clients

CONTEXT:
- Always use batch API for 50% discount
- Track costs for every API call
- Handle retries with exponential backoff
- Support async operations for parallelization
"""

from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Structured API response."""
    content: str
    tokens_used: int
    cost_usd: float
    model: str
    latency_ms: float


class CostTracker:
    """Track API costs across all calls."""
    
    def __init__(self):
        self.total_cost: float = 0.0
        self.calls: List[Dict[str, Any]] = []
    
    def add(self, cost: float, model: str, tokens: int) -> None:
        """Add a cost entry."""
        self.total_cost += cost
        self.calls.append({
            "cost": cost,
            "model": model,
            "tokens": tokens
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": self.total_cost,
            "total_calls": len(self.calls),
            "total_tokens": sum(c["tokens"] for c in self.calls)
        }


class APIClient:
    """Base API client template with cost tracking and batch support."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        use_batch: bool = True,
        cost_per_1k_input: float = 1.40,
        cost_per_1k_output: float = 7.00
    ):
        """
        Initialize API client.
        
        Args:
            api_key: API key for authentication
            model_name: Model identifier (e.g., "llama-3.1-405b")
            use_batch: Always True for 50% discount
            cost_per_1k_input: Cost per 1K input tokens
            cost_per_1k_output: Cost per 1K output tokens
        """
        self.api_key = api_key
        self.model_name = model_name
        self.use_batch = use_batch
        self.batch_discount = 0.5 if use_batch else 1.0
        self.cost_per_1k_input = cost_per_1k_input * self.batch_discount
        self.cost_per_1k_output = cost_per_1k_output * self.batch_discount
        self.cost_tracker = CostTracker()
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost before making API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Expected output tokens
            
        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> APIResponse:
        """
        Generate completion with cost tracking.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            
        Returns:
            APIResponse with content and cost info
        """
        # Estimate cost BEFORE calling (using tiktoken in real implementation)
        estimated_input_tokens = len(prompt) // 4  # Rough estimate
        estimated_cost = self.estimate_cost(estimated_input_tokens, max_tokens)
        
        logger.info(
            f"Estimated cost: ${estimated_cost:.4f} "
            f"(batch discount: {int(self.batch_discount * 100)}%)"
        )
        
        # Make API call (batch or immediate)
        if self.use_batch:
            response = await self._batch_generate(prompt, max_tokens, temperature)
        else:
            response = await self._immediate_generate(prompt, max_tokens, temperature)
        
        # Track actual cost
        actual_cost = self.estimate_cost(
            response["tokens_input"],
            response["tokens_output"]
        )
        self.cost_tracker.add(
            actual_cost,
            self.model_name,
            response["tokens_input"] + response["tokens_output"]
        )
        
        logger.info(f"Actual cost: ${actual_cost:.4f}")
        
        return APIResponse(
            content=response["content"],
            tokens_used=response["tokens_input"] + response["tokens_output"],
            cost_usd=actual_cost,
            model=self.model_name,
            latency_ms=response["latency_ms"]
        )
    
    async def _batch_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Generate using batch API (50% discount).
        
        Override this in subclass for specific API.
        """
        raise NotImplementedError("Implement in subclass (GroqClient, OpenAIClient)")
    
    async def _immediate_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Generate using immediate API (full price).
        
        Override this in subclass for specific API.
        """
        raise NotImplementedError("Implement in subclass (GroqClient, OpenAIClient)")
    
    async def batch_generate_multiple(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> List[APIResponse]:
        """
        Generate multiple completions in parallel with batch API.
        
        Args:
            prompts: List of prompts to process
            max_tokens: Max tokens per completion
            temperature: Sampling temperature
            
        Returns:
            List of APIResponse objects
        """
        # CRITICAL: Always use batch API for multiple requests
        assert self.use_batch, "Must use batch API for multiple requests!"
        
        # Parallel execution
        tasks = [
            self.generate(prompt, max_tokens, temperature)
            for prompt in prompts
        ]
        
        logger.info(f"Generating {len(prompts)} completions in parallel (batch API)")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any failures
        successful = [r for r in responses if not isinstance(r, Exception)]
        failed = [r for r in responses if isinstance(r, Exception)]
        
        if failed:
            logger.warning(f"{len(failed)} requests failed, {len(successful)} succeeded")
        
        return successful
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of all costs incurred."""
        return self.cost_tracker.get_summary()


# Example usage in actual implementation:
"""
from api_client_template import APIClient, APIResponse

class GroqClient(APIClient):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            model_name="llama-3.1-405b",
            use_batch=True,  # ALWAYS True
            cost_per_1k_input=2.80,  # Before discount
            cost_per_1k_output=14.00  # Before discount
        )
        self.client = Groq(api_key=api_key)
    
    async def _batch_generate(self, prompt, max_tokens, temperature):
        # Implement Groq-specific batch API call
        batch = await self.client.batches.create(...)
        return {
            "content": batch.output,
            "tokens_input": batch.usage.input_tokens,
            "tokens_output": batch.usage.output_tokens,
            "latency_ms": batch.latency_ms
        }
    
    async def _immediate_generate(self, prompt, max_tokens, temperature):
        # Should rarely use this (2x cost!)
        response = await self.client.chat.completions.create(...)
        return {
            "content": response.choices[0].message.content,
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
            "latency_ms": response.latency_ms
        }


# Usage:
client = GroqClient(api_key="your_key")

# Single generation
response = await client.generate("Explain photosynthesis", max_tokens=500)
print(f"Cost: ${response.cost_usd:.4f}")

# Batch generation (parallel)
prompts = ["Question 1", "Question 2", "Question 3"]
responses = await client.batch_generate_multiple(prompts)
print(f"Total cost: ${sum(r.cost_usd for r in responses):.2f}")

# Cost summary
summary = client.get_cost_summary()
print(f"Total spent: ${summary['total_cost_usd']:.2f}")
"""