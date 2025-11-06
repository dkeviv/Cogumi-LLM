"""
Cost Tracker
============

This module provides real-time cost tracking for LLM API usage.
Essential for staying within budget during distillation process.

Features:
- Real-time cost calculation
- Per-model pricing tracking
- Budget warnings and limits
- Cost optimization recommendations
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single cost tracking entry."""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    batch_mode: bool
    request_id: Optional[str] = None


@dataclass
class CostSummary:
    """Cost summary for reporting."""
    total_cost: float
    total_tokens: int
    total_requests: int
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float]
    tokens_by_model: Dict[str, int]
    batch_savings: float
    time_period: Tuple[float, float]


class CostTracker:
    """
    Tracks API costs in real-time with budget management.
    
    Features:
    - Real-time cost calculation
    - Budget limits and warnings
    - Detailed cost breakdown
    - Batch processing savings tracking
    """
    
    def __init__(self, budget_limit: float = 1000.0):
        self.budget_limit = budget_limit
        self.cost_entries: List[CostEntry] = []
        self.session_start = time.time()
        
        # Provider pricing (per 1M tokens)
        self.pricing = {
            # Groq pricing
            "groq": {
                "llama-3.3-70b-versatile": {"input": 2.80, "output": 14.00, "batch_input": 1.40, "batch_output": 7.00},
                "llama-3.1-405b-reasoning": {"input": 2.80, "output": 14.00, "batch_input": 1.40, "batch_output": 7.00},
            },
            
            # OpenAI pricing  
            "openai": {
                "gpt-4": {"input": 30.00, "output": 60.00, "batch_input": 15.00, "batch_output": 30.00},
                "gpt-4-turbo": {"input": 10.00, "output": 30.00, "batch_input": 5.00, "batch_output": 15.00},
                "gpt-4o": {"input": 2.50, "output": 10.00, "batch_input": 1.25, "batch_output": 5.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60, "batch_input": 0.075, "batch_output": 0.30},
                "o1-preview": {"input": 15.00, "output": 60.00, "batch_input": 7.50, "batch_output": 30.00},
                "o1-mini": {"input": 3.00, "output": 12.00, "batch_input": 1.50, "batch_output": 6.00},
            },
            
            # Together.ai pricing
            "together": {
                "Qwen/Qwen2.5-Coder-32B-Instruct": {"input": 0.30, "output": 0.30, "batch_input": 0.15, "batch_output": 0.15},
                "meta-llama/Llama-3.1-405B-Instruct": {"input": 3.50, "output": 3.50, "batch_input": 1.75, "batch_output": 1.75},
                "microsoft/WizardLM-2-8x22B": {"input": 1.20, "output": 1.20, "batch_input": 0.60, "batch_output": 0.60},
            }
        }
        
        # Cost tracking file
        self.cost_file = Path("data/cost_tracking.jsonl")
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing entries if file exists
        self._load_existing_entries()
        
    def _load_existing_entries(self):
        """Load existing cost entries from file."""
        if self.cost_file.exists():
            try:
                with open(self.cost_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry_data = json.loads(line)
                            entry = CostEntry(**entry_data)
                            self.cost_entries.append(entry)
                            
                logger.info(f"Loaded {len(self.cost_entries)} existing cost entries")
                
            except Exception as e:
                logger.warning(f"Failed to load existing cost entries: {e}")
                
    def _save_entry(self, entry: CostEntry):
        """Save cost entry to file."""
        try:
            with open(self.cost_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save cost entry: {e}")
            
    def track_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        batch_mode: bool = False,
        request_id: Optional[str] = None
    ) -> CostEntry:
        """
        Track cost for an API call.
        
        Args:
            provider: API provider name
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            batch_mode: Whether batch API was used
            request_id: Optional request identifier
            
        Returns:
            CostEntry with calculated costs
        """
        
        # Get pricing for model
        model_pricing = self._get_model_pricing(provider, model)
        
        if not model_pricing:
            logger.warning(f"No pricing data for {provider}/{model}, using default")
            model_pricing = {"input": 5.0, "output": 15.0, "batch_input": 2.5, "batch_output": 7.5}
            
        # Calculate costs based on batch mode
        if batch_mode:
            input_rate = model_pricing.get("batch_input", model_pricing["input"] * 0.5)
            output_rate = model_pricing.get("batch_output", model_pricing["output"] * 0.5)
        else:
            input_rate = model_pricing["input"]
            output_rate = model_pricing["output"]
            
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        total_cost = input_cost + output_cost
        
        # Create cost entry
        entry = CostEntry(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            batch_mode=batch_mode,
            request_id=request_id
        )
        
        # Add to tracking
        self.cost_entries.append(entry)
        self._save_entry(entry)
        
        # Check budget limits
        self._check_budget_limits()
        
        return entry
        
    def _get_model_pricing(self, provider: str, model: str) -> Optional[Dict[str, float]]:
        """Get pricing information for a specific model."""
        if provider in self.pricing and model in self.pricing[provider]:
            return self.pricing[provider][model]
            
        # Try partial model name matching
        if provider in self.pricing:
            for price_model, pricing in self.pricing[provider].items():
                if model.lower() in price_model.lower() or price_model.lower() in model.lower():
                    return pricing
                    
        return None
        
    def _check_budget_limits(self):
        """Check if budget limits are being approached."""
        current_cost = self.get_total_cost()
        
        # Warning thresholds
        if current_cost > self.budget_limit * 0.9:
            logger.critical(f"🚨 BUDGET ALERT: ${current_cost:.2f} spent (90% of ${self.budget_limit} limit)")
        elif current_cost > self.budget_limit * 0.75:
            logger.warning(f"⚠️  Budget warning: ${current_cost:.2f} spent (75% of ${self.budget_limit} limit)")
        elif current_cost > self.budget_limit * 0.5:
            logger.info(f"💰 Budget checkpoint: ${current_cost:.2f} spent (50% of ${self.budget_limit} limit)")
            
    def get_total_cost(self, since: Optional[float] = None) -> float:
        """Get total cost spent."""
        entries = self.cost_entries
        if since:
            entries = [e for e in entries if e.timestamp >= since]
            
        return sum(entry.total_cost for entry in entries)
        
    def get_session_cost(self) -> float:
        """Get cost for current session."""
        return self.get_total_cost(since=self.session_start)
        
    def get_cost_summary(
        self,
        since: Optional[float] = None,
        until: Optional[float] = None
    ) -> CostSummary:
        """Get detailed cost summary."""
        
        # Filter entries by time period
        entries = self.cost_entries
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        if until:
            entries = [e for e in entries if e.timestamp <= until]
            
        if not entries:
            return CostSummary(
                total_cost=0.0,
                total_tokens=0,
                total_requests=0,
                cost_by_provider={},
                cost_by_model={},
                tokens_by_model={},
                batch_savings=0.0,
                time_period=(since or 0, until or time.time())
            )
            
        # Calculate totals
        total_cost = sum(e.total_cost for e in entries)
        total_tokens = sum(e.total_tokens for e in entries)
        total_requests = len(entries)
        
        # Group by provider
        cost_by_provider = {}
        for entry in entries:
            provider = entry.provider
            cost_by_provider[provider] = cost_by_provider.get(provider, 0) + entry.total_cost
            
        # Group by model
        cost_by_model = {}
        tokens_by_model = {}
        for entry in entries:
            model = f"{entry.provider}/{entry.model}"
            cost_by_model[model] = cost_by_model.get(model, 0) + entry.total_cost
            tokens_by_model[model] = tokens_by_model.get(model, 0) + entry.total_tokens
            
        # Calculate batch savings
        batch_entries = [e for e in entries if e.batch_mode]
        regular_entries = [e for e in entries if not e.batch_mode]
        
        # Estimate what batch entries would have cost at regular rates
        batch_savings = 0.0
        for entry in batch_entries:
            regular_pricing = self._get_model_pricing(entry.provider, entry.model)
            if regular_pricing:
                regular_input_cost = (entry.input_tokens / 1_000_000) * regular_pricing["input"]
                regular_output_cost = (entry.output_tokens / 1_000_000) * regular_pricing["output"]
                regular_total = regular_input_cost + regular_output_cost
                batch_savings += regular_total - entry.total_cost
                
        return CostSummary(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_requests=total_requests,
            cost_by_provider=cost_by_provider,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            batch_savings=batch_savings,
            time_period=(
                min(e.timestamp for e in entries),
                max(e.timestamp for e in entries)
            )
        )
        
    def estimate_remaining_budget(self) -> Dict[str, float]:
        """Estimate how much budget remains and what it can buy."""
        current_cost = self.get_total_cost()
        remaining = max(0, self.budget_limit - current_cost)
        
        # Estimate tokens remaining at average cost
        if len(self.cost_entries) > 0:
            avg_cost_per_token = sum(e.total_cost for e in self.cost_entries) / sum(e.total_tokens for e in self.cost_entries)
            estimated_tokens = remaining / avg_cost_per_token if avg_cost_per_token > 0 else 0
        else:
            estimated_tokens = 0
            
        return {
            "remaining_budget": remaining,
            "budget_used": current_cost,
            "budget_percentage": (current_cost / self.budget_limit) * 100,
            "estimated_tokens_remaining": estimated_tokens
        }
        
    def get_cost_optimization_tips(self) -> List[str]:
        """Get personalized cost optimization recommendations."""
        
        summary = self.get_cost_summary()
        tips = []
        
        # Check batch usage
        batch_entries = len([e for e in self.cost_entries if e.batch_mode])
        total_entries = len(self.cost_entries)
        
        if total_entries > 0:
            batch_percentage = (batch_entries / total_entries) * 100
            
            if batch_percentage < 50:
                tips.append(f"💡 Use batch API more! Only {batch_percentage:.1f}% of requests use batch mode. Potential 50% savings.")
                
        # Check most expensive models
        if summary.cost_by_model:
            most_expensive = max(summary.cost_by_model.items(), key=lambda x: x[1])
            model_name, model_cost = most_expensive
            
            if model_cost > summary.total_cost * 0.5:
                tips.append(f"💰 Model '{model_name}' accounts for ${model_cost:.2f} ({model_cost/summary.total_cost*100:.1f}% of costs)")
                
        # Check for expensive providers
        if "openai" in summary.cost_by_provider and summary.cost_by_provider["openai"] > summary.total_cost * 0.3:
            openai_cost = summary.cost_by_provider["openai"]
            tips.append(f"🔄 Consider more Groq usage. OpenAI: ${openai_cost:.2f} ({openai_cost/summary.total_cost*100:.1f}%)")
            
        # Budget velocity check
        session_cost = self.get_session_cost()
        session_hours = (time.time() - self.session_start) / 3600
        
        if session_hours > 0:
            hourly_rate = session_cost / session_hours
            hours_to_budget = (self.budget_limit - self.get_total_cost()) / hourly_rate if hourly_rate > 0 else float('inf')
            
            if hours_to_budget < 24:
                tips.append(f"⏰ At current rate (${hourly_rate:.2f}/hr), budget will be exhausted in {hours_to_budget:.1f} hours")
                
        # Batch savings achieved
        if summary.batch_savings > 0:
            tips.append(f"✅ Batch API saved ${summary.batch_savings:.2f} so far!")
            
        return tips
        
    def print_cost_dashboard(self):
        """Print a formatted cost dashboard."""
        
        summary = self.get_cost_summary()
        budget_info = self.estimate_remaining_budget()
        tips = self.get_cost_optimization_tips()
        
        print("\n" + "="*60)
        print("💰 COST TRACKING DASHBOARD")
        print("="*60)
        
        # Budget overview
        print(f"🏦 Budget: ${budget_info['budget_used']:.2f} / ${self.budget_limit:.2f} ({budget_info['budget_percentage']:.1f}%)")
        print(f"💳 Remaining: ${budget_info['remaining_budget']:.2f}")
        print(f"📊 Session Cost: ${self.get_session_cost():.2f}")
        
        # Usage stats
        print(f"\n📈 Usage Stats:")
        print(f"   • Total Requests: {summary.total_requests:,}")
        print(f"   • Total Tokens: {summary.total_tokens:,}")
        print(f"   • Avg Cost/Token: ${(summary.total_cost/summary.total_tokens)*1000:.4f} per 1K tokens")
        
        # Provider breakdown
        if summary.cost_by_provider:
            print(f"\n🔗 By Provider:")
            for provider, cost in sorted(summary.cost_by_provider.items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / summary.total_cost) * 100
                print(f"   • {provider}: ${cost:.2f} ({percentage:.1f}%)")
                
        # Top models
        if summary.cost_by_model:
            print(f"\n🤖 Top Models by Cost:")
            top_models = sorted(summary.cost_by_model.items(), key=lambda x: x[1], reverse=True)[:5]
            for model, cost in top_models:
                percentage = (cost / summary.total_cost) * 100
                tokens = summary.tokens_by_model.get(model, 0)
                print(f"   • {model}: ${cost:.2f} ({percentage:.1f}%) | {tokens:,} tokens")
                
        # Savings
        if summary.batch_savings > 0:
            print(f"\n💎 Batch Savings: ${summary.batch_savings:.2f}")
            
        # Optimization tips
        if tips:
            print(f"\n💡 Optimization Tips:")
            for tip in tips[:3]:  # Show top 3 tips
                print(f"   {tip}")
                
        print("="*60 + "\n")