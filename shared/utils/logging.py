"""
Logging Configuration
====================

This module provides structured logging for the distillation pipeline.
Ensures comprehensive tracking and debugging capabilities.

Features:
- Structured JSON logging
- Performance metrics tracking
- Cost tracking integration
- Rich console output for development
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sys

# Try to import rich for better console output
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TaskID
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    
    Outputs logs in JSON format for easy parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """
    Specialized logger for tracking performance metrics.
    
    Tracks timing, memory usage, and other performance indicators
    throughout the distillation pipeline.
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times: Dict[str, float] = {}
        
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
        self.logger.info(
            f"Started operation: {operation_name}",
            extra={
                'extra_fields': {
                    'operation': operation_name,
                    'event_type': 'operation_start',
                    'timestamp': time.time()
                }
            }
        )
        
    def end_operation(
        self,
        operation_name: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """End timing an operation and log the duration."""
        
        if operation_name not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation_name}")
            return 0.0
            
        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        
        log_data = {
            'operation': operation_name,
            'event_type': 'operation_end',
            'duration_seconds': duration,
            'success': success,
            'timestamp': time.time()
        }
        
        if metadata:
            log_data.update(metadata)
            
        level = logging.INFO if success else logging.ERROR
        message = f"Completed operation: {operation_name} in {duration:.2f}s"
        
        if not success:
            message = f"Failed operation: {operation_name} after {duration:.2f}s"
            
        self.logger.log(
            level,
            message,
            extra={'extra_fields': log_data}
        )
        
        return duration
        
    def log_metrics(
        self,
        operation: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Log performance metrics for an operation."""
        
        log_data = {
            'operation': operation,
            'event_type': 'metrics',
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        self.logger.info(
            f"Metrics for {operation}: {metrics}",
            extra={'extra_fields': log_data}
        )


class CostLogger:
    """
    Specialized logger for tracking API costs and usage.
    
    Integrates with the cost tracking system to provide
    detailed cost analytics and budget monitoring.
    """
    
    def __init__(self, logger_name: str = "cost_tracking"):
        self.logger = logging.getLogger(logger_name)
        
    def log_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        batch_mode: bool = False,
        request_id: Optional[str] = None
    ) -> None:
        """Log an API call with cost information."""
        
        log_data = {
            'event_type': 'api_call',
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'batch_mode': batch_mode,
            'request_id': request_id,
            'timestamp': time.time()
        }
        
        self.logger.info(
            f"API call to {provider}/{model}: ${cost:.4f} ({input_tokens + output_tokens} tokens)",
            extra={'extra_fields': log_data}
        )
        
    def log_budget_alert(
        self,
        current_spend: float,
        budget_limit: float,
        alert_type: str = "warning"
    ) -> None:
        """Log budget alerts and warnings."""
        
        percentage = (current_spend / budget_limit) * 100 if budget_limit > 0 else 0
        
        log_data = {
            'event_type': 'budget_alert',
            'alert_type': alert_type,
            'current_spend': current_spend,
            'budget_limit': budget_limit,
            'percentage_used': percentage,
            'remaining_budget': budget_limit - current_spend,
            'timestamp': time.time()
        }
        
        level = logging.WARNING if alert_type == "warning" else logging.CRITICAL
        
        self.logger.log(
            level,
            f"Budget {alert_type}: ${current_spend:.2f} / ${budget_limit:.2f} ({percentage:.1f}%)",
            extra={'extra_fields': log_data}
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_rich: bool = True,
    structured_logging: bool = True
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the distillation pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_rich: Whether to use rich console output
        structured_logging: Whether to use structured JSON logging
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Convert log level string to level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        console_format = "%(message)s"
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        
        if structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
    # Create specialized loggers
    loggers = {
        'main': logging.getLogger('cogumi_llm'),
        'performance': logging.getLogger('performance'),
        'cost_tracking': logging.getLogger('cost_tracking'),
        'data_processing': logging.getLogger('data_processing'),
        'model_training': logging.getLogger('model_training'),
        'api_calls': logging.getLogger('api_calls'),
        'validation': logging.getLogger('validation')
    }
    
    # Log setup completion
    setup_message = f"Logging configured - Level: {log_level}"
    if log_file:
        setup_message += f", File: {log_file}"
    if enable_rich and RICH_AVAILABLE:
        setup_message += ", Rich output enabled"
    if structured_logging and log_file:
        setup_message += ", Structured JSON logging enabled"
        
    loggers['main'].info(setup_message)
    
    return loggers


def create_progress_tracker(description: str = "Processing") -> Optional[Progress]:
    """Create a Rich progress tracker if available."""
    
    if not RICH_AVAILABLE:
        return None
        
    progress = Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        "[progress.bar]",
        "[progress.remaining]{task.remaining}",
        console=Console(stderr=True),
        transient=False
    )
    
    return progress


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging."""
    
    import platform
    import psutil
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2)
    }
    
    logger.info(
        f"System info: {system_info}",
        extra={'extra_fields': {'event_type': 'system_info', **system_info}}
    )


def log_pipeline_phase(
    logger: logging.Logger,
    phase_name: str,
    phase_config: Dict[str, Any],
    start: bool = True
) -> None:
    """Log the start or end of a pipeline phase."""
    
    event_type = 'phase_start' if start else 'phase_end'
    action = 'Starting' if start else 'Completed'
    
    log_data = {
        'event_type': event_type,
        'phase_name': phase_name,
        'phase_config': phase_config,
        'timestamp': time.time()
    }
    
    logger.info(
        f"{action} pipeline phase: {phase_name}",
        extra={'extra_fields': log_data}
    )


# Pre-configured logger instances for convenience
def get_logger(name: str) -> logging.Logger:
    """Get a logger with standard configuration."""
    return logging.getLogger(name)


def get_performance_logger() -> PerformanceLogger:
    """Get a performance logger instance."""
    return PerformanceLogger()


def get_cost_logger() -> CostLogger:
    """Get a cost logger instance."""
    return CostLogger()


# Example usage patterns for documentation
if __name__ == "__main__":
    # Setup logging
    loggers = setup_logging(
        log_level="INFO",
        log_file="logs/pipeline.log",
        enable_rich=True,
        structured_logging=True
    )
    
    # Example usage
    main_logger = loggers['main']
    perf_logger = get_performance_logger()
    cost_logger = get_cost_logger()
    
    # Log system info
    log_system_info(main_logger)
    
    # Performance tracking example
    perf_logger.start_operation("data_loading")
    time.sleep(1)  # Simulate work
    perf_logger.end_operation("data_loading", success=True, metadata={'items_processed': 1000})
    
    # Cost tracking example
    cost_logger.log_api_call(
        provider="openai",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        cost=0.003,
        batch_mode=False
    )
    
    # Pipeline phase example
    log_pipeline_phase(
        main_logger,
        "Phase 1: Data Distillation",
        {"model": "gpt-4", "batch_size": 100},
        start=True
    )