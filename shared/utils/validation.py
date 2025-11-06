"""
Validation Framework
====================

This module provides comprehensive validation for the distillation pipeline.
Ensures data quality and model performance throughout the process.

Features:
- Input data validation
- Model output validation
- Training metrics validation
- Performance benchmarking
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    total_samples: int
    valid_samples: int
    avg_length: float
    min_length: int
    max_length: int
    language_distribution: Dict[str, float]
    encoding_issues: int
    duplicate_count: int
    quality_score: float


class InputDataValidator:
    """
    Validates input data for training and distillation.
    
    Checks for data quality, format consistency, and potential issues
    that could affect training performance.
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 8192):
        self.min_length = min_length
        self.max_length = max_length
        
    def validate_text_quality(self, text: str) -> ValidationResult:
        """Validate the quality of a single text sample."""
        
        errors = []
        warnings = []
        details = {}
        
        # Basic checks
        if not text or not text.strip():
            errors.append("Empty text")
            return ValidationResult(False, 0.0, errors, warnings, details)
            
        text = text.strip()
        details['length'] = len(text)
        details['word_count'] = len(text.split())
        
        # Length validation
        if len(text) < self.min_length:
            errors.append(f"Text too short: {len(text)} < {self.min_length}")
        elif len(text) > self.max_length:
            warnings.append(f"Text very long: {len(text)} > {self.max_length}")
            
        # Character encoding validation
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Invalid Unicode characters")
            
        # Language detection (basic)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        details['ascii_ratio'] = ascii_ratio
        
        if ascii_ratio < 0.7:
            warnings.append(f"Low ASCII ratio: {ascii_ratio:.2f} (possibly non-English)")
            
        # Content quality checks
        word_count = len(text.split())
        details['avg_word_length'] = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Check for repeated patterns
        lines = text.split('\n')
        if len(lines) > 10:
            repeated_lines = len(lines) - len(set(lines))
            if repeated_lines > len(lines) * 0.3:
                warnings.append(f"Many repeated lines: {repeated_lines}/{len(lines)}")
                
        # Calculate quality score
        score = 1.0
        
        # Penalize errors
        score -= len(errors) * 0.3
        
        # Penalize warnings
        score -= len(warnings) * 0.1
        
        # Bonus for good characteristics
        if 100 <= len(text) <= 2000:  # Good length range
            score += 0.1
            
        if 0.8 <= ascii_ratio <= 1.0:  # Good English content
            score += 0.1
            
        if 3 <= details['avg_word_length'] <= 7:  # Reasonable word lengths
            score += 0.1
            
        score = max(0.0, min(1.0, score))
        
        is_valid = len(errors) == 0 and score >= 0.5
        
        return ValidationResult(is_valid, score, errors, warnings, details)
        
    def validate_dataset(self, texts: List[str]) -> Tuple[DataQualityMetrics, List[ValidationResult]]:
        """Validate an entire dataset."""
        
        logger.info(f"Validating dataset with {len(texts)} samples")
        
        results = []
        valid_count = 0
        total_length = 0
        min_length = float('inf')
        max_length = 0
        encoding_issues = 0
        
        # Validate each text
        for i, text in enumerate(texts):
            result = self.validate_text_quality(text)
            results.append(result)
            
            if result.is_valid:
                valid_count += 1
                
            length = result.details.get('length', 0)
            total_length += length
            min_length = min(min_length, length)
            max_length = max(max_length, length)
            
            if any('encoding' in error.lower() for error in result.errors):
                encoding_issues += 1
                
            if i % 1000 == 0 and i > 0:
                logger.info(f"Validated {i}/{len(texts)} samples")
                
        # Calculate metrics
        avg_length = total_length / len(texts) if texts else 0
        quality_scores = [r.score for r in results]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Language distribution (basic)
        language_dist = {"english": 0.9, "other": 0.1}  # Simplified for now
        
        # Estimate duplicates (rough)
        duplicate_count = len(texts) - len(set(texts))
        
        metrics = DataQualityMetrics(
            total_samples=len(texts),
            valid_samples=valid_count,
            avg_length=avg_length,
            min_length=int(min_length) if min_length != float('inf') else 0,
            max_length=max_length,
            language_distribution=language_dist,
            encoding_issues=encoding_issues,
            duplicate_count=duplicate_count,
            quality_score=overall_quality
        )
        
        logger.info(f"Dataset validation complete: {valid_count}/{len(texts)} valid samples")
        
        return metrics, results


class ModelOutputValidator:
    """
    Validates model outputs for quality and consistency.
    
    Ensures generated text meets quality standards and follows
    expected patterns for training data.
    """
    
    def __init__(self):
        self.quality_patterns = {
            'has_content': re.compile(r'\w+'),
            'proper_sentences': re.compile(r'[.!?]\s*[A-Z]'),
            'balanced_punctuation': re.compile(r'[.!?]'),
            'reasonable_capitalization': re.compile(r'[A-Z]'),
        }
        
    def validate_response_format(self, response: str) -> ValidationResult:
        """Validate the format and structure of a model response."""
        
        errors = []
        warnings = []
        details = {}
        
        if not response or not response.strip():
            errors.append("Empty response")
            return ValidationResult(False, 0.0, errors, warnings, details)
            
        response = response.strip()
        details['length'] = len(response)
        
        # Check for minimum content
        if len(response) < 10:
            errors.append("Response too short")
            
        # Check for garbled output
        if len(response) > 0:
            non_printable = sum(1 for c in response if ord(c) < 32 and c not in '\n\t\r')
            if non_printable > len(response) * 0.05:
                errors.append("Too many non-printable characters")
                
        # Check for repetitive content
        words = response.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            details['repetition_ratio'] = repetition_ratio
            
            if repetition_ratio > 0.7:
                errors.append("Highly repetitive content")
            elif repetition_ratio > 0.5:
                warnings.append("Somewhat repetitive content")
                
        # Check for proper formatting
        sentence_count = len(re.findall(r'[.!?]', response))
        details['sentence_count'] = sentence_count
        
        if len(words) > 20 and sentence_count == 0:
            warnings.append("No sentence-ending punctuation")
            
        # Calculate quality score
        score = 1.0
        
        # Apply pattern checks
        for pattern_name, pattern in self.quality_patterns.items():
            matches = len(pattern.findall(response))
            details[f'{pattern_name}_count'] = matches
            
            if pattern_name == 'has_content' and matches == 0:
                score -= 0.5
            elif pattern_name == 'proper_sentences' and len(words) > 20 and matches == 0:
                score -= 0.2
                
        # Penalize errors and warnings
        score -= len(errors) * 0.3
        score -= len(warnings) * 0.1
        
        score = max(0.0, min(1.0, score))
        is_valid = len(errors) == 0 and score >= 0.6
        
        return ValidationResult(is_valid, score, errors, warnings, details)
        
    def validate_consistency(self, responses: List[str], prompts: List[str]) -> ValidationResult:
        """Validate consistency across multiple responses."""
        
        errors = []
        warnings = []
        details = {}
        
        if len(responses) != len(prompts):
            errors.append("Mismatch between responses and prompts count")
            return ValidationResult(False, 0.0, errors, warnings, details)
            
        # Validate each response
        response_scores = []
        for response in responses:
            result = self.validate_response_format(response)
            response_scores.append(result.score)
            
        details['individual_scores'] = response_scores
        details['avg_score'] = sum(response_scores) / len(response_scores) if response_scores else 0
        details['min_score'] = min(response_scores) if response_scores else 0
        details['score_variance'] = self._calculate_variance(response_scores)
        
        # Check for consistency issues
        if details['score_variance'] > 0.2:
            warnings.append("High variance in response quality")
            
        if details['min_score'] < 0.3:
            warnings.append("Some very low quality responses")
            
        # Check response length consistency
        lengths = [len(r) for r in responses]
        length_variance = self._calculate_variance(lengths)
        details['length_variance'] = length_variance
        
        if length_variance > 1000:  # High variance in length
            warnings.append("Inconsistent response lengths")
            
        score = details['avg_score']
        
        # Penalize high variance
        if details['score_variance'] > 0.3:
            score -= 0.2
            
        score = max(0.0, min(1.0, score))
        is_valid = len(errors) == 0 and score >= 0.7
        
        return ValidationResult(is_valid, score, errors, warnings, details)
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


class PerformanceValidator:
    """
    Validates model performance against benchmarks and targets.
    
    Tracks performance metrics and ensures models meet quality thresholds
    before proceeding to next training phase.
    """
    
    def __init__(self, target_accuracy: float = 0.95):
        self.target_accuracy = target_accuracy
        self.benchmark_tasks = [
            "math_problem_solving",
            "code_generation", 
            "reasoning",
            "factual_qa",
            "creative_writing"
        ]
        
    def validate_training_metrics(self, metrics: Dict[str, float]) -> ValidationResult:
        """Validate training metrics against targets."""
        
        errors = []
        warnings = []
        details = metrics.copy()
        
        required_metrics = ['loss', 'accuracy', 'perplexity']
        
        # Check for required metrics
        for metric in required_metrics:
            if metric not in metrics:
                errors.append(f"Missing required metric: {metric}")
                
        if errors:
            return ValidationResult(False, 0.0, errors, warnings, details)
            
        # Validate metric values
        loss = metrics.get('loss', float('inf'))
        accuracy = metrics.get('accuracy', 0.0)
        perplexity = metrics.get('perplexity', float('inf'))
        
        # Loss validation
        if loss > 10.0:
            errors.append(f"Loss too high: {loss}")
        elif loss > 5.0:
            warnings.append(f"Loss concerning: {loss}")
            
        # Accuracy validation
        if accuracy < 0.5:
            errors.append(f"Accuracy too low: {accuracy}")
        elif accuracy < self.target_accuracy:
            warnings.append(f"Accuracy below target: {accuracy} < {self.target_accuracy}")
            
        # Perplexity validation
        if perplexity > 50.0:
            errors.append(f"Perplexity too high: {perplexity}")
        elif perplexity > 20.0:
            warnings.append(f"Perplexity concerning: {perplexity}")
            
        # Calculate overall score
        score = accuracy  # Base score on accuracy
        
        # Penalize high loss
        if loss < 1.0:
            score += 0.1
        elif loss > 5.0:
            score -= 0.2
            
        # Penalize high perplexity
        if perplexity < 5.0:
            score += 0.1
        elif perplexity > 20.0:
            score -= 0.2
            
        score = max(0.0, min(1.0, score))
        is_valid = len(errors) == 0 and score >= 0.8
        
        return ValidationResult(is_valid, score, errors, warnings, details)
        
    def validate_compression_quality(
        self,
        original_size_mb: float,
        compressed_size_mb: float,
        performance_retention: float
    ) -> ValidationResult:
        """Validate model compression results."""
        
        errors = []
        warnings = []
        details = {
            'original_size_mb': original_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'compression_ratio': original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0,
            'performance_retention': performance_retention,
            'size_reduction_percent': ((original_size_mb - compressed_size_mb) / original_size_mb * 100) if original_size_mb > 0 else 0
        }
        
        # Target: 480MB or less
        target_size = 480.0
        
        if compressed_size_mb > target_size:
            errors.append(f"Compressed size {compressed_size_mb}MB exceeds target {target_size}MB")
            
        # Performance retention should be high
        if performance_retention < 0.95:
            warnings.append(f"Performance retention low: {performance_retention:.3f}")
            
        if performance_retention < 0.90:
            errors.append(f"Performance retention too low: {performance_retention:.3f}")
            
        # Compression ratio validation
        min_compression_ratio = 10.0  # At least 10x compression
        if details['compression_ratio'] < min_compression_ratio:
            warnings.append(f"Low compression ratio: {details['compression_ratio']:.1f}x")
            
        # Calculate score
        score = performance_retention  # Base on performance retention
        
        # Bonus for good compression
        if compressed_size_mb <= target_size:
            score += 0.1
            
        if details['compression_ratio'] >= min_compression_ratio:
            score += 0.1
            
        score = max(0.0, min(1.0, score))
        is_valid = len(errors) == 0 and score >= 0.85
        
        return ValidationResult(is_valid, score, errors, warnings, details)


def validate_pipeline_data(
    input_texts: List[str],
    output_texts: List[str],
    prompts: List[str]
) -> Dict[str, ValidationResult]:
    """
    Comprehensive validation for pipeline data.
    
    Args:
        input_texts: Original input texts
        output_texts: Generated/processed texts
        prompts: Prompts used for generation
        
    Returns:
        Dictionary of validation results for each component
    """
    
    results = {}
    
    # Input data validation
    input_validator = InputDataValidator()
    input_metrics, input_results = input_validator.validate_dataset(input_texts)
    
    results['input_data'] = ValidationResult(
        is_valid=input_metrics.quality_score >= 0.7,
        score=input_metrics.quality_score,
        errors=[],
        warnings=[],
        details={'metrics': input_metrics, 'individual_results': input_results}
    )
    
    # Output validation
    output_validator = ModelOutputValidator()
    
    # Validate individual outputs
    output_scores = []
    for output in output_texts:
        result = output_validator.validate_response_format(output)
        output_scores.append(result.score)
        
    # Validate consistency
    consistency_result = output_validator.validate_consistency(output_texts, prompts)
    
    results['output_quality'] = ValidationResult(
        is_valid=consistency_result.is_valid,
        score=consistency_result.score,
        errors=consistency_result.errors,
        warnings=consistency_result.warnings,
        details={'individual_scores': output_scores, 'consistency': consistency_result.details}
    )
    
    return results