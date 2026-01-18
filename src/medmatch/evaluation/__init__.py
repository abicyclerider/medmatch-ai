"""
Evaluation module for patient entity resolution.

This module provides tools for evaluating matching performance:
- EvaluationMetrics: Precision, recall, F1, accuracy calculations
- MatchEvaluator: Load ground truth and evaluate results

Phase 2.5 of the entity resolution system.
"""

from .metrics import EvaluationMetrics, MatchEvaluator

__all__ = [
    "EvaluationMetrics",
    "MatchEvaluator",
]
