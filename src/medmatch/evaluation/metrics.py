"""
Evaluation metrics for patient matching performance.

This module provides tools for evaluating entity resolution accuracy:
- EvaluationMetrics: Holds confusion matrix and calculated metrics
- MatchEvaluator: Loads ground truth and evaluates match results

Phase 2.5 of the entity resolution system.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from medmatch.matching.core import MatchResult


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for matching performance.

    Holds confusion matrix values and calculates derived metrics.

    Attributes:
        true_positives: Correctly identified matches
        true_negatives: Correctly identified non-matches
        false_positives: Non-matches incorrectly identified as matches
        false_negatives: Matches incorrectly identified as non-matches
        total_pairs: Total number of pairs evaluated
        difficulty: Optional difficulty level these metrics apply to
    """

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_pairs: int = 0
    difficulty: Optional[str] = None

    @property
    def precision(self) -> float:
        """
        Precision: TP / (TP + FP).

        Measures what fraction of predicted matches are correct.
        Returns 1.0 if no positive predictions made.
        """
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 1.0

    @property
    def recall(self) -> float:
        """
        Recall: TP / (TP + FN).

        Measures what fraction of actual matches were found.
        Returns 1.0 if no actual matches exist.
        """
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 1.0

    @property
    def f1_score(self) -> float:
        """
        F1 Score: 2 * (precision * recall) / (precision + recall).

        Harmonic mean of precision and recall.
        Returns 0.0 if both precision and recall are 0.
        """
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """
        Accuracy: (TP + TN) / total.

        Overall correctness of predictions.
        Returns 0.0 if no pairs evaluated.
        """
        total = self.total_pairs
        if total == 0:
            return 0.0
        correct = self.true_positives + self.true_negatives
        return correct / total

    @property
    def specificity(self) -> float:
        """
        Specificity: TN / (TN + FP).

        True negative rate - fraction of actual non-matches correctly identified.
        Returns 1.0 if no actual non-matches exist.
        """
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for export."""
        return {
            'difficulty': self.difficulty,
            'total_pairs': self.total_pairs,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'accuracy': round(self.accuracy, 4),
            'specificity': round(self.specificity, 4),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        header = f"Metrics ({self.difficulty})" if self.difficulty else "Metrics"
        return (
            f"{header}:\n"
            f"  Total Pairs: {self.total_pairs}\n"
            f"  TP: {self.true_positives}, TN: {self.true_negatives}, "
            f"FP: {self.false_positives}, FN: {self.false_negatives}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1 Score: {self.f1_score:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}"
        )


@dataclass
class ErrorCase:
    """
    Represents a matching error for analysis.

    Attributes:
        record_1_id: First record ID
        record_2_id: Second record ID
        error_type: 'false_positive' or 'false_negative'
        predicted: What the matcher predicted
        actual: What the ground truth says
        confidence: Matcher's confidence score
        stage: Which pipeline stage made the decision
        difficulty: Difficulty level of the pair
        explanation: Matcher's explanation
    """
    record_1_id: str
    record_2_id: str
    error_type: str  # 'false_positive' or 'false_negative'
    predicted: bool
    actual: bool
    confidence: float
    stage: str
    difficulty: Optional[str] = None
    explanation: str = ""


class MatchEvaluator:
    """
    Evaluates matching results against ground truth.

    Loads ground truth from CSV and provides methods to evaluate
    MatchResult objects against it.

    Ground Truth CSV Format:
        Required columns:
        - record_id: Unique record identifier
        - patient_id: Ground truth patient identifier
        - match_group: Records with same match_group should match
        - difficulty: Difficulty level (easy/medium/hard/ambiguous)

        Records with the same match_group or patient_id are considered
        matches. All other pairs are considered non-matches.

    Example:
        >>> evaluator = MatchEvaluator("data/synthetic/ground_truth.csv")
        >>> metrics = evaluator.evaluate(results)
        >>> print(f"Accuracy: {metrics.accuracy:.2%}")

        >>> by_difficulty = evaluator.evaluate_by_difficulty(results)
        >>> for diff, m in by_difficulty.items():
        ...     print(f"{diff}: {m.accuracy:.2%}")

        >>> errors = evaluator.find_errors(results)
        >>> false_positives = [e for e in errors if e.error_type == 'false_positive']
        >>> print(f"False positives: {len(false_positives)}")
    """

    def __init__(self, ground_truth_path: str):
        """
        Initialize evaluator with ground truth file.

        Args:
            ground_truth_path: Path to ground_truth.csv

        Raises:
            FileNotFoundError: If ground truth file doesn't exist
        """
        self.ground_truth_path = Path(ground_truth_path)
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {ground_truth_path}"
            )

        # Load ground truth data
        self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        """Load and index ground truth data."""
        # record_id -> row data
        self.record_data: Dict[str, Dict[str, Any]] = {}

        # match_group -> set of record_ids
        self.match_groups: Dict[str, Set[str]] = defaultdict(set)

        # patient_id -> set of record_ids (alternative grouping)
        self.patient_records: Dict[str, Set[str]] = defaultdict(set)

        # record_id -> difficulty
        self.difficulties: Dict[str, str] = {}

        with open(self.ground_truth_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_id = row['record_id']
                patient_id = row['patient_id']
                match_group = row['match_group']
                difficulty = row.get('difficulty', 'unknown')

                self.record_data[record_id] = row
                self.match_groups[match_group].add(record_id)
                self.patient_records[patient_id].add(record_id)
                self.difficulties[record_id] = difficulty

    def should_match(self, record_1_id: str, record_2_id: str) -> bool:
        """
        Check if two records should match according to ground truth.

        Two records should match if they belong to the same patient
        (have the same patient_id).

        Args:
            record_1_id: First record ID
            record_2_id: Second record ID

        Returns:
            True if records are from the same patient

        Raises:
            KeyError: If either record ID is not in ground truth
        """
        if record_1_id not in self.record_data:
            raise KeyError(f"Record {record_1_id} not in ground truth")
        if record_2_id not in self.record_data:
            raise KeyError(f"Record {record_2_id} not in ground truth")

        patient_1 = self.record_data[record_1_id]['patient_id']
        patient_2 = self.record_data[record_2_id]['patient_id']

        return patient_1 == patient_2

    def get_pair_difficulty(
        self,
        record_1_id: str,
        record_2_id: str,
    ) -> str:
        """
        Get difficulty level for a record pair.

        Uses the harder difficulty of the two records.

        Args:
            record_1_id: First record ID
            record_2_id: Second record ID

        Returns:
            Difficulty level ('easy', 'medium', 'hard', 'ambiguous')
        """
        difficulty_order = ['easy', 'medium', 'hard', 'ambiguous']

        d1 = self.difficulties.get(record_1_id, 'unknown')
        d2 = self.difficulties.get(record_2_id, 'unknown')

        # Return the harder difficulty
        try:
            idx1 = difficulty_order.index(d1) if d1 in difficulty_order else 4
            idx2 = difficulty_order.index(d2) if d2 in difficulty_order else 4
            max_idx = max(idx1, idx2)
            return difficulty_order[max_idx] if max_idx < 4 else d1
        except (ValueError, IndexError):
            return d1

    def get_all_true_match_pairs(self) -> Set[Tuple[str, str]]:
        """
        Get all pairs of records that should match.

        Returns:
            Set of (record_id_1, record_id_2) tuples where record_id_1 < record_id_2
        """
        pairs = set()
        for patient_id, record_ids in self.patient_records.items():
            record_list = sorted(record_ids)
            for i, r1 in enumerate(record_list):
                for r2 in record_list[i+1:]:
                    pairs.add((r1, r2))
        return pairs

    def evaluate(self, results: List[MatchResult]) -> EvaluationMetrics:
        """
        Evaluate match results against ground truth.

        Args:
            results: List of MatchResult objects from the matcher

        Returns:
            EvaluationMetrics with confusion matrix and calculated metrics
        """
        metrics = EvaluationMetrics(total_pairs=len(results))

        for result in results:
            predicted = result.is_match
            actual = self.should_match(result.record_1_id, result.record_2_id)

            if predicted and actual:
                metrics.true_positives += 1
            elif not predicted and not actual:
                metrics.true_negatives += 1
            elif predicted and not actual:
                metrics.false_positives += 1
            else:  # not predicted and actual
                metrics.false_negatives += 1

        return metrics

    def evaluate_by_difficulty(
        self,
        results: List[MatchResult],
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate results broken down by difficulty level.

        Args:
            results: List of MatchResult objects

        Returns:
            Dictionary mapping difficulty -> EvaluationMetrics
        """
        # Group results by difficulty
        by_difficulty: Dict[str, List[MatchResult]] = defaultdict(list)

        for result in results:
            difficulty = self.get_pair_difficulty(
                result.record_1_id,
                result.record_2_id
            )
            by_difficulty[difficulty].append(result)

        # Calculate metrics for each difficulty
        metrics_by_difficulty = {}
        for difficulty, diff_results in by_difficulty.items():
            metrics = self.evaluate(diff_results)
            metrics.difficulty = difficulty
            metrics_by_difficulty[difficulty] = metrics

        return metrics_by_difficulty

    def evaluate_by_stage(
        self,
        results: List[MatchResult],
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate results broken down by decision stage.

        Args:
            results: List of MatchResult objects

        Returns:
            Dictionary mapping stage -> EvaluationMetrics
        """
        by_stage: Dict[str, List[MatchResult]] = defaultdict(list)

        for result in results:
            by_stage[result.stage].append(result)

        metrics_by_stage = {}
        for stage, stage_results in by_stage.items():
            metrics = self.evaluate(stage_results)
            metrics.difficulty = stage  # Reuse difficulty field for stage label
            metrics_by_stage[stage] = metrics

        return metrics_by_stage

    def find_errors(
        self,
        results: List[MatchResult],
    ) -> List[ErrorCase]:
        """
        Find all false positives and false negatives.

        Args:
            results: List of MatchResult objects

        Returns:
            List of ErrorCase objects for analysis
        """
        errors = []

        for result in results:
            predicted = result.is_match
            actual = self.should_match(result.record_1_id, result.record_2_id)

            if predicted != actual:
                error_type = 'false_positive' if predicted else 'false_negative'
                difficulty = self.get_pair_difficulty(
                    result.record_1_id,
                    result.record_2_id
                )

                errors.append(ErrorCase(
                    record_1_id=result.record_1_id,
                    record_2_id=result.record_2_id,
                    error_type=error_type,
                    predicted=predicted,
                    actual=actual,
                    confidence=result.confidence,
                    stage=result.stage,
                    difficulty=difficulty,
                    explanation=result.explanation,
                ))

        return errors

    def generate_report(
        self,
        results: List[MatchResult],
        verbose: bool = False,
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: List of MatchResult objects
            verbose: Include detailed error analysis

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ENTITY RESOLUTION EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall metrics
        overall = self.evaluate(results)
        lines.append("OVERALL METRICS")
        lines.append("-" * 40)
        lines.append(f"Total Pairs Evaluated: {overall.total_pairs}")
        lines.append(f"True Positives:  {overall.true_positives}")
        lines.append(f"True Negatives:  {overall.true_negatives}")
        lines.append(f"False Positives: {overall.false_positives}")
        lines.append(f"False Negatives: {overall.false_negatives}")
        lines.append("")
        lines.append(f"Precision: {overall.precision:.4f} ({overall.precision:.2%})")
        lines.append(f"Recall:    {overall.recall:.4f} ({overall.recall:.2%})")
        lines.append(f"F1 Score:  {overall.f1_score:.4f} ({overall.f1_score:.2%})")
        lines.append(f"Accuracy:  {overall.accuracy:.4f} ({overall.accuracy:.2%})")
        lines.append("")

        # By difficulty
        lines.append("METRICS BY DIFFICULTY")
        lines.append("-" * 40)
        by_diff = self.evaluate_by_difficulty(results)
        difficulty_order = ['easy', 'medium', 'hard', 'ambiguous']

        for diff in difficulty_order:
            if diff in by_diff:
                m = by_diff[diff]
                lines.append(f"{diff.upper()}:")
                lines.append(f"  Pairs: {m.total_pairs}")
                lines.append(f"  Accuracy: {m.accuracy:.4f} ({m.accuracy:.2%})")
                lines.append(f"  Precision: {m.precision:.4f}, Recall: {m.recall:.4f}, F1: {m.f1_score:.4f}")
                lines.append("")

        # Check targets
        lines.append("TARGET ACHIEVEMENT")
        lines.append("-" * 40)
        targets = {
            'easy': 0.95,
            'medium': 0.85,
            'hard': 0.70,
            'ambiguous': 0.70,
        }
        for diff, target in targets.items():
            if diff in by_diff:
                actual = by_diff[diff].accuracy
                status = "PASS" if actual >= target else "FAIL"
                lines.append(f"  {diff}: {actual:.2%} (target: {target:.0%}) [{status}]")
        lines.append("")

        # By stage
        lines.append("METRICS BY DECISION STAGE")
        lines.append("-" * 40)
        by_stage = self.evaluate_by_stage(results)
        for stage, m in sorted(by_stage.items()):
            lines.append(f"{stage}:")
            lines.append(f"  Pairs: {m.total_pairs} ({m.total_pairs/overall.total_pairs:.1%} of total)")
            lines.append(f"  Accuracy: {m.accuracy:.2%}")
            lines.append("")

        # Error analysis
        if verbose:
            errors = self.find_errors(results)
            if errors:
                lines.append("ERROR ANALYSIS")
                lines.append("-" * 40)

                fp_errors = [e for e in errors if e.error_type == 'false_positive']
                fn_errors = [e for e in errors if e.error_type == 'false_negative']

                lines.append(f"False Positives: {len(fp_errors)}")
                for e in fp_errors[:5]:  # Show first 5
                    lines.append(f"  {e.record_1_id} ↔ {e.record_2_id} [{e.stage}] conf={e.confidence:.2f}")
                if len(fp_errors) > 5:
                    lines.append(f"  ... and {len(fp_errors) - 5} more")
                lines.append("")

                lines.append(f"False Negatives: {len(fn_errors)}")
                for e in fn_errors[:5]:  # Show first 5
                    lines.append(f"  {e.record_1_id} ↔ {e.record_2_id} [{e.stage}] conf={e.confidence:.2f}")
                if len(fn_errors) > 5:
                    lines.append(f"  ... and {len(fn_errors) - 5} more")
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_summary_stats(self, results: List[MatchResult]) -> Dict[str, Any]:
        """
        Get summary statistics as a dictionary.

        Useful for programmatic access and JSON export.

        Args:
            results: List of MatchResult objects

        Returns:
            Dictionary with all metrics and breakdowns
        """
        overall = self.evaluate(results)
        by_diff = self.evaluate_by_difficulty(results)
        by_stage = self.evaluate_by_stage(results)
        errors = self.find_errors(results)

        return {
            'overall': overall.to_dict(),
            'by_difficulty': {k: v.to_dict() for k, v in by_diff.items()},
            'by_stage': {k: v.to_dict() for k, v in by_stage.items()},
            'error_count': {
                'false_positives': sum(1 for e in errors if e.error_type == 'false_positive'),
                'false_negatives': sum(1 for e in errors if e.error_type == 'false_negative'),
            },
            'targets': {
                'easy': {'target': 0.95, 'actual': by_diff.get('easy', EvaluationMetrics()).accuracy},
                'medium': {'target': 0.85, 'actual': by_diff.get('medium', EvaluationMetrics()).accuracy},
                'hard': {'target': 0.70, 'actual': by_diff.get('hard', EvaluationMetrics()).accuracy},
                'ambiguous': {'target': 0.70, 'actual': by_diff.get('ambiguous', EvaluationMetrics()).accuracy},
            },
        }
