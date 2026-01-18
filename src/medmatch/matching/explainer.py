"""
Human-readable explanations for match results.

This module provides the MatchExplainer class for generating clear,
actionable explanations from MatchResult objects. Used for clinical
decision support and audit trails.

Phase 2.5 of the entity resolution system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .core import MatchResult


@dataclass
class ExplanationConfig:
    """
    Configuration for explanation generation.

    Attributes:
        verbose: Include all evidence details
        show_scores: Show numerical scores
        show_flags: Show warning flags
        show_stage: Show which pipeline stage decided
        max_evidence_fields: Maximum evidence fields to show in brief mode
    """
    verbose: bool = False
    show_scores: bool = True
    show_flags: bool = True
    show_stage: bool = True
    max_evidence_fields: int = 5


class MatchExplainer:
    """
    Generates human-readable explanations from MatchResult objects.

    Provides formatted explanations suitable for:
    - Clinical decision support
    - Audit trails
    - Error analysis
    - User feedback

    Example:
        >>> explainer = MatchExplainer()
        >>> explanation = explainer.explain(result)
        >>> print(explanation)

        PROBABLE MATCH (confidence: 0.87)
        Records: R0001 ↔ R0002

        Decision Stage: scoring

        Evidence:
        - name_first: 0.95 (known_variation: William → Bill)
        - name_last: 1.00 (exact_match)
        ...
    """

    # Map match types to display labels
    MATCH_TYPE_LABELS = {
        "definite": "DEFINITE MATCH",
        "exact": "EXACT MATCH",
        "probable": "PROBABLE MATCH",
        "possible": "POSSIBLE MATCH",
        "no_match": "NO MATCH",
        "uncertain": "UNCERTAIN",
    }

    # Map stage names to display labels
    STAGE_LABELS = {
        "rules": "Deterministic Rules",
        "scoring": "Feature Scoring",
        "ai": "AI Medical Fingerprinting",
        "blocking": "Blocking",
        "unknown": "Unknown",
    }

    # Common method descriptions for evidence
    METHOD_DESCRIPTIONS = {
        "exact_match": "exact match",
        "exact": "exact match",
        "nickname_match": "nickname variation",
        "known_variation": "known name variation",
        "typo_match": "likely typo (high similarity)",
        "typo": "likely typo",
        "soundex_match": "sounds similar (phonetic match)",
        "soundex": "phonetic match",
        "partial_match": "partial match",
        "low_similarity": "low similarity",
        "no_match": "no match",
        "transposed_digits": "transposed digits",
        "month_day_swap": "month/day swapped",
        "year_typo": "year typo",
        "same_street_city": "same street and city",
        "same_city_state": "same city and state",
        "same_zip": "same ZIP code",
        "different_address": "different address",
    }

    def __init__(self, config: Optional[ExplanationConfig] = None):
        """
        Initialize the explainer.

        Args:
            config: Configuration for explanation generation. Defaults to
                    standard configuration if not provided.
        """
        self.config = config or ExplanationConfig()

    def explain(
        self,
        result: MatchResult,
        verbose: Optional[bool] = None,
    ) -> str:
        """
        Generate human-readable explanation from a MatchResult.

        Args:
            result: The MatchResult to explain
            verbose: Override config verbose setting

        Returns:
            Formatted multi-line explanation string

        Example:
            >>> explanation = explainer.explain(result)
            >>> print(explanation)
        """
        use_verbose = verbose if verbose is not None else self.config.verbose
        lines = []

        # Header: Match decision and confidence
        header = self._format_header(result)
        lines.append(header)
        lines.append("")

        # Record IDs
        lines.append(f"Records: {result.record_1_id} ↔ {result.record_2_id}")
        lines.append("")

        # Decision stage
        if self.config.show_stage:
            stage_label = self.STAGE_LABELS.get(result.stage, result.stage)
            lines.append(f"Decision Stage: {stage_label}")
            lines.append("")

        # Rules triggered
        if result.rules_triggered:
            rules_str = ", ".join(result.rules_triggered)
            lines.append(f"Rules Applied: {rules_str}")
            lines.append("")

        # Evidence breakdown
        evidence_section = self._format_evidence(result.evidence, use_verbose)
        if evidence_section:
            lines.append("Evidence:")
            lines.extend(evidence_section)
            lines.append("")

        # AI reasoning (if present)
        if result.ai_reasoning:
            lines.append("AI Analysis:")
            lines.append(f"  {result.ai_reasoning}")
            if result.medical_similarity is not None:
                lines.append(f"  Medical Similarity: {result.medical_similarity:.2f}")
            lines.append("")

        # Warning flags
        if self.config.show_flags and result.flags:
            flags_str = ", ".join(result.flags)
            lines.append(f"Flags: {flags_str}")
            lines.append("")

        # Recommendation
        recommendation = self._get_recommendation(result)
        lines.append(f"Recommendation: {recommendation}")

        return "\n".join(lines)

    def explain_brief(self, result: MatchResult) -> str:
        """
        Generate a brief one-line explanation.

        Args:
            result: The MatchResult to explain

        Returns:
            Single-line summary string

        Example:
            >>> brief = explainer.explain_brief(result)
            >>> print(brief)
            PROBABLE MATCH (0.87): R0001 ↔ R0002 [scoring]
        """
        match_label = self.MATCH_TYPE_LABELS.get(result.match_type, result.match_type.upper())
        return (
            f"{match_label} ({result.confidence:.2f}): "
            f"{result.record_1_id} ↔ {result.record_2_id} [{result.stage}]"
        )

    def explain_batch(
        self,
        results: List[MatchResult],
        show_all: bool = False,
    ) -> str:
        """
        Generate summary report for multiple results.

        Args:
            results: List of MatchResult objects
            show_all: If True, include explanation for every result.
                      If False, only show summary statistics.

        Returns:
            Formatted summary report

        Example:
            >>> report = explainer.explain_batch(results)
            >>> print(report)
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ENTITY RESOLUTION SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Statistics
        total = len(results)
        matches = sum(1 for r in results if r.is_match)
        non_matches = total - matches

        lines.append(f"Total Pairs Evaluated: {total}")
        lines.append(f"Matches Found: {matches}")
        lines.append(f"Non-Matches: {non_matches}")
        lines.append("")

        # By match type
        by_type: Dict[str, int] = {}
        for r in results:
            by_type[r.match_type] = by_type.get(r.match_type, 0) + 1

        lines.append("By Match Type:")
        for match_type, count in sorted(by_type.items()):
            label = self.MATCH_TYPE_LABELS.get(match_type, match_type)
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        lines.append("")

        # By stage
        by_stage: Dict[str, int] = {}
        for r in results:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1

        lines.append("By Decision Stage:")
        for stage, count in sorted(by_stage.items()):
            label = self.STAGE_LABELS.get(stage, stage)
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        lines.append("")

        # Average confidence
        if results:
            avg_conf = sum(r.confidence for r in results) / len(results)
            lines.append(f"Average Confidence: {avg_conf:.3f}")
            lines.append("")

        # Detailed results (if requested)
        if show_all:
            lines.append("-" * 60)
            lines.append("DETAILED RESULTS")
            lines.append("-" * 60)
            for i, result in enumerate(results, 1):
                lines.append("")
                lines.append(f"[{i}/{total}]")
                lines.append(self.explain(result))
                lines.append("-" * 40)

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_header(self, result: MatchResult) -> str:
        """Format the header line with match decision and confidence."""
        match_label = self.MATCH_TYPE_LABELS.get(
            result.match_type,
            result.match_type.upper()
        )

        if self.config.show_scores:
            return f"{match_label} (confidence: {result.confidence:.2f})"
        else:
            return match_label

    def _format_evidence(
        self,
        evidence: Dict[str, Any],
        verbose: bool,
    ) -> List[str]:
        """Format evidence dictionary into readable lines."""
        if not evidence:
            return ["  (no evidence recorded)"]

        lines = []
        shown = 0
        max_fields = len(evidence) if verbose else self.config.max_evidence_fields

        for field_name, value in evidence.items():
            if shown >= max_fields:
                remaining = len(evidence) - shown
                lines.append(f"  ... and {remaining} more fields")
                break

            line = self._format_evidence_field(field_name, value)
            lines.append(f"  {line}")
            shown += 1

        return lines

    def _format_evidence_field(self, field_name: str, value: Any) -> str:
        """Format a single evidence field."""
        # Handle tuple format (score, method) from comparators
        if isinstance(value, tuple) and len(value) == 2:
            score, method = value
            method_desc = self.METHOD_DESCRIPTIONS.get(method, method)
            return f"- {field_name}: {score:.2f} ({method_desc})"

        # Handle dict format (from scoring breakdown)
        elif isinstance(value, dict):
            if 'score' in value and 'method' in value:
                method_desc = self.METHOD_DESCRIPTIONS.get(
                    value['method'],
                    value['method']
                )
                return f"- {field_name}: {value['score']:.2f} ({method_desc})"
            elif 'score' in value:
                return f"- {field_name}: {value['score']:.2f}"
            else:
                return f"- {field_name}: {value}"

        # Handle numeric scores
        elif isinstance(value, (int, float)):
            return f"- {field_name}: {value:.2f}"

        # Handle string values
        elif isinstance(value, str):
            return f"- {field_name}: {value}"

        # Handle boolean
        elif isinstance(value, bool):
            return f"- {field_name}: {'Yes' if value else 'No'}"

        # Fallback
        else:
            return f"- {field_name}: {value}"

    def _get_recommendation(self, result: MatchResult) -> str:
        """Generate recommendation based on match result."""
        if result.match_type == "definite" or result.match_type == "exact":
            return "These records refer to the same patient. Safe to merge."

        elif result.match_type == "probable":
            if result.flags:
                return (
                    "These records likely refer to the same patient, "
                    "but manual review recommended due to flagged concerns."
                )
            return "These records likely refer to the same patient."

        elif result.match_type == "possible":
            return (
                "These records may refer to the same patient. "
                "Manual review recommended before merging."
            )

        elif result.match_type == "uncertain":
            return (
                "Insufficient evidence to determine if records match. "
                "Requires human review."
            )

        else:  # no_match
            return "These records appear to be different patients."


def format_match_for_display(result: MatchResult) -> str:
    """
    Convenience function for quick formatting.

    Args:
        result: MatchResult to format

    Returns:
        Formatted explanation string

    Example:
        >>> print(format_match_for_display(result))
    """
    explainer = MatchExplainer()
    return explainer.explain(result)
