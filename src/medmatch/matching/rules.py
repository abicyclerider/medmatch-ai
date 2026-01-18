"""
Deterministic matching rules for patient entity resolution.

This module implements rule-based matching for clear matches and non-matches.
Rules provide fast, explainable decisions without requiring ML models or
expensive computations.

Rule types:
1. NO-MATCH rules: Identify obvious non-matches (gender mismatch, large age gap)
2. MATCH rules: Identify definite matches (exact demographics, strong identifiers)

Rules are applied in order: NO-MATCH first (faster), then MATCH.
If no rule fires, the case is uncertain and passed to next stage (scoring/AI).
"""

from typing import Optional, NamedTuple, List, Protocol
from datetime import date

from .core import PatientRecord, MatchResult
from .comparators import NameComparator, DateComparator


class RuleResult(NamedTuple):
    """Result of applying a matching rule."""

    decision: Optional[bool]  # True=match, False=no_match, None=uncertain
    confidence: float  # 0.0-1.0
    rule_name: str
    explanation: str


class MatchRule(Protocol):
    """Protocol for matching rules."""

    def apply(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> RuleResult:
        """
        Apply rule to a pair of records.

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            RuleResult with decision, confidence, rule name, and explanation
        """
        ...


# =============================================================================
# NO-MATCH RULES (Identify obvious non-matches)
# =============================================================================


class GenderMismatchRule:
    """
    Automatic no-match if genders differ.

    Rationale:
    - Gender is stable and rarely changes
    - Gender errors are uncommon in medical records
    - If gender differs, very unlikely to be same patient

    Exception: Missing gender is not a mismatch (data quality issue)
    """

    def apply(self, record1: PatientRecord, record2: PatientRecord) -> RuleResult:
        """Apply gender mismatch rule."""
        # If either gender is missing, can't determine mismatch
        if not record1.gender or not record2.gender:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="GenderMismatchRule",
                explanation="Cannot determine: missing gender",
            )

        # If genders match, rule doesn't apply
        if record1.gender == record2.gender:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="GenderMismatchRule",
                explanation="Genders match - rule not applicable",
            )

        # Genders differ → no match
        return RuleResult(
            decision=False,
            confidence=0.99,
            rule_name="GenderMismatchRule",
            explanation=f"Gender mismatch: {record1.gender} ≠ {record2.gender}",
        )


class LargeAgeDifferentNameRule:
    """
    No-match if age difference >5 years AND names are dissimilar.

    Rationale:
    - Large age gap + different names = likely different people
    - Small age gaps might be data errors (typos in DOB)
    - Similar names with age gap might be parent-child

    Thresholds:
    - Age difference: >5 years
    - Name similarity: <0.5 (dissimilar)
    """

    def __init__(self):
        self.name_comp = NameComparator()

    def apply(self, record1: PatientRecord, record2: PatientRecord) -> RuleResult:
        """Apply large age + different name rule."""
        # Calculate age difference
        age_diff = abs(record1.age - record2.age)

        # Check age threshold
        if age_diff <= 5:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="LargeAgeDifferentNameRule",
                explanation=f"Age difference {age_diff} years ≤ 5 - rule not applicable",
            )

        # Compare names (first + last)
        first_score, first_method = self.name_comp.compare(
            record1.name_first, record2.name_first
        )
        last_score, last_method = self.name_comp.compare(
            record1.name_last, record2.name_last
        )

        # Average name score
        name_score = (first_score + last_score) / 2

        # Check name threshold
        if name_score >= 0.5:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="LargeAgeDifferentNameRule",
                explanation=f"Age difference {age_diff} years but names similar ({name_score:.2f})",
            )

        # Large age gap + dissimilar names → no match
        return RuleResult(
            decision=False,
            confidence=0.95,
            rule_name="LargeAgeDifferentNameRule",
            explanation=f"Age difference {age_diff} years + dissimilar names ({name_score:.2f})",
        )


# =============================================================================
# MATCH RULES (Identify definite matches)
# =============================================================================


class ExactMatchRule:
    """
    Exact match on name + DOB + gender.

    Rationale:
    - Exact match on all demographics is very strong signal
    - Extremely unlikely to be different people

    Requirements:
    - Exact first name (score=1.0)
    - Exact last name (score=1.0)
    - Exact DOB (score=1.0)
    - Same gender
    """

    def __init__(self):
        self.name_comp = NameComparator()
        self.date_comp = DateComparator()

    def apply(self, record1: PatientRecord, record2: PatientRecord) -> RuleResult:
        """Apply exact match rule."""
        # Check gender
        if record1.gender != record2.gender:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="ExactMatchRule",
                explanation="Gender mismatch",
            )

        # Check names
        first_score, first_method = self.name_comp.compare(
            record1.name_first, record2.name_first
        )
        last_score, last_method = self.name_comp.compare(
            record1.name_last, record2.name_last
        )

        if first_score < 1.0 or last_score < 1.0:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="ExactMatchRule",
                explanation=f"Name not exact match (first:{first_score:.2f}, last:{last_score:.2f})",
            )

        # Check DOB
        dob_score, dob_method = self.date_comp.compare(
            record1.date_of_birth, record2.date_of_birth
        )

        if dob_score < 1.0:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="ExactMatchRule",
                explanation=f"DOB not exact match ({dob_score:.2f})",
            )

        # All exact → match
        return RuleResult(
            decision=True,
            confidence=0.99,
            rule_name="ExactMatchRule",
            explanation=f"Exact match: {record1.full_name}, DOB={record1.date_of_birth}, Gender={record1.gender}",
        )


class MRNNameMatchRule:
    """
    Match if same MRN + similar name.

    Rationale:
    - MRNs should be unique within a system
    - Require name confirmation to avoid MRN reuse/errors
    - Name similarity threshold: 0.8 (allows minor typos)

    Note: MRNs are system-specific, so this mainly catches records
    from the same source system.
    """

    def __init__(self):
        self.name_comp = NameComparator()

    def apply(self, record1: PatientRecord, record2: PatientRecord) -> RuleResult:
        """Apply MRN + name match rule."""
        # Check MRN
        if not record1.mrn or not record2.mrn:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="MRNNameMatchRule",
                explanation="Missing MRN",
            )

        if record1.mrn != record2.mrn:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="MRNNameMatchRule",
                explanation="MRN mismatch",
            )

        # Check name similarity
        first_score, first_method = self.name_comp.compare(
            record1.name_first, record2.name_first
        )
        last_score, last_method = self.name_comp.compare(
            record1.name_last, record2.name_last
        )

        name_score = (first_score + last_score) / 2

        if name_score < 0.8:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="MRNNameMatchRule",
                explanation=f"Same MRN but name dissimilar ({name_score:.2f})",
            )

        # Same MRN + similar name → match
        return RuleResult(
            decision=True,
            confidence=0.95,
            rule_name="MRNNameMatchRule",
            explanation=f"Same MRN ({record1.mrn}) + similar name ({name_score:.2f})",
        )


class SSNNameDOBMatchRule:
    """
    Match if same SSN + name + DOB.

    Rationale:
    - SSN last 4 + name + DOB is very strong identifier
    - SSN alone could have collisions (last 4 only)
    - Combined with name and DOB is highly discriminative

    Requirements:
    - Same SSN_last4
    - Name similarity ≥ 0.8
    - DOB similarity ≥ 0.9 (allows minor typos)
    """

    def __init__(self):
        self.name_comp = NameComparator()
        self.date_comp = DateComparator()

    def apply(self, record1: PatientRecord, record2: PatientRecord) -> RuleResult:
        """Apply SSN + name + DOB match rule."""
        # Check SSN
        if not record1.ssn_last4 or not record2.ssn_last4:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="SSNNameDOBMatchRule",
                explanation="Missing SSN",
            )

        if record1.ssn_last4 != record2.ssn_last4:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="SSNNameDOBMatchRule",
                explanation="SSN mismatch",
            )

        # Check name similarity
        first_score, first_method = self.name_comp.compare(
            record1.name_first, record2.name_first
        )
        last_score, last_method = self.name_comp.compare(
            record1.name_last, record2.name_last
        )

        name_score = (first_score + last_score) / 2

        if name_score < 0.8:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="SSNNameDOBMatchRule",
                explanation=f"Same SSN but name dissimilar ({name_score:.2f})",
            )

        # Check DOB similarity
        dob_score, dob_method = self.date_comp.compare(
            record1.date_of_birth, record2.date_of_birth
        )

        if dob_score < 0.9:
            return RuleResult(
                decision=None,
                confidence=0.0,
                rule_name="SSNNameDOBMatchRule",
                explanation=f"Same SSN + name but DOB dissimilar ({dob_score:.2f})",
            )

        # Same SSN + name + DOB → match
        return RuleResult(
            decision=True,
            confidence=0.97,
            rule_name="SSNNameDOBMatchRule",
            explanation=f"Same SSN ({record1.ssn_last4}) + name ({name_score:.2f}) + DOB ({dob_score:.2f})",
        )


# =============================================================================
# RULE ENGINE (Orchestrates rule application)
# =============================================================================


class RuleEngine:
    """
    Orchestrate application of matching rules.

    Rules are applied in order:
    1. NO-MATCH rules first (faster to eliminate)
    2. MATCH rules second (identify definite matches)

    If any rule fires (returns decision != None), that decision is final.
    If no rules fire, return None (uncertain, pass to next stage).
    """

    def __init__(self):
        """Initialize rule engine with all rules."""
        # NO-MATCH rules (applied first)
        self.no_match_rules: List[MatchRule] = [
            GenderMismatchRule(),
            LargeAgeDifferentNameRule(),
        ]

        # MATCH rules (applied second)
        self.match_rules: List[MatchRule] = [
            ExactMatchRule(),
            MRNNameMatchRule(),
            SSNNameDOBMatchRule(),
        ]

    def evaluate(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> Optional[MatchResult]:
        """
        Evaluate a pair of records using all rules.

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            MatchResult if any rule fires, None if uncertain

        Example:
            >>> engine = RuleEngine()
            >>> r1 = PatientRecord(...)  # John Smith, M, 1980-03-15
            >>> r2 = PatientRecord(...)  # Jane Doe, F, 1980-03-15
            >>> result = engine.evaluate(r1, r2)
            >>> result.is_match
            False
            >>> result.rules_triggered
            ['GenderMismatchRule']
        """
        # Try NO-MATCH rules first
        for rule in self.no_match_rules:
            rule_result = rule.apply(record1, record2)

            if rule_result.decision is not None:
                # Rule fired - return result
                return self._create_match_result(
                    record1, record2, rule_result
                )

        # Try MATCH rules second
        for rule in self.match_rules:
            rule_result = rule.apply(record1, record2)

            if rule_result.decision is not None:
                # Rule fired - return result
                return self._create_match_result(
                    record1, record2, rule_result
                )

        # No rules fired - uncertain
        return None

    def _create_match_result(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
        rule_result: RuleResult,
    ) -> MatchResult:
        """
        Create MatchResult from RuleResult.

        Args:
            record1: First patient record
            record2: Second patient record
            rule_result: Result from rule application

        Returns:
            MatchResult with full evidence and explanation
        """
        # Determine match type based on confidence
        if rule_result.decision:
            # Match
            if rule_result.confidence >= 0.95:
                match_type = "exact"
            elif rule_result.confidence >= 0.85:
                match_type = "probable"
            else:
                match_type = "possible"
        else:
            # No match
            match_type = "no_match"

        return MatchResult(
            record_1_id=record1.record_id,
            record_2_id=record2.record_id,
            is_match=rule_result.decision,
            confidence=rule_result.confidence,
            match_type=match_type,
            evidence={
                "rule_fired": rule_result.rule_name,
                "rule_explanation": rule_result.explanation,
            },
            stage="rules",
            rules_triggered=[rule_result.rule_name],
            explanation=rule_result.explanation,
        )
