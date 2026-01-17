"""
Patient entity resolution module.

This module provides AI-powered patient matching capabilities to prevent
wrong-patient medical errors. It implements a hybrid multi-stage matching
pipeline combining deterministic rules, feature-based scoring, and AI-powered
medical fingerprinting.

Main Classes:
    PatientRecord: Unified patient record combining demographics and medical history
    MatchResult: Output of record comparison with confidence and evidence
    PatientMatcher: Main matching orchestrator (to be implemented in Phase 2.2+)

Example:
    >>> from medmatch.matching import PatientRecord, PatientMatcher
    >>> from medmatch.data.models.patient import Demographics
    >>>
    >>> # Load demographics
    >>> demo1 = Demographics(record_id="R001", ...)
    >>> demo2 = Demographics(record_id="R002", ...)
    >>>
    >>> # Create patient records
    >>> record1 = PatientRecord.from_demographics(demo1)
    >>> record2 = PatientRecord.from_demographics(demo2)
    >>>
    >>> # Match records
    >>> matcher = PatientMatcher(use_ai=True, confidence_threshold=0.85)
    >>> result = matcher.match_pair(record1, record2)
    >>>
    >>> print(f"Match: {result.is_match}, Confidence: {result.confidence:.2f}")
    >>> print(result.explanation)
"""

from .core import PatientRecord, MatchResult
from .comparators import (
    NameComparator,
    DateComparator,
    AddressComparator,
    PhoneComparator,
    EmailComparator,
)

__all__ = [
    # Core models
    "PatientRecord",
    "MatchResult",
    # Comparators
    "NameComparator",
    "DateComparator",
    "AddressComparator",
    "PhoneComparator",
    "EmailComparator",
    # Main matcher (to be added in Phase 2.2+)
    # "PatientMatcher",
]
