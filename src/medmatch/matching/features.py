"""
Feature extraction for patient matching.

This module extracts numerical features from patient record pairs for
weighted confidence scoring. Features capture similarity across demographics,
contact information, and identifiers.

Key Classes:
    FeatureVector: Structured feature representation with 15+ numerical features
    FeatureExtractor: Extracts features using existing comparators for consistency
"""

from dataclasses import dataclass
from typing import Optional

from .core import PatientRecord
from .comparators import (
    NameComparator,
    DateComparator,
    AddressComparator,
    PhoneComparator,
    EmailComparator,
)


@dataclass
class FeatureVector:
    """
    Numerical features extracted from a patient record pair.

    Features are normalized to [0.0, 1.0] range where:
    - 1.0 = perfect match
    - 0.0 = complete mismatch
    - None = missing data (cannot compare)

    Attributes:
        # Name features (3)
        name_first_score: First name similarity (exact, variation, typo, soundex)
        name_last_score: Last name similarity
        name_middle_score: Middle name similarity (if available)

        # Date features (2)
        dob_score: Date of birth similarity (exact, transposed, swapped)
        age_difference: Absolute age difference in years

        # Contact features (3)
        phone_score: Phone number similarity (normalized)
        email_score: Email address similarity
        address_score: Address similarity (multi-level matching)

        # Identifier features (2 boolean)
        mrn_match: Medical record numbers match exactly
        ssn_match: SSN last 4 digits match

        # Metadata (2)
        name_first_method: Match method for explainability
        name_last_method: Match method for explainability
        dob_method: Match method for explainability

        # Quality flags (2)
        has_missing_fields: True if critical fields missing
        has_data_errors: True if data quality issues detected
    """

    # Name features
    name_first_score: Optional[float] = None
    name_last_score: Optional[float] = None
    name_middle_score: Optional[float] = None

    # Date features
    dob_score: Optional[float] = None
    age_difference: Optional[int] = None

    # Contact features
    phone_score: Optional[float] = None
    email_score: Optional[float] = None
    address_score: Optional[float] = None

    # Identifier features (boolean)
    mrn_match: bool = False
    ssn_match: bool = False

    # Match methods (for explainability)
    name_first_method: Optional[str] = None
    name_last_method: Optional[str] = None
    dob_method: Optional[str] = None
    address_method: Optional[str] = None

    # Quality flags
    has_missing_fields: bool = False
    has_data_errors: bool = False

    def to_dict(self) -> dict:
        """
        Convert feature vector to dictionary.

        Returns:
            Dictionary with feature names and values

        Example:
            >>> features = FeatureVector(name_first_score=0.95, ...)
            >>> features.to_dict()
            {'name_first_score': 0.95, 'name_last_score': 1.0, ...}
        """
        return {
            'name_first_score': self.name_first_score,
            'name_last_score': self.name_last_score,
            'name_middle_score': self.name_middle_score,
            'dob_score': self.dob_score,
            'age_difference': self.age_difference,
            'phone_score': self.phone_score,
            'email_score': self.email_score,
            'address_score': self.address_score,
            'mrn_match': self.mrn_match,
            'ssn_match': self.ssn_match,
            'has_missing_fields': self.has_missing_fields,
            'has_data_errors': self.has_data_errors,
        }


class FeatureExtractor:
    """
    Extract numerical features from patient record pairs.

    Uses existing comparators from Phase 2.1 for consistency with rules.
    All comparators return (score, method) tuples for explainability.

    Example:
        >>> extractor = FeatureExtractor()
        >>> record1 = PatientRecord(...)
        >>> record2 = PatientRecord(...)
        >>> features = extractor.extract(record1, record2)
        >>> print(f"Name match: {features.name_first_score:.2f} ({features.name_first_method})")
        Name match: 0.95 (known_variation)
    """

    def __init__(self):
        """
        Initialize feature extractor with comparators.

        Creates instances of all comparators from Phase 2.1 for consistent
        similarity calculations across the pipeline.

        Comparators:
            - NameComparator: Handles exact matches, nicknames, typos, soundex
            - DateComparator: Handles twins, transposed digits, month/day swaps
            - AddressComparator: Multi-level matching (exact, street+city, zip)
            - PhoneComparator: Normalized phone number comparison
            - EmailComparator: Case-insensitive email comparison
        """
        self.name_comparator = NameComparator()
        self.date_comparator = DateComparator()
        self.address_comparator = AddressComparator()
        self.phone_comparator = PhoneComparator()
        self.email_comparator = EmailComparator()

    def extract(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> FeatureVector:
        """
        Extract all features from a record pair.

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            FeatureVector with all extracted features

        Example:
            >>> r1 = PatientRecord(...)  # John Smith, 1980-03-15
            >>> r2 = PatientRecord(...)  # John Smith, 1980-03-15
            >>> features = extractor.extract(r1, r2)
            >>> features.name_first_score
            1.0
            >>> features.dob_score
            1.0
        """
        features = FeatureVector()

        # Extract name features
        features.name_first_score, features.name_first_method = self._extract_name_first(record1, record2)
        features.name_last_score, features.name_last_method = self._extract_name_last(record1, record2)
        features.name_middle_score, _ = self._extract_name_middle(record1, record2)

        # Extract date features
        features.dob_score, features.dob_method = self._extract_dob(record1, record2)
        features.age_difference = self._extract_age_difference(record1, record2)

        # Extract contact features
        features.phone_score, _ = self._extract_phone(record1, record2)
        features.email_score, _ = self._extract_email(record1, record2)
        features.address_score, features.address_method = self._extract_address(record1, record2)

        # Extract identifier features
        features.mrn_match = self._extract_mrn_match(record1, record2)
        features.ssn_match = self._extract_ssn_match(record1, record2)

        # Extract quality flags
        features.has_missing_fields = self._has_missing_fields(record1, record2)
        features.has_data_errors = self._has_data_errors(record1, record2)

        return features

    def _extract_name_first(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract first name similarity score and method."""
        if not r1.name_first or not r2.name_first:
            return None, None
        return self.name_comparator.compare(r1.name_first, r2.name_first)

    def _extract_name_last(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract last name similarity score and method."""
        if not r1.name_last or not r2.name_last:
            return None, None
        return self.name_comparator.compare(r1.name_last, r2.name_last)

    def _extract_name_middle(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract middle name similarity score and method."""
        # Both must have middle names to compare
        if not r1.name_middle or not r2.name_middle:
            return None, None
        return self.name_comparator.compare(r1.name_middle, r2.name_middle)

    def _extract_dob(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract date of birth similarity score and method."""
        if not r1.date_of_birth or not r2.date_of_birth:
            return None, None
        return self.date_comparator.compare(r1.date_of_birth, r2.date_of_birth)

    def _extract_age_difference(self, r1: PatientRecord, r2: PatientRecord) -> Optional[int]:
        """Extract absolute age difference in years."""
        if r1.age is None or r2.age is None:
            return None
        return abs(r1.age - r2.age)

    def _extract_phone(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract phone number similarity score and method."""
        if not r1.phone or not r2.phone:
            return None, None
        return self.phone_comparator.compare(r1.phone, r2.phone)

    def _extract_email(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract email similarity score and method."""
        if not r1.email or not r2.email:
            return None, None
        return self.email_comparator.compare(r1.email, r2.email)

    def _extract_address(self, r1: PatientRecord, r2: PatientRecord) -> tuple[Optional[float], Optional[str]]:
        """Extract address similarity score and method."""
        if not r1.address or not r2.address:
            return None, None
        return self.address_comparator.compare(r1.address, r2.address)

    def _extract_mrn_match(self, r1: PatientRecord, r2: PatientRecord) -> bool:
        """Check if medical record numbers match exactly."""
        if not r1.mrn or not r2.mrn:
            return False
        return r1.mrn.strip().upper() == r2.mrn.strip().upper()

    def _extract_ssn_match(self, r1: PatientRecord, r2: PatientRecord) -> bool:
        """Check if SSN last 4 digits match."""
        if not r1.ssn_last4 or not r2.ssn_last4:
            return False
        return r1.ssn_last4.strip() == r2.ssn_last4.strip()

    def _has_missing_fields(self, r1: PatientRecord, r2: PatientRecord) -> bool:
        """
        Check if critical fields are missing.

        Critical fields: first name, last name, date of birth
        """
        critical_fields_r1 = [r1.name_first, r1.name_last, r1.date_of_birth]
        critical_fields_r2 = [r2.name_first, r2.name_last, r2.date_of_birth]

        return any(f is None or f == '' for f in critical_fields_r1 + critical_fields_r2)

    def _has_data_errors(self, r1: PatientRecord, r2: PatientRecord) -> bool:
        """
        Check if data quality issues detected.

        Uses data_quality_flag from Demographics if available.
        """
        # Check data quality flags
        if hasattr(r1, 'data_quality_flag') and r1.data_quality_flag:
            if r1.data_quality_flag.lower() in ['error', 'invalid', 'corrupted']:
                return True

        if hasattr(r2, 'data_quality_flag') and r2.data_quality_flag:
            if r2.data_quality_flag.lower() in ['error', 'invalid', 'corrupted']:
                return True

        return False
