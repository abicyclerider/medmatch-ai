"""
Core data models for entity resolution.

This module defines the fundamental data structures used for patient matching:
- PatientRecord: Unified wrapper combining Demographics + MedicalRecord
- MatchResult: Output of comparing two patient records
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import date

from ..data.models.patient import Demographics, MedicalRecord, Address, MedicalCondition, Surgery


@dataclass
class PatientRecord:
    """
    Unified patient record for matching.

    Combines demographic information with medical history into a single
    comparable entity. This is the primary input for matching algorithms.

    The record can be created from Demographics alone or combined with
    MedicalRecord for AI-powered medical fingerprinting.

    Attributes:
        record_id: Unique identifier for this record
        name_first: First name
        name_middle: Middle name (optional)
        name_last: Last name
        name_suffix: Suffix like Jr, Sr, II (optional)
        date_of_birth: Date of birth
        gender: Gender (M/F/X)
        mrn: Medical Record Number (system-specific)
        ssn_last4: Last 4 digits of SSN (optional)
        phone: Phone number (optional)
        email: Email address (optional)
        address: Address information (optional)
        record_source: System this record came from
        record_date: When this record was created/updated
        data_quality_flag: Quality indicator (clean, typo, etc.)
        conditions: List of medical conditions (from MedicalRecord)
        medications: List of medications (from MedicalRecord)
        surgeries: List of surgical procedures (from MedicalRecord)
        allergies: List of allergies (from MedicalRecord)

    Properties:
        medical_signature: Human-readable medical summary for AI comparison
            (returns empty string if no medical history available)

    Example:
        >>> # From demographics only
        >>> demo = Demographics(record_id="R001", name_first="John", ...)
        >>> record = PatientRecord.from_demographics(demo)
        >>>
        >>> # With medical history
        >>> medical = MedicalRecord(record_id="R001", ...)
        >>> record = PatientRecord.from_demographics(demo, medical)
        >>> print(record.medical_signature)  # AI-readable summary
    """

    record_id: str

    # Demographics (from Demographics model)
    name_first: str
    name_middle: Optional[str]
    name_last: str
    name_suffix: Optional[str]
    date_of_birth: date
    gender: str

    # Identifiers
    mrn: str
    ssn_last4: Optional[str]

    # Contact
    phone: Optional[str]
    email: Optional[str]
    address: Optional[Address]

    # Metadata
    record_source: str
    record_date: date
    data_quality_flag: Optional[str]

    # Medical history (from MedicalRecord if available)
    conditions: List[MedicalCondition] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    surgeries: List[Surgery] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)

    # Cached normalized versions (for performance)
    _normalized_name: Optional[str] = field(default=None, repr=False)
    _normalized_dob: Optional[date] = field(default=None, repr=False)

    @classmethod
    def from_demographics(
        cls,
        demo: Demographics,
        medical: Optional[MedicalRecord] = None
    ) -> "PatientRecord":
        """
        Create PatientRecord from Demographics (and optionally MedicalRecord).

        Args:
            demo: Demographics object
            medical: Optional MedicalRecord object

        Returns:
            PatientRecord with combined information

        Example:
            >>> demo = Demographics(record_id="R001", ...)
            >>> medical = MedicalRecord(record_id="R001", ...)
            >>> record = PatientRecord.from_demographics(demo, medical)
        """
        return cls(
            record_id=demo.record_id,
            name_first=demo.name_first,
            name_middle=demo.name_middle,
            name_last=demo.name_last,
            name_suffix=demo.name_suffix,
            date_of_birth=demo.date_of_birth,
            gender=demo.gender,
            mrn=demo.mrn,
            ssn_last4=demo.ssn_last4,
            phone=demo.phone,
            email=demo.email,
            address=demo.address,
            record_source=demo.record_source,
            record_date=demo.record_date,
            data_quality_flag=demo.data_quality_flag,
            conditions=(
                medical.medical_history.conditions
                if medical and medical.medical_history
                else []
            ),
            medications=(
                medical.medical_history.medications
                if medical and medical.medical_history
                else []
            ),
            surgeries=(
                medical.medical_history.surgeries
                if medical and medical.medical_history
                else []
            ),
            allergies=(
                medical.medical_history.allergies
                if medical and medical.medical_history
                else []
            ),
        )

    @property
    def full_name(self) -> str:
        """Generate full name string."""
        parts = [self.name_first]
        if self.name_middle:
            parts.append(self.name_middle)
        parts.append(self.name_last)
        if self.name_suffix:
            parts.append(self.name_suffix)
        return " ".join(parts)

    @property
    def normalized_name(self) -> str:
        """
        Normalized full name for comparison.

        Returns lowercase name with extra whitespace removed.
        Cached for performance.
        """
        if self._normalized_name is None:
            parts = [self.name_first, self.name_last]
            if self.name_middle:
                parts.insert(1, self.name_middle)
            self._normalized_name = " ".join(parts).lower().strip()
        return self._normalized_name

    @property
    def medical_signature(self) -> str:
        """
        Medical history summary for AI analysis.

        Generates a concise text representation of medical history including:
        - Conditions with onset years
        - Medications (top 5)
        - Surgeries with dates
        - Allergies

        Returns:
            Human-readable medical summary string

        Example:
            "Conditions: Hypertension (since 2015), Type 2 Diabetes (since 2018);
             Medications: Lisinopril 10mg, Metformin 500mg;
             Allergies: Penicillin"
        """
        parts = []

        if self.conditions:
            cond_strs = [
                f"{c.name} (since {c.onset_year})" if c.onset_year else c.name
                for c in self.conditions
            ]
            parts.append(f"Conditions: {', '.join(cond_strs)}")

        if self.medications:
            # Limit to top 5 to keep summary concise
            meds = self.medications[:5]
            parts.append(f"Medications: {', '.join(meds)}")
            if len(self.medications) > 5:
                parts[-1] += f" (+{len(self.medications) - 5} more)"

        if self.surgeries:
            surg_strs = [f"{s.procedure} ({s.date.year})" for s in self.surgeries]
            parts.append(f"Surgeries: {', '.join(surg_strs)}")

        if self.allergies:
            parts.append(f"Allergies: {', '.join(self.allergies)}")

        return "; ".join(parts) if parts else "No medical history available"

    @property
    def age(self) -> int:
        """Calculate current age."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )


@dataclass
class MatchResult:
    """
    Result of comparing two patient records.

    Contains the matching decision, confidence score, supporting evidence,
    and human-readable explanation.

    Attributes:
        record_1_id: ID of first record
        record_2_id: ID of second record
        is_match: True if determined to be same patient
        confidence: Confidence score (0.0 to 1.0)
        match_type: Category ("exact", "probable", "possible", "no_match")
        evidence: Dict of field-by-field comparison scores
        stage: Which stage produced this result ("blocking", "rules", "scoring", "ai")
        rules_triggered: List of rule names that were applied
        medical_similarity: Optional medical history similarity score (0.0-1.0)
        ai_reasoning: Optional AI explanation text
        explanation: Human-readable explanation
        flags: Warning flags ("twin_risk", "common_name", "data_error", etc.)
    """

    record_1_id: str
    record_2_id: str

    # Decision
    is_match: bool
    confidence: float  # 0.0 to 1.0
    match_type: str  # "exact", "probable", "possible", "no_match"

    # Evidence (field-by-field scores)
    evidence: Dict[str, Any]

    # Which rules/stages were applied
    stage: str = "unknown"  # "blocking", "rules", "scoring", "ai"
    rules_triggered: List[str] = field(default_factory=list)

    # Medical fingerprint (if AI used)
    medical_similarity: Optional[float] = None
    ai_reasoning: Optional[str] = None

    # Explanation
    explanation: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for export/serialization.

        Returns:
            Dictionary representation suitable for JSON/CSV export
        """
        return {
            'record_1_id': self.record_1_id,
            'record_2_id': self.record_2_id,
            'is_match': self.is_match,
            'confidence': self.confidence,
            'match_type': self.match_type,
            'stage': self.stage,
            'explanation': self.explanation,
            'flags': ','.join(self.flags) if self.flags else '',
            'evidence': str(self.evidence),
            'rules_triggered': ','.join(self.rules_triggered) if self.rules_triggered else '',
            'medical_similarity': self.medical_similarity,
        }
