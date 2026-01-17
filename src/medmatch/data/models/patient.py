"""
Pydantic models for patient data structures.

These models define the schema for synthetic patient records,
ensuring data validation and type safety.
"""

from datetime import date, datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class Address(BaseModel):
    """Patient address information."""
    street: str
    city: str
    state: str
    zip_code: str

    def __str__(self):
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"


class MedicalCondition(BaseModel):
    """A medical condition/diagnosis."""
    name: str
    abbreviation: Optional[str] = None
    onset_year: Optional[int] = None
    status: Literal["active", "resolved", "chronic"] = "active"

    def clinical_text(self, use_abbrev: bool = False) -> str:
        """Generate clinical text representation."""
        text = self.abbreviation if (use_abbrev and self.abbreviation) else self.name
        if self.onset_year:
            text += f" (since {self.onset_year})"
        return text


class Surgery(BaseModel):
    """Surgical procedure history."""
    procedure: str
    date: date
    location: Optional[str] = None

    def __str__(self):
        return f"{self.procedure} ({self.date.year})"


class LabResult(BaseModel):
    """Laboratory test result."""
    test_name: str
    value: float
    unit: str
    reference_range: str
    is_abnormal: bool = False


class VitalSigns(BaseModel):
    """Patient vital signs."""
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature_f: Optional[float] = None
    oxygen_saturation: Optional[int] = None
    height_inches: Optional[float] = None
    weight_lbs: Optional[float] = None

    @property
    def bp_string(self) -> Optional[str]:
        """Format blood pressure as string."""
        if self.blood_pressure_systolic and self.blood_pressure_diastolic:
            return f"{self.blood_pressure_systolic}/{self.blood_pressure_diastolic}"
        return None

    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight available."""
        if self.height_inches and self.weight_lbs:
            return round((self.weight_lbs / (self.height_inches ** 2)) * 703, 1)
        return None


class MedicalHistory(BaseModel):
    """Complete medical history for a patient."""
    conditions: List[MedicalCondition] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    surgeries: List[Surgery] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: Optional[str] = None


class ImagingStudy(BaseModel):
    """Radiology/imaging study."""
    modality: str  # X-Ray, CT, MRI, Ultrasound, etc.
    body_part: str
    date: date
    findings: str
    impression: Optional[str] = None


class MedicalRecord(BaseModel):
    """Complete medical record for a single encounter/visit."""
    record_id: str
    patient_id: str  # Ground truth patient ID

    # Visit information
    record_source: str  # "Emergency Department", "Primary Care", "Lab", etc.
    record_date: datetime

    # Clinical content
    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None
    medical_history: Optional[MedicalHistory] = None
    vital_signs: Optional[VitalSigns] = None

    # Assessment and plan
    assessment: Optional[str] = None
    diagnosis_codes: List[str] = Field(default_factory=list)  # ICD-10 codes
    plan: Optional[str] = None

    # Test results
    lab_results: List[LabResult] = Field(default_factory=list)
    imaging_studies: List[ImagingStudy] = Field(default_factory=list)

    # Clinical notes (free text)
    clinical_notes: Optional[str] = None

    def model_dump_json(self, **kwargs):
        """Override to handle datetime serialization."""
        return super().model_dump_json(by_alias=True, exclude_none=True, **kwargs)


class Demographics(BaseModel):
    """Patient demographic information (as stored in different systems)."""
    record_id: str
    patient_id: str  # Ground truth patient ID

    # Name components
    name_first: str
    name_middle: Optional[str] = None
    name_last: str
    name_suffix: Optional[str] = None  # Jr, Sr, II, III, etc.

    # Core demographics
    date_of_birth: date
    gender: Literal["M", "F", "X"]

    # Identifiers
    mrn: str  # Medical Record Number (system-specific)
    ssn_last4: Optional[str] = None

    # Contact information
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Address] = None

    # Metadata
    record_source: str  # Which system this record came from
    record_date: date  # When this record was created/updated
    data_quality_flag: Optional[str] = None  # "clean", "name_variation", "typo", etc.

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
    def age(self) -> int:
        """Calculate current age."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )

    @field_validator('ssn_last4')
    @classmethod
    def validate_ssn(cls, v):
        """Validate SSN last 4 digits."""
        if v is not None and len(v) != 4:
            raise ValueError("SSN last 4 must be exactly 4 digits")
        return v


class GroundTruth(BaseModel):
    """Ground truth mapping for evaluation."""
    record_id: str
    patient_id: str
    match_group: str  # Group ID for records that should match
    notes: Optional[str] = None  # Explanation of variation/edge case

    # Flags for test case categorization
    is_common_name: bool = False
    is_name_variation: bool = False
    is_twin: bool = False
    is_family_member: bool = False
    has_data_error: bool = False
    difficulty: Literal["easy", "medium", "hard", "ambiguous"] = "easy"


class Patient(BaseModel):
    """
    Complete patient entity (ground truth).

    This represents the true patient, with multiple demographic records
    and medical records associated with them.
    """
    patient_id: str

    # True patient information
    true_first_name: str
    true_middle_name: Optional[str] = None
    true_last_name: str
    true_suffix: Optional[str] = None
    true_dob: date
    true_gender: Literal["M", "F", "X"]

    # Collection of records for this patient
    demographic_records: List[Demographics] = Field(default_factory=list)
    medical_records: List[MedicalRecord] = Field(default_factory=list)

    # Patient characteristics for generation
    has_common_name: bool = False
    has_twin: bool = False
    twin_id: Optional[str] = None
    family_member_ids: List[str] = Field(default_factory=list)

    @property
    def true_full_name(self) -> str:
        """Get true full name."""
        parts = [self.true_first_name]
        if self.true_middle_name:
            parts.append(self.true_middle_name)
        parts.append(self.true_last_name)
        if self.true_suffix:
            parts.append(self.true_suffix)
        return " ".join(parts)
