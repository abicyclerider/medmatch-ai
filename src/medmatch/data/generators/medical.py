"""
Medical record generator using AI-assisted content generation.

Uses Gemini API to generate realistic clinical narratives and medical records.
"""

import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
from dotenv import load_dotenv
import google.genai as genai

from ..models import (
    MedicalRecord,
    MedicalHistory,
    MedicalCondition,
    Surgery,
    VitalSigns,
    LabResult,
    ImagingStudy,
)
from ..utils.medical_utils import MedicalScenarioGenerator


class MedicalRecordGenerator:
    """Generate realistic medical records with AI assistance."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_ai: bool = True,
        requests_per_minute: int = 5
    ):
        """
        Initialize medical record generator.

        Args:
            api_key: Google AI API key (if None, loads from environment)
            use_ai: If False, uses rule-based generation only (no API calls)
            requests_per_minute: Max API requests per minute (default: 5 for free tier)
                                Set to 15 for paid tier, or higher if you have quota
        """
        self.use_ai = use_ai
        self.record_counter = 0
        self.scenario_gen = MedicalScenarioGenerator()

        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.delay_between_calls = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_api_call_time = 0

        if use_ai:
            # Load API key
            if api_key is None:
                load_dotenv()
                api_key = os.getenv('GOOGLE_AI_API_KEY')

            if not api_key:
                print("Warning: No API key found. Falling back to rule-based generation.")
                self.use_ai = False
            else:
                self.client = genai.Client(api_key=api_key)
                print(f"AI generation enabled with rate limit: {requests_per_minute} requests/minute")

    def _next_record_id(self) -> str:
        """Generate next sequential record ID."""
        self.record_counter += 1
        return f"R{self.record_counter:04d}"

    def _rate_limit_delay(self):
        """Apply rate limiting delay before making API call."""
        if not self.use_ai or self.delay_between_calls == 0:
            return

        # Calculate time since last API call
        time_since_last_call = time.time() - self.last_api_call_time

        # If not enough time has passed, wait
        if time_since_last_call < self.delay_between_calls:
            sleep_time = self.delay_between_calls - time_since_last_call
            time.sleep(sleep_time)

        # Update last call time
        self.last_api_call_time = time.time()

    def generate_medical_history(
        self,
        age: int,
        gender: str,
        conditions: Optional[List[Dict]] = None
    ) -> MedicalHistory:
        """
        Generate complete medical history for a patient.

        Args:
            age: Patient age
            gender: Patient gender (M/F/X)
            conditions: Pre-specified conditions (if None, generates random)

        Returns:
            MedicalHistory object
        """
        # Generate or use provided conditions
        if conditions is None:
            conditions = self.scenario_gen.generate_chronic_conditions(age)

        # Convert to MedicalCondition objects
        condition_objs = [
            MedicalCondition(
                name=c["name"],
                abbreviation=c.get("abbreviation"),
                onset_year=c.get("onset_year"),
                status=c.get("status", "active")
            )
            for c in conditions
        ]

        # Generate related medications
        medications = self.scenario_gen.generate_medications(conditions)

        # Generate allergies
        allergies = self.scenario_gen.generate_allergies()

        # Generate surgical history
        surgery_dicts = self.scenario_gen.generate_surgeries(age)
        surgeries = [
            Surgery(
                procedure=s["procedure"],
                date=s["date"]
            )
            for s in surgery_dicts
        ]

        # Generate family history
        family_history = self.scenario_gen.generate_family_history()

        # Social history (simplified)
        social_history = self._generate_social_history()

        return MedicalHistory(
            conditions=condition_objs,
            medications=medications,
            allergies=allergies,
            surgeries=surgeries,
            family_history=family_history,
            social_history=social_history
        )

    def _generate_social_history(self) -> str:
        """Generate simple social history."""
        smoking = random.choice([
            "Non-smoker",
            "Former smoker (quit 5 years ago)",
            "Current smoker (1 PPD)",
        ])
        alcohol = random.choice([
            "No alcohol use",
            "Social drinker",
            "Occasional alcohol use",
        ])
        return f"{smoking}. {alcohol}."

    def generate_clinical_notes_ai(
        self,
        patient_name: str,
        age: int,
        gender: str,
        medical_history: MedicalHistory,
        chief_complaint: str,
        use_abbreviations: bool = None
    ) -> str:
        """
        Generate clinical notes using AI.

        Args:
            patient_name: Patient name
            age: Patient age
            gender: Patient gender
            medical_history: Patient's medical history
            chief_complaint: Reason for visit
            use_abbreviations: If True, uses medical abbreviations heavily

        Returns:
            Clinical notes text
        """
        if not self.use_ai:
            return self._generate_clinical_notes_rule_based(
                patient_name, age, gender, medical_history, chief_complaint, use_abbreviations
            )

        if use_abbreviations is None:
            use_abbreviations = random.choice([True, False])

        # Build condition list for prompt
        conditions_text = ", ".join([
            c.clinical_text(use_abbrev=use_abbreviations)
            for c in medical_history.conditions
        ])

        medications_text = ", ".join(medical_history.medications[:3])  # First 3

        prompt = f"""Generate a realistic clinical note for a patient encounter.

Patient: {age}-year-old {gender}, {patient_name}
Chief Complaint: {chief_complaint}
Past Medical History: {conditions_text if conditions_text else "No significant PMH"}
Current Medications: {medications_text if medications_text else "None"}

Requirements:
- Write 2-3 sentences describing the history of present illness
- {"Use medical abbreviations where appropriate (HTN, T2DM, etc.)" if use_abbreviations else "Use full medical terminology"}
- Keep it concise and realistic
- Don't include a full H&P format, just the narrative portion

Example style: "58M with PMHx of HTN and T2DM presents with acute onset substernal chest pressure..."

Generate the clinical note:"""

        try:
            # Apply rate limiting delay
            self._rate_limit_delay()

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"AI generation failed: {e}. Using rule-based fallback.")
            return self._generate_clinical_notes_rule_based(
                patient_name, age, gender, medical_history, chief_complaint, use_abbreviations
            )

    def _generate_clinical_notes_rule_based(
        self,
        patient_name: str,
        age: int,
        gender: str,
        medical_history: MedicalHistory,
        chief_complaint: str,
        use_abbreviations: bool = None
    ) -> str:
        """Generate clinical notes using templates (fallback)."""
        if use_abbreviations is None:
            use_abbreviations = random.choice([True, False])

        gender_short = gender.upper()

        # Build PMH text
        if medical_history.conditions:
            pmh_parts = [
                c.clinical_text(use_abbrev=use_abbreviations)
                for c in medical_history.conditions[:3]  # Top 3
            ]
            pmh = f"PMHx of {', '.join(pmh_parts)}"
        else:
            pmh = "No significant PMH"

        templates = [
            f"{age}{gender_short} with {pmh} presents with {chief_complaint.lower()}. Symptoms began {random.choice(['today', 'yesterday', '2 days ago'])}.",
            f"{age}{gender_short} patient with history of {pmh} complains of {chief_complaint.lower()}. No prior episodes.",
            f"Patient is a {age}{gender_short} with {pmh} presenting for evaluation of {chief_complaint.lower()}.",
        ]

        return random.choice(templates)

    def generate_medical_record(
        self,
        record_id: str,
        patient_id: str,
        patient_name: str,
        age: int,
        gender: str,
        medical_history: MedicalHistory,
        record_source: str = "Primary Care",
        use_abbreviations: bool = None
    ) -> MedicalRecord:
        """
        Generate a complete medical record for an encounter.

        Args:
            record_id: Record identifier
            patient_id: Patient identifier
            patient_name: Patient name (for AI generation)
            age: Patient age
            gender: Patient gender
            medical_history: Patient's medical history
            record_source: Type of encounter
            use_abbreviations: Whether to use abbreviations in notes

        Returns:
            MedicalRecord object
        """
        # Generate chief complaint
        chief_complaint = self.scenario_gen.generate_chief_complaint()

        # Generate clinical notes
        clinical_notes = self.generate_clinical_notes_ai(
            patient_name, age, gender, medical_history, chief_complaint, use_abbreviations
        )

        # Generate vital signs
        vital_dict = self.scenario_gen.generate_vital_signs(
            age,
            [{"name": c.name} for c in medical_history.conditions]
        )
        vital_signs = VitalSigns(**vital_dict)

        # Generate assessment (simplified)
        if medical_history.conditions:
            # Use first condition as primary diagnosis
            primary_condition = medical_history.conditions[0]
            assessment = primary_condition.name
        else:
            assessment = "Routine examination, no acute findings"

        # Generate plan
        plan_options = [
            "Continue current medications, follow up in 3 months",
            "Labs ordered, imaging scheduled, return PRN",
            "Medication adjustment, recheck in 2 weeks",
            "Referral to specialist, continue monitoring",
        ]
        plan = random.choice(plan_options)

        # Record date (sometime in the past year)
        days_ago = random.randint(0, 365)
        record_date = datetime.now() - timedelta(days=days_ago)

        return MedicalRecord(
            record_id=record_id,
            patient_id=patient_id,
            record_source=record_source,
            record_date=record_date,
            chief_complaint=chief_complaint,
            medical_history=medical_history,
            vital_signs=vital_signs,
            clinical_notes=clinical_notes,
            assessment=assessment,
            plan=plan,
        )

    def generate_variant_record(
        self,
        base_record: MedicalRecord,
        use_different_abbreviations: bool = True
    ) -> MedicalRecord:
        """
        Generate a variant of an existing medical record.

        This creates a record for the same patient but from a different
        source/time, possibly with different terminology or abbreviations.

        Args:
            base_record: The original medical record
            use_different_abbreviations: Toggle abbreviation style

        Returns:
            New medical record with variations
        """
        # Same medical history, but might format differently
        new_notes = self.generate_clinical_notes_ai(
            "Patient",  # Generic name for variant
            40,  # Generic age
            "M",  # Generic gender
            base_record.medical_history,
            base_record.chief_complaint or "Follow-up",
            use_abbreviations=use_different_abbreviations
        )

        # Different record source
        sources = ["Lab", "Radiology", "Specialist Office", "Telemedicine"]
        new_source = random.choice(sources)

        # Different date
        days_offset = random.randint(30, 180)
        new_date = base_record.record_date + timedelta(days=days_offset)

        record_id = self._next_record_id()

        return MedicalRecord(
            record_id=record_id,
            patient_id=base_record.patient_id,
            record_source=new_source,
            record_date=new_date,
            chief_complaint="Follow-up visit",
            medical_history=base_record.medical_history,
            vital_signs=base_record.vital_signs,  # Reuse for simplicity
            clinical_notes=new_notes,
            assessment=base_record.assessment,
            plan="Continue current management",
        )
