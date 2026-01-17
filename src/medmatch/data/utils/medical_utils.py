"""
Medical terminology and condition utilities for synthetic data generation.

Handles medical abbreviations, condition templates, and realistic medical scenarios.
"""

import random
from typing import List, Dict, Tuple, Optional
from datetime import date


# Common medical conditions with their abbreviations
MEDICAL_CONDITIONS = {
    "Hypertension": {"abbrev": "HTN", "prevalence": "high", "chronic": True},
    "Type 2 Diabetes": {"abbrev": "T2DM", "prevalence": "high", "chronic": True},
    "Type 1 Diabetes": {"abbrev": "T1DM", "prevalence": "low", "chronic": True},
    "Coronary Artery Disease": {"abbrev": "CAD", "prevalence": "medium", "chronic": True},
    "Congestive Heart Failure": {"abbrev": "CHF", "prevalence": "medium", "chronic": True},
    "Chronic Obstructive Pulmonary Disease": {"abbrev": "COPD", "prevalence": "medium", "chronic": True},
    "Asthma": {"abbrev": "asthma", "prevalence": "medium", "chronic": True},
    "Gastroesophageal Reflux Disease": {"abbrev": "GERD", "prevalence": "high", "chronic": True},
    "Hyperlipidemia": {"abbrev": "HLD", "prevalence": "high", "chronic": True},
    "Hypothyroidism": {"abbrev": "hypothyroid", "prevalence": "medium", "chronic": True},
    "Atrial Fibrillation": {"abbrev": "AFib", "prevalence": "medium", "chronic": True},
    "Chronic Kidney Disease": {"abbrev": "CKD", "prevalence": "medium", "chronic": True},
    "Osteoarthritis": {"abbrev": "OA", "prevalence": "high", "chronic": True},
    "Depression": {"abbrev": "MDD", "prevalence": "medium", "chronic": True},
    "Anxiety Disorder": {"abbrev": "GAD", "prevalence": "medium", "chronic": True},
    "Obesity": {"abbrev": "obesity", "prevalence": "high", "chronic": True},
}

# Acute conditions for emergency department visits
ACUTE_CONDITIONS = {
    "Acute Coronary Syndrome": {"abbrev": "ACS", "urgent": True},
    "Myocardial Infarction": {"abbrev": "MI", "urgent": True},
    "Pneumonia": {"abbrev": "PNA", "urgent": True},
    "Urinary Tract Infection": {"abbrev": "UTI", "urgent": False},
    "Cellulitis": {"abbrev": "cellulitis", "urgent": False},
    "Acute Bronchitis": {"abbrev": "acute bronchitis", "urgent": False},
    "Gastroenteritis": {"abbrev": "gastroenteritis", "urgent": False},
}

# Medication mappings to conditions
CONDITION_MEDICATIONS = {
    "Hypertension": [
        "Lisinopril 10mg daily",
        "Amlodipine 5mg daily",
        "Losartan 50mg daily",
        "Metoprolol 25mg BID",
        "HCTZ 25mg daily",
    ],
    "Type 2 Diabetes": [
        "Metformin 500mg BID",
        "Metformin 1000mg BID",
        "Glipizide 5mg daily",
        "Insulin glargine 20 units qHS",
    ],
    "Type 1 Diabetes": [
        "Insulin aspart 6 units TID with meals",
        "Insulin glargine 15 units qHS",
    ],
    "Hyperlipidemia": [
        "Atorvastatin 20mg qHS",
        "Simvastatin 40mg qHS",
        "Rosuvastatin 10mg daily",
    ],
    "Hypothyroidism": [
        "Levothyroxine 50mcg daily",
        "Levothyroxine 75mcg daily",
        "Levothyroxine 100mcg daily",
    ],
    "GERD": [
        "Omeprazole 20mg daily",
        "Pantoprazole 40mg daily",
        "Famotidine 20mg BID",
    ],
    "Asthma": [
        "Albuterol inhaler 2 puffs q4-6h PRN",
        "Fluticasone inhaler 2 puffs BID",
    ],
    "COPD": [
        "Albuterol inhaler PRN",
        "Tiotropium 18mcg inhaled daily",
    ],
    "Atrial Fibrillation": [
        "Warfarin 5mg daily",
        "Apixaban 5mg BID",
        "Metoprolol 50mg BID",
    ],
    "Depression": [
        "Sertraline 50mg daily",
        "Escitalopram 10mg daily",
        "Fluoxetine 20mg daily",
    ],
    "Anxiety Disorder": [
        "Escitalopram 10mg daily",
        "Buspirone 15mg BID",
    ],
}

# Common allergies
COMMON_ALLERGIES = [
    "Penicillin (rash)",
    "Sulfa drugs (hives)",
    "Codeine (nausea)",
    "Latex (contact dermatitis)",
    "Shellfish (anaphylaxis)",
    "NKDA",  # No known drug allergies
]

# Common surgical procedures
COMMON_SURGERIES = [
    "Appendectomy",
    "Cholecystectomy",
    "Hernia repair",
    "Knee arthroscopy",
    "Coronary artery bypass graft",
    "Hip replacement",
    "Hysterectomy",
    "C-section",
    "Tonsillectomy",
    "Cataract surgery",
]

# Family history patterns
FAMILY_HISTORY_CONDITIONS = [
    "Father: MI at age 55",
    "Mother: breast cancer at age 62",
    "Father: Type 2 diabetes",
    "Mother: hypertension",
    "Sibling: asthma",
    "Maternal grandmother: Alzheimer's disease",
    "Paternal grandfather: stroke",
]


class MedicalScenarioGenerator:
    """Generate realistic medical scenarios for synthetic patients."""

    @staticmethod
    def generate_chronic_conditions(age: int, num_conditions: Optional[int] = None) -> List[Dict]:
        """
        Generate age-appropriate chronic conditions.

        Args:
            age: Patient age
            num_conditions: Number of conditions (if None, randomly determined)

        Returns:
            List of condition dictionaries with name, abbreviation, onset
        """
        if num_conditions is None:
            # Older patients tend to have more conditions
            if age < 30:
                num_conditions = random.randint(0, 1)
            elif age < 50:
                num_conditions = random.randint(0, 2)
            elif age < 70:
                num_conditions = random.randint(1, 3)
            else:
                num_conditions = random.randint(2, 5)

        # Weight by prevalence
        high_prev = [k for k, v in MEDICAL_CONDITIONS.items() if v["prevalence"] == "high"]
        medium_prev = [k for k, v in MEDICAL_CONDITIONS.items() if v["prevalence"] == "medium"]
        low_prev = [k for k, v in MEDICAL_CONDITIONS.items() if v["prevalence"] == "low"]

        # Create weighted pool
        pool = high_prev * 3 + medium_prev * 2 + low_prev

        # Select random conditions
        selected = random.sample(pool, min(num_conditions, len(pool)))

        conditions = []
        current_year = date.today().year
        for condition_name in selected:
            condition_info = MEDICAL_CONDITIONS[condition_name]

            # Generate onset year (sometime before current age)
            max_years_ago = min(age - 18, 30)  # At least 18, max 30 years ago
            if max_years_ago > 0:
                years_ago = random.randint(1, max_years_ago)
                onset_year = current_year - years_ago
            else:
                onset_year = current_year - 1

            conditions.append({
                "name": condition_name,
                "abbreviation": condition_info["abbrev"],
                "onset_year": onset_year,
                "status": "chronic"
            })

        return conditions

    @staticmethod
    def generate_medications(conditions: List[Dict]) -> List[str]:
        """
        Generate medications appropriate for the patient's conditions.

        Args:
            conditions: List of patient conditions

        Returns:
            List of medication strings
        """
        medications = []

        for condition in conditions:
            condition_name = condition["name"]
            if condition_name in CONDITION_MEDICATIONS:
                # Pick 1-2 medications for this condition
                available_meds = CONDITION_MEDICATIONS[condition_name]
                num_meds = min(random.randint(1, 2), len(available_meds))
                selected_meds = random.sample(available_meds, num_meds)
                medications.extend(selected_meds)

        return medications

    @staticmethod
    def generate_allergies(num_allergies: Optional[int] = None) -> List[str]:
        """
        Generate patient allergies.

        Args:
            num_allergies: Number of allergies (if None, randomly determined)

        Returns:
            List of allergy strings
        """
        if num_allergies is None:
            # Most patients have 0-2 allergies
            num_allergies = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]

        if num_allergies == 0:
            return ["NKDA"]

        # Don't include NKDA if there are allergies
        allergy_pool = [a for a in COMMON_ALLERGIES if a != "NKDA"]
        return random.sample(allergy_pool, min(num_allergies, len(allergy_pool)))

    @staticmethod
    def generate_surgeries(age: int, num_surgeries: Optional[int] = None) -> List[Dict]:
        """
        Generate surgical history.

        Args:
            age: Patient age
            num_surgeries: Number of surgeries (if None, randomly determined)

        Returns:
            List of surgery dictionaries with procedure, date
        """
        if num_surgeries is None:
            # Chance of surgeries increases with age
            if age < 30:
                num_surgeries = random.choices([0, 1], weights=[0.7, 0.3])[0]
            elif age < 50:
                num_surgeries = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            else:
                num_surgeries = random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.2, 0.2])[0]

        surgeries = []
        current_year = date.today().year

        for _ in range(num_surgeries):
            procedure = random.choice(COMMON_SURGERIES)

            # Surgery sometime in the past (at least 1 year ago, max age-10 years)
            max_years_ago = min(age - 10, 40)
            if max_years_ago > 1:
                years_ago = random.randint(1, max_years_ago)
            else:
                years_ago = 1

            surgery_year = current_year - years_ago
            surgery_month = random.randint(1, 12)
            surgery_day = random.randint(1, 28)

            surgeries.append({
                "procedure": procedure,
                "date": date(surgery_year, surgery_month, surgery_day)
            })

        return surgeries

    @staticmethod
    def generate_family_history(num_conditions: Optional[int] = None) -> List[str]:
        """
        Generate family medical history.

        Args:
            num_conditions: Number of family history items

        Returns:
            List of family history strings
        """
        if num_conditions is None:
            num_conditions = random.randint(0, 3)

        if num_conditions == 0:
            return []

        return random.sample(FAMILY_HISTORY_CONDITIONS,
                           min(num_conditions, len(FAMILY_HISTORY_CONDITIONS)))

    @staticmethod
    def generate_vital_signs(age: int, conditions: List[Dict]) -> Dict:
        """
        Generate realistic vital signs based on age and conditions.

        Args:
            age: Patient age
            conditions: List of patient conditions

        Returns:
            Dictionary of vital signs
        """
        # Check for conditions that affect vitals
        has_htn = any("Hypertension" in c["name"] for c in conditions)
        has_diabetes = any("Diabetes" in c["name"] for c in conditions)

        # Blood pressure
        if has_htn:
            # Elevated BP for hypertensive patients
            bp_sys = random.randint(140, 170)
            bp_dia = random.randint(85, 100)
        else:
            # Normal BP
            bp_sys = random.randint(110, 135)
            bp_dia = random.randint(70, 85)

        # Heart rate
        hr = random.randint(60, 95)

        # Respiratory rate
        rr = random.randint(12, 20)

        # Temperature (mostly normal)
        temp = round(random.uniform(97.5, 98.8), 1)

        # O2 saturation
        o2_sat = random.randint(95, 99)

        # Height (in inches)
        height = random.randint(60, 75)

        # Weight (in lbs) - higher if obese/diabetic
        if has_diabetes or any("Obesity" in c["name"] for c in conditions):
            weight = random.randint(180, 280)
        else:
            weight = random.randint(120, 200)

        return {
            "blood_pressure_systolic": bp_sys,
            "blood_pressure_diastolic": bp_dia,
            "heart_rate": hr,
            "respiratory_rate": rr,
            "temperature_f": temp,
            "oxygen_saturation": o2_sat,
            "height_inches": height,
            "weight_lbs": weight,
        }

    @staticmethod
    def generate_chief_complaint() -> str:
        """Generate a chief complaint for an encounter."""
        complaints = [
            "Chest pain",
            "Shortness of breath",
            "Abdominal pain",
            "Headache",
            "Back pain",
            "Cough",
            "Fever",
            "Dizziness",
            "Nausea and vomiting",
            "Routine checkup",
            "Medication refill",
            "Follow-up visit",
        ]
        return random.choice(complaints)

    @staticmethod
    def format_condition_text(condition: Dict, use_abbreviation: bool = None) -> str:
        """
        Format a condition as clinical text.

        Args:
            condition: Condition dictionary
            use_abbreviation: If True, uses abbreviation; if None, random

        Returns:
            Formatted condition string
        """
        if use_abbreviation is None:
            use_abbreviation = random.choice([True, False])

        if use_abbreviation and condition.get("abbreviation"):
            text = condition["abbreviation"]
        else:
            text = condition["name"]

        if condition.get("onset_year"):
            text += f" (since {condition['onset_year']})"

        return text
