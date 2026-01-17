"""
Edge case generators for challenging patient matching scenarios.

Generates twins, family members, common name collisions, and other
difficult cases for testing entity resolution algorithms.
"""

import random
from datetime import date
from typing import Tuple, List, Optional
from faker import Faker

from ..models import Patient
from ..utils.name_utils import (
    COMMON_MALE_NAMES,
    COMMON_FEMALE_NAMES,
    COMMON_LAST_NAMES,
)
from ..utils.date_utils import DateGenerator


class EdgeCaseGenerator:
    """Generate challenging edge cases for patient matching."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize edge case generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)

        self.faker = Faker()
        self.date_gen = DateGenerator()
        self.patient_counter = 0

    def _next_patient_id(self) -> str:
        """Generate next sequential patient ID."""
        self.patient_counter += 1
        return f"P{self.patient_counter:04d}"

    def generate_twin_pair(
        self,
        gender: Optional[str] = None,
        identical: bool = None
    ) -> Tuple[Patient, Patient]:
        """
        Generate a pair of twins with same/similar DOB and shared characteristics.

        Args:
            gender: Gender for twins (if None, random; if specified, both have same gender for identical)
            identical: If True, identical twins (same gender, very similar names)

        Returns:
            Tuple of (twin1, twin2)
        """
        if identical is None:
            identical = random.choice([True, False])

        # Identical twins must be same gender
        if identical:
            if gender is None:
                gender = random.choice(["M", "F"])
            gender1 = gender2 = gender
        else:
            # Fraternal twins might be different genders
            if gender is None:
                gender1 = random.choice(["M", "F"])
                gender2 = random.choice(["M", "F"])
            else:
                gender1 = gender2 = gender

        # Generate DOB
        dob1 = self.date_gen.generate_dob(min_age=20, max_age=70)
        dob2 = self.date_gen.generate_twin_dob(dob1)

        # Shared last name
        last_name = random.choice(COMMON_LAST_NAMES)

        # Generate names
        if gender1 == "M":
            first_name1 = self._generate_twin_name(gender1, identical)
        else:
            first_name1 = self._generate_twin_name(gender1, identical)

        if gender2 == "M":
            first_name2 = self._generate_twin_name(gender2, identical, sibling_name=first_name1)
        else:
            first_name2 = self._generate_twin_name(gender2, identical, sibling_name=first_name1)

        # Middle names
        middle1 = random.choice(["A", "B", "C", "D", "E", "J", "L", "M"])
        middle2 = random.choice(["A", "B", "C", "D", "E", "J", "L", "M"])

        # Suffixes for differentiation (sometimes)
        suffix1 = suffix2 = None
        if random.random() < 0.2:  # 20% chance
            suffix1 = "I"
            suffix2 = "II"

        # Create twin 1
        twin1_id = self._next_patient_id()
        twin2_id = self._next_patient_id()

        twin1 = Patient(
            patient_id=twin1_id,
            true_first_name=first_name1,
            true_middle_name=middle1,
            true_last_name=last_name,
            true_suffix=suffix1,
            true_dob=dob1,
            true_gender=gender1,
            has_twin=True,
            twin_id=twin2_id
        )

        twin2 = Patient(
            patient_id=twin2_id,
            true_first_name=first_name2,
            true_middle_name=middle2,
            true_last_name=last_name,
            true_suffix=suffix2,
            true_dob=dob2,
            true_gender=gender2,
            has_twin=True,
            twin_id=twin1_id
        )

        return twin1, twin2

    def _generate_twin_name(
        self,
        gender: str,
        identical: bool,
        sibling_name: Optional[str] = None
    ) -> str:
        """Generate a name for a twin."""
        if identical and sibling_name:
            # Similar sounding names for identical twins
            similar_pairs = {
                "M": [("James", "Jason"), ("Michael", "Mitchell"), ("Daniel", "David")],
                "F": [("Mary", "Marie"), ("Jessica", "Jennifer"), ("Emily", "Emma")]
            }
            pairs = similar_pairs.get(gender, [])
            if pairs:
                pair = random.choice(pairs)
                # Return the one that's not the sibling name
                return pair[1] if sibling_name == pair[0] else pair[0]

        # Otherwise, just a random name
        if gender == "M":
            return random.choice(COMMON_MALE_NAMES)
        else:
            return random.choice(COMMON_FEMALE_NAMES)

    def generate_sibling_pair(self) -> Tuple[Patient, Patient]:
        """
        Generate a pair of siblings with shared last name and similar demographics.

        Returns:
            Tuple of (sibling1, sibling2)
        """
        # Shared last name
        last_name = random.choice(COMMON_LAST_NAMES)

        # Random genders
        gender1 = random.choice(["M", "F"])
        gender2 = random.choice(["M", "F"])

        # Generate DOBs with age gap
        dob1 = self.date_gen.generate_dob(min_age=25, max_age=70)
        dob2 = self.date_gen.generate_sibling_dob(dob1, min_gap_years=2, max_gap_years=8)

        # Different first names
        if gender1 == "M":
            first1 = random.choice(COMMON_MALE_NAMES)
        else:
            first1 = random.choice(COMMON_FEMALE_NAMES)

        if gender2 == "M":
            first2 = random.choice(COMMON_MALE_NAMES)
        else:
            first2 = random.choice(COMMON_FEMALE_NAMES)

        # Middle names
        middle1 = random.choice(["A", "B", "C", "D", "E", "J", "L", "M"])
        middle2 = random.choice(["A", "B", "C", "D", "E", "J", "L", "M"])

        sib1_id = self._next_patient_id()
        sib2_id = self._next_patient_id()

        sibling1 = Patient(
            patient_id=sib1_id,
            true_first_name=first1,
            true_middle_name=middle1,
            true_last_name=last_name,
            true_dob=dob1,
            true_gender=gender1,
            family_member_ids=[sib2_id]
        )

        sibling2 = Patient(
            patient_id=sib2_id,
            true_first_name=first2,
            true_middle_name=middle2,
            true_last_name=last_name,
            true_dob=dob2,
            true_gender=gender2,
            family_member_ids=[sib1_id]
        )

        return sibling1, sibling2

    def generate_parent_child_pair(self) -> Tuple[Patient, Patient]:
        """
        Generate a parent-child pair (e.g., John Smith Sr. and John Smith Jr.).

        Returns:
            Tuple of (parent, child)
        """
        # Same gender for Jr/Sr naming
        gender = random.choice(["M", "F"])

        # Shared last name
        last_name = random.choice(COMMON_LAST_NAMES)

        # Shared first name (Jr/Sr pattern)
        if gender == "M":
            first_name = random.choice(COMMON_MALE_NAMES)
        else:
            first_name = random.choice(COMMON_FEMALE_NAMES)

        # Middle name (might be same or different)
        if random.random() < 0.7:
            # Same middle name
            middle_parent = middle_child = random.choice(["A", "B", "C", "J", "L", "M"])
        else:
            # Different middle names
            middle_parent = random.choice(["A", "B", "C", "J", "L", "M"])
            middle_child = random.choice(["A", "B", "C", "J", "L", "M"])

        # Generate ages (parent 25-45 years older)
        child_dob = self.date_gen.generate_dob(min_age=20, max_age=50)
        age_gap_years = random.randint(25, 45)
        parent_dob = date(
            child_dob.year - age_gap_years,
            child_dob.month,
            child_dob.day
        )

        parent_id = self._next_patient_id()
        child_id = self._next_patient_id()

        parent = Patient(
            patient_id=parent_id,
            true_first_name=first_name,
            true_middle_name=middle_parent,
            true_last_name=last_name,
            true_suffix="Sr",
            true_dob=parent_dob,
            true_gender=gender,
            family_member_ids=[child_id]
        )

        child = Patient(
            patient_id=child_id,
            true_first_name=first_name,
            true_middle_name=middle_child,
            true_last_name=last_name,
            true_suffix="Jr",
            true_dob=child_dob,
            true_gender=gender,
            family_member_ids=[parent_id]
        )

        return parent, child

    def generate_common_name_collision(
        self,
        num_patients: int = 3
    ) -> List[Patient]:
        """
        Generate multiple DIFFERENT patients with the same/very similar names.

        This is the classic wrong-patient error scenario: multiple John Smiths.

        Args:
            num_patients: Number of patients with similar names

        Returns:
            List of patients with similar names
        """
        # Pick a common name combination
        gender = random.choice(["M", "F"])
        if gender == "M":
            first_name = random.choice(COMMON_MALE_NAMES[:5])  # Most common
        else:
            first_name = random.choice(COMMON_FEMALE_NAMES[:5])

        last_name = random.choice(COMMON_LAST_NAMES[:10])  # Most common

        patients = []
        for i in range(num_patients):
            # Same first and last, but different middle, DOB, etc.
            middle = random.choice(["A", "B", "C", "D", "E", "J", "L", "M", "R", "S"])

            # Different DOBs (but could be close!)
            dob = self.date_gen.generate_dob(min_age=25, max_age=70)

            # Sometimes add suffix for differentiation
            suffix = None
            if i > 0 and random.random() < 0.3:
                suffix = random.choice(["Jr", "Sr", "II", "III"])

            patient_id = self._next_patient_id()

            patient = Patient(
                patient_id=patient_id,
                true_first_name=first_name,
                true_middle_name=middle,
                true_last_name=last_name,
                true_suffix=suffix,
                true_dob=dob,
                true_gender=gender,
                has_common_name=True
            )

            patients.append(patient)

        return patients

    def generate_same_name_same_dob(self) -> Tuple[Patient, Patient]:
        """
        Generate two DIFFERENT patients with same name AND same DOB.

        This is the ultimate challenge case - twins born same day at same hospital,
        or pure coincidence. Algorithm MUST use medical history to differentiate.

        Returns:
            Tuple of (patient1, patient2)
        """
        # Same name
        gender = random.choice(["M", "F"])
        if gender == "M":
            first_name = random.choice(COMMON_MALE_NAMES)
        else:
            first_name = random.choice(COMMON_FEMALE_NAMES)

        last_name = random.choice(COMMON_LAST_NAMES)

        # SAME DOB
        dob = self.date_gen.generate_dob(min_age=25, max_age=70)

        # Different middle names (only differentiator besides medical history)
        middle1 = random.choice(["A", "B", "C"])
        middle2 = random.choice(["D", "E", "F"])

        patient1_id = self._next_patient_id()
        patient2_id = self._next_patient_id()

        patient1 = Patient(
            patient_id=patient1_id,
            true_first_name=first_name,
            true_middle_name=middle1,
            true_last_name=last_name,
            true_dob=dob,
            true_gender=gender,
            has_common_name=True
        )

        patient2 = Patient(
            patient_id=patient2_id,
            true_first_name=first_name,
            true_middle_name=middle2,
            true_last_name=last_name,
            true_dob=dob,
            true_gender=gender,
            has_common_name=True
        )

        return patient1, patient2
