"""
Demographics generator for synthetic patient data.

Generates realistic patient demographic records with variations,
errors, and edge cases for testing entity resolution.
"""

import random
from datetime import date
from typing import Optional, List
from faker import Faker

from ..models import Demographics, Address
from ..utils.name_utils import (
    NameGenerator,
    COMMON_MALE_NAMES,
    COMMON_FEMALE_NAMES,
    COMMON_LAST_NAMES,
)
from ..utils.date_utils import DateFormatter, DateErrorGenerator


class DemographicsGenerator:
    """Generate synthetic patient demographics with realistic variations."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize demographics generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)

        self.faker = Faker()
        self.name_gen = NameGenerator(seed)
        self.record_counter = 0

    def _next_record_id(self) -> str:
        """Generate next sequential record ID."""
        self.record_counter += 1
        return f"R{self.record_counter:04d}"

    def _generate_mrn(self) -> str:
        """Generate a realistic Medical Record Number."""
        # Format: 8 digits
        return str(random.randint(10000000, 99999999))

    def _generate_ssn_last4(self) -> str:
        """Generate last 4 digits of SSN."""
        return str(random.randint(1000, 9999))

    def _generate_address(self) -> Address:
        """Generate a random address using Faker."""
        return Address(
            street=self.faker.street_address(),
            city=self.faker.city(),
            state=self.faker.state_abbr(),
            zip_code=self.faker.zipcode()
        )

    def _format_phone(self, phone: str) -> str:
        """Format phone number in various styles."""
        # Remove all non-digits
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) >= 10:
            digits = digits[:10]  # Take first 10 digits

            # Various phone formats
            formats = [
                f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",  # (555) 123-4567
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",     # 555-123-4567
                f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",     # 555.123.4567
                digits,                                          # 5551234567
            ]
            return random.choice(formats)
        return phone

    def generate_base_demographics(
        self,
        patient_id: str,
        first_name: str,
        middle_name: Optional[str],
        last_name: str,
        suffix: Optional[str],
        dob: date,
        gender: str,
        is_common_name: bool = False
    ) -> Demographics:
        """
        Generate a base demographic record with clean data.

        Args:
            patient_id: Ground truth patient ID
            first_name: First name
            middle_name: Middle name (optional)
            last_name: Last name
            suffix: Suffix (Jr, Sr, etc.)
            dob: Date of birth
            gender: Gender (M/F/X)
            is_common_name: Whether this is a common name

        Returns:
            Demographics record
        """
        return Demographics(
            record_id=self._next_record_id(),
            patient_id=patient_id,
            name_first=first_name,
            name_middle=middle_name,
            name_last=last_name,
            name_suffix=suffix,
            date_of_birth=dob,
            gender=gender,
            mrn=self._generate_mrn(),
            ssn_last4=self._generate_ssn_last4(),
            phone=self._format_phone(self.faker.phone_number()),
            email=self.faker.email() if random.random() < 0.7 else None,
            address=self._generate_address(),
            record_source=random.choice([
                "Primary Care",
                "Emergency Department",
                "Outpatient Clinic",
                "Hospital Registration"
            ]),
            record_date=date.today(),
            data_quality_flag="clean"
        )

    def generate_name_variation(
        self,
        base_record: Demographics,
        variation_type: str
    ) -> Demographics:
        """
        Create a demographic record with name variation.

        Args:
            base_record: The clean base record
            variation_type: Type of variation
                          ("nickname", "no_middle", "middle_initial",
                           "accent", "married_name")

        Returns:
            New demographics record with name variation
        """
        # Copy base attributes
        new_first = base_record.name_first
        new_middle = base_record.name_middle
        new_last = base_record.name_last

        # Apply variation
        if variation_type == "nickname":
            new_first, new_middle, new_last = self.name_gen.apply_variation(
                new_first, new_middle, new_last, "nickname"
            )
        elif variation_type == "no_middle":
            new_middle = None
        elif variation_type == "middle_initial":
            if new_middle and len(new_middle) > 1:
                new_middle = new_middle[0]
        elif variation_type == "accent":
            new_first, new_middle, new_last = self.name_gen.apply_variation(
                new_first, new_middle, new_last, "accent"
            )
        elif variation_type == "married_name":
            # Generate a new last name
            new_last = random.choice(COMMON_LAST_NAMES)

        return Demographics(
            record_id=self._next_record_id(),
            patient_id=base_record.patient_id,
            name_first=new_first,
            name_middle=new_middle,
            name_last=new_last,
            name_suffix=base_record.name_suffix,
            date_of_birth=base_record.date_of_birth,
            gender=base_record.gender,
            mrn=self._generate_mrn(),  # Different MRN (different system)
            ssn_last4=base_record.ssn_last4,  # Same SSN
            phone=self._format_phone(self.faker.phone_number()),  # Might have new phone
            email=self.faker.email() if random.random() < 0.5 else base_record.email,
            address=base_record.address if random.random() < 0.7 else self._generate_address(),
            record_source=random.choice([
                "Lab System",
                "Radiology",
                "Pharmacy",
                "Specialist Office"
            ]),
            record_date=date.today(),
            data_quality_flag=f"name_variation_{variation_type}"
        )

    def generate_data_error(
        self,
        base_record: Demographics,
        error_type: str
    ) -> Demographics:
        """
        Create a demographic record with data entry error.

        Args:
            base_record: The clean base record
            error_type: Type of error
                       ("dob_typo", "name_typo", "name_misspelling",
                        "mrn_error", "transposed_names")

        Returns:
            New demographics record with error
        """
        new_first = base_record.name_first
        new_middle = base_record.name_middle
        new_last = base_record.name_last
        new_dob = base_record.date_of_birth

        if error_type == "dob_typo":
            # Apply date error
            new_dob = DateErrorGenerator.apply_random_error(base_record.date_of_birth)
        elif error_type == "name_typo":
            # Transpose characters in last name
            new_last, _, _ = self.name_gen.apply_variation(
                new_first, new_middle, new_last, "typo"
            )
        elif error_type == "name_misspelling":
            # Apply common misspelling
            _, _, new_last = self.name_gen.apply_variation(
                new_first, new_middle, new_last, "misspelling"
            )
        elif error_type == "mrn_error":
            # This will be handled with a different MRN
            pass
        elif error_type == "transposed_names":
            # Swap first and last name (data entry error)
            new_first, new_last = new_last, new_first

        return Demographics(
            record_id=self._next_record_id(),
            patient_id=base_record.patient_id,
            name_first=new_first,
            name_middle=new_middle,
            name_last=new_last,
            name_suffix=base_record.name_suffix,
            date_of_birth=new_dob,
            gender=base_record.gender,
            mrn=self._generate_mrn(),  # Always different MRN
            ssn_last4=base_record.ssn_last4,  # SSN should match
            phone=base_record.phone,  # Phone might match
            email=base_record.email,
            address=base_record.address,
            record_source="Emergency Department",  # Often errors in ED
            record_date=date.today(),
            data_quality_flag=f"data_error_{error_type}"
        )

    def generate_different_source(
        self,
        base_record: Demographics,
    ) -> Demographics:
        """
        Create a record from a different source system.

        This simulates the same patient in different hospital systems
        with minor data inconsistencies.

        Args:
            base_record: The base record

        Returns:
            New demographics record from different source
        """
        # Might have slight variations due to different data entry
        variations = ["clean", "middle_initial", "no_middle"]
        variation = random.choice(variations)

        new_middle = base_record.name_middle
        if variation == "middle_initial" and new_middle and len(new_middle) > 1:
            new_middle = new_middle[0]
        elif variation == "no_middle":
            new_middle = None

        # Might have moved or updated contact info
        has_moved = random.random() < 0.3
        has_new_phone = random.random() < 0.4

        return Demographics(
            record_id=self._next_record_id(),
            patient_id=base_record.patient_id,
            name_first=base_record.name_first,
            name_middle=new_middle,
            name_last=base_record.name_last,
            name_suffix=base_record.name_suffix,
            date_of_birth=base_record.date_of_birth,
            gender=base_record.gender,
            mrn=self._generate_mrn(),  # Different system = different MRN
            ssn_last4=base_record.ssn_last4,
            phone=self._format_phone(self.faker.phone_number()) if has_new_phone else base_record.phone,
            email=self.faker.email() if random.random() < 0.3 else base_record.email,
            address=self._generate_address() if has_moved else base_record.address,
            record_source=random.choice([
                "External Hospital",
                "Imaging Center",
                "Surgery Center",
                "Urgent Care"
            ]),
            record_date=date.today(),
            data_quality_flag="different_source"
        )
