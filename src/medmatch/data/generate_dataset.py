#!/usr/bin/env python3
"""
Main script for generating synthetic patient dataset.

This script orchestrates the generation of a complete synthetic dataset
with challenging edge cases for testing patient matching algorithms.
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import date, datetime
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from .models import Patient, Demographics, MedicalRecord, GroundTruth
from .generators import DemographicsGenerator, MedicalRecordGenerator, EdgeCaseGenerator
from .utils import DateGenerator
from .utils.name_utils import COMMON_MALE_NAMES, COMMON_FEMALE_NAMES, COMMON_LAST_NAMES


class SyntheticDatasetGenerator:
    """Orchestrates generation of complete synthetic patient dataset."""

    def __init__(
        self,
        num_patients: int = 75,
        output_dir: str = "data/synthetic",
        seed: int = 42,
        use_ai: bool = True,
        api_rate_limit: int = 5
    ):
        """
        Initialize dataset generator.

        Args:
            num_patients: Target number of unique patients
            output_dir: Directory for output files
            seed: Random seed for reproducibility
            use_ai: Whether to use AI for medical record generation
            api_rate_limit: Max API requests per minute (5 for free tier, 15+ for paid)
        """
        self.num_patients = num_patients
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.use_ai = use_ai
        self.api_rate_limit = api_rate_limit

        # Set random seed
        random.seed(seed)

        # Initialize generators
        self.demo_gen = DemographicsGenerator(seed=seed)
        self.medical_gen = MedicalRecordGenerator(
            use_ai=use_ai,
            requests_per_minute=api_rate_limit
        )
        self.edge_gen = EdgeCaseGenerator(seed=seed)
        self.date_gen = DateGenerator()

        # Storage
        self.patients: List[Patient] = []
        self.all_demographics: List[Demographics] = []
        self.all_medical_records: List[MedicalRecord] = []
        self.ground_truth: List[GroundTruth] = []

        # Statistics
        self.stats = {
            "total_patients": 0,
            "total_demographic_records": 0,
            "total_medical_records": 0,
            "edge_cases": {
                "twins": 0,
                "siblings": 0,
                "parent_child": 0,
                "common_name_collision": 0,
                "same_name_same_dob": 0,
            },
            "variations": {
                "name_variations": 0,
                "data_errors": 0,
                "different_source": 0,
            }
        }

    def generate_dataset(self):
        """Generate the complete synthetic dataset."""
        print(f"Generating synthetic dataset with ~{self.num_patients} patients...")
        print(f"Random seed: {self.seed}")
        print(f"AI-assisted generation: {self.use_ai}")
        print()

        # Calculate distribution of edge cases
        num_twins_pairs = int(self.num_patients * 0.10)  # 10% twins
        num_sibling_pairs = int(self.num_patients * 0.05)  # 5% siblings
        num_parent_child = int(self.num_patients * 0.05)  # 5% parent-child
        num_common_collisions = int(self.num_patients * 0.20)  # 20% common names
        num_same_name_dob = 2  # 2 pairs of same name + same DOB

        print("Edge case distribution:")
        print(f"  - Twin pairs: {num_twins_pairs}")
        print(f"  - Sibling pairs: {num_sibling_pairs}")
        print(f"  - Parent-child pairs: {num_parent_child}")
        print(f"  - Common name collisions: {num_common_collisions // 3} groups of 3")
        print(f"  - Same name + same DOB: {num_same_name_dob} pairs")
        print()

        # Generate edge cases
        self._generate_twins(num_twins_pairs)
        self._generate_siblings(num_sibling_pairs)
        self._generate_parent_child(num_parent_child)
        self._generate_common_name_collisions(num_common_collisions)
        self._generate_same_name_same_dob(num_same_name_dob)

        # Fill remaining with regular patients
        remaining = self.num_patients - len(self.patients)
        self._generate_regular_patients(remaining)

        print(f"\nGenerated {len(self.patients)} unique patients")

        # Generate demographic and medical records for each patient
        print("\nGenerating demographic and medical records...")
        for patient in tqdm(self.patients, desc="Processing patients"):
            self._generate_patient_records(patient)

        print(f"\nGenerated {len(self.all_demographics)} demographic records")
        print(f"Generated {len(self.all_medical_records)} medical records")

        # Create ground truth
        self._create_ground_truth()

        # Print statistics
        self._print_statistics()

    def _generate_twins(self, num_pairs: int):
        """Generate twin pairs."""
        print(f"Generating {num_pairs} twin pairs...")
        for _ in range(num_pairs):
            twin1, twin2 = self.edge_gen.generate_twin_pair()
            self.patients.extend([twin1, twin2])
            self.stats["edge_cases"]["twins"] += 2

    def _generate_siblings(self, num_pairs: int):
        """Generate sibling pairs."""
        print(f"Generating {num_pairs} sibling pairs...")
        for _ in range(num_pairs):
            sib1, sib2 = self.edge_gen.generate_sibling_pair()
            self.patients.extend([sib1, sib2])
            self.stats["edge_cases"]["siblings"] += 2

    def _generate_parent_child(self, num_pairs: int):
        """Generate parent-child pairs."""
        print(f"Generating {num_pairs} parent-child pairs...")
        for _ in range(num_pairs):
            parent, child = self.edge_gen.generate_parent_child_pair()
            self.patients.extend([parent, child])
            self.stats["edge_cases"]["parent_child"] += 2

    def _generate_common_name_collisions(self, num_patients: int):
        """Generate common name collisions (groups of 3)."""
        num_groups = num_patients // 3
        print(f"Generating {num_groups} common name collision groups...")
        for _ in range(num_groups):
            collision_group = self.edge_gen.generate_common_name_collision(num_patients=3)
            self.patients.extend(collision_group)
            self.stats["edge_cases"]["common_name_collision"] += 3

    def _generate_same_name_same_dob(self, num_pairs: int):
        """Generate same name + same DOB pairs."""
        print(f"Generating {num_pairs} same name + same DOB pairs...")
        for _ in range(num_pairs):
            p1, p2 = self.edge_gen.generate_same_name_same_dob()
            self.patients.extend([p1, p2])
            self.stats["edge_cases"]["same_name_same_dob"] += 2

    def _generate_regular_patients(self, num_patients: int):
        """Generate regular patients without special edge cases."""
        if num_patients <= 0:
            return

        print(f"Generating {num_patients} regular patients...")
        for _ in range(num_patients):
            patient = self._create_random_patient()
            self.patients.append(patient)

    def _create_random_patient(self) -> Patient:
        """Create a random patient."""
        gender = random.choice(["M", "F"])

        if gender == "M":
            first = random.choice(COMMON_MALE_NAMES)
        else:
            first = random.choice(COMMON_FEMALE_NAMES)

        last = random.choice(COMMON_LAST_NAMES)
        middle = random.choice(["A", "B", "C", "D", "E", "J", "L", "M", "R", "S", None])
        dob = self.date_gen.generate_dob(min_age=18, max_age=85)

        patient_id = f"P{len(self.patients) + 1:04d}"

        return Patient(
            patient_id=patient_id,
            true_first_name=first,
            true_middle_name=middle,
            true_last_name=last,
            true_dob=dob,
            true_gender=gender
        )

    def _generate_patient_records(self, patient: Patient):
        """Generate demographic and medical records for a patient."""
        # Generate medical history first
        age = patient.age if hasattr(patient, 'age') else (
            date.today().year - patient.true_dob.year
        )

        medical_history = self.medical_gen.generate_medical_history(
            age=age,
            gender=patient.true_gender
        )

        # Generate 2-5 demographic records per patient
        # Change to (2, 3) for faster generation or (3, 7) for more variations
        num_demo_records = random.randint(2, 5)

        # First record: clean baseline
        base_demo = self.demo_gen.generate_base_demographics(
            patient_id=patient.patient_id,
            first_name=patient.true_first_name,
            middle_name=patient.true_middle_name,
            last_name=patient.true_last_name,
            suffix=patient.true_suffix,
            dob=patient.true_dob,
            gender=patient.true_gender,
            is_common_name=patient.has_common_name
        )
        self.all_demographics.append(base_demo)
        patient.demographic_records.append(base_demo)

        # Generate variations
        for _ in range(num_demo_records - 1):
            variation_type = random.choice([
                "name_variation",
                "data_error",
                "different_source"
            ])

            if variation_type == "name_variation":
                variant_type = random.choice([
                    "nickname", "no_middle", "middle_initial", "accent"
                ])
                variant = self.demo_gen.generate_name_variation(base_demo, variant_type)
                self.stats["variations"]["name_variations"] += 1
            elif variation_type == "data_error":
                error_type = random.choice([
                    "dob_typo", "name_typo", "name_misspelling"
                ])
                variant = self.demo_gen.generate_data_error(base_demo, error_type)
                self.stats["variations"]["data_errors"] += 1
            else:  # different_source
                variant = self.demo_gen.generate_different_source(base_demo)
                self.stats["variations"]["different_source"] += 1

            self.all_demographics.append(variant)
            patient.demographic_records.append(variant)

        # Generate 1-2 medical records per patient
        # Change to (1, 1) for faster generation or (2, 4) for more records
        num_medical_records = random.randint(1, 2)

        for i in range(num_medical_records):
            record = self.medical_gen.generate_medical_record(
                record_id=f"MR{len(self.all_medical_records) + 1:04d}",
                patient_id=patient.patient_id,
                patient_name=patient.true_full_name,
                age=age,
                gender=patient.true_gender,
                medical_history=medical_history,
                record_source=random.choice([
                    "Primary Care", "Emergency Department",
                    "Specialist Office", "Hospital"
                ]),
                use_abbreviations=random.choice([True, False])
            )
            self.all_medical_records.append(record)
            patient.medical_records.append(record)

    def _create_ground_truth(self):
        """Create ground truth mapping for all records."""
        print("\nCreating ground truth mappings...")

        match_group_counter = 0

        for patient in self.patients:
            match_group_counter += 1
            match_group = f"G{match_group_counter:04d}"

            # Create entry for each demographic record
            for demo_record in patient.demographic_records:
                # Determine difficulty
                if patient.has_twin or any(
                    p.patient_id != patient.patient_id and
                    p.true_first_name == patient.true_first_name and
                    p.true_last_name == patient.true_last_name and
                    p.true_dob == patient.true_dob
                    for p in self.patients
                ):
                    difficulty = "ambiguous"
                elif "data_error" in demo_record.data_quality_flag:
                    difficulty = "hard"
                elif "name_variation" in demo_record.data_quality_flag:
                    difficulty = "medium"
                else:
                    difficulty = "easy"

                gt = GroundTruth(
                    record_id=demo_record.record_id,
                    patient_id=patient.patient_id,
                    match_group=match_group,
                    notes=demo_record.data_quality_flag,
                    is_common_name=patient.has_common_name,
                    is_twin=patient.has_twin,
                    has_data_error="error" in demo_record.data_quality_flag,
                    difficulty=difficulty
                )
                self.ground_truth.append(gt)

    def _print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total unique patients: {len(self.patients)}")
        print(f"Total demographic records: {len(self.all_demographics)}")
        print(f"Total medical records: {len(self.all_medical_records)}")
        print(f"Avg demographic records per patient: {len(self.all_demographics) / len(self.patients):.1f}")
        print(f"Avg medical records per patient: {len(self.all_medical_records) / len(self.patients):.1f}")
        print()
        print("Edge Cases:")
        for case_type, count in self.stats["edge_cases"].items():
            print(f"  - {case_type}: {count}")
        print()
        print("Variations:")
        for var_type, count in self.stats["variations"].items():
            print(f"  - {var_type}: {count}")
        print()

        # Difficulty distribution
        difficulty_counts = {}
        for gt in self.ground_truth:
            difficulty_counts[gt.difficulty] = difficulty_counts.get(gt.difficulty, 0) + 1

        print("Difficulty Distribution:")
        for difficulty, count in sorted(difficulty_counts.items()):
            pct = (count / len(self.ground_truth)) * 100
            print(f"  - {difficulty}: {count} ({pct:.1f}%)")
        print()

    def save_dataset(self):
        """Save dataset to files."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving dataset to {self.output_dir}...")

        # Save demographics to CSV
        demo_dicts = []
        for demo in self.all_demographics:
            demo_dict = demo.model_dump()
            # Convert address to string
            if demo_dict.get('address'):
                demo_dict['address_street'] = demo_dict['address']['street']
                demo_dict['address_city'] = demo_dict['address']['city']
                demo_dict['address_state'] = demo_dict['address']['state']
                demo_dict['address_zip'] = demo_dict['address']['zip_code']
                del demo_dict['address']
            demo_dicts.append(demo_dict)

        df_demographics = pd.DataFrame(demo_dicts)
        demographics_path = self.output_dir / "synthetic_demographics.csv"
        df_demographics.to_csv(demographics_path, index=False)
        print(f"  ✓ Saved demographics: {demographics_path}")

        # Save medical records to JSON
        medical_records_data = []
        for record in self.all_medical_records:
            record_dict = json.loads(record.model_dump_json())
            medical_records_data.append(record_dict)

        medical_path = self.output_dir / "synthetic_medical_records.json"
        with open(medical_path, 'w') as f:
            json.dump(medical_records_data, f, indent=2, default=str)
        print(f"  ✓ Saved medical records: {medical_path}")

        # Save ground truth to CSV
        gt_dicts = [gt.model_dump() for gt in self.ground_truth]
        df_ground_truth = pd.DataFrame(gt_dicts)
        gt_path = self.output_dir / "ground_truth.csv"
        df_ground_truth.to_csv(gt_path, index=False)
        print(f"  ✓ Saved ground truth: {gt_path}")

        # Save metadata
        metadata = {
            "generation_date": datetime.now().isoformat(),
            "seed": self.seed,
            "use_ai": self.use_ai,
            "num_patients": len(self.patients),
            "num_demographic_records": len(self.all_demographics),
            "num_medical_records": len(self.all_medical_records),
            "statistics": self.stats
        }
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {metadata_path}")

        print("\nDataset generation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient dataset for MedMatch AI"
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=75,
        help="Number of unique patients to generate (default: 75)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI-assisted generation (use rule-based only)"
    )
    parser.add_argument(
        "--api-rate-limit",
        type=int,
        default=5,
        help="Max API requests per minute (default: 5 for free tier, 15+ for paid)"
    )

    args = parser.parse_args()

    generator = SyntheticDatasetGenerator(
        num_patients=args.num_patients,
        output_dir=args.output_dir,
        seed=args.seed,
        use_ai=not args.no_ai,
        api_rate_limit=args.api_rate_limit
    )

    generator.generate_dataset()
    generator.save_dataset()


if __name__ == "__main__":
    main()
