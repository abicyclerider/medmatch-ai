#!/usr/bin/env python3
"""
Command-line interface for batch patient matching.

This script provides a production-ready CLI for running the entity resolution
pipeline on datasets. Suitable for batch processing and Kaggle submission.

Usage:
    python scripts/run_matcher.py \\
        --demographics data/synthetic/synthetic_demographics.csv \\
        --medical-records data/synthetic/synthetic_medical_records.json \\
        --output results.json \\
        --use-ai \\
        --progress

See --help for full options.
"""

import sys
import os
import argparse
import json
import csv
import logging
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from tqdm import tqdm

from medmatch.matching import PatientMatcher, PatientRecord, MatchExplainer
from medmatch.data.models.patient import Demographics, MedicalRecord, Address, MedicalHistory, MedicalCondition, Surgery


def setup_logging(log_level: str) -> logging.Logger:
    """Configure logging for the CLI."""
    logger = logging.getLogger('run_matcher')
    logger.setLevel(getattr(logging, log_level.upper()))

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_demographics(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load demographics CSV."""
    logger.info(f"Loading demographics from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} demographic records")
    return df


def load_medical_records(json_path: Optional[Path], logger: logging.Logger) -> Dict[str, dict]:
    """Load medical records JSON and index by patient_id."""
    if not json_path or not json_path.exists():
        logger.info("No medical records provided")
        return {}

    logger.info(f"Loading medical records from {json_path}")
    with open(json_path, 'r') as f:
        medical_data = json.load(f)

    # Index by patient_id (use first record for each patient)
    medical_by_patient = {}
    for mr in medical_data:
        patient_id = mr['patient_id']
        if patient_id not in medical_by_patient:
            medical_by_patient[patient_id] = mr

    logger.info(f"Loaded {len(medical_by_patient)} medical records")
    return medical_by_patient


def create_patient_records(
    df: pd.DataFrame,
    medical_by_patient: Dict[str, dict],
    logger: logging.Logger
) -> List[PatientRecord]:
    """Convert DataFrame to PatientRecord list."""
    logger.info("Creating PatientRecord objects...")
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading records", disable=False):
        # Parse date of birth
        dob_str = row['date_of_birth']
        if isinstance(dob_str, str):
            dob = date.fromisoformat(dob_str)
        else:
            dob = dob_str

        # Parse record date
        rec_date_str = row.get('record_date')
        if pd.isna(rec_date_str):
            rec_date = date.today()
        elif isinstance(rec_date_str, str):
            rec_date = date.fromisoformat(rec_date_str.split('T')[0])
        else:
            rec_date = rec_date_str

        # Create Address if available
        address = None
        if pd.notna(row.get('address_street')):
            address = Address(
                street=row['address_street'],
                city=row.get('address_city', ''),
                state=row.get('address_state', ''),
                zip_code=str(row.get('address_zip', '')),
            )

        # Create Demographics object
        demo = Demographics(
            record_id=row['record_id'],
            patient_id=row['patient_id'],
            name_first=row['name_first'],
            name_middle=row.get('name_middle') if pd.notna(row.get('name_middle')) else None,
            name_last=row['name_last'],
            name_suffix=row.get('name_suffix') if pd.notna(row.get('name_suffix')) else None,
            date_of_birth=dob,
            gender=row['gender'],
            mrn=str(row['mrn']),
            ssn_last4=str(row['ssn_last4']) if pd.notna(row.get('ssn_last4')) else None,
            phone=row.get('phone') if pd.notna(row.get('phone')) else None,
            email=row.get('email') if pd.notna(row.get('email')) else None,
            address=address,
            record_source=row.get('record_source', 'unknown'),
            record_date=rec_date,
            data_quality_flag=row.get('data_quality_flag') if pd.notna(row.get('data_quality_flag')) else None,
        )

        # Get medical record for this patient (if available)
        medical = None
        patient_id = row['patient_id']
        if patient_id in medical_by_patient:
            mr_data = medical_by_patient[patient_id]

            # Parse medical history
            mh_data = mr_data.get('medical_history', {})
            conditions = [
                MedicalCondition(
                    name=c['name'],
                    abbreviation=c.get('abbreviation'),
                    onset_year=c.get('onset_year'),
                    status=c.get('status', 'active')
                ) for c in mh_data.get('conditions', [])
            ]
            surgeries = [
                Surgery(
                    procedure=s['procedure'],
                    date=date.fromisoformat(s['date']) if s.get('date') else None
                ) for s in mh_data.get('surgeries', [])
            ]
            medical_history = MedicalHistory(
                conditions=conditions,
                medications=mh_data.get('medications', []),
                allergies=mh_data.get('allergies', []),
                surgeries=surgeries,
                family_history=mh_data.get('family_history', []),
                social_history=mh_data.get('social_history', '')
            )

            # Create MedicalRecord
            medical = MedicalRecord(
                record_id=mr_data['record_id'],
                patient_id=mr_data['patient_id'],
                record_source=mr_data.get('record_source', 'unknown'),
                record_date=date.fromisoformat(mr_data['record_date'].split('T')[0]),
                chief_complaint=mr_data.get('chief_complaint'),
                medical_history=medical_history,
                assessment=mr_data.get('assessment'),
                plan=mr_data.get('plan'),
                clinical_notes=mr_data.get('clinical_notes'),
            )

        records.append(PatientRecord.from_demographics(demo, medical))

    logger.info(f"Created {len(records)} PatientRecord objects")
    with_medical = sum(1 for r in records if r.conditions or r.medications)
    logger.info(f"Records with medical history: {with_medical}/{len(records)}")

    return records


def run_matching(
    records: List[PatientRecord],
    matcher: PatientMatcher,
    logger: logging.Logger,
    show_progress: bool = True
) -> List:
    """Run matching pipeline on all records."""
    logger.info("Running entity resolution pipeline...")
    logger.info(f"Configuration: blocking={matcher.use_blocking}, rules={matcher.use_rules}, "
                f"scoring={matcher.use_scoring}, ai={matcher.use_ai}")

    results = matcher.match_datasets(records, show_progress=show_progress)

    logger.info(f"Generated {len(results)} match results")
    return results


def format_json_output(
    results: List,
    matcher: PatientMatcher,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Format results as JSON."""
    logger.info("Formatting results as JSON...")

    stats = matcher.get_stats(results)

    output = {
        "metadata": {
            "total_pairs": len(results),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "blocking": matcher.use_blocking,
                "rules": matcher.use_rules,
                "scoring": matcher.use_scoring,
                "ai": matcher.use_ai,
            },
        },
        "results": [
            {
                "record1_id": r.record_1_id,
                "record2_id": r.record_2_id,
                "is_match": r.is_match,
                "confidence": round(r.confidence, 3),
                "match_type": r.match_type,
                "stage": r.stage,
                "explanation": r.explanation,
            }
            for r in results
        ],
        "summary": {
            "matches": stats['matches'],
            "non_matches": stats['no_matches'],
            "by_stage": stats['by_stage'],
            "by_match_type": stats['by_match_type'],
            "avg_confidence": stats['avg_confidence'],
        },
    }

    return output


def format_csv_output(
    results: List,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Format results as CSV-compatible rows."""
    logger.info("Formatting results as CSV...")

    rows = [
        {
            "record1_id": r.record_1_id,
            "record2_id": r.record_2_id,
            "is_match": r.is_match,
            "confidence": round(r.confidence, 3),
            "match_type": r.match_type,
            "stage": r.stage,
        }
        for r in results
    ]

    return rows


def format_verbose_output(
    results: List,
    matcher: PatientMatcher,
    logger: logging.Logger
) -> str:
    """Format results with verbose explanations."""
    logger.info("Formatting verbose output...")

    explainer = MatchExplainer()
    output = []

    # Overall statistics
    stats = matcher.get_stats(results)
    output.append("=" * 70)
    output.append("ENTITY RESOLUTION RESULTS")
    output.append("=" * 70)
    output.append(f"\nTotal Pairs: {stats['total_pairs']}")
    output.append(f"Matches: {stats['matches']}")
    output.append(f"Non-matches: {stats['no_matches']}")
    output.append(f"Average Confidence: {stats['avg_confidence']:.3f}")
    output.append(f"\nBy Stage: {stats['by_stage']}")
    output.append(f"By Match Type: {stats['by_match_type']}")
    output.append("\n" + "=" * 70)
    output.append("INDIVIDUAL RESULTS")
    output.append("=" * 70)

    # Individual results
    for i, result in enumerate(results, 1):
        output.append(f"\n[{i}] " + explainer.explain(result, verbose=True))
        output.append("-" * 70)

    return "\n".join(output)


def save_output(
    output: Any,
    output_path: Optional[Path],
    output_format: str,
    logger: logging.Logger
):
    """Save output to file or stdout."""
    if output_path is None:
        # Output to stdout
        if output_format == 'json':
            print(json.dumps(output, indent=2))
        elif output_format == 'csv':
            if output:
                writer = csv.DictWriter(sys.stdout, fieldnames=output[0].keys())
                writer.writeheader()
                writer.writerows(output)
        else:  # verbose
            print(output)
        return

    logger.info(f"Saving output to {output_path}")

    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    elif output_format == 'csv':
        with open(output_path, 'w', newline='') as f:
            if output:
                writer = csv.DictWriter(f, fieldnames=output[0].keys())
                writer.writeheader()
                writer.writerows(output)
    else:  # verbose
        with open(output_path, 'w') as f:
            f.write(output)

    logger.info(f"Output saved successfully")


def load_custom_config(path: Path, logger: logging.Logger) -> dict:
    """Load custom configuration from JSON file."""
    logger.info(f"Loading custom configuration from {path}")
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run entity resolution matching on patient datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all stages
  python scripts/run_matcher.py \\
    --demographics data/synthetic/synthetic_demographics.csv \\
    --medical-records data/synthetic/synthetic_medical_records.json \\
    --output results.json

  # Fast mode (no AI)
  python scripts/run_matcher.py \\
    --demographics data/demographics.csv \\
    --output results.csv \\
    --format csv \\
    --no-ai

  # With custom configuration
  python scripts/run_matcher.py \\
    --demographics data/demographics.csv \\
    --scoring-weights config/custom_weights.json \\
    --scoring-thresholds config/custom_thresholds.json \\
    --output results.json

  # Verbose output to stdout
  python scripts/run_matcher.py \\
    --demographics data/demographics.csv \\
    --format verbose
        """
    )

    # Input files
    parser.add_argument(
        '--demographics',
        type=str,
        required=True,
        help='Path to demographics CSV file'
    )
    parser.add_argument(
        '--medical-records',
        type=str,
        help='Path to medical records JSON file (optional)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'verbose'],
        default='json',
        help='Output format (default: json)'
    )

    # Pipeline configuration
    parser.add_argument(
        '--no-blocking',
        action='store_true',
        help='Disable blocking (all pairs, slow for large datasets)'
    )
    parser.add_argument(
        '--no-rules',
        action='store_true',
        help='Disable deterministic rules'
    )
    parser.add_argument(
        '--no-scoring',
        action='store_true',
        help='Disable feature scoring'
    )
    parser.add_argument(
        '--use-ai',
        action='store_true',
        help='Enable AI medical fingerprinting (requires GOOGLE_AI_API_KEY)'
    )

    # AI configuration
    parser.add_argument(
        '--ai-model',
        type=str,
        default='gemini-2.5-flash',
        help='AI model name (default: gemini-2.5-flash)'
    )
    parser.add_argument(
        '--api-rate-limit',
        type=int,
        default=0,
        help='API rate limit in requests/minute (0=unlimited, default: 0)'
    )

    # Custom configuration
    parser.add_argument(
        '--scoring-weights',
        type=str,
        help='Path to custom scoring weights JSON file'
    )
    parser.add_argument(
        '--scoring-thresholds',
        type=str,
        help='Path to custom scoring thresholds JSON file'
    )

    # Display options
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress bar during matching'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    try:
        # Load data
        df_demographics = load_demographics(Path(args.demographics), logger)

        medical_by_patient = {}
        if args.medical_records:
            medical_by_patient = load_medical_records(Path(args.medical_records), logger)

        records = create_patient_records(df_demographics, medical_by_patient, logger)

        # Load custom configuration
        scoring_weights = None
        if args.scoring_weights:
            scoring_weights = load_custom_config(Path(args.scoring_weights), logger)

        scoring_thresholds = None
        if args.scoring_thresholds:
            scoring_thresholds = load_custom_config(Path(args.scoring_thresholds), logger)

        # Create matcher
        logger.info("Initializing PatientMatcher...")
        matcher = PatientMatcher(
            use_blocking=not args.no_blocking,
            use_rules=not args.no_rules,
            use_scoring=not args.no_scoring,
            use_ai=args.use_ai,
            scoring_weights=scoring_weights,
            scoring_thresholds=scoring_thresholds,
            ai_model=args.ai_model,
            api_rate_limit=args.api_rate_limit,
        )

        # Run matching
        results = run_matching(records, matcher, logger, show_progress=args.progress)

        # Format output
        if args.format == 'json':
            output = format_json_output(results, matcher, logger)
        elif args.format == 'csv':
            output = format_csv_output(results, logger)
        else:  # verbose
            output = format_verbose_output(results, matcher, logger)

        # Save output
        output_path = Path(args.output) if args.output else None
        save_output(output, output_path, args.format, logger)

        logger.info("Matching complete!")

    except Exception as e:
        logger.error(f"Error during matching: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
