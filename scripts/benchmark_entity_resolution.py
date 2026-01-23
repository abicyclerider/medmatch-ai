#!/usr/bin/env python3
"""
Entity Resolution AI Benchmark

Efficiently benchmarks AI performance on the actual entity resolution problem:
1. Runs pipeline once (blocking → rules → scoring) to identify AI-bound pairs
2. Extracts ambiguous pairs (demographic score 0.50-0.90)
3. Benchmarks only the AI comparisons with different configurations

This avoids re-running the deterministic stages repeatedly.

Usage:
    python scripts/benchmark_entity_resolution.py
    python scripts/benchmark_entity_resolution.py --mode batched --batch-size 3
    python scripts/benchmark_entity_resolution.py --model medgemma:1.5-4b  # full model
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm
import requests

# MedMatch imports
from medmatch.matching import PatientRecord, PatientMatcher
from medmatch.matching.features import FeatureExtractor
from medmatch.matching.scoring import ConfidenceScorer
from medmatch.evaluation import MatchEvaluator


@dataclass
class AmbiguousPair:
    """A pair that needs AI evaluation."""
    record1: PatientRecord
    record2: PatientRecord
    demographic_score: float
    ground_truth_match: bool
    difficulty: str

    @property
    def medical_history_1(self) -> str:
        return self.record1.medical_signature or ""

    @property
    def medical_history_2(self) -> str:
        return self.record2.medical_signature or ""


@dataclass
class AIBenchmarkResult:
    """Result from AI benchmark."""
    mode: str
    model: str
    total_pairs: int
    total_time: float
    avg_time_per_pair: float
    throughput_per_minute: float

    # Accuracy metrics
    correct: int
    accuracy: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # By difficulty
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "total_pairs": self.total_pairs,
            "total_time": round(self.total_time, 2),
            "avg_time_per_pair": round(self.avg_time_per_pair, 2),
            "throughput_per_minute": round(self.throughput_per_minute, 1),
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy_by_difficulty": {
                k: round(v, 4) for k, v in self.accuracy_by_difficulty.items()
            },
        }


class EntityResolutionBenchmark:
    """Benchmarks AI performance on entity resolution."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.records: List[PatientRecord] = []
        self.ambiguous_pairs: List[AmbiguousPair] = []
        self.evaluator: Optional[MatchEvaluator] = None

    def load_data(self) -> None:
        """Load synthetic dataset and convert to PatientRecords."""
        print("Loading synthetic dataset...")

        demographics_path = self.data_dir / 'synthetic_demographics.csv'
        medical_path = self.data_dir / 'synthetic_medical_records.json'
        ground_truth_path = self.data_dir / 'ground_truth.csv'

        df_demo = pd.read_csv(demographics_path)

        # Load medical records
        medical_by_patient = {}
        if medical_path.exists():
            with open(medical_path, 'r') as f:
                medical_data = json.load(f)
            for mr in medical_data:
                patient_id = mr['patient_id']
                if patient_id not in medical_by_patient:
                    medical_by_patient[patient_id] = mr

        # Convert to PatientRecords
        self.records = self._convert_to_records(df_demo, medical_by_patient)
        print(f"  Loaded {len(self.records)} records")

        # Count records with medical history
        with_medical = sum(1 for r in self.records if r.medical_signature)
        print(f"  Records with medical history: {with_medical}")

        # Load evaluator
        self.evaluator = MatchEvaluator(str(ground_truth_path))
        print(f"  Ground truth loaded")

    def _convert_to_records(self, df: pd.DataFrame, medical_by_patient: dict) -> List[PatientRecord]:
        """Convert DataFrame to PatientRecord list with medical records."""
        from medmatch.data.models.patient import (
            Demographics, Address, MedicalRecord, MedicalHistory,
            MedicalCondition, Surgery
        )

        records = []
        for _, row in df.iterrows():
            # Parse dates
            dob = date.fromisoformat(row['date_of_birth']) if isinstance(row['date_of_birth'], str) else row['date_of_birth']
            rec_date_str = row.get('record_date')
            if pd.isna(rec_date_str):
                rec_date = date.today()
            elif isinstance(rec_date_str, str):
                rec_date = date.fromisoformat(rec_date_str.split('T')[0])
            else:
                rec_date = rec_date_str

            # Create Address
            address = None
            if pd.notna(row.get('address_street')):
                address = Address(
                    street=row['address_street'],
                    city=row.get('address_city', ''),
                    state=row.get('address_state', ''),
                    zip_code=str(row.get('address_zip', '')),
                )

            # Create Demographics
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

            # Get medical record
            medical = None
            patient_id = row['patient_id']
            if patient_id in medical_by_patient:
                mr_data = medical_by_patient[patient_id]
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

        return records

    def identify_ambiguous_pairs(self) -> None:
        """Run pipeline without AI to identify pairs that would go to AI."""
        print("\nIdentifying ambiguous pairs (running blocking → rules → scoring)...")

        # Run matcher without AI
        matcher = PatientMatcher(
            use_blocking=True,
            use_rules=True,
            use_scoring=True,
            use_ai=False,
        )

        # Get candidate pairs from blocking
        pairs = matcher.blocker.generate_candidate_pairs(self.records)
        print(f"  Blocking produced {len(pairs)} candidate pairs")

        # Score each pair and identify ambiguous ones
        feature_extractor = FeatureExtractor()
        scorer = ConfidenceScorer()

        self.ambiguous_pairs = []
        rules_decided = 0
        scoring_clear = 0

        for record1, record2 in tqdm(pairs, desc="  Scoring pairs"):
            # Check if rules would decide
            rule_result = matcher.rule_engine.evaluate(record1, record2)
            if rule_result is not None:
                rules_decided += 1
                continue

            # Get demographic score
            features = feature_extractor.extract(record1, record2)
            demo_score, _ = scorer.score(features)

            # Check if ambiguous (would go to AI)
            if 0.50 <= demo_score <= 0.90:
                # Get ground truth
                should_match = self.evaluator.should_match(
                    record1.record_id, record2.record_id
                )
                difficulty = self.evaluator.get_pair_difficulty(
                    record1.record_id, record2.record_id
                )

                self.ambiguous_pairs.append(AmbiguousPair(
                    record1=record1,
                    record2=record2,
                    demographic_score=demo_score,
                    ground_truth_match=should_match,
                    difficulty=difficulty,
                ))
            else:
                scoring_clear += 1

        print(f"\n  Pipeline breakdown:")
        print(f"    Rules decided: {rules_decided} pairs")
        print(f"    Scoring decided (clear): {scoring_clear} pairs")
        print(f"    AI needed (ambiguous): {len(self.ambiguous_pairs)} pairs")

        # Show difficulty breakdown of ambiguous pairs
        difficulty_counts = {}
        for pair in self.ambiguous_pairs:
            difficulty_counts[pair.difficulty] = difficulty_counts.get(pair.difficulty, 0) + 1
        print(f"\n  Ambiguous pairs by difficulty:")
        for diff, count in sorted(difficulty_counts.items()):
            print(f"    {diff}: {count}")

    def benchmark_standard(self, model: str = "medgemma:1.5-4b-q4") -> AIBenchmarkResult:
        """Benchmark standard mode (one API call per pair)."""
        print(f"\nBenchmarking STANDARD mode ({len(self.ambiguous_pairs)} pairs)...")

        from medmatch.matching.ai_client import OllamaClient
        client = OllamaClient(model=model)

        results = []
        times = []

        for pair in tqdm(self.ambiguous_pairs, desc="  Standard"):
            start = time.time()
            medical_score, reasoning = client.compare_medical_histories(
                pair.medical_history_1, pair.medical_history_2
            )
            elapsed = time.time() - start
            times.append(elapsed)

            # Calculate combined score (60% demo + 40% medical)
            combined = 0.6 * pair.demographic_score + 0.4 * medical_score
            predicted_match = combined >= 0.65

            results.append({
                "pair": pair,
                "medical_score": medical_score,
                "combined_score": combined,
                "predicted_match": predicted_match,
                "actual_match": pair.ground_truth_match,
                "correct": predicted_match == pair.ground_truth_match,
            })

        return self._compute_metrics(results, times, "standard", model)

    def benchmark_batched(
        self,
        model: str = "medgemma:1.5-4b-q4",
        batch_size: int = 3,
    ) -> AIBenchmarkResult:
        """Benchmark batched mode (multiple pairs per API call)."""
        print(f"\nBenchmarking BATCHED mode (batch_size={batch_size})...")

        # Use the BatchedOllamaClient from benchmark script
        from benchmark_medgemma_quantization import BatchedOllamaClient
        client = BatchedOllamaClient(model=model, batch_size=batch_size)

        results = []
        times = []

        # Process in batches
        num_batches = (len(self.ambiguous_pairs) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="  Batched"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.ambiguous_pairs))
            batch_pairs = self.ambiguous_pairs[start_idx:end_idx]

            # Build batch input
            batch_input = [
                (pair.medical_history_1, pair.medical_history_2)
                for pair in batch_pairs
            ]

            start = time.time()
            batch_results = client.compare_batch(batch_input)
            elapsed = time.time() - start

            # Distribute time across pairs in batch
            time_per_pair = elapsed / len(batch_pairs)

            for pair, (medical_score, _) in zip(batch_pairs, batch_results):
                times.append(time_per_pair)

                combined = 0.6 * pair.demographic_score + 0.4 * medical_score
                predicted_match = combined >= 0.65

                results.append({
                    "pair": pair,
                    "medical_score": medical_score,
                    "combined_score": combined,
                    "predicted_match": predicted_match,
                    "actual_match": pair.ground_truth_match,
                    "correct": predicted_match == pair.ground_truth_match,
                })

        return self._compute_metrics(results, times, f"batched (size={batch_size})", model)

    def _compute_metrics(
        self,
        results: List[dict],
        times: List[float],
        mode: str,
        model: str,
    ) -> AIBenchmarkResult:
        """Compute accuracy metrics from results."""
        total_time = sum(times)
        avg_time = total_time / len(results) if results else 0
        throughput = 60.0 / avg_time if avg_time > 0 else 0

        # Confusion matrix
        tp = sum(1 for r in results if r["predicted_match"] and r["actual_match"])
        tn = sum(1 for r in results if not r["predicted_match"] and not r["actual_match"])
        fp = sum(1 for r in results if r["predicted_match"] and not r["actual_match"])
        fn = sum(1 for r in results if not r["predicted_match"] and r["actual_match"])

        correct = tp + tn
        accuracy = correct / len(results) if results else 0

        # By difficulty
        by_difficulty = {}
        difficulty_groups = {}
        for r in results:
            diff = r["pair"].difficulty
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(r)

        for diff, group in difficulty_groups.items():
            diff_correct = sum(1 for r in group if r["correct"])
            by_difficulty[diff] = diff_correct / len(group) if group else 0

        return AIBenchmarkResult(
            mode=mode,
            model=model,
            total_pairs=len(results),
            total_time=total_time,
            avg_time_per_pair=avg_time,
            throughput_per_minute=throughput,
            correct=correct,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy_by_difficulty=by_difficulty,
        )

    def print_result(self, result: AIBenchmarkResult) -> None:
        """Print benchmark result in readable format."""
        print(f"\n{'='*60}")
        print(f"Mode: {result.mode} | Model: {result.model}")
        print(f"{'='*60}")
        print(f"  Pairs evaluated: {result.total_pairs}")
        print(f"  Total time: {result.total_time:.1f}s")
        print(f"  Avg per pair: {result.avg_time_per_pair:.2f}s")
        print(f"  Throughput: {result.throughput_per_minute:.1f}/min")
        print()
        print(f"  Accuracy: {result.accuracy:.1%} ({result.correct}/{result.total_pairs})")
        print(f"  TP: {result.true_positives}, TN: {result.true_negatives}, "
              f"FP: {result.false_positives}, FN: {result.false_negatives}")
        print()
        print(f"  By difficulty:")
        for diff in ['easy', 'medium', 'hard', 'ambiguous']:
            if diff in result.accuracy_by_difficulty:
                print(f"    {diff}: {result.accuracy_by_difficulty[diff]:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution AI Benchmark")
    parser.add_argument("--mode", choices=["standard", "batched", "compare"],
                        default="compare", help="Benchmark mode")
    parser.add_argument("--model", default="medgemma:1.5-4b-q4",
                        help="Ollama model name")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Batch size for batched mode")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "synthetic"),
                        help="Path to synthetic data directory")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")

    args = parser.parse_args()

    print("="*60)
    print("ENTITY RESOLUTION AI BENCHMARK")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.mode in ["batched", "compare"]:
        print(f"Batch size: {args.batch_size}")

    # Initialize benchmark
    benchmark = EntityResolutionBenchmark(Path(args.data_dir))

    try:
        # Load data
        benchmark.load_data()

        # Identify ambiguous pairs
        benchmark.identify_ambiguous_pairs()

        if not benchmark.ambiguous_pairs:
            print("\nNo ambiguous pairs found - all decided by rules/scoring!")
            return

        # Run benchmarks
        results = {}

        if args.mode == "standard":
            result = benchmark.benchmark_standard(model=args.model)
            benchmark.print_result(result)
            results["standard"] = result.to_dict()

        elif args.mode == "batched":
            result = benchmark.benchmark_batched(
                model=args.model, batch_size=args.batch_size
            )
            benchmark.print_result(result)
            results["batched"] = result.to_dict()

        elif args.mode == "compare":
            # Run both and compare
            standard = benchmark.benchmark_standard(model=args.model)
            benchmark.print_result(standard)
            results["standard"] = standard.to_dict()

            batched = benchmark.benchmark_batched(
                model=args.model, batch_size=args.batch_size
            )
            benchmark.print_result(batched)
            results["batched"] = batched.to_dict()

            # Print comparison
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            speedup = standard.avg_time_per_pair / batched.avg_time_per_pair
            accuracy_diff = batched.accuracy - standard.accuracy
            print(f"  Speedup: {speedup:.2f}x faster with batching")
            print(f"  Accuracy difference: {accuracy_diff:+.1%}")
            print(f"  Standard throughput: {standard.throughput_per_minute:.1f}/min")
            print(f"  Batched throughput: {batched.throughput_per_minute:.1f}/min")

        # Save results
        if args.save:
            output = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "total_ambiguous_pairs": len(benchmark.ambiguous_pairs),
                "results": results,
            }
            output_path = Path(args.data_dir).parent / "entity_resolution_benchmark.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to: {output_path}")

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama server")
        print("Make sure Ollama is running: brew services start ollama")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
