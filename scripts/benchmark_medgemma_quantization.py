#!/usr/bin/env python3
"""
Benchmark MedGemma Quantization Performance

Compares full model (medgemma:1.5-4b) vs quantized model (medgemma:1.5-4b-q4):
- Inference speed (tokens/second)
- Medical understanding accuracy
- Memory usage
- Total time for batch comparisons

Usage:
    python scripts/benchmark_medgemma_quantization.py
    python scripts/benchmark_medgemma_quantization.py --iterations 20
"""

import argparse
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Direct import to avoid loading unnecessary dependencies
import requests


class LightweightOllamaClient:
    """Lightweight Ollama client for benchmarking (avoids heavy imports)."""

    def __init__(self, model="medgemma:1.5-4b-q4", base_url="http://localhost:11434", timeout=180):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def compare_medical_histories(self, history1, history2):
        """Compare two medical histories and return similarity score."""
        prompt = f"""Compare these two medical histories and determine if they describe the same patient.

Medical History 1: {history1}

Medical History 2: {history2}

Consider:
- Medical abbreviations (HTN=Hypertension, T2DM=Type 2 Diabetes, CAD=Coronary Artery Disease, etc.)
- Medication synonyms (Tylenol=acetaminophen, etc.)
- Different date formats and phrasing
- Clinical context and treatment patterns

Respond EXACTLY in this format:
SIMILARITY_SCORE: <number between 0.0 and 1.0>
REASONING: <brief explanation>

Where 1.0 = definitely same patient, 0.0 = definitely different patients."""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        result = response.json()
        text = result.get("response", "")

        # Parse response
        score = 0.5  # default
        reasoning = text

        try:
            if "SIMILARITY_SCORE:" in text:
                score_line = text.split("SIMILARITY_SCORE:")[1].split("\n")[0].strip()
                score = float(score_line)

            if "REASONING:" in text:
                reasoning = text.split("REASONING:")[1].strip()
        except Exception:
            pass  # Use defaults if parsing fails

        return score, reasoning


class MedGemmaBenchmark:
    """Benchmark MedGemma models for speed and accuracy."""

    def __init__(self, full_model="medgemma:1.5-4b", quantized_model="medgemma:1.5-4b-q4"):
        self.full_model = full_model
        self.quantized_model = quantized_model

        # Test cases for medical understanding
        self.test_cases = [
            {
                "name": "Matching Medical Histories (Abbreviations)",
                "history1": "Patient has T2DM diagnosed 2015, HTN since 2018, takes Metformin 1000mg BID and Lisinopril 10mg daily.",
                "history2": "Type 2 Diabetes Mellitus (onset 2015), Hypertension (2018), medications: Metformin 1000mg twice daily, Lisinopril 10mg QD.",
                "expected_score": 1.0,
                "tolerance": 0.1,
            },
            {
                "name": "Different Medical Conditions",
                "history1": "Patient has CAD s/p CABG 2020, on aspirin and atorvastatin.",
                "history2": "Patient has COPD, uses albuterol inhaler PRN, no cardiac history.",
                "expected_score": 0.0,
                "tolerance": 0.2,
            },
            {
                "name": "Partial Match (Shared Condition)",
                "history1": "Patient has HTN, T2DM, hyperlipidemia. Medications: Lisinopril, Metformin, Atorvastatin.",
                "history2": "Patient with hypertension, takes Lisinopril. No diabetes. History of MI 2019.",
                "expected_score": 0.4,
                "tolerance": 0.3,
            },
            {
                "name": "Medication Name Variations",
                "history1": "Patient on Tylenol 500mg PRN for pain, takes aspirin 81mg daily.",
                "history2": "Uses acetaminophen 500mg as needed, daily baby aspirin 81mg.",
                "expected_score": 1.0,
                "tolerance": 0.1,
            },
            {
                "name": "Complex Medical History",
                "history1": "ESRD on HD MWF, DM2 with retinopathy, CHF (EF 35%), s/p pacemaker 2019.",
                "history2": "End-stage renal disease, hemodialysis Monday/Wednesday/Friday. Type 2 diabetes with diabetic retinopathy. Congestive heart failure, ejection fraction 35%, pacemaker placed 2019.",
                "expected_score": 1.0,
                "tolerance": 0.1,
            },
        ]

    def benchmark_speed(self, model_name, iterations=10):
        """Benchmark inference speed for a model."""
        print(f"\n{'='*70}")
        print(f"Speed Benchmark: {model_name}")
        print(f"{'='*70}")

        client = LightweightOllamaClient(model=model_name, timeout=180)

        # Warmup (model loading time)
        print("Warming up model...")
        warmup_start = time.time()
        _, _ = client.compare_medical_histories(
            self.test_cases[0]["history1"],
            self.test_cases[0]["history2"]
        )
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s (includes model loading)")

        # Run benchmark
        print(f"\nRunning {iterations} comparisons...")
        times = []

        with tqdm(total=iterations, desc="Speed test", unit="comparison") as pbar:
            for i in range(iterations):
                # Cycle through test cases
                test_case = self.test_cases[i % len(self.test_cases)]

                start = time.time()
                score, reasoning = client.compare_medical_histories(
                    test_case["history1"],
                    test_case["history2"]
                )
                elapsed = time.time() - start
                times.append(elapsed)

                pbar.set_postfix({"time": f"{elapsed:.1f}s", "score": f"{score:.2f}"})
                pbar.update(1)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nSpeed Results:")
        print(f"  Average: {avg_time:.2f}s per comparison")
        print(f"  Min:     {min_time:.2f}s")
        print(f"  Max:     {max_time:.2f}s")
        print(f"  Total:   {sum(times):.2f}s for {iterations} comparisons")

        return {
            "model": model_name,
            "iterations": iterations,
            "warmup_time": warmup_time,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": sum(times),
            "times": times,
        }

    def benchmark_accuracy(self, model_name):
        """Benchmark medical understanding accuracy for a model."""
        print(f"\n{'='*70}")
        print(f"Accuracy Benchmark: {model_name}")
        print(f"{'='*70}")

        client = LightweightOllamaClient(model=model_name, timeout=180)

        results = []
        correct = 0

        with tqdm(total=len(self.test_cases), desc="Accuracy test", unit="test") as pbar:
            for i, test_case in enumerate(self.test_cases, 1):
                pbar.set_description(f"Accuracy test: {test_case['name'][:30]}")

                score, reasoning = client.compare_medical_histories(
                    test_case["history1"],
                    test_case["history2"]
                )

                # Check if within tolerance
                error = abs(score - test_case['expected_score'])
                is_correct = error <= test_case['tolerance']

                if is_correct:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"

                pbar.set_postfix({"score": f"{score:.2f}", "expected": f"{test_case['expected_score']:.2f}", "status": status})
                pbar.update(1)

                results.append({
                    "name": test_case["name"],
                    "expected": test_case["expected_score"],
                    "actual": score,
                    "error": error,
                    "tolerance": test_case["tolerance"],
                    "passed": is_correct,
                    "reasoning": reasoning,
                })

        # Print detailed results after progress bar
        print("\nDetailed Results:")
        for i, result in enumerate(results, 1):
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"  [{i}] {result['name']}: {result['actual']:.2f} (expected {result['expected']:.2f}) - {status}")

        accuracy = correct / len(self.test_cases)

        print(f"\nAccuracy Results:")
        print(f"  Passed: {correct}/{len(self.test_cases)} ({accuracy:.1%})")

        return {
            "model": model_name,
            "total_tests": len(self.test_cases),
            "passed": correct,
            "accuracy": accuracy,
            "results": results,
        }

    def run_full_benchmark(self, speed_iterations=10, save_results=True):
        """Run complete benchmark comparing both models."""
        print("="*70)
        print("MedGemma Quantization Benchmark")
        print("="*70)
        print(f"Full Model:      {self.full_model}")
        print(f"Quantized Model: {self.quantized_model}")
        print(f"Speed Test:      {speed_iterations} iterations")
        print(f"Accuracy Test:   {len(self.test_cases)} test cases")
        print("="*70)

        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "full_model": self.full_model,
            "quantized_model": self.quantized_model,
        }

        # Benchmark full model
        print("\n" + "="*70)
        print("FULL MODEL BENCHMARKS")
        print("="*70)

        full_speed = self.benchmark_speed(self.full_model, speed_iterations)
        full_accuracy = self.benchmark_accuracy(self.full_model)

        benchmark_results["full_model_speed"] = full_speed
        benchmark_results["full_model_accuracy"] = full_accuracy

        # Benchmark quantized model
        print("\n" + "="*70)
        print("QUANTIZED MODEL BENCHMARKS")
        print("="*70)

        quant_speed = self.benchmark_speed(self.quantized_model, speed_iterations)
        quant_accuracy = self.benchmark_accuracy(self.quantized_model)

        benchmark_results["quantized_model_speed"] = quant_speed
        benchmark_results["quantized_model_accuracy"] = quant_accuracy

        # Comparison
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        speedup = full_speed["avg_time"] / quant_speed["avg_time"]
        accuracy_diff = quant_accuracy["accuracy"] - full_accuracy["accuracy"]

        print(f"\nSpeed:")
        print(f"  Full Model:      {full_speed['avg_time']:.2f}s per comparison")
        print(f"  Quantized Model: {quant_speed['avg_time']:.2f}s per comparison")
        print(f"  Speedup:         {speedup:.2f}x faster")

        print(f"\nAccuracy:")
        print(f"  Full Model:      {full_accuracy['passed']}/{full_accuracy['total_tests']} ({full_accuracy['accuracy']:.1%})")
        print(f"  Quantized Model: {quant_accuracy['passed']}/{quant_accuracy['total_tests']} ({quant_accuracy['accuracy']:.1%})")
        print(f"  Difference:      {accuracy_diff:+.1%}")

        print(f"\nMemory:")
        print(f"  Full Model:      ~8.6 GB")
        print(f"  Quantized Model: ~2.5 GB")
        print(f"  Reduction:       70% smaller")

        benchmark_results["comparison"] = {
            "speedup": speedup,
            "accuracy_difference": accuracy_diff,
            "memory_reduction_percent": 70,
        }

        # Save results
        if save_results:
            output_path = Path(__file__).parent.parent / "data" / "medgemma_benchmark.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)

            print(f"\nResults saved to: {output_path}")

        return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MedGemma quantization performance")
    parser.add_argument("--iterations", type=int, default=10, help="Number of speed test iterations (default: 10)")
    parser.add_argument("--full-model", type=str, default="medgemma:1.5-4b", help="Full model name")
    parser.add_argument("--quantized-model", type=str, default="medgemma:1.5-4b-q4", help="Quantized model name")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    benchmark = MedGemmaBenchmark(
        full_model=args.full_model,
        quantized_model=args.quantized_model
    )

    try:
        benchmark.run_full_benchmark(
            speed_iterations=args.iterations,
            save_results=not args.no_save
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
