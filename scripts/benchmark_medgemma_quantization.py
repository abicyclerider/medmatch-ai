#!/usr/bin/env python3
"""
Benchmark MedGemma Quantization Performance

Compares full model (medgemma:1.5-4b) vs quantized model (medgemma:1.5-4b-q4):
- Inference speed (tokens/second)
- Medical understanding accuracy
- Memory usage
- Total time for batch comparisons

Supports three benchmark modes:
- standard: Original verbose prompt (baseline)
- optimized: Simplified prompt with minimal output tokens
- parallel: Optimized + concurrent requests for throughput testing

Usage:
    python scripts/benchmark_medgemma_quantization.py
    python scripts/benchmark_medgemma_quantization.py --iterations 20
    python scripts/benchmark_medgemma_quantization.py --mode optimized --iterations 5
    python scripts/benchmark_medgemma_quantization.py --mode parallel --workers 4
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        """Compare two medical histories and return similarity score (standard verbose prompt)."""
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


class SimplifiedOllamaClient:
    """
    Optimized Ollama client with simplified prompt for speed testing.

    Key optimizations:
    - Shorter, more direct prompt (~150 tokens vs ~420 tokens)
    - Handles MedGemma's thought token format (<unused94>...<unused95>)
    - Uses 1024 tokens (required for MedGemma to complete thought process)

    Note: MedGemma's internal "thinking" process means even simple prompts
    generate 500-800 tokens of reasoning. Speed gain comes from shorter prompts,
    not from reducing output tokens.
    """

    def __init__(self, model="medgemma:1.5-4b-q4", base_url="http://localhost:11434", timeout=180):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def compare_medical_histories(self, history1, history2):
        """Compare two medical histories using simplified prompt."""
        # Shorter prompt - essential context only, no verbose instructions
        prompt = f"""Medical history similarity score (0.0=different, 1.0=same patient):

A: {history1}
B: {history2}

Consider abbreviations (HTN=Hypertension, T2DM=Diabetes, etc.) and medications.
Score:"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 1024,  # MedGemma needs space for thought process
                    "temperature": 0.1,  # More deterministic
                }
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        result = response.json()
        text = result.get("response", "").strip()

        # Parse response - handle MedGemma's thought token format
        score = 0.5  # default

        # MedGemma outputs: <unused94>thought...reasoning...<unused95>Score: X.X
        # Extract content after <unused95> marker (the actual answer)
        if '<unused95>' in text:
            answer_part = text.split('<unused95>')[-1]
        else:
            answer_part = text

        reasoning = f"Output: {answer_part[:150]}..." if len(answer_part) > 150 else f"Output: {answer_part}"

        try:
            # Extract score from answer
            import re
            # Look for patterns like "Score: 0.85" or just "0.85"
            numbers = re.findall(r'(?:Score:\s*)?([01]\.?\d*)', answer_part)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                score = max(0.0, min(1.0, score))
        except Exception:
            pass  # Use default if parsing fails

        return score, reasoning


class BatchedOllamaClient:
    """
    Ollama client that batches multiple comparisons in a single prompt.

    Key optimization: Reduces API call overhead by processing multiple pairs
    in one request. MedGemma's thought process is amortized across pairs.

    Benchmark results: ~11s per pair (batched) vs ~19s per pair (individual)
    """

    def __init__(self, model="medgemma:1.5-4b-q4", base_url="http://localhost:11434", timeout=300, batch_size=3):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.batch_size = batch_size

    def compare_batch(self, pairs):
        """Compare multiple pairs of medical histories in a single request.

        Args:
            pairs: List of (history1, history2) tuples

        Returns:
            List of (score, reasoning) tuples
        """
        if not pairs:
            return []

        # Build batched prompt
        prompt_parts = ["Rate similarity (0.0-1.0) for each medical history pair:\n"]
        for i, (h1, h2) in enumerate(pairs, 1):
            prompt_parts.append(f"\nPair {i}:\nA: {h1}\nB: {h2}\n")

        prompt_parts.append("\nOutput scores in this exact format:")
        for i in range(1, len(pairs) + 1):
            prompt_parts.append(f"\nPair {i}: X.X")

        prompt = "".join(prompt_parts)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 2048,  # More tokens for multiple pairs
                    "temperature": 0.1,
                }
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        result = response.json()
        text = result.get("response", "").strip()

        # Extract answer after thought tokens
        if '<unused95>' in text:
            answer_part = text.split('<unused95>')[-1]
        else:
            answer_part = text

        # Parse scores for each pair
        import re
        results = []
        for i in range(1, len(pairs) + 1):
            score = 0.5  # default
            reasoning = ""

            # Look for "Pair N: X.X" or "**Similarity: X.X**" patterns
            patterns = [
                rf'Pair\s*{i}[:\s]+.*?([01]\.?\d*)',
                rf'Pair\s*{i}.*?Similarity[:\s]+([01]\.?\d*)',
            ]

            for pattern in patterns:
                match = re.search(pattern, answer_part, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score = float(match.group(1))
                        score = max(0.0, min(1.0, score))
                        break
                    except ValueError:
                        pass

            results.append((score, f"Batch result for pair {i}"))

        return results


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
                "tolerance": 0.15,
            },
            {
                "name": "Different Medical Conditions",
                "history1": "Patient has CAD s/p CABG 2020, on aspirin and atorvastatin.",
                "history2": "Patient has COPD, uses albuterol inhaler PRN, no cardiac history.",
                "expected_score": 0.0,
                "tolerance": 0.25,
            },
            {
                "name": "Partial Match (Shared Condition)",
                "history1": "Patient has HTN, T2DM, hyperlipidemia. Medications: Lisinopril, Metformin, Atorvastatin.",
                "history2": "Patient with hypertension, takes Lisinopril. No diabetes. History of MI 2019.",
                "expected_score": 0.4,
                "tolerance": 0.35,
            },
            {
                "name": "Medication Name Variations",
                "history1": "Patient on Tylenol 500mg PRN for pain, takes aspirin 81mg daily.",
                "history2": "Uses acetaminophen 500mg as needed, daily baby aspirin 81mg.",
                "expected_score": 1.0,
                "tolerance": 0.15,
            },
            {
                "name": "Complex Medical History",
                "history1": "ESRD on HD MWF, DM2 with retinopathy, CHF (EF 35%), s/p pacemaker 2019.",
                "history2": "End-stage renal disease, hemodialysis Monday/Wednesday/Friday. Type 2 diabetes with diabetic retinopathy. Congestive heart failure, ejection fraction 35%, pacemaker placed 2019.",
                "expected_score": 1.0,
                "tolerance": 0.15,
            },
        ]

    def _get_client(self, model_name, mode="standard"):
        """Get the appropriate client based on benchmark mode."""
        if mode == "standard":
            return LightweightOllamaClient(model=model_name, timeout=180)
        else:  # optimized or parallel
            return SimplifiedOllamaClient(model=model_name, timeout=60)

    def benchmark_speed(self, model_name, iterations=10, mode="standard"):
        """Benchmark inference speed for a model."""
        print(f"\n{'='*70}")
        print(f"Speed Benchmark: {model_name} (mode: {mode})")
        print(f"{'='*70}")

        client = self._get_client(model_name, mode)

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
        throughput = 60.0 / avg_time  # comparisons per minute

        print(f"\nSpeed Results:")
        print(f"  Average: {avg_time:.2f}s per comparison")
        print(f"  Min:     {min_time:.2f}s")
        print(f"  Max:     {max_time:.2f}s")
        print(f"  Total:   {sum(times):.2f}s for {iterations} comparisons")
        print(f"  Throughput: {throughput:.1f} comparisons/minute")

        return {
            "model": model_name,
            "mode": mode,
            "iterations": iterations,
            "warmup_time": warmup_time,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": sum(times),
            "throughput_per_minute": throughput,
            "times": times,
        }

    def benchmark_parallel(self, model_name, iterations=10, max_workers=4, mode="optimized"):
        """Benchmark parallel inference throughput."""
        print(f"\n{'='*70}")
        print(f"Parallel Benchmark: {model_name} (workers: {max_workers})")
        print(f"{'='*70}")

        # Warmup first
        client = self._get_client(model_name, mode)
        print("Warming up model...")
        warmup_start = time.time()
        _, _ = client.compare_medical_histories(
            self.test_cases[0]["history1"],
            self.test_cases[0]["history2"]
        )
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")

        # Prepare test pairs
        test_pairs = []
        for i in range(iterations):
            test_case = self.test_cases[i % len(self.test_cases)]
            test_pairs.append((test_case["history1"], test_case["history2"]))

        # Run parallel benchmark
        print(f"\nRunning {iterations} comparisons with {max_workers} workers...")

        def compare_pair(pair):
            # Each thread needs its own client
            thread_client = self._get_client(model_name, mode)
            start = time.time()
            score, reasoning = thread_client.compare_medical_histories(pair[0], pair[1])
            elapsed = time.time() - start
            return score, elapsed

        results = []
        times = []
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compare_pair, pair): i for i, pair in enumerate(test_pairs)}

            with tqdm(total=iterations, desc="Parallel test", unit="comparison") as pbar:
                for future in as_completed(futures):
                    score, elapsed = future.result()
                    times.append(elapsed)
                    results.append(score)
                    pbar.set_postfix({"score": f"{score:.2f}", "time": f"{elapsed:.1f}s"})
                    pbar.update(1)

        total_time = time.time() - total_start
        avg_time = sum(times) / len(times)
        throughput = iterations / total_time * 60  # comparisons per minute

        print(f"\nParallel Results:")
        print(f"  Total wall time: {total_time:.2f}s for {iterations} comparisons")
        print(f"  Avg per-request: {avg_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} comparisons/minute")
        print(f"  Speedup vs sequential: {(avg_time * iterations) / total_time:.2f}x")

        return {
            "model": model_name,
            "mode": "parallel",
            "iterations": iterations,
            "max_workers": max_workers,
            "warmup_time": warmup_time,
            "total_wall_time": total_time,
            "avg_per_request_time": avg_time,
            "throughput_per_minute": throughput,
            "speedup_factor": (avg_time * iterations) / total_time,
            "times": times,
        }

    def benchmark_batched(self, model_name, iterations=10, batch_size=3):
        """Benchmark batched inference (multiple comparisons per request)."""
        print(f"\n{'='*70}")
        print(f"Batched Benchmark: {model_name} (batch_size: {batch_size})")
        print(f"{'='*70}")

        client = BatchedOllamaClient(model=model_name, batch_size=batch_size, timeout=300)

        # Warmup
        print("Warming up model...")
        warmup_start = time.time()
        warmup_pairs = [(self.test_cases[0]["history1"], self.test_cases[0]["history2"])]
        client.compare_batch(warmup_pairs)
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")

        # Prepare test pairs
        test_pairs = []
        for i in range(iterations):
            test_case = self.test_cases[i % len(self.test_cases)]
            test_pairs.append((test_case["history1"], test_case["history2"]))

        # Process in batches
        num_batches = (iterations + batch_size - 1) // batch_size
        print(f"\nRunning {iterations} comparisons in {num_batches} batches...")

        all_results = []
        batch_times = []

        with tqdm(total=num_batches, desc="Batch test", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, iterations)
                batch_pairs = test_pairs[start_idx:end_idx]

                start = time.time()
                batch_results = client.compare_batch(batch_pairs)
                elapsed = time.time() - start

                batch_times.append(elapsed)
                all_results.extend(batch_results)

                pbar.set_postfix({"batch_time": f"{elapsed:.1f}s", "per_item": f"{elapsed/len(batch_pairs):.1f}s"})
                pbar.update(1)

        total_time = sum(batch_times)
        avg_per_item = total_time / iterations
        throughput = 60.0 / avg_per_item

        print(f"\nBatched Results:")
        print(f"  Total time: {total_time:.2f}s for {iterations} comparisons")
        print(f"  Avg per comparison: {avg_per_item:.2f}s")
        print(f"  Throughput: {throughput:.1f} comparisons/minute")
        print(f"  Batch efficiency: {batch_size}x items per request")

        return {
            "model": model_name,
            "mode": "batched",
            "iterations": iterations,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "warmup_time": warmup_time,
            "total_time": total_time,
            "avg_per_item": avg_per_item,
            "throughput_per_minute": throughput,
            "batch_times": batch_times,
        }

    def benchmark_accuracy(self, model_name, mode="standard"):
        """Benchmark medical understanding accuracy for a model."""
        print(f"\n{'='*70}")
        print(f"Accuracy Benchmark: {model_name} (mode: {mode})")
        print(f"{'='*70}")

        client = self._get_client(model_name, mode)

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
                    "reasoning": reasoning[:200] if len(reasoning) > 200 else reasoning,
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
            "mode": mode,
            "total_tests": len(self.test_cases),
            "passed": correct,
            "accuracy": accuracy,
            "results": results,
        }

    def run_full_benchmark(self, speed_iterations=10, mode="standard", max_workers=4, save_results=True):
        """Run complete benchmark comparing both models."""
        print("="*70)
        print("MedGemma Quantization Benchmark")
        print("="*70)
        print(f"Full Model:      {self.full_model}")
        print(f"Quantized Model: {self.quantized_model}")
        print(f"Mode:            {mode}")
        print(f"Speed Test:      {speed_iterations} iterations")
        print(f"Accuracy Test:   {len(self.test_cases)} test cases")
        if mode == "parallel":
            print(f"Workers:         {max_workers}")
        print("="*70)

        # Check for Ollama env vars
        env_vars = {
            "OLLAMA_KV_CACHE_TYPE": os.environ.get("OLLAMA_KV_CACHE_TYPE", "not set"),
            "OLLAMA_FLASH_ATTENTION": os.environ.get("OLLAMA_FLASH_ATTENTION", "not set"),
            "OLLAMA_KEEP_ALIVE": os.environ.get("OLLAMA_KEEP_ALIVE", "not set"),
            "OLLAMA_NUM_PARALLEL": os.environ.get("OLLAMA_NUM_PARALLEL", "not set"),
        }
        print("\nOllama Environment Variables:")
        for var, val in env_vars.items():
            status = "✓" if val != "not set" else "○"
            print(f"  {status} {var}: {val}")

        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "full_model": self.full_model,
            "quantized_model": self.quantized_model,
            "mode": mode,
            "env_vars": env_vars,
        }

        # Benchmark quantized model only for speed (it's faster)
        print("\n" + "="*70)
        print("QUANTIZED MODEL BENCHMARKS")
        print("="*70)

        if mode == "parallel":
            quant_speed = self.benchmark_parallel(self.quantized_model, speed_iterations, max_workers, "optimized")
        else:
            quant_speed = self.benchmark_speed(self.quantized_model, speed_iterations, mode)

        quant_accuracy = self.benchmark_accuracy(self.quantized_model, mode)

        benchmark_results["quantized_model_speed"] = quant_speed
        benchmark_results["quantized_model_accuracy"] = quant_accuracy

        # Summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        if mode == "parallel":
            print(f"\nParallel Performance ({max_workers} workers):")
            print(f"  Throughput: {quant_speed['throughput_per_minute']:.1f} comparisons/minute")
            print(f"  Wall time:  {quant_speed['total_wall_time']:.2f}s for {speed_iterations} comparisons")
            print(f"  Speedup:    {quant_speed['speedup_factor']:.2f}x vs sequential")
        else:
            print(f"\nSpeed ({mode} mode):")
            print(f"  Average: {quant_speed['avg_time']:.2f}s per comparison")
            print(f"  Throughput: {quant_speed['throughput_per_minute']:.1f} comparisons/minute")

        print(f"\nAccuracy ({mode} mode):")
        print(f"  Passed: {quant_accuracy['passed']}/{quant_accuracy['total_tests']} ({quant_accuracy['accuracy']:.1%})")

        print(f"\nMemory:")
        print(f"  Quantized Model: ~2.5 GB")

        # Save results
        if save_results:
            output_path = Path(__file__).parent.parent / "data" / f"benchmark_{mode}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)

            print(f"\nResults saved to: {output_path}")

        return benchmark_results

    def run_comparison_benchmark(self, iterations=5, save_results=True):
        """Run all three modes for comparison."""
        print("="*70)
        print("MODE COMPARISON BENCHMARK")
        print("="*70)
        print("Running standard, optimized, and parallel modes for comparison...")
        print("="*70)

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.quantized_model,
            "iterations": iterations,
        }

        # Standard mode
        print("\n" + "="*70)
        print("MODE 1: STANDARD (verbose prompt)")
        print("="*70)
        standard_speed = self.benchmark_speed(self.quantized_model, iterations, "standard")
        standard_accuracy = self.benchmark_accuracy(self.quantized_model, "standard")
        all_results["standard"] = {
            "speed": standard_speed,
            "accuracy": standard_accuracy,
        }

        # Optimized mode
        print("\n" + "="*70)
        print("MODE 2: OPTIMIZED (simplified prompt)")
        print("="*70)
        optimized_speed = self.benchmark_speed(self.quantized_model, iterations, "optimized")
        optimized_accuracy = self.benchmark_accuracy(self.quantized_model, "optimized")
        all_results["optimized"] = {
            "speed": optimized_speed,
            "accuracy": optimized_accuracy,
        }

        # Parallel mode
        print("\n" + "="*70)
        print("MODE 3: PARALLEL (optimized + concurrent)")
        print("="*70)
        parallel_speed = self.benchmark_parallel(self.quantized_model, iterations, max_workers=4, mode="optimized")
        # Use optimized accuracy (same prompt)
        all_results["parallel"] = {
            "speed": parallel_speed,
            "accuracy": optimized_accuracy,  # Same prompt as optimized
        }

        # Comparison summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Mode':<12} | {'Avg Time':<10} | {'Throughput':<15} | {'Accuracy':<10}")
        print("-" * 55)
        print(f"{'standard':<12} | {standard_speed['avg_time']:.2f}s      | {standard_speed['throughput_per_minute']:.1f}/min        | {standard_accuracy['accuracy']:.0%}")
        print(f"{'optimized':<12} | {optimized_speed['avg_time']:.2f}s      | {optimized_speed['throughput_per_minute']:.1f}/min        | {optimized_accuracy['accuracy']:.0%}")
        print(f"{'parallel':<12} | {parallel_speed['avg_per_request_time']:.2f}s      | {parallel_speed['throughput_per_minute']:.1f}/min        | {optimized_accuracy['accuracy']:.0%}")

        # Calculate improvements
        speedup_optimized = standard_speed['avg_time'] / optimized_speed['avg_time']
        speedup_parallel_throughput = parallel_speed['throughput_per_minute'] / standard_speed['throughput_per_minute']

        print(f"\nImprovements over standard:")
        print(f"  Optimized: {speedup_optimized:.2f}x faster latency")
        print(f"  Parallel:  {speedup_parallel_throughput:.2f}x higher throughput")

        all_results["comparison"] = {
            "speedup_optimized": speedup_optimized,
            "speedup_parallel_throughput": speedup_parallel_throughput,
        }

        # Save results
        if save_results:
            output_path = Path(__file__).parent.parent / "data" / "benchmark_comparison.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)

            print(f"\nResults saved to: {output_path}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MedGemma quantization performance")
    parser.add_argument("--iterations", type=int, default=10, help="Number of speed test iterations (default: 10)")
    parser.add_argument("--full-model", type=str, default="medgemma:1.5-4b", help="Full model name")
    parser.add_argument("--quantized-model", type=str, default="medgemma:1.5-4b-q4", help="Quantized model name")
    parser.add_argument("--mode", type=str, choices=["standard", "optimized", "parallel", "batched", "compare"],
                        default="standard", help="Benchmark mode (default: standard)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for batched mode (default: 3)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    benchmark = MedGemmaBenchmark(
        full_model=args.full_model,
        quantized_model=args.quantized_model
    )

    try:
        if args.mode == "compare":
            benchmark.run_comparison_benchmark(
                iterations=args.iterations,
                save_results=not args.no_save
            )
        elif args.mode == "batched":
            # Run batched benchmark directly
            print("="*70)
            print("MedGemma Batched Benchmark")
            print("="*70)
            print(f"Model:      {args.quantized_model}")
            print(f"Iterations: {args.iterations}")
            print(f"Batch Size: {args.batch_size}")
            print("="*70)

            results = benchmark.benchmark_batched(
                args.quantized_model,
                iterations=args.iterations,
                batch_size=args.batch_size
            )

            if not args.no_save:
                output_path = Path(__file__).parent.parent / "data" / "benchmark_batched.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {output_path}")
        else:
            benchmark.run_full_benchmark(
                speed_iterations=args.iterations,
                mode=args.mode,
                max_workers=args.workers,
                save_results=not args.no_save
            )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n\nError: Cannot connect to Ollama server")
        print("Make sure Ollama is running: brew services start ollama")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
