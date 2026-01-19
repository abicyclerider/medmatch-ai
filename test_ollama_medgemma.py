#!/usr/bin/env python3
"""
Test script for MedGemma via Ollama

This script verifies that:
1. Ollama is running and accessible
2. MedGemma model is loaded
3. Medical abbreviations are understood
4. API responses are correctly formatted
"""

import json
import requests
import sys

OLLAMA_URL = "http://localhost:11434"


def test_ollama_server():
    """Test if Ollama server is running"""
    print("Testing Ollama server connection...")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        print("✅ Ollama server is running")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running")
        print("   Start it with: brew services start ollama")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False


def test_medgemma_available():
    """Test if MedGemma model is available"""
    print("\nChecking for MedGemma model...")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])

        medgemma_found = any(
            "medgemma" in model.get("name", "").lower()
            for model in models
        )

        if medgemma_found:
            print("✅ MedGemma model found")
            for model in models:
                if "medgemma" in model.get("name", "").lower():
                    print(f"   Model: {model.get('name')}")
                    print(f"   Size: {model.get('size', 0) / (1024**3):.1f} GB")
            return True
        else:
            print("❌ MedGemma model not found")
            print("   Available models:", [m.get("name") for m in models])
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False


def test_medical_understanding():
    """Test MedGemma's medical knowledge"""
    print("\nTesting medical knowledge...")

    test_cases = [
        {
            "query": "What does HTN stand for in medical terminology? Answer in one sentence.",
            "expected_term": "hypertension",
        },
        {
            "query": "What does T2DM stand for in medical terminology? Answer in one sentence.",
            "expected_term": "type 2 diabetes",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['query'][:50]}...")

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "medgemma:1.5-4b",
                    "prompt": test_case["query"],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for factual responses
                    }
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "")

            # Extract the actual response (skip thought process if present)
            if "<unused95>" in response_text:
                response_text = response_text.split("<unused95>")[1].strip()

            print(f"Response: {response_text[:150]}...")

            # Check if expected term is in response
            if test_case["expected_term"].lower() in response_text.lower():
                print(f"✅ Correctly identified '{test_case['expected_term']}'")
            else:
                print(f"⚠️  Warning: Expected '{test_case['expected_term']}' not found in response")

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_medical_comparison():
    """Test medical record comparison (similar to our use case)"""
    print("\n\nTesting medical record comparison...")

    prompt = """Compare these two medical histories and determine if they describe the same patient.
Respond with a similarity score (0.0 to 1.0) and brief reasoning.

Record 1: Patient has Type 2 Diabetes Mellitus, Hypertension, currently on Metformin 500mg twice daily

Record 2: T2DM, HTN history, medications include Metformin

Format your response as:
SCORE: [0.0-1.0]
REASONING: [brief explanation]"""

    print("Comparing medical records...")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "medgemma:1.5-4b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                }
            },
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        # Extract actual response
        if "<unused95>" in response_text:
            response_text = response_text.split("<unused95>")[1].strip()

        print(f"\nResponse:\n{response_text}\n")

        # Check if response includes a score
        if "SCORE:" in response_text.upper() or any(
            char.isdigit() for char in response_text[:100]
        ):
            print("✅ Model provided structured comparison")
        else:
            print("⚠️  Model response format unexpected")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("MedGemma via Ollama - Test Suite")
    print("=" * 60)

    # Test 1: Server connection
    if not test_ollama_server():
        sys.exit(1)

    # Test 2: Model availability
    if not test_medgemma_available():
        sys.exit(1)

    # Test 3: Medical knowledge
    test_medical_understanding()

    # Test 4: Medical comparison (our use case)
    test_medical_comparison()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement OllamaClient in src/medmatch/matching/ai_client.py")
    print("2. Update PatientMatcher to use ai_backend='ollama'")
    print("3. Run benchmarks comparing Gemini vs MedGemma")


if __name__ == "__main__":
    main()
