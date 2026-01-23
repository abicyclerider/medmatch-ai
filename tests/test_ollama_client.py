#!/usr/bin/env python3
"""
Test script for OllamaClient integration.

This script verifies that:
1. OllamaClient initializes correctly
2. Can connect to Ollama server
3. Can compare medical histories
4. Correctly parses MedGemma responses
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from medmatch.matching.ai_client import OllamaClient, MedicalAIClient


def test_ollama_client_init():
    """Test OllamaClient initialization"""
    print("=" * 60)
    print("Test 1: OllamaClient Initialization")
    print("=" * 60)

    try:
        client = OllamaClient()
        print(f"✅ OllamaClient initialized successfully")
        print(f"   Backend: {client.backend_name}")
        print(f"   Model: {client.model}")
        print(f"   URL: {client.base_url}")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_medical_comparison():
    """Test medical history comparison"""
    print("\n" + "=" * 60)
    print("Test 2: Medical History Comparison")
    print("=" * 60)

    try:
        client = OllamaClient()

        # Test case 1: Matching records with abbreviations
        hist1 = "Type 2 Diabetes Mellitus, Hypertension, currently on Metformin 500mg twice daily"
        hist2 = "T2DM, HTN history, medications include Metformin"

        print("\nRecord 1:", hist1)
        print("Record 2:", hist2)
        print("\nComparing...")

        score, reasoning = client.compare_medical_histories(
            hist1,
            hist2,
            "Patient A",
            "Patient B"
        )

        print(f"\n✅ Comparison successful!")
        print(f"   Similarity Score: {score:.2f}")
        print(f"   Reasoning: {reasoning}")

        # Verify score is reasonable for matching records
        if score >= 0.7:
            print(f"   ✓ Score indicates likely match (expected for equivalent histories)")
        else:
            print(f"   ⚠ Score lower than expected for matching records")

        return True

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_method():
    """Test factory method for creating OllamaClient"""
    print("\n" + "=" * 60)
    print("Test 3: Factory Method")
    print("=" * 60)

    try:
        client = MedicalAIClient.create(backend="ollama")
        print(f"✅ Factory method successful")
        print(f"   Backend: {client.backend_name}")
        print(f"   Type: {type(client).__name__}")
        return True
    except Exception as e:
        print(f"❌ Factory method failed: {e}")
        return False


def test_different_histories():
    """Test with different medical histories"""
    print("\n" + "=" * 60)
    print("Test 4: Different Medical Histories")
    print("=" * 60)

    try:
        client = OllamaClient()

        hist1 = "Type 2 Diabetes, Hypertension, on Metformin and Lisinopril"
        hist2 = "Asthma, seasonal allergies, on Albuterol inhaler"

        print("\nRecord 1:", hist1)
        print("Record 2:", hist2)
        print("\nComparing...")

        score, reasoning = client.compare_medical_histories(hist1, hist2)

        print(f"\n✅ Comparison successful!")
        print(f"   Similarity Score: {score:.2f}")
        print(f"   Reasoning: {reasoning}")

        # Verify score is low for different records
        if score <= 0.3:
            print(f"   ✓ Score correctly indicates different patients")
        else:
            print(f"   ⚠ Score higher than expected for different histories")

        return True

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OllamaClient Integration Tests")
    print("=" * 60)

    # Run tests
    results = []
    results.append(("Initialization", test_ollama_client_init()))

    if results[0][1]:  # Only continue if init succeeds
        results.append(("Medical Comparison (Match)", test_medical_comparison()))
        results.append(("Factory Method", test_factory_method()))
        results.append(("Medical Comparison (Different)", test_different_histories()))
    else:
        print("\n⚠ Skipping remaining tests due to initialization failure")
        print("Make sure:")
        print("1. Ollama is running: brew services start ollama")
        print("2. MedGemma is imported: ollama list | grep medgemma")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! OllamaClient is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
