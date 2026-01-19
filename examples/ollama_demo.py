#!/usr/bin/env python3
"""
Demo: Using OllamaClient for medical history comparison.

This example shows how to use the OllamaClient with MedGemma running
in Ollama for privacy-preserving patient matching.

Prerequisites:
    1. Ollama installed and running
    2. MedGemma model imported in Ollama
    3. See docs/ollama_setup.md for setup instructions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from medmatch.matching import OllamaClient, MedicalAIClient


def demo_direct_instantiation():
    """Demo: Create OllamaClient directly"""
    print("=" * 60)
    print("Example 1: Direct Instantiation")
    print("=" * 60)

    # Create client
    client = OllamaClient(
        model="medgemma:1.5-4b",
        temperature=0.3,  # Low temperature for factual medical responses
    )

    print(f"✓ Created: {client.backend_name}")

    # Compare medical histories
    hist1 = "Type 2 Diabetes, Hypertension, on Metformin and Lisinopril"
    hist2 = "T2DM, HTN, medications: Metformin, Lisinopril"

    print(f"\nRecord 1: {hist1}")
    print(f"Record 2: {hist2}")

    score, reasoning = client.compare_medical_histories(hist1, hist2)

    print(f"\nSimilarity: {score:.2f}")
    print(f"Reasoning: {reasoning}")


def demo_factory_method():
    """Demo: Create OllamaClient via factory"""
    print("\n" + "=" * 60)
    print("Example 2: Factory Method (Recommended)")
    print("=" * 60)

    # Create client via factory
    client = MedicalAIClient.create(backend="ollama")

    print(f"✓ Created: {client.backend_name}")

    # Example: Pediatric vs Adult patient
    hist1 = "Pediatric asthma, seasonal allergies, on Albuterol PRN"
    hist2 = "COPD, chronic bronchitis, on Albuterol and Spiriva"

    print(f"\nRecord 1: {hist1}")
    print(f"Record 2: {hist2}")

    score, reasoning = client.compare_medical_histories(hist1, hist2)

    print(f"\nSimilarity: {score:.2f}")
    print(f"Reasoning: {reasoning}")


def demo_with_patient_names():
    """Demo: Include patient names for context"""
    print("\n" + "=" * 60)
    print("Example 3: With Patient Names")
    print("=" * 60)

    client = OllamaClient()

    hist1 = "Coronary Artery Disease, prior MI in 2020, on Aspirin and Atorvastatin"
    hist2 = "CAD history, heart attack 2020, medications include ASA 81mg, statin"

    print(f"\nJohn Smith: {hist1}")
    print(f"Smith, J: {hist2}")

    score, reasoning = client.compare_medical_histories(
        hist1,
        hist2,
        patient_name_1="John Smith",
        patient_name_2="Smith, J",
    )

    print(f"\nSimilarity: {score:.2f}")
    print(f"Reasoning: {reasoning}")


def demo_privacy_note():
    """Demo: Privacy-preserving local inference"""
    print("\n" + "=" * 60)
    print("Example 4: Privacy-Preserving Architecture")
    print("=" * 60)

    print("""
Key Benefits of Ollama + MedGemma:

✓ All data stays local (no API calls to external services)
✓ HIPAA-compliant (no data sent to Google, OpenAI, etc.)
✓ Fast inference (~1-2 seconds per comparison)
✓ No per-request costs (runs on your hardware)
✓ Offline capable (no internet required)

Architecture:
    Your App → HTTP (localhost:11434) → Ollama → MedGemma

All processing happens on your local machine. Patient data NEVER
leaves your system.

For production deployment with real patient data, ALWAYS use
OllamaClient or MedGemmaAIClient (Transformers). NEVER use
GeminiAIClient with real PHI.
    """)


def main():
    print("\n" + "=" * 60)
    print("OllamaClient Demo - Privacy-Preserving Patient Matching")
    print("=" * 60)

    try:
        # Run demos
        demo_direct_instantiation()
        demo_factory_method()
        demo_with_patient_names()
        demo_privacy_note()

        print("\n" + "=" * 60)
        print("✅ Demo Complete!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Use OllamaClient in PatientMatcher with ai_backend='ollama'")
        print("2. Run benchmarks comparing Gemini vs MedGemma (Task 7)")
        print("3. See docs/ollama_setup.md for more details")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Run: brew services start ollama")
        print("2. Is MedGemma loaded? Run: ollama list | grep medgemma")
        print("3. See docs/ollama_setup.md for setup instructions")
        sys.exit(1)


if __name__ == "__main__":
    main()
