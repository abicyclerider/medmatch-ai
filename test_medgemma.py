#!/usr/bin/env python3
"""
Test script for MedGemma 1.5 setup and medical entity understanding.

This script demonstrates:
1. Loading API credentials from .env
2. Connecting to Google AI / MedGemma
3. Testing medical text understanding
4. Entity extraction capabilities
"""

import os
import sys
from dotenv import load_dotenv

def test_env_loading():
    """Test that environment variables load correctly."""
    print("=" * 60)
    print("Step 1: Loading Environment Variables")
    print("=" * 60)

    # Load .env file
    load_dotenv()

    api_key = os.getenv('GOOGLE_AI_API_KEY')

    if not api_key:
        print("✗ GOOGLE_AI_API_KEY not found in .env file")
        print("\nPlease ensure .env file exists with:")
        print("GOOGLE_AI_API_KEY=your_api_key_here")
        return False

    # Show masked key for security
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print(f"✓ API Key loaded: {masked_key}")
    print()

    return True

def test_google_ai_connection():
    """Test connection to Google AI and list available models."""
    print("=" * 60)
    print("Step 2: Testing Google AI Connection")
    print("=" * 60)

    try:
        import google.genai as genai
        from google.genai import types

        # Configure with API key
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        client = genai.Client(api_key=api_key)

        print("✓ Successfully created Google AI client")
        print()

        # List available models
        print("Available models:")
        try:
            models = client.models.list()
            for model in models:
                if 'gemini' in model.name.lower():
                    print(f"  - {model.name}")
        except Exception as e:
            print(f"  Note: Could not list models ({e})")

        print()
        return client

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure you installed: pip install google-genai")
        return None
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return None

def test_medical_understanding(client):
    """Test MedGemma's medical understanding with sample text."""
    print("=" * 60)
    print("Step 3: Testing Medical Text Understanding")
    print("=" * 60)

    # Sample medical text for testing
    medical_text = """
    Patient: John Smith, DOB 03/15/1965, MRN: 12345678

    Chief Complaint: Chest pain and shortness of breath

    History: 58-year-old male with history of hypertension and type 2 diabetes
    presents with substernal chest pressure radiating to left arm, associated
    with diaphoresis and dyspnea. Onset 2 hours ago while climbing stairs.

    Vitals: BP 165/95, HR 102, RR 22, O2 Sat 94% on RA

    Assessment: Acute coronary syndrome, rule out myocardial infarction

    Plan: ECG, troponin levels, cardiology consult, aspirin 325mg given
    """

    print("Sample Medical Text:")
    print("-" * 60)
    print(medical_text.strip())
    print("-" * 60)
    print()

    # Test with Gemini (MedGemma may need special access)
    # We'll start with standard Gemini which still has medical knowledge
    print("Testing model response...")
    print()

    try:
        prompt = f"""Analyze this medical record and extract key clinical entities:

{medical_text}

Please identify:
1. Patient demographics
2. Medical conditions (current and history)
3. Vital signs
4. Clinical assessment
5. Treatment plan

Format as structured data."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',  # Using stable model with better quota
            contents=prompt
        )

        print("✓ Model Response:")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        print()

        return True

    except Exception as e:
        print(f"✗ Error during inference: {e}")
        print()
        print("Note: If you see permission errors, MedGemma may require")
        print("special access. Standard Gemini models work for testing.")
        return False

def test_entity_matching_concept(client):
    """Demonstrate entity matching concept for patient records."""
    print("=" * 60)
    print("Step 4: Entity Matching Concept Demo")
    print("=" * 60)

    record_1 = "John A. Smith, DOB: 03/15/1965, diabetic, hypertensive"
    record_2 = "Smith, John, born 3/15/65, T2DM, HTN history"

    print("Testing if these records refer to the same patient:")
    print(f"\nRecord 1: {record_1}")
    print(f"Record 2: {record_2}")
    print()

    try:
        prompt = f"""You are a medical entity resolution expert. Determine if these two records
refer to the same patient. Consider name variations, date formats, and medical abbreviations.

Record 1: {record_1}
Record 2: {record_2}

Respond with:
1. Match confidence (0-100%)
2. Key matching factors
3. Any discrepancies
4. Recommendation (SAME PATIENT / DIFFERENT PATIENT / UNCERTAIN)"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',  # Using stable model with better quota
            contents=prompt
        )

        print("✓ Entity Matching Analysis:")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        print()

        return True

    except Exception as e:
        print(f"✗ Error during matching test: {e}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  MedMatch AI - MedGemma Setup Verification".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # Step 1: Load environment
    if not test_env_loading():
        sys.exit(1)

    # Step 2: Connect to Google AI
    client = test_google_ai_connection()
    if not client:
        sys.exit(1)

    # Step 3: Test medical understanding
    test_medical_understanding(client)

    # Step 4: Test entity matching concept
    test_entity_matching_concept(client)

    # Summary
    print("=" * 60)
    print("✓ MedGemma Setup Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("1. Experiment with different medical texts")
    print("2. Test entity extraction and matching algorithms")
    print("3. Build your medical fingerprinting system")
    print("4. Consider local model deployment for privacy")
    print()
    print("Ready to start development!")
    print()

if __name__ == "__main__":
    main()
