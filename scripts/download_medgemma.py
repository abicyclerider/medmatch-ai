#!/usr/bin/env python3
"""
Download and verify MedGemma model from Hugging Face.

This script will:
1. Check if you have Hugging Face access
2. Attempt to download MedGemma-2B
3. Verify the model works on your Mac M3 Pro (MPS)

Usage:
    python scripts/download_medgemma.py
"""

import sys
import torch
from pathlib import Path


def check_device():
    """Check available compute devices."""
    print("="*80)
    print("DEVICE CHECK")
    print("="*80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Mac Metal) available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Will use device: {device} (Mac Metal GPU)")
    else:
        device = "cpu"
        print(f"⚠ Will use device: {device} (slower)")

    print()
    return device


def download_medgemma(model_name="google/medgemma-1.5-4b-it"):
    """
    Download MedGemma model from Hugging Face.

    Args:
        model_name: Model identifier on Hugging Face
    """
    print("="*80)
    print(f"DOWNLOADING {model_name}")
    print("="*80)
    print()
    print("This will download ~8GB of model files.")
    print("On first run, this may take 5-15 minutes depending on your connection.")
    print()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Download tokenizer first (small, fast)
        print(f"[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded successfully")
        print()

        # Download model (large, slow)
        print(f"[2/2] Downloading model (~8GB)...")
        print("This will take several minutes...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Reduce memory usage
            low_cpu_mem_usage=True,  # Optimize loading
            device_map="auto",  # Auto-select device
        )

        print(f"✓ Model loaded successfully")
        print()

        return tokenizer, model

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print()
        print("Please install required packages:")
        print("  pip install transformers>=4.40.0 accelerate>=0.27.0")
        sys.exit(1)

    except Exception as e:
        error_msg = str(e)

        # Check for common errors
        if "gated" in error_msg.lower() or "401" in error_msg:
            print(f"✗ Access denied to {model_name}")
            print()
            print("MedGemma is a gated model that requires approval.")
            print()
            print("To get access:")
            print("1. Create a Hugging Face account at https://huggingface.co/join")
            print("2. Request access at https://huggingface.co/google/medgemma-1.5-4b-it")
            print("3. Accept the license agreement (approval usually instant to 24 hours)")
            print("4. Create an access token at https://huggingface.co/settings/tokens")
            print("5. Login with: huggingface-cli login")
            print()
            print("See docs/medgemma_setup.md for detailed setup instructions")
            print()
            print("Alternative: Use Gemini API backend while waiting for approval")
            sys.exit(1)

        elif "404" in error_msg or "not found" in error_msg.lower():
            print(f"✗ Model not found: {model_name}")
            print()
            print("The model name may be incorrect or the model may not exist.")
            print("Check https://huggingface.co/google for available MedGemma models.")
            sys.exit(1)

        else:
            print(f"✗ Unexpected error: {e}")
            print()
            print("Full error details:")
            print(error_msg)
            sys.exit(1)


def test_inference(tokenizer, model, device):
    """
    Test model inference with a simple medical comparison.

    Args:
        tokenizer: Loaded tokenizer
        model: Loaded model
        device: Device to run on ("mps" or "cpu")
    """
    print("="*80)
    print("TESTING INFERENCE")
    print("="*80)
    print()

    # Simple test prompt
    test_prompt = """Compare these two medical histories:

Record 1: Patient has Type 2 Diabetes, Hypertension
Record 2: T2DM, HTN

Are these the same conditions? Answer yes or no."""

    print(f"Test prompt:")
    print(test_prompt)
    print()

    try:
        # Tokenize
        print("Tokenizing...")
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move to device
        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            model = model.to("mps")

        print(f"Generating response on {device}...")

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print()
        print("Model response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print()
        print("✓ Inference test successful!")
        print()

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        print()
        return False


def main():
    """Main execution."""
    print()
    print("MedGemma Model Download & Verification")
    print()

    # Check device
    device = check_device()

    # Download model
    tokenizer, model = download_medgemma()

    # Test inference
    success = test_inference(tokenizer, model, device)

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if success:
        print("✓ MedGemma-2B is ready to use!")
        print()
        print("Next steps:")
        print("1. The model is cached locally (~/.cache/huggingface/)")
        print("2. Future runs will load instantly (no re-download)")
        print("3. You can now use MedGemma in the matching pipeline")
        print()
        print("Example usage:")
        print("  from src.medmatch.matching.ai_client import MedicalAIClient")
        print("  client = MedicalAIClient.create(backend='medgemma')")
        print()
    else:
        print("⚠ Model downloaded but inference failed")
        print()
        print("You can still use Gemini API as a fallback:")
        print("  from src.medmatch.matching.ai_client import MedicalAIClient")
        print("  client = MedicalAIClient.create(backend='gemini')")
        print()

    print("="*80)


if __name__ == "__main__":
    main()
