#!/usr/bin/env python3
"""Verify that all critical dependencies are installed and working correctly."""

import sys

def check_pytorch():
    """Check PyTorch installation and Metal/MPS availability."""
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")

        # Check for MPS (Metal Performance Shaders) support
        if torch.backends.mps.is_available():
            print(f"✓ Metal/MPS acceleration: AVAILABLE")
            if torch.backends.mps.is_built():
                print(f"✓ MPS backend: BUILT-IN")
            else:
                print(f"⚠ MPS backend: Not built (unexpected)")
        else:
            print(f"⚠ Metal/MPS acceleration: NOT AVAILABLE")

        return True
    except ImportError as e:
        print(f"✗ PyTorch: NOT INSTALLED - {e}")
        return False

def check_google_ai():
    """Check Google Generative AI SDK."""
    try:
        import google.generativeai as genai
        print(f"✓ Google Generative AI SDK installed")
        return True
    except ImportError as e:
        print(f"✗ Google Generative AI: NOT INSTALLED - {e}")
        return False

def check_data_tools():
    """Check data processing libraries."""
    results = []

    libs = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'pydicom': 'pydicom',
    }

    for module, name in libs.items():
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
            results.append(True)
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            results.append(False)

    return all(results)

def check_dev_tools():
    """Check development tools."""
    results = []

    libs = {
        'jupyterlab': 'jupyterlab',
        'pytest': 'pytest',
        'black': 'black',
        'ruff': 'ruff',
        'dotenv': 'python-dotenv',
    }

    for module, name in libs.items():
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
            results.append(True)
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            results.append(False)

    return all(results)

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("MedMatch AI - Dependency Verification")
    print("=" * 60)
    print()

    print("Python Version:")
    print(f"  {sys.version}")
    print()

    print("Core ML Framework:")
    pytorch_ok = check_pytorch()
    print()

    print("Google AI SDK:")
    google_ok = check_google_ai()
    print()

    print("Data Processing Libraries:")
    data_ok = check_data_tools()
    print()

    print("Development Tools:")
    dev_ok = check_dev_tools()
    print()

    print("=" * 60)
    if all([pytorch_ok, google_ok, data_ok, dev_ok]):
        print("✓ All dependencies installed successfully!")
        print("=" * 60)
        return 0
    else:
        print("⚠ Some dependencies missing - review errors above")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
