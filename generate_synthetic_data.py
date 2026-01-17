#!/usr/bin/env python3
"""
Wrapper script to generate synthetic patient dataset.

Usage:
    python generate_synthetic_data.py [--num-patients N] [--no-ai]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medmatch.data.generate_dataset import main

if __name__ == "__main__":
    main()
