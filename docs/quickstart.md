# Quick Start Guide

Get started with MedMatch AI entity resolution in under 5 minutes.

## Prerequisites

- Python 3.12.4
- Git
- Google AI API key (optional, for AI features)

## 1. Installation

Clone the repository and set up the environment:

```bash
# Clone repository
git clone git@github.com:abicyclerider/medmatch-ai.git
cd medmatch-ai

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure API Key (Optional)

For AI medical fingerprinting, set up your Google AI API key:

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# Get one from: https://aistudio.google.com/apikey
echo "GOOGLE_AI_API_KEY=your_key_here" >> .env
```

AI is optional - the system works without it using deterministic rules and scoring.

## 3. Generate Synthetic Data

Create a test dataset to experiment with:

```bash
# Fast generation (30 seconds, no AI)
python generate_synthetic_data.py --no-ai

# Or with AI assistance (15-20 minutes)
python generate_synthetic_data.py --api-rate-limit 5
```

This creates:

- `data/synthetic/synthetic_demographics.csv` - Patient records
- `data/synthetic/synthetic_medical_records.json` - Medical histories
- `data/synthetic/ground_truth.csv` - Evaluation labels

## 4. Run Your First Match

Create a simple matching script ([examples/first_match.py](../examples/first_match.py)):

```python
from medmatch.matching import PatientMatcher
from medmatch.data.models.patient import Demographics
from datetime import date

# Create two patient records
record1 = Demographics(
    record_id="R001",
    patient_id="P001",
    name_first="John",
    name_middle="A",
    name_last="Smith",
    name_suffix=None,
    date_of_birth=date(1980, 1, 15),
    gender="M",
    mrn="MRN001",
    ssn_last4="1234",
    phone="555-0100",
    email="john.smith@email.com",
    address=None,
    record_source="Hospital A",
    record_date=date(2025, 1, 1),
    data_quality_flag=None,
)

record2 = Demographics(
    record_id="R002",
    patient_id="P001",
    name_first="Johnny",  # Nickname variation
    name_middle=None,
    name_last="Smith",
    name_suffix=None,
    date_of_birth=date(1980, 1, 15),
    gender="M",
    mrn="MRN001",
    ssn_last4="1234",
    phone="555-0100",
    email=None,
    address=None,
    record_source="Hospital B",
    record_date=date(2025, 1, 2),
    data_quality_flag=None,
)

# Convert to PatientRecord
from medmatch.matching import PatientRecord
patient1 = PatientRecord.from_demographics(record1)
patient2 = PatientRecord.from_demographics(record2)

# Create matcher (AI disabled for speed)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    use_ai=False,
)

# Match the records
result = matcher.match_pair(patient1, patient2)

# View results
print(f"Match: {result.is_match}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Match Type: {result.match_type}")
print(f"Stage: {result.stage}")
print(f"\nExplanation:")
print(result.explanation)
```

Run it:

```bash
python examples/first_match.py
```

Expected output:

```text
Match: True
Confidence: 0.95
Match Type: exact
Stage: rules

Explanation:
Same MRN (MRN001) + similar name (0.93)
```

## 5. Batch Matching with CLI

Process an entire dataset:

```bash
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results.json \
  --progress
```

View results:

```bash
# JSON format (full details)
cat results.json | jq '.summary'

# CSV format (spreadsheet-friendly)
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --output results.csv \
  --format csv

# Verbose format (human-readable)
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --format verbose | less
```

## 6. Enable AI Medical Fingerprinting

For hard cases, enable AI to compare medical histories:

```bash
# Requires GOOGLE_AI_API_KEY in .env
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results_with_ai.json \
  --use-ai \
  --api-rate-limit 0 \
  --progress
```

AI improves accuracy on ambiguous cases from ~80% to ~95%.

## 7. Evaluate Results

Analyze performance against ground truth:

```python
from medmatch.evaluation import MatchEvaluator

# Load evaluator with ground truth
evaluator = MatchEvaluator('data/synthetic/ground_truth.csv')

# Evaluate your results
metrics = evaluator.evaluate(results)

print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Precision: {metrics.precision:.2%}")
print(f"Recall: {metrics.recall:.2%}")
print(f"F1 Score: {metrics.f1_score:.2%}")

# Get detailed breakdown
by_difficulty = evaluator.evaluate_by_difficulty(results)
for difficulty, metrics in by_difficulty.items():
    print(f"{difficulty}: {metrics.accuracy:.2%}")
```

Or use the interactive notebook:

```bash
jupyter lab notebooks/01_entity_resolution_evaluation.ipynb
```

## What You've Accomplished

You now know how to:

- ✅ Install and configure MedMatch AI
- ✅ Generate synthetic patient data
- ✅ Match individual patient record pairs
- ✅ Process entire datasets with the CLI
- ✅ Enable AI medical fingerprinting
- ✅ Evaluate matching accuracy

## Next Steps

### Learn More About the System

- **[Matching Module README](../src/medmatch/matching/README.md)** - Comprehensive documentation of the 4-stage pipeline
- **[Scripts README](../scripts/README.md)** - CLI usage guide with all options
- **[Main README](../README.md)** - Project overview and current capabilities

### Customize Configuration

Adjust matching behavior with custom weights and thresholds:

```python
# Custom scoring weights (must sum to 1.0)
custom_weights = {
    'name_first': 0.10,
    'name_last': 0.15,
    'dob': 0.50,  # Emphasize date of birth
    'phone': 0.10,
    'mrn': 0.15,
}

# Custom thresholds (more conservative)
custom_thresholds = {
    'definite': 0.95,  # Higher bar for definite matches
    'probable': 0.85,
    'possible': 0.70,
}

matcher = PatientMatcher(
    use_scoring=True,
    scoring_weights=custom_weights,
    scoring_thresholds=custom_thresholds,
)
```

### Explore Advanced Features

- **Progressive pipeline** - Understand how blocking → rules → scoring → AI work together
- **Explainable decisions** - Generate human-readable explanations for each match
- **Error analysis** - Identify and understand false positives/negatives
- **Custom rules** - Add domain-specific matching rules

### Run Tests

Verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_blocking.py -v
pytest tests/test_rules.py -v
pytest tests/test_scoring.py -v
pytest tests/test_integration.py -v
```

### Contribute

Found a bug or have a feature request? Open an issue on GitHub!

## Common Issues

### API Key Not Found

**Error:** `ValueError: GOOGLE_AI_API_KEY not found in environment`

**Solution:**

```bash
# Make sure .env exists and has your key
cat .env

# Should contain:
# GOOGLE_AI_API_KEY=your_key_here

# If missing, add it:
echo "GOOGLE_AI_API_KEY=your_key_here" >> .env
```

### Slow Matching

**Problem:** Matching takes too long on large datasets

**Solution:**

```bash
# Ensure blocking is enabled (default)
python scripts/run_matcher.py --demographics data.csv --output results.json

# Disable AI for speed (still 90%+ accurate)
python scripts/run_matcher.py --demographics data.csv --output results.json
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'medmatch'`

**Solution:**

```bash
# Make sure you're in the project root
cd /Users/alex/repos/Kaggle/medmatch-ai

# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import src.medmatch; print('OK')"
```

## Need Help?

- **Documentation:** Check the READMEs in [src/medmatch/matching/](../src/medmatch/matching/README.md) and [scripts/](../scripts/README.md)
- **Examples:** See the evaluation notebook: [notebooks/01_entity_resolution_evaluation.ipynb](../notebooks/01_entity_resolution_evaluation.ipynb)
- **Issues:** Report bugs at [GitHub Issues](https://github.com/abicyclerider/medmatch-ai/issues)

---

**Ready to prevent wrong-patient medical errors?** Continue with the [Matching Module README](../src/medmatch/matching/README.md) for comprehensive documentation.
