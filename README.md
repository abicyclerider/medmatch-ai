# MedMatch AI

An AI-powered medical entity resolution system that prevents wrong-patient errors in healthcare using AI (Google Gemini API or local MedGemma via Ollama).

## Problem Statement

Wrong-patient medical errors (e.g., operating on the wrong John Smith) are "never events" that cause deaths. Current systems use basic name matching. MedMatch AI uses deep medical understanding to create "medical fingerprints" ensuring accurate patient matching at critical decision points.

## Approach

Instead of simple demographic matching, we use a hybrid 4-stage pipeline:

1. **Blocking** - Reduces O(n²) comparisons by 97% using phonetic and key-based strategies
2. **Deterministic Rules** - Fast exact matching for clear cases (74% of decisions)
3. **Feature Scoring** - Weighted confidence scores for moderate difficulty cases
4. **AI Medical Fingerprinting** - Deep medical history comparison using Gemini API or local MedGemma

The system understands:

- Medical abbreviations (T2DM = Type 2 Diabetes, HTN = Hypertension)
- Name variations and typos (Jennifer vs Jenny, Johnson vs Jonson)
- Date format differences and errors (03/15/1980 vs 3/15/80)
- Medical context (Metformin → Diabetes treatment)
- Clinical narratives and medication histories

## Project Status

✅ **Phase 1 Complete** - Synthetic data generation system
✅ **Phase 2 Complete** - AI-powered entity resolution (94.51% accuracy)
✅ **Phase 4 In Progress** - Local MedGemma deployment via Ollama (3/11 tasks complete)
📅 **Phase 3 Planned** - Advanced optimization and uncertainty quantification

### Current Capabilities

- **94.51% overall accuracy** on patient matching across all difficulty levels
- **4-stage progressive pipeline**: Blocking → Rules → Scoring → AI Medical Fingerprinting
- **97% efficiency improvement** through intelligent blocking (33,930 → 437 pairs)
- **100% accuracy on AI decisions** for medical history comparison
- **Explainable decisions** with confidence scores and human-readable explanations

## Entity Resolution Results

Our 4-stage progressive pipeline achieves exceptional accuracy on synthetic patient matching:

**Overall Performance (437 test pairs):**

- **Accuracy:** 94.51%
- **Precision:** 95%
- **Recall:** 96%
- **F1 Score:** 95%

**Accuracy by Difficulty:**

| Difficulty | Target | Achieved | Status |
| ---------- | ------ | -------- | ------ |
| Easy | 95% | **100.00%** | ✅ PASS |
| Medium | 85% | **100.00%** | ✅ PASS |
| Hard | 70% | **88.24%** | ✅ PASS |
| Ambiguous | 70% | **80.54%** | ✅ PASS |

**Pipeline Stages:**

1. **Blocking:** Reduces candidate pairs by 97% (33,930 → ~1,000) with 97.3% recall
2. **Deterministic Rules:** Handles 74% of decisions with 92.6% accuracy
3. **Feature Scoring:** Handles 0% of decisions (when AI enabled, otherwise 26% with 87.6% accuracy)
4. **AI Medical Fingerprinting:** Handles 26% of decisions with 100% accuracy

All targets exceeded! See [evaluation notebook](notebooks/01_entity_resolution_evaluation.ipynb) for detailed analysis.

## Tech Stack

- **Python 3.12.4** with venv
- **PyTorch 2.9.1** (Mac Metal/MPS acceleration)
- **AI Backends:**
  - **Ollama + MedGemma 1.5 4B** (local inference, HIPAA-compliant)
  - **Google Gemini API** (development/testing)
- **Data Processing:** pandas, numpy, pydicom
- **Development:** Jupyter Lab, pytest, black, ruff

## Setup

### 1. Clone and Create Virtual Environment
```bash
git clone git@github.com:abicyclerider/medmatch-ai.git
cd medmatch-ai
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure AI Backend

**Option A: Ollama (Recommended for Production)**

For HIPAA-compliant local inference:

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Download and import MedGemma (see docs/ollama_setup.md for details)
# Requires HuggingFace token with access to google/medgemma-1.5-4b-it
```

See [docs/ollama_setup.md](docs/ollama_setup.md) for complete setup instructions.

**Option B: Gemini API (Development/Testing)**

```bash
cp .env.example .env
# Edit .env and add your Google AI API key from https://aistudio.google.com/apikey
```

⚠️ **Privacy Warning**: Gemini API sends data to Google's servers. Only use with synthetic/anonymized data. For production with real patient data, use Ollama.

### 4. Verify Installation
```bash
python verify_install.py
python test_medgemma.py
```

### 5. Launch Jupyter Lab
```bash
jupyter lab
# Open notebooks/00_environment_test.ipynb
```

## Usage Examples

### Generate Synthetic Data

```bash
# Fast generation (30 seconds, rule-based)
python generate_synthetic_data.py --no-ai

# AI-assisted generation (15-20 minutes with rate limiting)
python generate_synthetic_data.py --api-rate-limit 5
```

### Run Entity Resolution (Python API)

```python
from medmatch.matching import PatientMatcher, PatientRecord
from medmatch.data.models.patient import Demographics
from datetime import date

# Load or create patient records
records = [...]  # Load from CSV or create manually

# Create matcher with Ollama (local MedGemma)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    use_ai=True,
    ai_backend="ollama",  # Local inference (HIPAA-compliant)
)

# Or use Gemini API (development/testing only)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    use_ai=True,
    ai_backend="gemini",  # Requires GOOGLE_AI_API_KEY in .env
)

# Match two specific records
result = matcher.match_pair(records[0], records[1])
print(f"Match: {result.is_match}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Stage: {result.stage}")
print(result.explanation)

# Batch matching on entire dataset
results = matcher.match_datasets(records, show_progress=True)
stats = matcher.get_stats(results)
print(f"Found {stats['matches']} matches in {stats['total_pairs']} pairs")
```

### Batch Matching (CLI)

```bash
# With Ollama (local MedGemma)
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results.json \
  --use-ai \
  --ai-backend ollama \
  --progress

# Or with Gemini API (development/testing)
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results.json \
  --use-ai \
  --ai-backend gemini \
  --progress
```

### Evaluate Results

```python
from medmatch.evaluation import MatchEvaluator

# Evaluate against ground truth
evaluator = MatchEvaluator('data/synthetic/ground_truth.csv')
metrics = evaluator.evaluate(results)

print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Precision: {metrics.precision:.2%}")
print(f"Recall: {metrics.recall:.2%}")

# Detailed analysis in Jupyter notebook
# See notebooks/01_entity_resolution_evaluation.ipynb
```

## Project Structure

```
medmatch-ai/
├── src/medmatch/          # Main package
│   ├── models/            # AI model integration
│   ├── matching/          # Entity resolution algorithms
│   ├── data/              # Data processing
│   └── utils/             # Utilities
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Test suite
├── data/                  # Local data (gitignored)
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## Development Workflow

1. **Prototype in notebooks** - Experiment with matching strategies
2. **Refactor to src/** - Move working code to package
3. **Write tests** - Ensure correctness
4. **Document** - Keep README and docs updated

## Roadmap

### Phase 1: Synthetic Data Generation ✅ COMPLETE

- ✅ Pydantic models for patient records
- ✅ Demographics generator with variations and edge cases
- ✅ AI-assisted medical record generation
- ✅ Ground truth labels for evaluation
- ✅ 261 records with 437 match pairs for testing

### Phase 2: Entity Resolution Algorithm ✅ COMPLETE

- ✅ **Phase 2.1:** Core infrastructure (PatientRecord, comparators)
- ✅ **Phase 2.2:** Blocking & deterministic rules
- ✅ **Phase 2.3:** Feature-based confidence scoring
- ✅ **Phase 2.4:** AI medical fingerprinting (Gemini API)
- ✅ **Phase 2.5:** Evaluation & explanation system
- ✅ **Phase 2.6:** Documentation & polish (current)

### Phase 3: Optimization & Advanced Features (Planned)

- [ ] Uncertainty quantification for borderline cases
- [ ] Advanced error analysis and correction
- [ ] Performance optimization for large datasets
- [ ] Ensemble methods combining multiple AI models
- [ ] Active learning for difficult cases

### Phase 4: MedGemma Integration & Production (Planned)

- [ ] Deploy MedGemma locally via Hugging Face
- [ ] Migrate from Gemini to MedGemma
- [ ] Benchmark accuracy improvements
- [ ] Optimize for Mac Metal performance
- [ ] Multi-modal support (imaging, lab data)
- [ ] Clinical workflow integration
- [ ] Privacy and HIPAA compliance
- [ ] Kaggle competition submission

## Documentation

- **[Matching Module README](src/medmatch/matching/README.md)** - Complete entity resolution system documentation with architecture, usage examples, and API reference
- **[Evaluation Notebook](notebooks/01_entity_resolution_evaluation.ipynb)** - Interactive analysis of matching performance with visualizations
- **[Quick Start Guide](docs/quickstart.md)** - 5-minute getting started guide for new users
- **[Project Context](.claude/CLAUDE.md)** - Detailed development history and design decisions
- **[Synthetic Data Plan](docs/synthetic_data_plan.md)** - Dataset specification and generation details

## Important Notes

### Medical Data Privacy
- **Development:** Use synthetic or anonymized data only
- **API Usage:** Be cautious - data sent to Google AI
- **Production:** Must use local deployment for real patient data (HIPAA)

### Current Limitations
- Using Gemini (general model) not MedGemma (medical specialist)
- API has rate limits (free tier)
- Not yet validated for clinical use

## License

[Add your license]

## Contributors

- Alex (Developer)
- Claude Sonnet 4.5 (AI Assistant)

---

**Built for the MedGemma Challenge - Preventing wrong-patient medical errors through AI**
