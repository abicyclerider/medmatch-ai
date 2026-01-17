# MedMatch AI - Project Context for Claude

## Project Overview

**Name:** MedMatch AI
**Goal:** Prevent wrong-patient medical errors using AI-powered entity resolution
**Competition:** Google MedGemma Challenge
**Developer:** Alex (advanced Python/ML experience)
**Status:** Initial setup complete, ready for development phase

## The Problem We're Solving

Wrong-patient medical errors (e.g., operating on the wrong John Smith) are "never events" that cause deaths. Current hospital systems use basic name/DOB matching. Our solution uses deep medical understanding to create "medical fingerprints" that ensure accurate patient matching at critical clinical decision points.

## Technical Approach

Instead of simple demographic matching, we use medical AI to understand:
- Medical histories and treatment patterns
- Clinical terminology and abbreviations (T2DM vs diabetic, HTN vs hypertensive)
- Name variations and different date formats
- Medical context across multiple data sources (records, imaging, lab results)

The system provides confidence scores and explainable reasoning for each match decision.

## Current Setup Status

### Environment (✅ Complete)
- **Python:** 3.12.4 in virtual environment (`venv/`)
- **Platform:** MacBook Pro with Metal/MPS acceleration
- **PyTorch:** 2.9.1 with Mac GPU support enabled
- **Working Directory:** `/Users/alex/repos/Kaggle/medmatch-ai`
- **Remote:** `git@github.com:abicyclerider/medmatch-ai.git`

### Dependencies Installed
- **ML Framework:** torch, torchvision (with MPS support)
- **Google AI:** google-genai (new SDK), google-generativeai (deprecated but installed)
- **Data Processing:** pandas, numpy, pillow, pydicom
- **Synthetic Data:** faker (realistic fake data), pydantic v2 (data validation), pyyaml
- **Development:** jupyterlab, ipykernel, pytest, black, ruff, python-dotenv
- **Utilities:** tqdm, scikit-learn, matplotlib, seaborn

See [requirements.txt](requirements.txt) for full list.

### API Configuration
- **Google AI API Key:** Stored in `.env` file (gitignored)
- **Current Model:** `gemini-2.5-flash` (stable model with good quota)
- **Target Model:** MedGemma 1.5 (will migrate later - see roadmap)
- **API Key Location:** Get from https://aistudio.google.com/apikey

### Project Structure

```text
medmatch-ai/
├── src/medmatch/          # Main package
│   ├── __init__.py
│   ├── models/            # AI model integration (empty - future)
│   ├── matching/          # Entity resolution algorithms (empty - next phase)
│   ├── data/              # ✅ Data generation complete
│   │   ├── models/        # Pydantic schemas (Demographics, MedicalRecord, Patient)
│   │   ├── generators/    # Demographics, medical, edge case generators
│   │   ├── utils/         # Name, date, medical terminology utilities
│   │   └── generate_dataset.py  # Main dataset generation orchestrator
│   └── utils/             # Shared utilities (empty)
├── notebooks/             # Jupyter notebooks for exploration
│   └── 00_environment_test.ipynb  # Environment validation
├── tests/                 # Test suite (empty)
├── data/                  # Local data (gitignored)
│   ├── synthetic/         # ✅ Generated datasets
│   │   ├── synthetic_demographics.csv       # Patient records with variations
│   │   ├── synthetic_medical_records.json   # Clinical encounters
│   │   ├── ground_truth.csv                 # Match labels for evaluation
│   │   ├── dataset_metadata.json            # Generation statistics
│   │   └── README.md                        # Dataset documentation
│   ├── raw/               # Future: real datasets
│   └── processed/         # Future: processed data
├── docs/                  # Documentation
│   └── synthetic_data_plan.md  # ✅ Dataset specification
├── .env                   # API keys (gitignored)
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies (updated with faker, pydantic, pyyaml)
├── generate_synthetic_data.py  # ✅ CLI wrapper for dataset generation
├── verify_install.py      # Dependency verification script
├── test_medgemma.py       # API connection test script
└── README.md              # Project documentation
```

### Git History

- **c5407a0:** Empty repo with .gitignore
- **c95329f:** Python package structure, requirements.txt
- **b3059c2:** Google AI integration, README, notebooks
- **62d2161:** Project roadmap documentation
- **ab35a3e:** Synthetic data implementation guide
- **8223755:** Project context file (this file)
- **7d0fc6b:** ✅ **Synthetic data generation system (3,424 lines, 17 files)**

## Proof of Concept Results

Successfully tested AI-powered patient matching with `test_medgemma.py`:

**Test Case:**
- Record 1: "John A. Smith, DOB: 03/15/1965, diabetic, hypertensive"
- Record 2: "Smith, John, born 3/15/65, T2DM, HTN history"

**AI Analysis:**
- Match Confidence: 98%
- Correctly identified same patient despite format differences
- Understood medical abbreviations (T2DM = Type 2 Diabetes, HTN = Hypertension)
- Provided explainable reasoning

This validates the core approach works with Gemini API.

## Important Design Decisions

### Using Gemini Instead of MedGemma (Currently)

**Decision:** Start with Gemini 2.5 Flash API, migrate to MedGemma later

**Rationale:**
- Gemini has good medical knowledge for prototyping
- Easy API access (no special permissions needed)
- Allows rapid iteration on matching algorithms
- MedGemma requires local deployment or Vertex AI
- Core algorithm logic is model-agnostic

**Migration Plan:**
1. Build matching algorithm with Gemini (current phase)
2. Deploy MedGemma locally via Hugging Face
3. Swap in MedGemma for production/competition submission
4. Benchmark accuracy improvements

### Privacy & Medical Data

**Current State:**
- Using Google AI API (data sent to Google)
- Only use synthetic/anonymized data for development
- Never use real patient data with API

**Production Requirements:**
- Must use local MedGemma deployment for real patient data
- HIPAA compliance requires on-premise processing
- Kaggle submission likely requires local model too

## Development Roadmap

### ✅ Phase 1: Synthetic Data Generation (COMPLETED)

**Status:** Complete (commit 7d0fc6b)

**What was built:**
- Pydantic v2 models for type-safe patient records
- Demographics generator with name variations, typos, format changes
- AI-assisted medical record generator with rate limiting (5 req/min default)
- Edge case generators: twins, siblings, parent-child, common names, same name+DOB
- Ground truth labels for evaluation
- CLI with `--no-ai` and `--api-rate-limit` flags

**Generated dataset includes:**
- 75 unique patients with 225-375 demographic records (2-5 per patient)
- 75-150 medical records with clinical narratives
- 50% edge cases (twins, common names, data errors, variations)
- Difficulty levels: easy, medium, hard, ambiguous
- Output: CSV (demographics) + JSON (medical records) + ground truth

**Key files:**
- [generate_synthetic_data.py](generate_synthetic_data.py) - CLI wrapper
- [src/medmatch/data/](src/medmatch/data/) - All generation code
- [data/synthetic/README.md](data/synthetic/README.md) - Dataset documentation
- [docs/synthetic_data_plan.md](docs/synthetic_data_plan.md) - Specification

**Commands:**
```bash
python generate_synthetic_data.py --no-ai  # Fast rule-based generation
python generate_synthetic_data.py --api-rate-limit 5  # With AI (slower)
```

### 🎯 Phase 2: Entity Resolution Algorithm (NEXT)

**Goal:** Build AI-powered patient matching system

**Tasks:**
1. Design matching algorithm architecture
2. Implement pairwise record comparison
3. Build confidence scoring system (0-100%)
4. Add explainable AI (why records match/don't match)
5. Handle medical terminology (T2DM = Type 2 Diabetes, HTN = Hypertension)
6. Test against ground truth labels
7. Measure accuracy on easy/medium/hard cases

**Success criteria:**
- 95%+ accuracy on easy cases
- 85%+ accuracy on medium cases
- 70%+ accuracy on hard cases
- Explainable reasoning for each decision

### Phase 3: Evaluation & Optimization

1. Build comprehensive evaluation metrics (precision, recall, F1)
2. Analyze failure cases (false positives/negatives)
3. Optimize matching logic based on error analysis
4. Add uncertainty quantification
5. Create benchmarking suite

### Phase 4: MedGemma Integration

1. Deploy MedGemma locally via Hugging Face
2. Optimize for Mac Metal/MPS performance
3. Swap Gemini for MedGemma in matching pipeline
4. Benchmark accuracy improvements
5. Prepare for Kaggle competition submission

## Key Files to Reference

### Core Documentation

- [README.md](README.md) - Complete project documentation
- [.claude/CLAUDE.md](.claude/CLAUDE.md) - This file (project context)
- [docs/synthetic_data_plan.md](docs/synthetic_data_plan.md) - Dataset specification

### Synthetic Data System

- [generate_synthetic_data.py](generate_synthetic_data.py) - CLI wrapper
- [src/medmatch/data/generate_dataset.py](src/medmatch/data/generate_dataset.py) - Main orchestrator
- [src/medmatch/data/models/patient.py](src/medmatch/data/models/patient.py) - Pydantic schemas
- [data/synthetic/README.md](data/synthetic/README.md) - Dataset documentation

### Testing & Validation

- [test_medgemma.py](test_medgemma.py) - API connection test with entity matching demo
- [verify_install.py](verify_install.py) - Dependency verification
- [notebooks/00_environment_test.ipynb](notebooks/00_environment_test.ipynb) - Interactive validation

### Configuration

- [requirements.txt](requirements.txt) - All Python dependencies
- [.env.example](.env.example) - Environment variable template
- [.gitignore](.gitignore) - Excludes generated data, venv, API keys

## Commands to Run

### Activate Environment

```bash
source venv/bin/activate
```

### Generate Synthetic Data

```bash
# Fast generation (30 seconds, rule-based)
python generate_synthetic_data.py --no-ai

# AI-assisted generation (15-20 minutes with rate limiting)
python generate_synthetic_data.py --api-rate-limit 5

# Custom configuration
python generate_synthetic_data.py --num-patients 100 --seed 42 --no-ai
```

### Test API Connection

```bash
python test_medgemma.py
```

### Verify Installation

```bash
python verify_install.py
```

### Launch Jupyter

```bash
jupyter lab
```

### Run Tests (when created)

```bash
pytest tests/
```

## User Preferences & Context

- **Experience Level:** Advanced Python/ML developer
- **Learning Style:** Wants to understand concepts, not just run commands
- **Preferred Tools:**
  - Dependency management: requirements.txt (simple, Kaggle-compatible)
  - Notebooks: Yes (Jupyter Lab for prototyping)
  - Code quality: black, ruff
  - Testing: pytest

## Common Questions & Answers

**Q: Why not use MedGemma from the start?**
A: Prototyping with Gemini API is faster. Will migrate once core logic is solid.

**Q: Is the API key safe?**
A: Yes, stored in `.env` which is gitignored. Never committed to git.

**Q: What's the difference between Gemini and MedGemma?**
A: MedGemma is fine-tuned on medical data (more accurate), Gemini is general-purpose (easier access).

**Q: Can we use this with real patient data?**
A: Not with API (privacy risk). Need local MedGemma deployment for production.

## Important Notes for Future Sessions

1. **Virtual environment must be activated** - Commands run in user's terminal, not Claude's bash sessions
2. **API quota limits** - Free tier has request limits, use `gemini-2.5-flash` for better quota
3. **Mac MPS acceleration** - PyTorch can use Metal GPU, verified working
4. **Git workflow** - Always commit meaningful checkpoints, include explanatory commit messages
5. **Security** - Never commit `.env`, always verify `.gitignore` is working

## Resources

- **Google AI Studio:** https://aistudio.google.com/apikey
- **Rate Limits:** https://ai.google.dev/gemini-api/docs/rate-limits
- **Usage Monitor:** https://ai.dev/rate-limit
- **MedGemma (future):** Will use Hugging Face transformers for local deployment

## Synthetic Dataset Details

**Generated via:** `python generate_synthetic_data.py --no-ai`

**Output files** (gitignored, regeneratable):

- `data/synthetic/synthetic_demographics.csv` - Patient demographic records with variations
- `data/synthetic/synthetic_medical_records.json` - Clinical encounters with medical narratives
- `data/synthetic/ground_truth.csv` - Match labels for evaluation
- `data/synthetic/dataset_metadata.json` - Generation statistics

**Dataset characteristics:**

- 75 unique patients (configurable via `--num-patients`)
- 2-5 demographic records per patient (225-375 total records)
- 1-2 medical records per patient (75-150 total encounters)
- ~50% edge cases: twins (7-8 pairs), common names (15-20 collisions), family members
- Difficulty distribution: ~30% easy, 30% medium, 30% hard, 10% ambiguous
- Name variations: nicknames, misspellings, format changes, accents
- Data errors: typos, transposed digits, missing middle names
- Medical terminology: Abbreviations (HTN, T2DM), synonyms, clinical notes

**How records map:**

- `record_id` in demographics → unique identifier for each record (R0001, R0002...)
- `patient_id` in demographics → ground truth patient (P0001, P0002...)
- `match_group` in ground_truth.csv → groups records that should match
- Multiple records with same `patient_id` = same person, different variations

**Usage notes:**

- Dataset is reproducible (same seed = same output)
- Use `--seed` to generate different variants
- Ground truth CSV provides evaluation labels
- Do NOT use `patient_id` when building matcher (that's cheating - it's the answer!)

---

**Last Updated:** 2026-01-17
**Current Phase:** Phase 2 - Entity Resolution Algorithm
**Previous Session:** Synthetic data generation system complete (commit 7d0fc6b)
**Next Session Should Focus On:** Building the AI-powered patient matching algorithm
