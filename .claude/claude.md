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
- **Development:** jupyterlab, ipykernel, pytest, black, ruff, python-dotenv
- **Utilities:** tqdm, scikit-learn, matplotlib, seaborn

See [requirements.txt](requirements.txt) for full list.

### API Configuration
- **Google AI API Key:** Stored in `.env` file (gitignored)
- **Current Model:** `gemini-2.5-flash` (stable model with good quota)
- **Target Model:** MedGemma 1.5 (will migrate later - see roadmap)
- **API Key Location:** Get from https://aistudio.google.com/apikey

### Project Structure

```
medmatch-ai/
├── src/medmatch/          # Main package (empty scaffolding)
│   ├── __init__.py
│   ├── models/            # AI model integration
│   ├── matching/          # Entity resolution algorithms
│   ├── data/              # Data processing pipelines
│   └── utils/             # Shared utilities
├── notebooks/             # Jupyter notebooks for exploration
│   └── 00_environment_test.ipynb  # Environment validation
├── tests/                 # Test suite (empty)
├── data/                  # Local data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── samples/
├── configs/               # Configuration files (empty)
├── docs/                  # Documentation (empty)
├── .env                   # API keys (gitignored)
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
├── verify_install.py      # Dependency verification script
├── test_medgemma.py       # API connection test script
└── README.md              # Project documentation
```

### Git History
- **Initial commit (c5407a0):** Empty repo with .gitignore
- **Setup commit (c95329f):** Python package structure, requirements.txt
- **Integration commit (b3059c2):** Google AI integration, README, notebooks

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

## Next Development Steps

### Immediate Tasks (Not Started)
1. Create synthetic patient record dataset
2. Build entity extraction pipeline
3. Develop matching confidence scoring system
4. Test with edge cases (same name+DOB, different people)
5. Build evaluation metrics and benchmarks

### Medium Term
1. Refine matching algorithm logic
2. Multi-signal matching (combine demographics + medical history)
3. Move prototype code from notebooks to `src/medmatch/`
4. Write unit tests
5. Create standard benchmark dataset

### Future (Before Competition)
1. Deploy MedGemma locally
2. Optimize for Mac Metal/MPS performance
3. Add multi-modal support (imaging, lab data)
4. Build explainable AI features
5. Create clinical workflow integration

## Key Files to Reference

- **[README.md](README.md):** Complete project documentation
- **[test_medgemma.py](test_medgemma.py):** Working API test with entity matching demo
- **[notebooks/00_environment_test.ipynb](notebooks/00_environment_test.ipynb):** Interactive environment validation
- **[verify_install.py](verify_install.py):** Dependency verification
- **[requirements.txt](requirements.txt):** All Python dependencies
- **[.env.example](.env.example):** Environment variable template

## Commands to Run

### Activate Environment
```bash
source venv/bin/activate
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

---

**Last Updated:** 2026-01-16
**Session:** Initial setup walkthrough completed
**Next Session Should Focus On:** Building synthetic patient dataset or first matching algorithm prototype
