# MedMatch AI

An AI-powered medical entity resolution system that prevents wrong-patient errors in healthcare using Google's Gemini/MedGemma models.

## Problem Statement

Wrong-patient medical errors (e.g., operating on the wrong John Smith) are "never events" that cause deaths. Current systems use basic name matching. MedMatch AI uses deep medical understanding to create "medical fingerprints" ensuring accurate patient matching at critical decision points.

## Approach

Instead of simple demographic matching, we use AI to understand:
- Medical histories and patterns
- Treatment timelines
- Imaging characteristics
- Clinical context and terminology
- Variations in names, dates, and abbreviations

## Project Status

**Current Phase:** Initial setup complete with Gemini API
- ✓ Development environment configured
- ✓ Google AI API integrated
- ✓ Basic entity extraction working
- ✓ Proof-of-concept matching demonstrated

**Next Phase:** Build production matching algorithms, then migrate to MedGemma

## Tech Stack

- **Python 3.12.4** with venv
- **PyTorch 2.9.1** (Mac Metal/MPS acceleration)
- **Google Gemini API** (prototyping, will migrate to MedGemma)
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

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Google AI API key from https://aistudio.google.com/apikey
```

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

### Phase 1: Core Algorithm (Current)
- [ ] Build patient record data structures
- [ ] Implement entity extraction pipeline
- [ ] Develop matching confidence scoring
- [ ] Create test datasets
- [ ] Build evaluation metrics

### Phase 2: MedGemma Integration
- [ ] Deploy MedGemma locally via Hugging Face
- [ ] Migrate from Gemini to MedGemma
- [ ] Benchmark accuracy improvements
- [ ] Optimize for Mac Metal performance

### Phase 3: Production Features
- [ ] Multi-modal support (imaging, lab data)
- [ ] Explainable AI for match decisions
- [ ] Clinical workflow integration
- [ ] Privacy and HIPAA compliance

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
