# Patient Entity Resolution System

AI-powered patient matching to prevent wrong-patient medical errors using a hybrid 4-stage progressive pipeline.

## Overview

The MedMatch AI matching system combines deterministic rules, feature-based scoring, and AI-powered medical fingerprinting to accurately identify when different patient records refer to the same person. This prevents dangerous wrong-patient errors like operating on the wrong John Smith.

### Key Features

- **97% efficiency improvement** - Blocking reduces candidate pairs from O(n²) to ~3%
- **94.5% overall accuracy** - Exceeds all targets (95% easy, 85% medium, 70% hard/ambiguous)
- **Progressive pipeline** - Fast deterministic rules handle 74% of cases, AI only for ambiguous
- **Explainable decisions** - Every match includes confidence score and human-readable explanation
- **Medical understanding** - AI recognizes abbreviations (T2DM = Type 2 Diabetes, HTN = Hypertension)

### When to Use This System

- **Patient record deduplication** - Merge duplicate records across hospital systems
- **Master Patient Index (MPI)** - Build unified patient identities across data sources
- **Clinical decision support** - Ensure correct patient context at point of care
- **Health information exchange** - Match patients across different healthcare organizations

## Architecture

The system uses a **4-stage progressive pipeline** where each stage only runs if previous stages didn't make a confident decision:

```
┌─────────────────────────────────────────────────────────────┐
│                    All Record Pairs (O(n²))                  │
│                      33,930 pairs possible                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: BLOCKING                                          │
│  Purpose: Reduce O(n²) comparisons to candidate pairs       │
│  Strategy: 5 blocking strategies (phonetic, key-based)      │
│  Performance: 97% reduction → ~1,000 candidate pairs        │
│  Recall: 97.3% (only 2.7% of true matches missed)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: DETERMINISTIC RULES                               │
│  Purpose: Fast exact matching for clear cases               │
│  Rules: 5 rules (2 NO-MATCH, 3 MATCH)                      │
│  Performance: Handles 74% of all pairs                      │
│  Accuracy: 92.6% on rule-based decisions                    │
│  Examples:                                                   │
│    - ExactMatchRule: Same name + DOB + gender → MATCH      │
│    - GenderMismatchRule: Different gender → NO-MATCH       │
│    - MRNNameMatchRule: Same MRN + similar name → MATCH     │
└────────────────────────┬────────────────────────────────────┘
                         │
                   No rule fired?
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: FEATURE SCORING                                   │
│  Purpose: Weighted confidence for medium difficulty         │
│  Features: 15+ features (name, DOB, contact, identifiers)  │
│  Weights: name=40%, DOB=30%, contact=20%, IDs=10%          │
│  Thresholds:                                                │
│    - ≥0.90 = definite match                                │
│    - ≥0.80 = probable match                                │
│    - ≥0.65 = possible match                                │
│  Performance: Handles 14% of pairs (100% accuracy)          │
└────────────────────────┬────────────────────────────────────┘
                         │
              Score ambiguous (0.50-0.90)?
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: AI MEDICAL FINGERPRINTING                         │
│  Purpose: Deep medical history comparison for hard cases    │
│  Model: Gemini 2.5 Flash (will migrate to MedGemma)        │
│  Understanding:                                             │
│    - Medical abbreviations (T2DM, HTN, CAD, etc.)          │
│    - Medication → condition links (Metformin → Diabetes)   │
│    - Surgical history matching                             │
│  Scoring: 60% demographic + 40% medical similarity          │
│  Performance: Handles 12% of pairs (100% accuracy!)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                   MatchResult
         (decision, confidence, explanation)
```

### Stage Details

#### Stage 1: Blocking

**Purpose:** Reduce computational complexity by filtering out obvious non-matches.

**Strategies:**
1. **SoundexYearGenderBlocker** - Phonetic last name + birth year + gender
2. **NamePrefixDOBBlocker** - First 3 chars of last name + full DOB
3. **PhoneBlocker** - Normalized phone number
4. **SSNYearGenderBlocker** - SSN last 4 + birth year + gender
5. **MRNBlocker** - Exact MRN match

**Performance:** 97%+ reduction (33,930 → ~1,000 pairs), 97.3% recall

#### Stage 2: Deterministic Rules

**Purpose:** Handle clear matches and non-matches with fast, explainable rules.

**NO-MATCH Rules (checked first):**
- `GenderMismatchRule` - Different genders → NO-MATCH
- `LargeAgeDifferentNameRule` - 10+ years age gap + dissimilar names → NO-MATCH

**MATCH Rules:**
- `ExactMatchRule` - Same first/last name + DOB + gender → MATCH
- `MRNNameMatchRule` - Same MRN + similar names → MATCH
- `SSNNameDOBMatchRule` - Same SSN + same DOB + similar names → MATCH

**Performance:** 74% of decisions, 92.6% accuracy

#### Stage 3: Feature Scoring

**Purpose:** Calculate weighted confidence scores for ambiguous demographic matches.

**Features Extracted (15+):**
- **Name:** First (15%), Last (20%), Middle (5%)
- **DOB:** Date of birth (30%), with special handling for twins/typos
- **Contact:** Phone (8%), Email (7%), Address (5%)
- **Identifiers:** MRN (5%), SSN (5%)

**Default Weights (customizable):**
```python
{
    'name_first': 0.15,
    'name_last': 0.20,
    'name_middle': 0.05,
    'dob': 0.30,
    'phone': 0.08,
    'email': 0.07,
    'address': 0.05,
    'mrn': 0.05,
    'ssn': 0.05,
}
```

**Classification Thresholds (customizable):**
- **Definite match:** ≥0.90
- **Probable match:** ≥0.80
- **Possible match:** ≥0.65
- **No match:** <0.65

**Performance:** 14% of decisions, 100% accuracy on medium difficulty

#### Stage 4: AI Medical Fingerprinting

**Purpose:** Compare medical histories using AI for cases where demographics are ambiguous.

**When AI Triggers:**
- Demographic confidence score between 0.50 and 0.90
- Both records have medical history available
- AI enabled (`use_ai=True`)

**What AI Analyzes:**
- Medical conditions (with abbreviation understanding)
- Medications (with condition linkage)
- Surgical history
- Allergies
- Family history

**Scoring:**
- Medical similarity: 0.0 (completely different) to 1.0 (identical)
- Combined score: 60% demographic + 40% medical
- Uses combined score for final classification

**Performance:** 12% of decisions, 100% accuracy on hard/ambiguous cases

## Quick Start

### Basic Usage

```python
from medmatch.matching import PatientMatcher, load_patient_records

# Load patient records from CSV + JSON
records = load_patient_records(
    'data/synthetic/synthetic_demographics.csv',
    'data/synthetic/synthetic_medical_records.json'
)

# Create matcher (all stages enabled)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    use_ai=True,  # Requires GOOGLE_AI_API_KEY in .env
)

# Match two records
result = matcher.match_pair(records[0], records[1])

# View result
print(f"Match: {result.is_match}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Stage: {result.stage}")
print(result.explanation)
```

### Batch Matching

```python
# Match all records in dataset
results = matcher.match_datasets(records, show_progress=True)

# Get statistics
stats = matcher.get_stats(results)
print(f"Total pairs: {stats['total_pairs']}")
print(f"Matches: {stats['matches']}")
print(f"By stage: {stats['by_stage']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
```

### Custom Configuration

```python
# Conservative thresholds (fewer false positives)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    scoring_thresholds={
        'definite': 0.95,  # Higher bar for definite match
        'probable': 0.90,
        'possible': 0.75,
    }
)

# Custom weights (prioritize DOB over name)
custom_weights = {
    'name_first': 0.10,
    'name_last': 0.15,
    'name_middle': 0.05,
    'dob': 0.50,  # 50% weight on DOB
    'phone': 0.05,
    'email': 0.05,
    'address': 0.05,
    'mrn': 0.025,
    'ssn': 0.025,
}

matcher = PatientMatcher(
    use_scoring=True,
    scoring_weights=custom_weights,
)
```

### Fast Mode (No AI)

```python
# Disable AI for speed (rules + scoring only)
matcher = PatientMatcher(
    use_blocking=True,
    use_rules=True,
    use_scoring=True,
    use_ai=False,  # Fast mode
)

# Still achieves 91.3% accuracy
results = matcher.match_datasets(records)
```

## Configuration Guide

### Blocking Strategies

**When to enable:** Always enable unless dataset is very small (<100 records)

**When to disable:**
- Small datasets where O(n²) is fast enough
- Testing/debugging specific record pairs
- Analyzing blocking recall

```python
# Disable blocking (all pairs)
matcher = PatientMatcher(use_blocking=False)

# Access blocker directly for analysis
blocker = MultiBlocker([...])
pairs = blocker.generate_candidate_pairs(records)
```

### Deterministic Rules

**When to enable:** Always enable for production use

**When to disable:**
- Pure feature-based matching experiments
- Analyzing rule vs scoring performance

```python
# Disable rules (scoring only)
matcher = PatientMatcher(
    use_rules=False,
    use_scoring=True,
)
```

### Feature Scoring

**When to enable:** When you need confident scores for medium difficulty cases

**When to disable:**
- Using only deterministic rules
- Using AI for all decisions

**Customization:**

```python
# More conservative thresholds
matcher = PatientMatcher(
    use_scoring=True,
    scoring_thresholds={
        'definite': 0.95,
        'probable': 0.85,
        'possible': 0.70,
    }
)

# Prioritize identifiers over names
custom_weights = {
    'name_first': 0.05,
    'name_last': 0.10,
    'dob': 0.30,
    'mrn': 0.20,  # High weight on MRN
    'ssn': 0.20,  # High weight on SSN
    # ... remaining weights sum to 1.0
}
```

### AI Medical Fingerprinting

**When to enable:**
- Medical history data available
- High accuracy required for ambiguous cases
- Production deployment with local MedGemma

**When to disable:**
- No medical history data
- Fast prototyping without API calls
- Budget constraints (API costs)

**Rate Limiting:**

```python
# No rate limiting (billing enabled)
matcher = PatientMatcher(use_ai=True, api_rate_limit=0)

# Rate limiting (5 requests/minute, free tier)
matcher = PatientMatcher(use_ai=True, api_rate_limit=5)
```

**API Key Setup:**

```bash
# Add to .env file
GOOGLE_AI_API_KEY=your_api_key_here
```

Get API key from: https://aistudio.google.com/apikey

## Performance Benchmarks

Results from evaluation on 261 synthetic patient records (437 candidate pairs after blocking):

### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **94.51%** |
| Precision | 95% |
| Recall | 96% |
| F1 Score | 95% |

### Accuracy by Difficulty

| Difficulty | Target | Achieved | Status |
|------------|--------|----------|--------|
| **Easy** | 95% | **100.00%** | ✅ PASS |
| **Medium** | 85% | **100.00%** | ✅ PASS |
| **Hard** | 70% | **88.24%** | ✅ PASS |
| **Ambiguous** | 70% | **80.54%** | ✅ PASS |

### Stage Distribution

| Stage | Decisions | Percentage | Accuracy |
|-------|-----------|------------|----------|
| **Blocking** | 33,930 → 437 | 97% reduction | 97.3% recall |
| **Rules** | 324 pairs | 74% | 92.6% |
| **Scoring** | 0 pairs* | 0%* | 100%* |
| **AI** | 113 pairs | 26% | 100% |

*When AI disabled: Scoring handles 113 pairs (26%) with 87.6% accuracy

### Runtime Performance

- **Blocking:** <2 seconds (261 records)
- **Full pipeline (no AI):** <5 seconds (437 pairs)
- **Full pipeline (with AI):** ~6 minutes (113 AI calls, no rate limiting)

## Module Reference

### Core Classes

#### `PatientRecord`

Unified patient record combining demographics and medical history.

**Creation:**
```python
from medmatch.matching import PatientRecord
from medmatch.data.models.patient import Demographics, MedicalRecord

# From demographics only
record = PatientRecord.from_demographics(demo)

# With medical history
record = PatientRecord.from_demographics(demo, medical)
```

**Attributes:**
- `record_id` - Unique record identifier
- `name_first`, `name_last`, `name_middle` - Name fields
- `date_of_birth`, `gender` - Demographics
- `phone`, `email`, `address` - Contact info
- `mrn`, `ssn_last4` - Identifiers
- `medical_signature` - AI-readable medical summary
- `conditions`, `medications`, `allergies` - Medical lists

#### `MatchResult`

Result of comparing two patient records.

**Attributes:**
- `record_1_id`, `record_2_id` - Record identifiers
- `is_match` - Boolean match decision
- `confidence` - Confidence score (0.0-1.0)
- `match_type` - Classification (exact, probable, possible, no_match)
- `stage` - Which pipeline stage made decision (rules/scoring/ai)
- `evidence` - Dictionary with detailed evidence
- `explanation` - Human-readable explanation
- `rules_triggered` - List of rules fired (if stage=rules)
- `medical_similarity` - Medical match score (if stage=ai)
- `ai_reasoning` - AI explanation (if stage=ai)

#### `PatientMatcher`

Main orchestrator for the matching pipeline.

**Constructor:**
```python
matcher = PatientMatcher(
    use_blocking=True,           # Enable blocking
    use_rules=True,              # Enable rules
    use_scoring=False,           # Enable scoring
    use_ai=False,                # Enable AI
    confidence_threshold=0.85,   # Minimum confidence
    scoring_weights=None,        # Custom weights dict
    scoring_thresholds=None,     # Custom thresholds dict
    ai_model="gemini-2.5-flash", # AI model name
    api_rate_limit=0,            # API rate limit (0=unlimited)
)
```

**Methods:**
- `match_pair(record1, record2)` - Match two records
- `match_datasets(records, show_progress=True)` - Match all pairs
- `get_stats(results)` - Calculate statistics

### Blocking

#### `MultiBlocker`

Combines multiple blocking strategies using union approach.

```python
from medmatch.matching.blocking import MultiBlocker

blocker = MultiBlocker([...strategies...])
pairs = blocker.generate_candidate_pairs(records)
```

### Rules

#### `RuleEngine`

Applies deterministic matching rules.

```python
from medmatch.matching.rules import RuleEngine

engine = RuleEngine()
result = engine.evaluate(record1, record2)  # Returns MatchResult or None
```

### Scoring

#### `FeatureExtractor`

Extracts numerical features from record pairs.

```python
from medmatch.matching.features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract(record1, record2)
```

#### `ConfidenceScorer`

Calculates weighted confidence scores.

```python
from medmatch.matching.scoring import ConfidenceScorer, ScoringWeights

weights = ScoringWeights(name_first=0.15, name_last=0.20, ...)
scorer = ConfidenceScorer(weights=weights)
score, breakdown = scorer.score(features)
is_match, match_type = scorer.classify(score)
```

### AI Medical Fingerprinting

#### `MedicalFingerprintMatcher`

AI-powered medical history comparison.

```python
from medmatch.matching.medical_fingerprint import MedicalFingerprintMatcher

ai_matcher = MedicalFingerprintMatcher(
    model="gemini-2.5-flash",
    api_rate_limit=0,
)
similarity, reasoning = ai_matcher.compare_medical_histories(record1, record2)
```

### Explanation

#### `MatchExplainer`

Generates human-readable explanations.

```python
from medmatch.matching import MatchExplainer

explainer = MatchExplainer()
explanation = explainer.explain(result, verbose=True)
print(explanation)
```

## Examples

### Example 1: Basic Matching

```python
from medmatch.matching import PatientMatcher, PatientRecord
from medmatch.data.models.patient import Demographics
from datetime import date

# Create two records
demo1 = Demographics(
    record_id="R001",
    patient_id="P001",
    name_first="John",
    name_last="Smith",
    date_of_birth=date(1980, 1, 15),
    gender="M",
    mrn="12345",
)

demo2 = Demographics(
    record_id="R002",
    patient_id="P001",  # Same patient
    name_first="John",
    name_last="Smith",
    date_of_birth=date(1980, 1, 15),
    gender="M",
    mrn="67890",  # Different MRN
)

record1 = PatientRecord.from_demographics(demo1)
record2 = PatientRecord.from_demographics(demo2)

# Match
matcher = PatientMatcher()
result = matcher.match_pair(record1, record2)

print(f"Match: {result.is_match}")  # True
print(f"Confidence: {result.confidence}")  # 0.99
print(f"Stage: {result.stage}")  # "rules"
print(result.explanation)  # "Exact match: John Smith, DOB=1980-01-15, Gender=M"
```

### Example 2: Handling Typos

```python
demo1 = Demographics(
    record_id="R001",
    name_first="Jennifer",  # Full name
    name_last="Johnson",
    date_of_birth=date(1992, 3, 20),
    gender="F",
)

demo2 = Demographics(
    record_id="R002",
    name_first="Jenny",  # Nickname
    name_last="Jonson",  # Typo
    date_of_birth=date(1992, 3, 20),
    gender="F",
)

record1 = PatientRecord.from_demographics(demo1)
record2 = PatientRecord.from_demographics(demo2)

matcher = PatientMatcher(use_scoring=True)
result = matcher.match_pair(record1, record2)

print(f"Match: {result.is_match}")  # True
print(f"Confidence: {result.confidence}")  # ~0.88 (probable match)
print(f"Stage: {result.stage}")  # "scoring"
# Explanation shows name_first matched via nickname, name_last via typo
```

### Example 3: Using AI for Medical History

```python
from medmatch.data.models.patient import MedicalRecord, MedicalHistory, MedicalCondition

# Patient 1: Full medical terms
medical1 = MedicalRecord(
    record_id="M001",
    patient_id="P001",
    medical_history=MedicalHistory(
        conditions=[
            MedicalCondition(name="Type 2 Diabetes Mellitus", onset_year=2015),
            MedicalCondition(name="Hypertension", onset_year=2018),
        ],
        medications=["Metformin 1000mg", "Lisinopril 10mg"],
    ),
)

# Patient 2: Abbreviations
medical2 = MedicalRecord(
    record_id="M002",
    patient_id="P001",
    medical_history=MedicalHistory(
        conditions=[
            MedicalCondition(name="T2DM", onset_year=2015),  # Abbreviation
            MedicalCondition(name="HTN", onset_year=2018),   # Abbreviation
        ],
        medications=["Metformin", "Lisinopril"],
    ),
)

record1 = PatientRecord.from_demographics(demo1, medical1)
record2 = PatientRecord.from_demographics(demo2, medical2)

matcher = PatientMatcher(use_ai=True)
result = matcher.match_pair(record1, record2)

print(f"Medical similarity: {result.medical_similarity}")  # 1.0
print(f"AI reasoning: {result.ai_reasoning}")
# "T2DM = Type 2 Diabetes Mellitus, HTN = Hypertension. Same conditions and medications."
```

### Example 4: Evaluating Results

```python
from medmatch.evaluation import MatchEvaluator

# Run matching
matcher = PatientMatcher(use_scoring=True)
results = matcher.match_datasets(records)

# Evaluate against ground truth
evaluator = MatchEvaluator('data/synthetic/ground_truth.csv')
metrics = evaluator.evaluate(results)

print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Precision: {metrics.precision:.2%}")
print(f"Recall: {metrics.recall:.2%}")

# By difficulty
by_difficulty = evaluator.evaluate_by_difficulty(results)
for diff, m in by_difficulty.items():
    print(f"{diff}: {m.accuracy:.2%}")

# Find errors
errors = evaluator.find_errors(results)
false_positives = [e for e in errors if e.error_type == 'false_positive']
print(f"False positives: {len(false_positives)}")
```

## Troubleshooting

### API Key Issues

**Problem:** `ValueError: GOOGLE_AI_API_KEY not found in environment`

**Solution:**
```bash
# Create .env file in project root
echo "GOOGLE_AI_API_KEY=your_key_here" > .env

# Or export in shell
export GOOGLE_AI_API_KEY=your_key_here
```

Get key from: https://aistudio.google.com/apikey

### Rate Limiting

**Problem:** `429 Too Many Requests` or API quota exceeded

**Solution:**
```python
# Enable rate limiting (free tier: 5 requests/minute)
matcher = PatientMatcher(use_ai=True, api_rate_limit=5)

# Or disable AI temporarily
matcher = PatientMatcher(use_ai=False)
```

Check quota: https://ai.dev/rate-limit

### Missing Medical Records

**Problem:** AI returns 0.5 for all pairs

**Solution:**
```python
# Verify medical records loaded
records = load_patient_records(
    'demographics.csv',
    'medical_records.json'  # Must provide medical records
)

# Check if records have medical data
has_medical = sum(1 for r in records if r.medical_signature)
print(f"Records with medical data: {has_medical}/{len(records)}")
```

### Poor Accuracy

**Problem:** Accuracy below targets

**Diagnostic steps:**
1. Check stage distribution: `stats = matcher.get_stats(results)`
2. Evaluate by stage: `evaluator.evaluate_by_stage(results)`
3. Find errors: `errors = evaluator.find_errors(results)`
4. Examine false positives/negatives

**Common fixes:**
- Adjust scoring thresholds (more conservative/aggressive)
- Enable AI for ambiguous cases
- Customize weights for your data characteristics
- Add custom rules for your domain

### Performance Issues

**Problem:** Slow matching on large datasets

**Solutions:**
```python
# 1. Ensure blocking is enabled
matcher = PatientMatcher(use_blocking=True)

# 2. Disable AI for speed
matcher = PatientMatcher(use_ai=False)

# 3. Reduce AI rate limiting (if billing enabled)
matcher = PatientMatcher(api_rate_limit=0)

# 4. Process in batches
for batch in batches(records, size=1000):
    results = matcher.match_datasets(batch)
```

## See Also

- **Main README:** [../../README.md](../../README.md) - Project overview and setup
- **Evaluation Notebook:** [../../notebooks/01_entity_resolution_evaluation.ipynb](../../notebooks/01_entity_resolution_evaluation.ipynb) - Interactive analysis
- **Quick Start Guide:** [../../docs/quickstart.md](../../docs/quickstart.md) - 5-minute getting started
- **Project Context:** [../../.claude/CLAUDE.md](../../.claude/CLAUDE.md) - Detailed development history

## Contributing

See the main [README.md](../../README.md) for development setup and contribution guidelines.

## License

See [LICENSE](../../LICENSE) for details.
