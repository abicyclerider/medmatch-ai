# MedMatch AI Scripts

Command-line tools for batch patient matching and entity resolution.

## run_matcher.py

Production-ready CLI for running the entity resolution pipeline on datasets.

### Quick Start

```bash
# Basic usage with all stages
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results.json \
  --progress
```

### Command-Line Options

**Input Files:**

- `--demographics PATH` - Demographics CSV file (required)
- `--medical-records PATH` - Medical records JSON file (optional)

**Output Options:**

- `--output PATH` - Output file path (default: stdout)
- `--format {json,csv,verbose}` - Output format (default: json)

**Pipeline Configuration:**

- `--no-blocking` - Disable blocking (all pairs, slow)
- `--no-rules` - Disable deterministic rules
- `--no-scoring` - Disable feature scoring
- `--use-ai` - Enable AI medical fingerprinting

**AI Configuration:**

- `--ai-model MODEL` - AI model name (default: gemini-2.5-flash)
- `--api-rate-limit N` - API rate limit in requests/minute (default: 0=unlimited)

**Custom Configuration:**

- `--scoring-weights PATH` - Custom scoring weights JSON file
- `--scoring-thresholds PATH` - Custom scoring thresholds JSON file

**Display Options:**

- `--progress` - Show progress bar during matching
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: INFO)

### Examples

**Fast Mode (No AI):**

```bash
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --output results.csv \
  --format csv \
  --progress
```

**With AI Medical Fingerprinting:**

```bash
# Requires GOOGLE_AI_API_KEY in .env
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --medical-records data/synthetic/synthetic_medical_records.json \
  --output results.json \
  --use-ai \
  --api-rate-limit 0 \
  --progress
```

**Custom Configuration:**

```bash
# Create custom weights file: config/custom_weights.json
{
  "name_first": 0.10,
  "name_last": 0.15,
  "dob": 0.50,
  "phone": 0.10,
  "mrn": 0.15
}

# Create custom thresholds file: config/custom_thresholds.json
{
  "definite": 0.95,
  "probable": 0.85,
  "possible": 0.70
}

# Run with custom configuration
python scripts/run_matcher.py \
  --demographics data/demographics.csv \
  --scoring-weights config/custom_weights.json \
  --scoring-thresholds config/custom_thresholds.json \
  --output results.json
```

**Verbose Output to Stdout:**

```bash
python scripts/run_matcher.py \
  --demographics data/synthetic/synthetic_demographics.csv \
  --format verbose \
  --progress | less
```

### Output Formats

**JSON (default):**

```json
{
  "metadata": {
    "total_pairs": 437,
    "timestamp": "2026-01-18T10:30:00",
    "config": {
      "blocking": true,
      "rules": true,
      "scoring": true,
      "ai": false
    }
  },
  "results": [
    {
      "record1_id": "R0001",
      "record2_id": "R0002",
      "is_match": true,
      "confidence": 0.95,
      "match_type": "definite",
      "stage": "rules",
      "explanation": "Exact match: John Smith, DOB=1980-01-15, Gender=M"
    }
  ],
  "summary": {
    "matches": 388,
    "non_matches": 49,
    "by_stage": {"rules": 324, "scoring": 113},
    "by_match_type": {"exact": 318, "probable": 55, "no_match": 64},
    "avg_confidence": 0.908
  }
}
```

**CSV:**

```csv
record1_id,record2_id,is_match,confidence,match_type,stage
R0001,R0002,True,0.95,definite,rules
R0003,R0004,False,0.35,no_match,scoring
...
```

**Verbose:**

```
======================================================================
ENTITY RESOLUTION RESULTS
======================================================================

Total Pairs: 437
Matches: 388
Non-matches: 49
Average Confidence: 0.908

By Stage: {'rules': 324, 'scoring': 113}
By Match Type: {'exact': 318, 'probable': 55, 'no_match': 64}

======================================================================
INDIVIDUAL RESULTS
======================================================================

[1] EXACT MATCH (confidence: 0.99)

Records: R0001 ↔ R0002

Decision Stage: Deterministic Rules

Rules Applied: ExactMatchRule

Evidence:
  - rule_fired: ExactMatchRule
  - rule_explanation: Exact match: John Smith, DOB=1980-01-15, Gender=M

Recommendation: These records refer to the same patient. Safe to merge.
----------------------------------------------------------------------
...
```

### Error Handling

The script includes comprehensive error handling:

- **Missing API Key:** Clear error message if AI enabled without `GOOGLE_AI_API_KEY`
- **Invalid File Paths:** File not found errors with helpful messages
- **API Failures:** Graceful degradation with logging
- **Invalid Data:** Validation errors with row/column information

### Performance Notes

- **Blocking:** Reduces O(n²) to ~3% of pairs (97% reduction)
- **Without AI:** Processes 437 pairs in <5 seconds
- **With AI:** Time depends on API rate limit (0=unlimited is fastest)
- **Memory:** Efficient for datasets up to 10,000 records

### Troubleshooting

**Problem:** `ValueError: GOOGLE_AI_API_KEY not found in environment`

**Solution:**

```bash
# Add to .env file
echo "GOOGLE_AI_API_KEY=your_key_here" > .env

# Or export in shell
export GOOGLE_AI_API_KEY=your_key_here
```

**Problem:** Slow matching on large datasets

**Solution:**

```bash
# Ensure blocking is enabled (default)
python scripts/run_matcher.py --demographics data.csv --output results.json

# Disable AI for speed
python scripts/run_matcher.py --demographics data.csv --output results.json  # AI is off by default
```

**Problem:** Rate limit errors (429)

**Solution:**

```bash
# Add rate limiting (5 requests/minute for free tier)
python scripts/run_matcher.py \
  --demographics data.csv \
  --use-ai \
  --api-rate-limit 5
```

## benchmark_entity_resolution.py

Efficiently benchmarks AI performance on the actual entity resolution problem.

### How It Works

Instead of re-running the full pipeline repeatedly, this script:

1. **Runs deterministic stages once** (blocking → rules → scoring)
2. **Identifies ambiguous pairs** (demographic score 0.50-0.90) that need AI
3. **Benchmarks only the AI layer** with different configurations

This is much faster than running the full pipeline for each configuration.

### Quick Start

```bash
# Compare standard vs batched mode
python scripts/benchmark_entity_resolution.py --mode compare --save

# Batched mode only (faster)
python scripts/benchmark_entity_resolution.py --mode batched --batch-size 3

# Standard mode only (baseline)
python scripts/benchmark_entity_resolution.py --mode standard

# Use full (non-quantized) model
python scripts/benchmark_entity_resolution.py --model medgemma:1.5-4b
```

### Command-Line Options

- `--mode {standard,batched,compare}` - Benchmark mode (default: compare)
- `--model MODEL` - Ollama model name (default: medgemma:1.5-4b-q4)
- `--batch-size N` - Batch size for batched mode (default: 3)
- `--data-dir PATH` - Path to synthetic data directory
- `--save` - Save results to JSON file

### Output

```
============================================================
ENTITY RESOLUTION AI BENCHMARK
============================================================

Pipeline breakdown:
  Rules decided: 324 pairs
  Scoring decided (clear): 0 pairs
  AI needed (ambiguous): 113 pairs

Ambiguous pairs by difficulty:
  ambiguous: 54
  hard: 59

============================================================
Mode: batched (size=3) | Model: medgemma:1.5-4b-q4
============================================================
  Pairs evaluated: 113
  Total time: 285.3s
  Avg per pair: 2.52s
  Throughput: 23.8/min

  Accuracy: 85.0% (96/113)
  TP: 45, TN: 51, FP: 8, FN: 9

  By difficulty:
    hard: 88.1%
    ambiguous: 81.5%
```

### Why This Approach?

- **Efficient:** Only benchmarks the AI layer, not the whole pipeline
- **Realistic:** Uses actual entity resolution pairs, not synthetic test cases
- **Actionable:** Shows accuracy impact of speed optimizations

## benchmark_medgemma_quantization.py

Low-level inference speed benchmark for MedGemma models.

### Quick Start

```bash
# Standard mode (baseline)
python scripts/benchmark_medgemma_quantization.py --mode standard --iterations 5

# Batched mode (recommended)
python scripts/benchmark_medgemma_quantization.py --mode batched --iterations 9 --batch-size 3

# Parallel mode
python scripts/benchmark_medgemma_quantization.py --mode parallel --iterations 6 --workers 2

# Compare all modes
python scripts/benchmark_medgemma_quantization.py --mode compare --iterations 5
```

### Command-Line Options

- `--mode {standard,optimized,parallel,batched,compare}` - Benchmark mode
- `--iterations N` - Number of test iterations (default: 10)
- `--full-model MODEL` - Full model name (default: medgemma:1.5-4b)
- `--quantized-model MODEL` - Quantized model name (default: medgemma:1.5-4b-q4)
- `--workers N` - Number of parallel workers (default: 2)
- `--batch-size N` - Batch size for batched mode (default: 3)
- `--no-save` - Don't save results to file

### Test Cases

The benchmark uses 5 medical history comparison test cases:

1. **Matching (abbreviations):** T2DM/HTN vs Type 2 Diabetes/Hypertension
2. **Different conditions:** CAD vs COPD
3. **Partial match:** Shared HTN only
4. **Medication variations:** Tylenol vs acetaminophen
5. **Complex history:** ESRD/DM2/CHF with abbreviations

## See Also

- [Main README](../README.md) - Project overview and setup
- [Matching Module README](../src/medmatch/matching/README.md) - Entity resolution documentation
- [Evaluation Notebook](../notebooks/01_entity_resolution_evaluation.ipynb) - Interactive analysis
- [Ollama Setup Guide](../docs/ollama_setup.md) - Local MedGemma deployment
