# MedGemma Quantization Benchmark

This script benchmarks the performance difference between the full MedGemma model and the quantized (Q4) version.

## Prerequisites

1. **Ollama installed and running:**
   ```bash
   brew services start ollama
   ```

2. **Both models available:**
   ```bash
   ollama list
   # Should show:
   # medgemma:1.5-4b       (full model, 8.6 GB)
   # medgemma:1.5-4b-q4    (quantized, 2.5 GB)
   ```

3. **Python environment activated:**
   ```bash
   source venv/bin/activate
   ```

## Running the Benchmark

### Quick Test (5 iterations each):
```bash
python3 scripts/benchmark_medgemma_quantization.py --iterations 5
```

### Full Test (10 iterations each):
```bash
python3 scripts/benchmark_medgemma_quantization.py --iterations 10
```

### Custom Models:
```bash
python3 scripts/benchmark_medgemma_quantization.py \
    --full-model medgemma:1.5-4b \
    --quantized-model medgemma:1.5-4b-q4 \
    --iterations 5
```

## What It Tests

### Speed Test
- Warmup run (measures model loading time)
- N iterations of medical history comparisons
- Reports: average, min, max inference time

### Accuracy Test (5 test cases)
1. **Medical Abbreviations**: HTN/T2DM vs full names
2. **Different Conditions**: CAD vs COPD (should score 0.0)
3. **Partial Match**: Shared HTN only (should score ~0.4)
4. **Medication Synonyms**: Tylenol vs acetaminophen
5. **Complex Case**: ESRD/DM2/CHF with abbreviations

Each test checks if the score is within tolerance of expected value.

## Expected Results

### Speed
- **Full Model**: ~50-60 seconds per comparison
- **Quantized Model**: ~15-25 seconds per comparison
- **Speedup**: 2-4x faster

### Accuracy
- **Full Model**: 4-5/5 tests passing (80-100%)
- **Quantized Model**: 4-5/5 tests passing (80-100%)
- **Difference**: Minimal (<5% accuracy loss)

### Memory
- **Full Model**: 8.6 GB on disk, ~10 GB loaded
- **Quantized Model**: 2.5 GB on disk, ~3 GB loaded
- **Reduction**: 70% smaller

## Output Files

Results are saved to:
- `data/medgemma_benchmark.json` - Full benchmark results in JSON format
- `data/benchmark_output.txt` - Console output (if using `tee`)

## Estimated Runtime

- **5 iterations**: ~8-10 minutes total
  - Full model: ~5 minutes (warmup + 5 speed + 5 accuracy)
  - Quantized model: ~3 minutes (warmup + 5 speed + 5 accuracy)

- **10 iterations**: ~15-18 minutes total
  - Full model: ~10 minutes
  - Quantized model: ~5 minutes

## Troubleshooting

### Timeout Errors
If you get timeout errors, the default 180s timeout might be too short. Edit the script and increase:
```python
def __init__(self, model="medgemma:1.5-4b-q4", base_url="http://localhost:11434", timeout=300):
```

### Ollama Not Running
```bash
brew services start ollama
ollama ps  # Should show running models
```

### Model Not Found
```bash
ollama list  # Check available models
# If missing, download:
huggingface-cli download mradermacher/medgemma-1.5-4b-it-GGUF medgemma-1.5-4b-it.Q4_K_M.gguf --local-dir ~/Downloads/medgemma-q4
ollama create medgemma:1.5-4b-q4 -f Modelfile
```

## Example Output

```
======================================================================
MedGemma Quantization Benchmark
======================================================================
Full Model:      medgemma:1.5-4b
Quantized Model: medgemma:1.5-4b-q4
Speed Test:      5 iterations
Accuracy Test:   5 test cases
======================================================================

FULL MODEL BENCHMARKS
Speed test: 100%|██████████| 5/5 [04:14<00:00, 50.8s/comparison]
Accuracy test: 100%|██████████| 5/5 [04:02<00:00, 48.5s/test]

QUANTIZED MODEL BENCHMARKS
Speed test: 100%|██████████| 5/5 [01:35<00:00, 19.0s/comparison]
Accuracy test: 100%|██████████| 5/5 [01:28<00:00, 17.6s/test]

COMPARISON SUMMARY
  Speedup:         2.67x faster
  Accuracy:        Same (4/5 passed both)
  Memory:          70% smaller
```
