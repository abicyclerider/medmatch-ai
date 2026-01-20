# Ollama Setup Guide for MedGemma

## Overview

This guide shows how to run MedGemma 1.5 locally using Ollama as the inference server.

**Architecture:**
```
medmatch-ai (your app) → HTTP API → Ollama (server) → MedGemma model
```

## Status

✅ Ollama installed and running
⏳ MedGemma download and import (next steps)

## Why This Approach?

MedGemma is a **gated model** on HuggingFace - it requires approval and can't be redistributed on Ollama's public registry. We need to:

1. Download the model from HuggingFace (you already have access)
2. Import it into Ollama using a Modelfile
3. Serve it locally via Ollama's API

## Prerequisites

- ✅ Ollama installed: `brew install ollama` (DONE)
- ✅ Ollama service running: `brew services start ollama` (DONE)
- ⏳ HuggingFace account with MedGemma access approved
- ⏳ HuggingFace CLI token

## Step 1: Login to HuggingFace

```bash
# Activate your virtual environment
source venv/bin/activate

# Login to HuggingFace (you'll be prompted for your token)
huggingface-cli login
```

**Get your token:** https://huggingface.co/settings/tokens

When prompted:
- Paste your access token (Read permission is sufficient)
- Press Enter
- Choose "Y" to add token to git credentials (optional)

**Verify login:**
```bash
huggingface-cli whoami
```

## Step 2: Download MedGemma from HuggingFace

```bash
# Create directory for models
mkdir -p ~/.ollama/models/huggingface

# Download MedGemma 1.5 4B model (~8GB)
huggingface-cli download google/medgemma-1.5-4b-it \
    --local-dir ~/.ollama/models/huggingface/medgemma-1.5-4b-it \
    --local-dir-use-symlinks False
```

**This will download:**
- Model weights (~8GB)
- Tokenizer files
- Configuration files

**Expected time:** 5-15 minutes depending on connection

## Step 3: Create Ollama Modelfile

Create a Modelfile to import the model into Ollama:

```bash
cat > /tmp/medgemma-modelfile <<'EOF'
FROM ~/.ollama/models/huggingface/medgemma-1.5-4b-it

TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM """You are MedGemma, a medical AI assistant fine-tuned on clinical data. You provide accurate, evidence-based medical information and analysis."""
EOF
```

**Template notes:**
- Uses Gemma's chat format with `<start_of_turn>` tokens
- Supports system prompts for medical context
- Sets reasonable defaults for medical text generation

## Step 4: Import Model into Ollama

```bash
# Create the model in Ollama
ollama create medgemma:1.5-4b -f /tmp/medgemma-modelfile
```

This will:
- Import the HuggingFace model
- Apply the template and parameters
- Make it available via `ollama run medgemma:1.5-4b`

**Expected time:** 2-5 minutes

## Step 5: Test the Model

```bash
# Quick test
ollama run medgemma:1.5-4b "What is the difference between Type 1 and Type 2 Diabetes?"
```

**Expected behavior:**
- Model loads into memory (~4-8GB RAM usage)
- Generates medical response
- First run slower (loading), subsequent runs fast

**Example output:**
```
Type 1 Diabetes is an autoimmune condition where the pancreas produces little or no insulin...
Type 2 Diabetes is characterized by insulin resistance and relative insulin deficiency...
```

## Step 6: Test via API (Python)

```python
import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'medgemma:1.5-4b',
        'prompt': 'What does HTN stand for in medical terminology?',
        'stream': False
    }
)

print(response.json()['response'])
# Expected: "HTN stands for Hypertension..."
```

## Verification Checklist

- [ ] HuggingFace login successful (`huggingface-cli whoami`)
- [ ] MedGemma downloaded (~8GB in `~/.ollama/models/huggingface/medgemma-1.5-4b-it/`)
- [ ] Ollama model created (`ollama list` shows `medgemma:1.5-4b`)
- [ ] Model runs in CLI (`ollama run medgemma:1.5-4b "test"`)
- [ ] API responds (`curl http://localhost:11434/api/tags`)

## Troubleshooting

### Error: "401 Unauthorized" when downloading

**Cause:** Not logged in to HuggingFace or access not approved

**Solution:**
1. Check login: `huggingface-cli whoami`
2. Verify access at https://huggingface.co/google/medgemma-1.5-4b-it
3. Re-login: `huggingface-cli login`

### Error: "Out of memory" when running model

**Cause:** Model doesn't fit in available RAM

**Solution:**
1. Close other applications
2. Use quantized version (4-bit):
   ```bash
   # Add to Modelfile:
   PARAMETER quantization Q4_K_M
   ```
3. Your Mac M3 Pro with 18GB should handle the 4B model fine

### Error: "Model not found" in Ollama

**Cause:** Import step failed or model name incorrect

**Solution:**
1. List models: `ollama list`
2. Re-run import: `ollama create medgemma:1.5-4b -f /tmp/medgemma-modelfile`
3. Check logs: `tail -f ~/Library/Logs/Ollama/server.log`

### Model loads slowly

**Normal behavior:**
- First load: 10-30 seconds (loading weights into memory)
- Subsequent loads: 1-5 seconds (cached)
- Inference: 0.5-2 seconds per response

**Tip:** Keep model loaded by running `ollama run medgemma:1.5-4b` in background

## Model Management

```bash
# List installed models
ollama list

# Show model details
ollama show medgemma:1.5-4b

# Remove model (frees disk space)
ollama rm medgemma:1.5-4b

# Stop running model (frees memory)
ollama stop medgemma:1.5-4b

# View running models
ollama ps
```

## API Endpoints

Ollama provides OpenAI-compatible API:

- **Generate:** `POST http://localhost:11434/api/generate`
- **Chat:** `POST http://localhost:11434/api/chat`
- **List Models:** `GET http://localhost:11434/api/tags`
- **Model Info:** `POST http://localhost:11434/api/show`

## Next Steps

Once MedGemma is running in Ollama:

1. **Implement OllamaClient** (Task 3 in plan)
   - Add to `src/medmatch/matching/ai_client.py`
   - HTTP client calling `http://localhost:11434/api/generate`

2. **Update PatientMatcher**
   - Use `ai_backend="ollama"` instead of `"gemini"`

3. **Run benchmarks**
   - Compare Gemini API vs local MedGemma
   - Measure latency, accuracy, cost

## Performance Tuning

Ollama supports several environment variables that can significantly improve inference speed:

### Environment Variables

Add these to your shell profile (`~/.zshrc` or `~/.bashrc`) or set before starting Ollama:

```bash
# KV Cache Quantization - halves memory for key/value cache (20-30% speedup)
export OLLAMA_KV_CACHE_TYPE=q8_0

# Flash Attention - more efficient attention mechanism (10-15% speedup)
export OLLAMA_FLASH_ATTENTION=1

# Keep model loaded - prevents unloading between requests (saves 8s load time)
export OLLAMA_KEEP_ALIVE=24h

# Parallel requests - enables concurrent inference (2-4x throughput)
export OLLAMA_NUM_PARALLEL=4
```

### Applying Changes

After setting environment variables, restart Ollama:

```bash
# If running as service
brew services restart ollama

# Or if running manually, stop and restart
pkill ollama
ollama serve
```

### Verify Settings

Check if variables are being used:

```bash
# Check Ollama version and settings
ollama --version

# Check server logs
tail -f ~/Library/Logs/Ollama/server.log
```

### Prompt Optimization

For fastest inference, minimize output tokens:

```python
# Slow: Asks for explanation (~500 tokens output)
prompt = "Compare these medical histories and explain your reasoning..."

# Fast: Only requests score (~10 tokens output)
prompt = "Score similarity 0.0-1.0. Output ONLY a number.\nHistory 1: ...\nHistory 2: ...\nScore:"
```

**Impact:** Reducing output from 500 to 10 tokens can improve latency from 19s to ~3-5s.

### Parallel Processing

With `OLLAMA_NUM_PARALLEL=4`, you can make concurrent requests:

```python
from concurrent.futures import ThreadPoolExecutor

def compare_batch(pairs, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: client.compare(*p), pairs))
    return results
```

**Note:** Parallel requests improve throughput, not individual latency.

### Benchmark Results (Mac M3 Pro)

| Configuration | Avg Latency | Throughput |
|---------------|-------------|------------|
| Default + verbose prompt | 19.08s | 3.1/min |
| Env vars + simplified prompt | ~4s | 15/min |
| Env vars + simplified + parallel | ~4s | 60/min |

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Ollama Modelfile Syntax](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama Performance Tuning Guide](https://github.com/ollama/ollama/blob/main/docs/faq.md)

---

**Last Updated:** 2026-01-20
