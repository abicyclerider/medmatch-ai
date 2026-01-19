# MedGemma Local Setup Guide

## Overview

This guide will help you set up local MedGemma deployment for MedMatch AI. MedGemma is Google's medical fine-tuned version of Gemma, optimized for healthcare applications.

## Available Models

| Model | Parameters | Size | Memory Required | Recommended For |
|-------|-----------|------|-----------------|-----------------|
| `medgemma-1.5-4b-it` | 4B | ~8GB | 12GB+ RAM | **Recommended** - Good balance |
| `medgemma-4b-it` | 4B | ~8GB | 12GB+ RAM | Alternative (older version) |
| `medgemma-27b-it` | 27B | ~54GB | 32GB+ RAM | Advanced (requires more memory) |

**For this project, we'll use `medgemma-1.5-4b-it` (latest 4B instruction-tuned model).**

Your Mac M3 Pro with 18GB RAM is perfect for the 4B model!

## Step 1: Create Hugging Face Account

1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up with email or GitHub
3. Verify your email address

## Step 2: Request MedGemma Access

MedGemma is a **gated model** that requires approval:

1. Visit [https://huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
2. Click **"Agree and access repository"** button
3. Read and accept the license agreement
4. Wait for approval (usually instant to 24 hours)

You'll receive an email when access is granted.

## Step 3: Create Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Name it: `medmatch-ai` (or any name you prefer)
4. Select token type: **"Read"** (sufficient for downloading models)
5. Click **"Generate token"**
6. **Copy the token** (you won't be able to see it again!)

## Step 4: Login to Hugging Face CLI

In your terminal, run:

```bash
# Activate virtual environment
source venv/bin/activate

# Install huggingface-cli (if not already installed)
pip install huggingface-hub

# Login with your token
huggingface-cli login
```

When prompted:
- Paste your access token
- Press Enter
- Choose "Y" to add token to git credentials (optional)

You should see: `✓ Login successful`

## Step 5: Download MedGemma

Run the download script:

```bash
python scripts/download_medgemma.py
```

**What will happen:**
- Downloads ~8GB of model files (5-15 minutes depending on connection)
- Caches model locally at `~/.cache/huggingface/hub/`
- Tests inference on your Mac Metal GPU (MPS)
- Verifies everything works

**First download is slow, but future runs will be instant!**

## Step 6: Verify Installation

After download completes, test the installation:

```python
from src.medmatch.matching.ai_client import MedicalAIClient

# Create MedGemma client
client = MedicalAIClient.create(
    backend="medgemma",
    model="medgemma-1.5-4b-it",
    device="mps"  # Mac Metal GPU
)

# Test medical comparison
score, reasoning = client.compare_medical_histories(
    "Patient has Type 2 Diabetes, Hypertension, on Metformin",
    "T2DM, HTN, medications: Metformin"
)

print(f"Similarity: {score:.2f}")
print(f"Reasoning: {reasoning}")
```

Expected output:
```
Similarity: 0.95
Reasoning: Both records describe the same chronic conditions...
```

## Troubleshooting

### Error: "401 Unauthorized" or "Access denied"

**Cause:** You haven't been granted access to MedGemma yet.

**Solution:**
1. Check your email for approval confirmation
2. Verify you accepted the license at https://huggingface.co/google/medgemma-1.5-4b-it
3. Wait up to 24 hours for approval
4. Try logging in again: `huggingface-cli login`

### Error: "Out of memory"

**Cause:** Model doesn't fit in RAM.

**Solution:** Enable quantization to reduce memory usage by 50%:

```python
client = MedicalAIClient.create(
    backend="medgemma",
    model="medgemma-1.5-4b-it",
    device="mps",
    use_quantization=True  # Reduces to 4-bit
)
```

Or use Gemini API as fallback while waiting for better hardware.

### Error: "MPS not available"

**Cause:** PyTorch MPS backend not enabled.

**Solution:**
1. Check PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`
2. Should return `True` on Mac M1/M2/M3
3. If False, reinstall PyTorch: `pip install --upgrade torch`

### Download is slow

**Normal:** Model is 8GB, can take 5-15 minutes on typical connections.

**Tips:**
- Use wired connection if possible
- Download during off-peak hours
- Model is cached after first download (subsequent loads are instant)

## Alternative: Use Gemini API While Waiting

If you're waiting for MedGemma access approval, use Gemini API:

```python
client = MedicalAIClient.create(
    backend="gemini",
    model="gemini-2.5-flash"
)
```

You can switch to MedGemma later by changing one line - the interfaces are identical!

## Hardware Requirements

### Minimum (CPU mode)
- 16GB RAM
- 20GB disk space
- Will work but slow (~5-10s per comparison)

### Recommended (GPU mode)
- Mac M1/M2/M3 with 18GB+ RAM ✓ **You have this!**
- 20GB disk space
- Fast inference (~0.5-1s per comparison)

### Optimal
- Mac M1 Pro/Max/Ultra with 32GB+ RAM
- NVMe SSD for caching
- Ultra-fast inference (~0.3-0.5s per comparison)

## Next Steps

Once MedGemma is downloaded:

1. **Run benchmarks:** Compare Gemini vs MedGemma
   ```bash
   python scripts/benchmark_ai_models.py --gemini --medgemma
   ```

2. **Use in matching pipeline:**
   ```bash
   python scripts/run_matcher.py --use-ai --ai-backend medgemma
   ```

3. **Enjoy zero API costs!** 🎉

## Resources

- [MedGemma Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Hugging Face Gated Models Guide](https://huggingface.co/docs/hub/models-gated)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Last Updated:** 2026-01-18
