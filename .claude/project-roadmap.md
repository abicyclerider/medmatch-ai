# MedMatch AI - Complete Project Roadmap

## Project Vision

**Goal:** Build an AI-powered medical entity resolution system that prevents wrong-patient errors in healthcare

**Problem:** Wrong-patient errors (e.g., operating on the wrong John Smith) are "never events" that cause deaths

**Solution:** Use medical AI to create "medical fingerprints" that ensure accurate patient matching at critical clinical decision points

---

## Current Progress

✅ **Phase 0: Environment Setup** (COMPLETE)
- Python 3.12.4 with venv
- PyTorch 2.9.1 with Mac Metal/MPS acceleration
- Google AI API configured (Gemini 2.5 Flash)
- Project structure created
- Proof of concept: 98% confidence patient matching

🔄 **Phase 1: Synthetic Data Generation** (IN PROGRESS)
- Create generation script
- Generate 50 patients with ~100 records
- Output JSON, CSV, ground truth files
- Build validation notebook

---

## Complete Development Roadmap

### Phase 1: Synthetic Data Generation (Current)
**Status:** In progress
**Goal:** Create test dataset for algorithm development
**Deliverables:**
- 50 unique patients with format variations
- Ground truth labels for validation
- Exploration notebook
- Dataset documentation

**Key files:**
- `src/medmatch/data/generate_synthetic.py`
- `data/samples/patients_raw.json`
- `data/samples/ground_truth.csv`
- `notebooks/01_dataset_exploration.ipynb`

---

### Phase 2: Basic Matching Algorithm Prototype
**Status:** Not started
**Goal:** Build first matching algorithm using Gemini API
**Estimated time:** 2-3 days

**What to build:**

1. **Record Comparison Function** (`src/medmatch/matching/comparator.py`)
   - Load two patient records
   - Send to Gemini for comparison
   - Parse confidence score and reasoning
   - Return match decision (True/False/Uncertain)

2. **Batch Matching Script** (`src/medmatch/matching/batch_matcher.py`)
   - Load entire dataset
   - Compare all record pairs
   - Generate match predictions
   - Output results with confidence scores

3. **Evaluation Module** (`src/medmatch/matching/evaluator.py`)
   - Compare predictions to ground truth
   - Calculate metrics:
     - Precision (% of predicted matches that are correct)
     - Recall (% of actual matches found)
     - F1 Score (harmonic mean of precision/recall)
     - Accuracy (overall correctness)
   - Identify false positives and false negatives

4. **Results Notebook** (`notebooks/02_matching_results.ipynb`)
   - Run batch matcher on dataset
   - Visualize results
   - Analyze errors
   - Tune confidence thresholds

**Success criteria:**
- Achieves >90% precision and >90% recall on simple dataset
- Can process 100 records in reasonable time
- Clear understanding of where algorithm fails

**Key decisions:**
- How to structure prompts for best results
- What confidence threshold to use (e.g., >80% = match)
- How to handle uncertain cases

---

### Phase 3: Prompt Engineering & Optimization
**Status:** Not started
**Goal:** Improve matching accuracy through better prompting
**Estimated time:** 1-2 days

**What to experiment with:**

1. **Prompt Variations:**
   - Different instruction formats
   - Include/exclude specific fields
   - Add medical context
   - Test chain-of-thought reasoning

2. **Multi-Signal Matching:**
   - Name similarity weight
   - DOB match weight
   - Medical condition overlap weight
   - MRN consideration

3. **Confidence Calibration:**
   - Map AI confidence to actual accuracy
   - Adjust thresholds per field type
   - Handle missing data gracefully

4. **Experiment Tracking:**
   - Log all prompt variations
   - Track performance metrics
   - Document what works/doesn't work

**Deliverables:**
- Optimized prompt templates
- Confidence threshold guidelines
- Performance comparison notebook

---

### Phase 4: Expand Dataset Complexity
**Status:** Not started
**Goal:** Test algorithm with harder matching scenarios
**Estimated time:** 2 days

**Dataset expansions:**

1. **Medium Complexity Dataset:**
   - Add typos and misspellings
   - Include missing data (blank fields)
   - Add nickname variations (William → Bill)
   - More medical conditions (5-10 per patient)
   - Include vitals and medications

2. **Hard Complexity Dataset:**
   - Same name + DOB but different patients (edge case!)
   - Transposed digits in dates/MRNs
   - Multiple middle names
   - Married name changes
   - International date formats

3. **Evaluation:**
   - Re-run matching on new datasets
   - Measure performance degradation
   - Identify specific failure modes

**Success criteria:**
- Algorithm handles medium complexity with >80% F1
- Correctly identifies edge cases (same name+DOB, different person)
- Gracefully degrades with hard cases

---

### Phase 5: Local MedGemma Deployment
**Status:** Not started
**Goal:** Replace Gemini API with local MedGemma for better medical accuracy
**Estimated time:** 2-3 days

**Implementation:**

1. **Setup MedGemma:**
   - Add transformers to requirements
   - Download MedGemma from Hugging Face
   - Configure for Mac Metal/MPS
   - Optimize memory usage

2. **Adapter Layer:**
   - Create unified interface for both Gemini and MedGemma
   - Allow easy switching between models
   - Maintain same API contract

3. **Benchmark:**
   - Run same datasets through both models
   - Compare accuracy (expect MedGemma to be better)
   - Compare speed (expect local to be slower)
   - Measure resource usage

4. **Migration:**
   - Switch default to MedGemma
   - Keep Gemini as fallback
   - Update documentation

**Key files:**
- `src/medmatch/models/medgemma_local.py`
- `src/medmatch/models/model_interface.py`
- `notebooks/03_model_comparison.ipynb`

---

### Phase 6: Multi-Modal Matching (Advanced)
**Status:** Future
**Goal:** Include imaging and lab data in matching
**Estimated time:** 1-2 weeks

**Capabilities:**

1. **Imaging Integration:**
   - Load medical images (X-rays, CT scans)
   - Use MedGemma's vision capabilities
   - Match based on imaging characteristics
   - Combine with demographic matching

2. **Lab Data:**
   - Include lab values (glucose, A1C, etc.)
   - Temporal patterns (trends over time)
   - Abnormal value patterns

3. **Timeline Matching:**
   - Admission/discharge dates
   - Treatment sequences
   - Medication history

**Use case:** "These records have similar demographics, but different imaging patterns → likely different patients"

---

### Phase 7: Explainable AI & Safety
**Status:** Future
**Goal:** Make match decisions transparent and safe for clinical use
**Estimated time:** 1 week

**Features:**

1. **Match Explanation:**
   - Show which fields contributed to decision
   - Highlight discrepancies
   - Provide evidence for match/no-match

2. **Confidence Visualization:**
   - Color-coded confidence levels
   - Field-by-field similarity scores
   - Uncertainty indicators

3. **Safety Checks:**
   - Flag ambiguous cases for human review
   - Never auto-merge on low confidence
   - Audit trail of all decisions

4. **Clinical Workflow:**
   - Integration points in EHR systems
   - Pre-operative verification
   - Emergency department patient identification

---

### Phase 8: Production & Kaggle Submission
**Status:** Future
**Goal:** Prepare for competition submission and real-world deployment
**Estimated time:** 1 week

**Deliverables:**

1. **Kaggle Submission:**
   - Follow competition requirements
   - Submit MedGemma-based solution
   - Document approach
   - Create presentation/demo

2. **Production Readiness:**
   - API endpoints for matching
   - Batch processing capabilities
   - Performance optimization
   - Error handling and logging

3. **Documentation:**
   - Technical documentation
   - User guide
   - Deployment instructions
   - Privacy/HIPAA compliance notes

---

## Key Decision Points

### Now (Phase 1-2):
- ✅ Use Gemini for prototyping (faster iteration)
- ✅ Start with simple dataset (build foundation)
- ✅ Focus on core matching logic

### Soon (Phase 3-5):
- Switch to MedGemma for accuracy
- Expand dataset complexity
- Optimize prompts and thresholds

### Later (Phase 6-8):
- Add multi-modal capabilities
- Build production features
- Prepare for deployment

---

## Success Metrics

### Technical Metrics:
- **Precision:** >95% (few false matches)
- **Recall:** >95% (find all true matches)
- **F1 Score:** >95%
- **Speed:** <1 second per comparison
- **Critical safety:** 0% false positives on edge cases (same name+DOB, different patient)

### Competition Metrics:
- High ranking in MedGemma challenge
- Novel approach (medical fingerprinting)
- Clinical applicability

### Real-World Impact:
- Prevents wrong-patient errors
- Deployable in hospital EHR systems
- HIPAA compliant (local processing)
- Explainable decisions (clinicians trust it)

---

## Resource Requirements

### Current Phase (1-2):
- Time: 1-2 weeks
- Resources: Gemini API (free tier sufficient)
- Skills: Python, data generation, prompt engineering

### Future Phases:
- Local compute: Mac with 16GB+ RAM for MedGemma
- Storage: ~10GB for models
- API costs: Minimal with local deployment
- Skills: ML model deployment, healthcare domain knowledge

---

## Risk Mitigation

### Technical Risks:
- **Risk:** MedGemma too slow locally
  **Mitigation:** Optimize with Metal/MPS, batch processing

- **Risk:** Low accuracy on edge cases
  **Mitigation:** Expand training data, human-in-loop for ambiguous cases

- **Risk:** API rate limits
  **Mitigation:** Switch to local MedGemma early

### Competition Risks:
- **Risk:** Other teams use similar approach
  **Mitigation:** Focus on novel "medical fingerprinting" concept

- **Risk:** Competition timeline pressure
  **Mitigation:** Modular design allows parallel development

### Deployment Risks:
- **Risk:** Privacy concerns with patient data
  **Mitigation:** Local-only deployment, no cloud APIs

- **Risk:** Clinical adoption resistance
  **Mitigation:** Explainable AI, audit trails, gradual rollout

---

## Next Immediate Steps

**After synthetic data generation completes:**

1. **Day 1-2:** Build basic matching algorithm (Phase 2)
   - Start with `notebooks/02_matching_prototype.ipynb`
   - Experiment with different prompts
   - Get first accuracy numbers

2. **Day 3-4:** Optimize and evaluate (Phase 3)
   - Improve prompts
   - Tune confidence thresholds
   - Document best practices

3. **Day 5-7:** Expand dataset complexity (Phase 4)
   - Generate medium complexity dataset
   - Test algorithm robustness
   - Identify failure modes

4. **Week 2:** MedGemma migration (Phase 5)
   - Set up local deployment
   - Benchmark vs Gemini
   - Switch to production model

---

## Reference Documents

- **Full project context:** `.claude/claude.md`
- **Synthetic data plan:** `.claude/synthetic-data-plan.md`
- **Current plan:** `.claude/plans/velvet-discovering-corbato.md`
- **Main README:** `README.md`

---

**This roadmap provides the complete journey from where you are now (data generation) to a production-ready system!**
