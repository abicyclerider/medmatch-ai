# Synthetic Patient Dataset Generation - Implementation Guide

## Current Session Status

**Completed:**
- ✅ Full environment setup (Python 3.12.4, venv, all dependencies)
- ✅ Google AI API configured and tested (Gemini 2.5 Flash)
- ✅ Proof of concept: 98% confidence patient matching demonstrated
- ✅ Plan created and approved for synthetic data generation

**Next Task:** Build synthetic patient dataset generation system

---

## Quick Start Commands

```bash
# Activate environment
cd /Users/alex/repos/Kaggle/medmatch-ai
source venv/bin/activate

# Run generation script (when created)
python src/medmatch/data/generate_synthetic.py

# Explore dataset (when created)
jupyter lab
# Open: notebooks/01_dataset_exploration.ipynb
```

---

## Implementation Tasks (In Order)

### Phase 1: Create `src/medmatch/data/generate_synthetic.py`

**File location:** `/Users/alex/repos/Kaggle/medmatch-ai/src/medmatch/data/generate_synthetic.py`

**What it does:**
- Generates 50 unique base patients
- Creates 1-2 duplicate records per patient with format variations
- Outputs ~100-150 total records

**Key components to implement:**

1. **Name lists:**
   - 50 male first names, 50 female first names
   - 100 common last names
   - Random middle initials (A-Z)

2. **Name format generator (4 variations):**
   - Standard: "John A. Smith"
   - Last-First: "Smith, John A."
   - No Middle: "Smith, John"
   - Initials: "J. A. Smith"

3. **Date generator:**
   - Random dates 1935-2005 (ages 20-90)
   - 4 format variations:
     - Standard: "03/15/1965"
     - Short: "3/15/65"
     - Written: "March 15, 1965"
     - ISO: "1965-03-15"

4. **Medical conditions:**
   - Pool: Diabetes, Hypertension, Heart disease, Asthma, Arthritis
   - Each patient: 0-3 conditions
   - Multiple terminology options per condition:
     - Diabetes: ["diabetic", "T2DM", "Type 2 Diabetes", "diabetes mellitus"]
     - Hypertension: ["hypertensive", "HTN", "high blood pressure", "hypertension"]
     - Heart disease: ["CAD", "coronary artery disease", "heart disease"]
     - Asthma: ["asthma", "asthmatic", "reactive airway disease"]
     - Arthritis: ["arthritis", "OA", "osteoarthritis"]

5. **MRN generator:**
   - 8-digit random numbers (unique)
   - 3 format variations:
     - Plain: "12345678"
     - Prefix: "MRN-12345678"
     - Short: "MR12345678"

**Output structure per record:**
```python
{
    "record_id": "rec_001",      # Unique per record
    "patient_id": "pat_001",     # Same for duplicates
    "name": "John A. Smith",
    "dob": "03/15/1965",
    "mrn": "12345678",
    "gender": "M",
    "conditions": ["diabetic", "hypertensive"],
    "format_variant": {
        "name_format": "standard",
        "date_format": "standard",
        "mrn_format": "plain"
    }
}
```

**Required imports:**
```python
import random
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
```

---

### Phase 2: Generate 3 Output Files

**1. `data/samples/patients_raw.json`**
- Full JSON array of all records
- Nested structure with complete details

**2. `data/samples/patients_raw.csv`**
- Flattened format
- Columns: record_id, patient_id, name, dob, mrn, gender, condition_1, condition_2, condition_3

**3. `data/samples/ground_truth.csv`**
- Match validation data
- Columns: record_id_1, record_id_2, should_match, patient_id, reason
- Example rows:
  ```csv
  record_id_1,record_id_2,should_match,patient_id,reason
  rec_001,rec_002,True,pat_001,Same patient with format variations
  rec_001,rec_003,False,,Different patients
  ```

**4. `data/samples/README.md`**
- Dataset statistics
- Generation methodology
- Usage instructions

---

### Phase 3: Create `notebooks/01_dataset_exploration.ipynb`

**Purpose:** Validate and explore generated dataset

**Sections:**

1. **Load Data**
```python
import pandas as pd
import json

# Load JSON
with open('../data/samples/patients_raw.json') as f:
    records = json.load(f)

# Load CSV
df = pd.read_csv('../data/samples/patients_raw.csv')
ground_truth = pd.read_csv('../data/samples/ground_truth.csv')
```

2. **Display Statistics**
- Total records
- Unique patients
- Duplicate count per patient
- Age distribution
- Gender split
- Condition frequency

3. **Visualizations**
- Matplotlib/seaborn charts:
  - Age histogram
  - Gender pie chart
  - Top 10 conditions bar chart
  - Format variation breakdown

4. **Ground Truth Validation**
- Verify all duplicates have same patient_id
- Check for data leakage
- Test random pairs

5. **Gemini API Test**
```python
from dotenv import load_dotenv
import os
import google.genai as genai

load_dotenv()
client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'))

# Test with a matching pair
record_1 = records[0]
record_2 = records[1]  # Should be same patient

prompt = f"""Compare these patient records. Do they refer to the same person?

Record 1: {record_1['name']}, DOB: {record_1['dob']}, Conditions: {', '.join(record_1['conditions'])}
Record 2: {record_2['name']}, DOB: {record_2['dob']}, Conditions: {', '.join(record_2['conditions'])}

Provide confidence (0-100%) and reasoning."""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt
)
print(response.text)
```

---

## Data Generation Methodology Summary

**Core Principle:** Controlled realistic variation

**Approach:**
1. Generate 50 base patients with canonical data
2. For each patient, create 1-2 duplicates
3. Apply format variations to duplicates (name, date, terminology)
4. Track ground truth (patient_id) for validation
5. No typos, no missing data (easy matches only)

**Why this works:**
- Tests format handling (core challenge)
- Maintains ground truth for accuracy measurement
- Realistic medical terminology variations
- Scalable to larger datasets

---

## File Locations

**To create:**
- `/Users/alex/repos/Kaggle/medmatch-ai/src/medmatch/data/generate_synthetic.py`
- `/Users/alex/repos/Kaggle/medmatch-ai/data/samples/patients_raw.json`
- `/Users/alex/repos/Kaggle/medmatch-ai/data/samples/patients_raw.csv`
- `/Users/alex/repos/Kaggle/medmatch-ai/data/samples/ground_truth.csv`
- `/Users/alex/repos/Kaggle/medmatch-ai/data/samples/README.md`
- `/Users/alex/repos/Kaggle/medmatch-ai/notebooks/01_dataset_exploration.ipynb`

**To update:**
- `/Users/alex/repos/Kaggle/medmatch-ai/README.md` - Add usage section
- `/Users/alex/repos/Kaggle/medmatch-ai/.claude/claude.md` - Note dataset creation

---

## Success Criteria

Dataset ready when:
- ✓ Script runs without errors
- ✓ 50 unique patients, ~100-150 total records
- ✓ All 3 output files generated
- ✓ Ground truth correctly labels matches
- ✓ Exploration notebook validates data
- ✓ Gemini test shows high confidence matching
- ✓ Committed to git

---

## Git Commit Plan

**Commit message:**
```
Add synthetic patient dataset generation

- Create generation script with name/date/condition variations
- Generate 50 unique patients with ~100 total records
- Include ground truth labels for match validation
- Add exploration notebook for dataset analysis
- Simple format (demographics + 2-3 conditions) for initial testing

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Files to commit:**
- src/medmatch/data/generate_synthetic.py
- data/samples/*.json, *.csv, README.md
- notebooks/01_dataset_exploration.ipynb
- Updated README.md
- Updated .claude/claude.md

---

## Next Steps After Dataset

Once dataset is complete:
1. Build first matching algorithm prototype
2. Experiment with prompting strategies
3. Develop confidence scoring system
4. Create evaluation metrics (precision, recall, F1)
5. Expand to medium complexity dataset

---

## Key Context for New Sessions

- **Project:** MedMatch AI - prevent wrong-patient medical errors
- **Approach:** Use medical AI for entity resolution
- **Current model:** Gemini 2.5 Flash (will migrate to MedGemma later)
- **Developer:** Alex (advanced Python/ML experience)
- **Environment:** Mac with MPS/Metal acceleration
- **Dataset stage:** Building first simple synthetic dataset
- **Plan:** Located at `/Users/alex/.claude/plans/velvet-discovering-corbato.md`

---

## Estimated Time

- Phase 1: 20-30 min (generation script)
- Phase 2: 10 min (output files)
- Phase 3: 20 min (validation notebook)
- Phase 4: 5 min (git commit)
- **Total: ~55-65 minutes**

---

**Ready to implement!** Start with Phase 1: creating the generation script.
