# Synthetic Data Generation Plan

## Overview

Generate a high-quality synthetic patient dataset (50-100 patients) to develop and test the MedMatch AI entity resolution system. The dataset will include challenging edge cases that stress-test patient matching algorithms.

## Dataset Specifications

### Size & Scope
- **Total Patients:** 50-100 unique patients
- **Records per Patient:** 2-5 records (simulating data from different systems)
- **Total Records:** ~200-400 patient records
- **Format:** CSV (demographics) + JSON (medical records)

### Challenging Test Scenarios

1. **Common Names (20-30% of dataset)**
   - Multiple "John Smith", "Maria Garcia", "Mohammed Ahmed" patients
   - Same name + same DOB but different people (twins born same day)
   - Forces algorithm to rely on medical history differentiation

2. **Name Variations (15-20% of dataset)**
   - Same patient across records: "William" → "Bill" → "Billy"
   - Married name changes: "Sarah Johnson" → "Sarah Martinez"
   - Middle name inconsistencies: "John A. Smith" vs "John Smith" vs "J. Smith"
   - Cultural name variations: "José" vs "Jose" (accents)

3. **Twins & Family Members (10-15% of dataset)**
   - Identical twins: same DOB, similar names, shared genetic conditions
   - Siblings: similar names, close DOBs, same address
   - Parent-child: Jr/Sr suffixes, shared medical facility

4. **Data Entry Errors (20-25% of dataset)**
   - DOB typos: 03/15/1965 → 03/15/1956 (transposed digits)
   - Name misspellings: "Schmidt" → "Smith"
   - MRN digit swaps
   - Missing middle initials or suffixes

### Data Fields

#### Demographics (CSV: `synthetic_demographics.csv`)
```csv
record_id,patient_id,name_first,name_middle,name_last,name_suffix,dob,gender,mrn,ssn_last4,phone,email,address_street,address_city,address_state,address_zip,record_source,record_date,data_quality_flag
```

**Fields:**
- `record_id`: Unique record identifier (R001, R002, ...)
- `patient_id`: Ground truth patient ID (P001, P002, ...) - for evaluation only
- `name_*`: Name components (allows variation testing)
- `dob`: Date of birth (various formats: MM/DD/YYYY, YYYY-MM-DD, etc.)
- `gender`: M/F/X
- `mrn`: Medical record number (system-specific, can have errors)
- `ssn_last4`: Last 4 of SSN (sometimes available)
- `contact`: Phone/email (can be outdated/inconsistent)
- `address_*`: Address components (people move)
- `record_source`: Which system/hospital (ER, Lab, Primary Care, Radiology)
- `record_date`: When this record was created
- `data_quality_flag`: Indicates if this record has intentional errors

#### Medical Records (JSON: `synthetic_medical_records.json`)
```json
{
  "record_id": "R001",
  "patient_id": "P001",
  "chief_complaint": "Chest pain and shortness of breath",
  "medical_history": {
    "conditions": [
      {"name": "Hypertension", "abbrev": "HTN", "onset": "2010"},
      {"name": "Type 2 Diabetes", "abbrev": "T2DM", "onset": "2015"}
    ],
    "medications": ["Metformin 500mg BID", "Lisinopril 10mg daily"],
    "allergies": ["Penicillin (rash)"],
    "surgeries": [
      {"procedure": "Appendectomy", "date": "1985-07-12"}
    ],
    "family_history": ["Father: MI at age 55", "Mother: breast cancer"]
  },
  "vital_signs": {
    "bp": "165/95",
    "hr": 102,
    "rr": 22,
    "temp": "98.6F",
    "o2_sat": "94% on RA",
    "height": "5'10\"",
    "weight": "220 lbs",
    "bmi": 31.6
  },
  "clinical_notes": "58M with PMHx of HTN and T2DM presents with...",
  "lab_results": {
    "date": "2024-01-15",
    "tests": [
      {"name": "Troponin I", "value": 0.8, "unit": "ng/mL", "reference": "<0.04"},
      {"name": "HbA1c", "value": 7.2, "unit": "%", "reference": "<5.7"}
    ]
  },
  "imaging": [
    {
      "modality": "Chest X-Ray",
      "date": "2024-01-15",
      "findings": "Mild cardiomegaly, no acute infiltrates"
    }
  ],
  "assessment": "Acute coronary syndrome, NSTEMI",
  "plan": "Cardiology consult, cath lab, ASA 325mg given",
  "record_source": "Emergency Department",
  "record_date": "2024-01-15T14:30:00"
}
```

#### Ground Truth Mapping (CSV: `ground_truth.csv`)
```csv
record_id,patient_id,match_group,notes
R001,P001,G001,Primary record - clean data
R002,P001,G001,Same patient - name variation (Bill vs William)
R003,P001,G001,Same patient - typo in DOB
R004,P002,G002,Different patient - same name as P001
```

**Purpose:** Evaluation dataset to measure matching algorithm accuracy

## Implementation Approach

### Phase 1: Data Schema & Templates
1. Define Pydantic models for data validation
2. Create template structures for different record types
3. Set up data quality flag system

### Phase 2: Demographics Generation
1. Build name generator with variation logic
   - Common surnames from census data
   - Name variation mappings (William→Bill→Billy)
   - Cultural name patterns
2. Generate DOB with intentional error injection
3. Create contact info with realistic inconsistencies
4. Build address generator with move tracking

### Phase 3: Medical History Generation (AI-Assisted)
1. Use Gemini API to generate realistic medical scenarios
2. Create condition-medication mappings
3. Generate clinical notes with proper medical terminology
4. Add abbreviation variations (HTN vs Hypertension)
5. Ensure medical coherence (age-appropriate conditions)

### Phase 4: Record Variation Generation
1. For each patient, create 2-5 variant records
2. Apply different types of variations:
   - Clean baseline record
   - Name variation record
   - Data entry error record
   - Different source system record
3. Maintain medical history consistency across variants

### Phase 5: Edge Case Injection
1. Create twin pairs
2. Generate common name collisions
3. Add family member relationships
4. Insert ambiguous cases (same name+DOB, different people)

### Phase 6: Quality Validation
1. Verify JSON schema compliance
2. Check medical coherence (no 5-year-olds with prostate cancer)
3. Ensure adequate challenge distribution
4. Generate summary statistics

## File Structure

```
data/
├── synthetic/
│   ├── synthetic_demographics.csv       # All demographic records
│   ├── synthetic_medical_records.json   # Rich medical data
│   ├── ground_truth.csv                 # Evaluation labels
│   ├── dataset_metadata.json            # Generation params & stats
│   └── README.md                        # Dataset documentation
└── generation/
    ├── templates/                       # Data templates
    ├── name_lists/                      # Name resources
    ├── medical_conditions.json          # Condition templates
    └── generation_config.yaml           # Generation parameters
```

## Code Structure

```
src/medmatch/data/
├── __init__.py
├── generators/
│   ├── __init__.py
│   ├── demographics.py     # Demographics generation
│   ├── medical.py          # Medical history generation (AI-assisted)
│   ├── variations.py       # Record variation logic
│   └── edge_cases.py       # Special case generation
├── models/
│   ├── __init__.py
│   ├── patient.py          # Pydantic patient models
│   └── record.py           # Record schemas
├── utils/
│   ├── __init__.py
│   ├── name_utils.py       # Name variation logic
│   ├── date_utils.py       # Date format variations
│   └── medical_utils.py    # Medical term mappings
└── generate_dataset.py     # Main generation script
```

## Key Design Principles

### Medical Realism
- Use proper medical terminology and abbreviations
- Age-appropriate conditions (no pediatric geriatric diseases)
- Medication-condition alignment
- Realistic vital sign ranges
- Coherent clinical narratives

### Challenge Distribution
- 30% easy matches (same format, minor variations)
- 40% medium matches (name variations, different formats)
- 20% hard matches (errors, abbreviations, sparse data)
- 10% ambiguous cases (same name+DOB different people)

### Privacy & Ethics
- All data is purely synthetic
- No real patient information used
- Random demographic distributions
- Medical scenarios from textbooks/training cases

### Reproducibility
- Seed random number generators
- Document generation parameters
- Version control generation code
- Track dataset versions

## Success Metrics

A successful synthetic dataset will:
1. ✅ Generate without errors or crashes
2. ✅ Pass schema validation (100% compliance)
3. ✅ Include all specified edge cases
4. ✅ Have medically coherent records (human review)
5. ✅ Support algorithm testing (positive/negative matching cases)
6. ✅ Be reproducible (same seed → same data)

## Next Steps After Generation

1. **Visual Inspection:** Manually review sample records
2. **Baseline Algorithm:** Run simple name+DOB matching
3. **Error Analysis:** Identify where baseline fails
4. **Iterate:** Adjust generation if needed
5. **Scale Up:** Generate larger datasets once validated

## Estimated Timeline

- **Phase 1-2 (Schema + Demographics):** 2-3 hours implementation
- **Phase 3 (Medical AI Generation):** 3-4 hours implementation + API time
- **Phase 4-5 (Variations + Edge Cases):** 2-3 hours implementation
- **Phase 6 (Validation):** 1-2 hours
- **Total:** Single focused work session

## API Usage Considerations

- **Gemini API Calls:** ~50-100 calls for medical history generation
- **Rate Limits:** Free tier should be sufficient
- **Cost:** Negligible for this dataset size
- **Fallback:** Can generate simpler medical histories if API issues

## Dependencies

Already installed:
- pandas (CSV/data manipulation)
- numpy (random generation)
- python-dotenv (API keys)
- google-genai (medical text generation)

May need to add:
- faker (realistic fake data generation)
- pydantic (data validation)
- pyyaml (config files)

---

**Status:** Planning complete, ready for implementation
**Last Updated:** 2026-01-16
