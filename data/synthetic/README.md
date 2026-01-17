# Synthetic Patient Dataset

This directory contains synthetic patient data generated for testing the MedMatch AI entity resolution system.

## Files

- **synthetic_demographics.csv**: Demographic records (name, DOB, contact info) from various systems
- **synthetic_medical_records.json**: Complete medical records with clinical notes, conditions, vitals
- **ground_truth.csv**: Ground truth mapping showing which records belong to the same patient
- **dataset_metadata.json**: Generation parameters and statistics

## Dataset Characteristics

### Size
- ~75 unique patients
- ~225-375 demographic records (3-5 per patient)
- ~75-150 medical records (1-2 per patient)

### Edge Cases (Challenging Scenarios)

1. **Twins (10%)**: Same/similar names, same/very close DOB
2. **Siblings (5%)**: Shared last name, similar demographics
3. **Parent-Child (5%)**: Jr/Sr patterns, shared names
4. **Common Name Collisions (20%)**: Multiple "John Smith" patients
5. **Same Name + Same DOB**: Ultimate challenge - same demographics, different people

### Variations in Records

1. **Name Variations (30-40%)**:
   - Nicknames (William → Bill)
   - Middle name inconsistencies
   - Married name changes
   - Accent variations (José vs Jose)

2. **Data Entry Errors (20-25%)**:
   - DOB typos (transposed digits)
   - Name misspellings
   - Missing fields

3. **Different Sources (30-40%)**:
   - Same patient in different hospital systems
   - Different formatting conventions
   - Updated contact information

### Difficulty Distribution

- **Easy (30%)**: Clean data, same format
- **Medium (40%)**: Name variations, different formats
- **Hard (20%)**: Data errors, abbreviations
- **Ambiguous (10%)**: Same name+DOB, twins, minimal differentiators

## Usage

### Loading Demographics
```python
import pandas as pd

demographics = pd.read_csv('data/synthetic/synthetic_demographics.csv')
print(demographics.head())
```

### Loading Medical Records
```python
import json

with open('data/synthetic/synthetic_medical_records.json', 'r') as f:
    medical_records = json.load(f)
```

### Loading Ground Truth
```python
import pandas as pd

ground_truth = pd.read_csv('data/synthetic/ground_truth.csv')

# Get all records for a specific patient
patient_records = ground_truth[ground_truth['patient_id'] == 'P0001']
```

### Evaluation Example
```python
# Check if two records should match
def should_match(record_id1, record_id2, ground_truth):
    """Check if two records belong to the same patient."""
    match_group1 = ground_truth[ground_truth['record_id'] == record_id1]['match_group'].values[0]
    match_group2 = ground_truth[ground_truth['record_id'] == record_id2]['match_group'].values[0]
    return match_group1 == match_group2

# Example
result = should_match('R0001', 'R0002', ground_truth)
print(f"Records match: {result}")
```

## Data Fields

### Demographics CSV
- `record_id`: Unique record identifier
- `patient_id`: Ground truth patient ID (for evaluation)
- `name_first`, `name_middle`, `name_last`, `name_suffix`: Name components
- `date_of_birth`: Date of birth (various formats)
- `gender`: M/F/X
- `mrn`: Medical record number (system-specific)
- `ssn_last4`: Last 4 digits of SSN
- `phone`, `email`: Contact information
- `address_*`: Address components
- `record_source`: Which system created this record
- `record_date`: When record was created
- `data_quality_flag`: Type of variation ("clean", "name_variation_nickname", etc.)

### Medical Records JSON
- `record_id`: Links to demographics
- `patient_id`: Ground truth patient ID
- `record_source`: Type of encounter
- `record_date`: Encounter date
- `chief_complaint`: Reason for visit
- `medical_history`: Conditions, medications, allergies, surgeries
- `vital_signs`: BP, HR, RR, temp, height, weight
- `clinical_notes`: AI-generated clinical narrative
- `assessment`: Diagnosis
- `plan`: Treatment plan

### Ground Truth CSV
- `record_id`: Record identifier
- `patient_id`: Ground truth patient ID
- `match_group`: Group ID for matching records
- `notes`: Description of variation type
- `is_common_name`, `is_twin`, `is_family_member`, `has_data_error`: Flags
- `difficulty`: easy/medium/hard/ambiguous

## Privacy & Ethics

- **100% synthetic data** - no real patient information
- Generated using AI and algorithmic methods
- Safe for development, testing, and public sharing
- Medical scenarios based on common textbook cases

## Regenerating Dataset

To generate a new dataset:

```bash
# Default: 75 patients with AI-assisted generation
python generate_synthetic_data.py

# Custom size
python generate_synthetic_data.py --num-patients 100

# Without AI (faster, less realistic)
python generate_synthetic_data.py --no-ai

# Different random seed
python generate_synthetic_data.py --seed 123
```

## Known Limitations

1. **Medical realism**: AI-generated notes are plausible but may not match real clinical workflows
2. **Demographic diversity**: Name/location distributions may not reflect real populations
3. **Simplified medical histories**: Real patients have more complex longitudinal histories
4. **Limited imaging data**: Not included in current version

## Version Info

See `dataset_metadata.json` for:
- Generation date
- Random seed used
- Exact counts and statistics
- Generator version
