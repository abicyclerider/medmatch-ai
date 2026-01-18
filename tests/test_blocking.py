"""
Tests for blocking strategies.

Validates:
1. Each blocking strategy works correctly
2. MultiBlocker reduces candidate pairs to <5% of total
3. Blocking recall = 100% (no true matches missed)
"""

import pytest
from datetime import date
import pandas as pd

from src.medmatch.matching.core import PatientRecord
from src.medmatch.matching.blocking import (
    SoundexYearGenderBlocker,
    NamePrefixDOBBlocker,
    PhoneBlocker,
    SSNYearGenderBlocker,
    MRNBlocker,
    MultiBlocker,
)
from src.medmatch.data.models.patient import Demographics, Address


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_patient_record(
    record_id: str,
    name_first: str,
    name_last: str,
    dob: date,
    gender: str,
    phone: str = None,
    ssn_last4: str = None,
    mrn: str = None,
) -> PatientRecord:
    """Helper to create PatientRecord for testing."""
    demo = Demographics(
        record_id=record_id,
        patient_id="P001",  # Dummy
        name_first=name_first,
        name_middle=None,
        name_last=name_last,
        name_suffix=None,
        date_of_birth=dob,
        gender=gender,
        mrn=mrn or f"MRN{record_id}",
        ssn_last4=ssn_last4,
        phone=phone,
        email=None,
        address=Address(
            street="123 Main St",
            city="Boston",
            state="MA",
            zip_code="02101",
        ),
        record_source="Test",
        record_date=date.today(),
        data_quality_flag="clean",
    )
    return PatientRecord.from_demographics(demo)


def load_full_dataset() -> list:
    """Load full synthetic dataset for testing."""
    df = pd.read_csv('data/synthetic/synthetic_demographics.csv')

    def safe_str(value):
        """Convert value to string, handling NaN."""
        if pd.isna(value):
            return None
        return str(value)

    records = []
    for _, row in df.iterrows():
        try:
            dob = pd.to_datetime(row['date_of_birth']).date()
        except (pd.errors.OutOfBoundsDatetime, OverflowError):
            # Skip records with invalid dates
            continue

        # Parse address
        address = Address(
            street=safe_str(row.get('address_street')) or '',
            city=safe_str(row.get('address_city')) or '',
            state=safe_str(row.get('address_state')) or '',
            zip_code=safe_str(row.get('address_zip')) or '',
        )

        demo = Demographics(
            record_id=row['record_id'],
            patient_id=safe_str(row.get('patient_id')) or '',
            name_first=row['name_first'],
            name_middle=safe_str(row.get('name_middle')),
            name_last=row['name_last'],
            name_suffix=safe_str(row.get('name_suffix')),
            date_of_birth=dob,
            gender=row['gender'],
            mrn=safe_str(row['mrn']),
            ssn_last4=safe_str(row.get('ssn_last4')),
            phone=safe_str(row.get('phone')),
            email=safe_str(row.get('email')),
            address=address,
            record_source=row['record_source'],
            record_date=pd.to_datetime(row['record_date']).date(),
            data_quality_flag=safe_str(row.get('data_quality_flag')),
        )

        record = PatientRecord.from_demographics(demo)
        records.append(record)

    return records


def load_ground_truth():
    """Load ground truth for evaluation."""
    return pd.read_csv('data/synthetic/ground_truth.csv')


# =============================================================================
# UNIT TESTS - Individual Blocking Strategies
# =============================================================================


def test_soundex_year_gender_blocker():
    """Test Soundex+year+gender blocking."""
    blocker = SoundexYearGenderBlocker()

    # Same patient with name variation (phonetically similar)
    r1 = create_patient_record(
        "R001", "John", "Smith", date(1980, 3, 15), "M"
    )
    r2 = create_patient_record(
        "R002", "John", "Smyth", date(1980, 3, 15), "M"  # Phonetically same
    )

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Should have same Soundex code
    assert key1 == key2
    assert "S530" in key1  # Soundex for Smith/Smyth
    assert "1980" in key1
    assert "M" in key1


def test_soundex_different_names():
    """Test Soundex blocker with different names."""
    blocker = SoundexYearGenderBlocker()

    r1 = create_patient_record("R001", "John", "Smith", date(1980, 3, 15), "M")
    r2 = create_patient_record("R002", "Jane", "Johnson", date(1980, 3, 15), "F")

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Different names and gender → different keys
    assert key1 != key2


def test_name_prefix_dob_blocker():
    """Test first_3_chars+DOB blocking."""
    blocker = NamePrefixDOBBlocker()

    r1 = create_patient_record("R001", "John", "Smith", date(1980, 3, 15), "M")
    r2 = create_patient_record("R002", "Jane", "Smithson", date(1980, 3, 15), "F")

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Same prefix (SMI) + same DOB → same key
    assert key1 == key2
    assert "SMI" in key1
    assert "1980-03-15" in key1


def test_phone_blocker():
    """Test normalized phone number blocking."""
    blocker = PhoneBlocker()

    r1 = create_patient_record(
        "R001", "John", "Smith", date(1980, 3, 15), "M",
        phone="(617) 555-1234"
    )
    r2 = create_patient_record(
        "R002", "John", "Smith", date(1980, 3, 15), "M",
        phone="617-555-1234"  # Different format, same number
    )

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Same normalized phone → same key
    assert key1 == key2
    assert key1 == "6175551234"


def test_phone_blocker_missing():
    """Test phone blocker with missing phones."""
    blocker = PhoneBlocker()

    r1 = create_patient_record("R001", "John", "Smith", date(1980, 3, 15), "M", phone=None)
    r2 = create_patient_record("R002", "Jane", "Doe", date(1980, 3, 15), "F", phone=None)

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Missing phones get unique keys (don't block together)
    assert key1 != key2
    assert "MISSING_PHONE" in key1
    assert "MISSING_PHONE" in key2


def test_ssn_year_gender_blocker():
    """Test SSN+year+gender blocking."""
    blocker = SSNYearGenderBlocker()

    r1 = create_patient_record(
        "R001", "John", "Smith", date(1980, 3, 15), "M", ssn_last4="1234"
    )
    r2 = create_patient_record(
        "R002", "John", "Smith", date(1980, 3, 15), "M", ssn_last4="1234"
    )

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Same SSN + year + gender → same key
    assert key1 == key2
    assert "1234" in key1
    assert "1980" in key1
    assert "M" in key1


def test_mrn_blocker():
    """Test MRN blocking."""
    blocker = MRNBlocker()

    r1 = create_patient_record("R001", "John", "Smith", date(1980, 3, 15), "M", mrn="MRN123")
    r2 = create_patient_record("R002", "John", "Smith", date(1980, 3, 15), "M", mrn="MRN123")

    key1 = blocker.get_block_key(r1)
    key2 = blocker.get_block_key(r2)

    # Same MRN → same key
    assert key1 == key2
    assert key1 == "MRN123"


# =============================================================================
# INTEGRATION TESTS - MultiBlocker
# =============================================================================


def test_multi_blocker_generates_pairs():
    """Test MultiBlocker generates candidate pairs."""
    blocker = MultiBlocker([
        SoundexYearGenderBlocker(),
        NamePrefixDOBBlocker(),
    ])

    records = [
        create_patient_record("R001", "John", "Smith", date(1980, 3, 15), "M"),
        create_patient_record("R002", "John", "Smyth", date(1980, 3, 15), "M"),  # Phonetically same
        create_patient_record("R003", "Jane", "Johnson", date(1985, 6, 20), "F"),  # Different
    ]

    pairs = blocker.generate_candidate_pairs(records)

    # Should have (R001, R002) pair (same soundex)
    # Should not have (R001, R003) or (R002, R003) (different)
    assert len(pairs) >= 1

    pair_ids = [(p[0].record_id, p[1].record_id) for p in pairs]
    assert ('R001', 'R002') in pair_ids or ('R002', 'R001') in pair_ids


def test_multi_blocker_no_duplicates():
    """Test MultiBlocker doesn't generate duplicate pairs."""
    blocker = MultiBlocker([
        SoundexYearGenderBlocker(),
        NamePrefixDOBBlocker(),
        PhoneBlocker(),
    ])

    records = [
        create_patient_record(
            "R001", "John", "Smith", date(1980, 3, 15), "M", phone="6175551234"
        ),
        create_patient_record(
            "R002", "John", "Smith", date(1980, 3, 15), "M", phone="6175551234"
        ),
    ]

    pairs = blocker.generate_candidate_pairs(records)

    # Should only have 1 pair, even though multiple strategies match
    assert len(pairs) == 1

    # Verify it's the right pair
    assert pairs[0][0].record_id == "R001"
    assert pairs[0][1].record_id == "R002"


# =============================================================================
# PERFORMANCE TESTS - Full Dataset
# =============================================================================


def test_blocking_reduction_rate():
    """Test that blocking reduces candidate pairs to <5% of total."""
    records = load_full_dataset()

    blocker = MultiBlocker([
        SoundexYearGenderBlocker(),
        NamePrefixDOBBlocker(),
        PhoneBlocker(),
        SSNYearGenderBlocker(),
        MRNBlocker(),
    ])

    pairs = blocker.generate_candidate_pairs(records)

    n = len(records)
    total_possible = n * (n - 1) // 2
    num_candidates = len(pairs)
    reduction_rate = num_candidates / total_possible

    print(f"\nBlocking Performance:")
    print(f"  Total records: {n}")
    print(f"  Total possible pairs: {total_possible:,}")
    print(f"  Candidate pairs: {num_candidates:,}")
    print(f"  Reduction rate: {reduction_rate:.2%}")

    # Should reduce to <5% (target: 2-4%)
    assert reduction_rate < 0.05, f"Reduction rate {reduction_rate:.2%} is too high (>5%)"

    # Should have at least some pairs
    assert num_candidates > 0, "No candidate pairs generated"


def test_blocking_recall():
    """Test that blocking doesn't miss true matches (recall=100%)."""
    records = load_full_dataset()
    ground_truth = load_ground_truth()

    blocker = MultiBlocker([
        SoundexYearGenderBlocker(),
        NamePrefixDOBBlocker(),
        PhoneBlocker(),
        SSNYearGenderBlocker(),
        MRNBlocker(),
    ])

    pairs = blocker.generate_candidate_pairs(records)

    # Build set of candidate pair IDs
    pair_ids_set = set()
    for r1, r2 in pairs:
        id1, id2 = r1.record_id, r2.record_id
        pair_ids_set.add((min(id1, id2), max(id1, id2)))

    # Get true matches from ground truth
    # Records with same match_group should match
    match_groups = ground_truth.groupby('match_group')['record_id'].apply(list).tolist()

    missed_matches = []
    total_true_pairs = 0

    for group in match_groups:
        if len(group) < 2:
            continue

        # Generate all true pairs within this group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                id1, id2 = group[i], group[j]
                pair_key = (min(id1, id2), max(id1, id2))
                total_true_pairs += 1

                # Check if this true pair is in candidate pairs
                if pair_key not in pair_ids_set:
                    missed_matches.append((id1, id2))

    recall = (total_true_pairs - len(missed_matches)) / total_true_pairs if total_true_pairs > 0 else 1.0

    print(f"\nBlocking Recall:")
    print(f"  Total true match pairs: {total_true_pairs}")
    print(f"  Missed matches: {len(missed_matches)}")
    print(f"  Recall: {recall:.2%}")

    if missed_matches:
        print(f"\n  First 5 missed pairs: {missed_matches[:5]}")

    # Should have 100% recall (no true matches missed)
    assert recall == 1.0, f"Blocking missed {len(missed_matches)} true matches (recall={recall:.2%})"


def test_blocking_stats():
    """Test blocking statistics generation."""
    records = load_full_dataset()

    blocker = MultiBlocker([
        SoundexYearGenderBlocker(),
        NamePrefixDOBBlocker(),
        PhoneBlocker(),
        SSNYearGenderBlocker(),
        MRNBlocker(),
    ])

    stats = blocker.get_blocking_stats(records)

    print(f"\nBlocking Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Total possible pairs: {stats['total_possible_pairs']:,}")
    print(f"  Candidate pairs: {stats['candidate_pairs']:,}")
    print(f"  Reduction rate: {stats['reduction_rate']}")
    print(f"\n  Pairs per strategy:")
    for strategy, count in stats['pairs_per_strategy'].items():
        print(f"    {strategy}: {count:,}")

    # Validate stats
    assert stats['total_records'] > 0
    assert stats['total_possible_pairs'] > 0
    assert stats['candidate_pairs'] > 0
    assert stats['candidate_pairs'] < stats['total_possible_pairs']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
