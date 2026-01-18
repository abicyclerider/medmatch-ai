"""
Tests for AI-powered medical fingerprint matching (Phase 2.4).

These tests verify:
1. MedicalFingerprintMatcher initialization
2. Rate limiter functionality
3. AI understanding of medical abbreviations
4. Response parsing robustness
5. Accuracy on hard/ambiguous cases (target 70%+)

Tests that require API calls are marked with @pytest.mark.api
and can be skipped with: pytest -m "not api"
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from medmatch.matching.medical_fingerprint import (
    MedicalFingerprintMatcher,
    RateLimiter,
)
from datetime import datetime
from medmatch.matching.core import PatientRecord
from medmatch.data.models.patient import (
    Demographics,
    MedicalRecord,
    MedicalHistory,
    MedicalCondition,
    Address,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_genai():
    """Mock the google.genai module for testing without API calls."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Client.return_value = mock_client
    return mock_module, mock_client


def create_patient_record(
    record_id: str,
    first_name: str,
    last_name: str,
    dob: date,
    gender: str = "M",
    conditions: list = None,
    medications: list = None,
) -> PatientRecord:
    """Helper to create PatientRecord for testing."""
    demo = Demographics(
        record_id=record_id,
        patient_id="P001",
        name_first=first_name,
        name_last=last_name,
        date_of_birth=dob,
        gender=gender,
        mrn="MRN12345",
        record_source="test",
        record_date=date.today(),
    )

    # Create medical history if conditions provided
    medical = None
    if conditions or medications:
        cond_objs = [
            MedicalCondition(name=c, onset_year=2020)
            for c in (conditions or [])
        ]
        med_history = MedicalHistory(
            conditions=cond_objs,
            medications=medications or [],
            surgeries=[],
            allergies=[],
        )
        medical = MedicalRecord(
            record_id=record_id,
            patient_id="P001",
            record_source="test",
            record_date=datetime.now(),
            medical_history=med_history,
        )

    return PatientRecord.from_demographics(demo, medical)


# ============================================================================
# Rate Limiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_disabled(self):
        """Test that rate limiter with 0 does nothing."""
        limiter = RateLimiter(requests_per_minute=0)

        import time
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be nearly instant

    def test_rate_limiter_enabled(self):
        """Test that rate limiter enforces delay."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 per second

        import time
        # First call should be instant
        limiter.wait_if_needed()

        # Second call should wait
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start

        # Should have waited ~1 second (allow 0.9-1.2 tolerance)
        assert elapsed >= 0.9


# ============================================================================
# MedicalFingerprintMatcher Initialization Tests
# ============================================================================

class TestMedicalFingerprintMatcherInit:
    """Tests for matcher initialization."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        # Clear env and use empty key parameter
        with patch.dict(os.environ, {'GOOGLE_AI_API_KEY': ''}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_AI_API_KEY"):
                MedicalFingerprintMatcher(api_key="")

    def test_init_with_api_key_parameter(self, mock_genai):
        """Test initialization with explicit API key."""
        mock_module, mock_client = mock_genai

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            matcher = MedicalFingerprintMatcher(api_key="test_key_12345")

            assert matcher.api_key == "test_key_12345"
            assert matcher.model == "gemini-2.5-flash"


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestResponseParsing:
    """Tests for AI response parsing robustness."""

    @pytest.fixture
    def matcher(self, mock_genai):
        """Create matcher with mocked API."""
        mock_module, mock_client = mock_genai

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            return MedicalFingerprintMatcher(api_key="test_key")

    def test_parse_standard_response(self, matcher):
        """Test parsing standard formatted response."""
        response = """SIMILARITY_SCORE: 0.85
REASONING: Both records show hypertension and diabetes with matching medications."""

        score, reasoning = matcher._parse_response(response)

        assert score == 0.85
        assert "hypertension" in reasoning.lower()

    def test_parse_percentage_format(self, matcher):
        """Test parsing percentage format (85%)."""
        response = """SIMILARITY_SCORE: 85%
REASONING: Very similar medical profiles."""

        score, reasoning = matcher._parse_response(response)

        assert score == 0.85
        assert "similar" in reasoning.lower()

    def test_parse_high_score(self, matcher):
        """Test parsing score above 100 (percentage) gets converted and clamped."""
        response = """SIMILARITY_SCORE: 150%
REASONING: Identical histories."""

        score, reasoning = matcher._parse_response(response)

        assert score == 1.0  # Clamped after percentage conversion

    def test_parse_negative_score(self, matcher):
        """Test parsing negative score gets clamped to 0."""
        response = """SIMILARITY_SCORE: -0.5
REASONING: No match."""

        score, reasoning = matcher._parse_response(response)

        assert score == 0.0  # Clamped

    def test_parse_multiline_reasoning(self, matcher):
        """Test parsing reasoning that spans multiple lines."""
        response = """SIMILARITY_SCORE: 0.7
REASONING: The medical histories show significant overlap.
Both patients have cardiovascular conditions and take similar medications."""

        score, reasoning = matcher._parse_response(response)

        assert score == 0.7
        assert len(reasoning) > 20  # Should capture multi-line

    def test_parse_malformed_response(self, matcher):
        """Test parsing malformed response returns defaults."""
        response = "This is not a properly formatted response at all."

        score, reasoning = matcher._parse_response(response)

        # Should return safe defaults
        assert 0.0 <= score <= 1.0


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    """Tests for prompt construction."""

    @pytest.fixture
    def matcher(self, mock_genai):
        """Create matcher with mocked API."""
        mock_module, mock_client = mock_genai

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            return MedicalFingerprintMatcher(api_key="test_key")

    def test_prompt_includes_names(self, matcher):
        """Test that prompt includes patient names."""
        prompt = matcher._build_comparison_prompt(
            "John Smith", "Conditions: Diabetes",
            "J. Smith", "Conditions: T2DM",
        )

        assert "John Smith" in prompt
        assert "J. Smith" in prompt

    def test_prompt_includes_medical_signatures(self, matcher):
        """Test that prompt includes medical signatures."""
        prompt = matcher._build_comparison_prompt(
            "John Smith", "Conditions: Hypertension; Medications: Lisinopril",
            "J. Smith", "Conditions: HTN; Medications: Lisinopril 10mg",
        )

        assert "Hypertension" in prompt
        assert "HTN" in prompt
        assert "Lisinopril" in prompt

    def test_prompt_includes_abbreviation_guide(self, matcher):
        """Test that prompt includes medical abbreviation explanations."""
        prompt = matcher._build_comparison_prompt(
            "Test", "test", "Test", "test"
        )

        # Should include common abbreviations
        assert "T2DM" in prompt
        assert "HTN" in prompt
        assert "MI" in prompt


# ============================================================================
# Medical History Comparison Tests (Mocked)
# ============================================================================

class TestMedicalHistoryComparison:
    """Tests for medical history comparison logic."""

    def test_both_records_no_history(self, mock_genai):
        """Test comparison when neither record has medical history."""
        mock_module, mock_client = mock_genai

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            matcher = MedicalFingerprintMatcher(api_key="test_key")

            r1 = create_patient_record("R001", "John", "Smith", date(1980, 1, 1))
            r2 = create_patient_record("R002", "John", "Smith", date(1980, 1, 1))

            score, reasoning = matcher.compare_medical_histories(r1, r2)

            # Should return 0.5 (uncertain) when no history
            assert score == 0.5
            assert "medical history" in reasoning.lower()

    def test_one_record_no_history(self, mock_genai):
        """Test comparison when one record has no medical history."""
        mock_module, mock_client = mock_genai

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            matcher = MedicalFingerprintMatcher(api_key="test_key")

            r1 = create_patient_record(
                "R001", "John", "Smith", date(1980, 1, 1),
                conditions=["Diabetes"],
            )
            r2 = create_patient_record("R002", "John", "Smith", date(1980, 1, 1))

            score, reasoning = matcher.compare_medical_histories(r1, r2)

            assert score == 0.5
            assert "medical history" in reasoning.lower()

    def test_api_call_made_with_histories(self, mock_genai):
        """Test that API is called when both records have history."""
        mock_module, mock_client = mock_genai

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = """SIMILARITY_SCORE: 0.9
REASONING: Both have diabetes and hypertension with matching medications."""
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            matcher = MedicalFingerprintMatcher(api_key="test_key")
            matcher.client = mock_client

            r1 = create_patient_record(
                "R001", "John", "Smith", date(1980, 1, 1),
                conditions=["Diabetes", "Hypertension"],
                medications=["Metformin 500mg", "Lisinopril 10mg"],
            )
            r2 = create_patient_record(
                "R002", "John", "Smith", date(1980, 1, 1),
                conditions=["T2DM", "HTN"],
                medications=["Metformin 500mg BID", "Lisinopril"],
            )

            score, reasoning = matcher.compare_medical_histories(r1, r2)

            # API should have been called
            assert mock_client.models.generate_content.called
            assert score == 0.9
            assert "diabetes" in reasoning.lower() or "matching" in reasoning.lower()

    def test_api_error_returns_fallback(self, mock_genai):
        """Test that API errors return graceful fallback."""
        mock_module, mock_client = mock_genai

        # Mock API error
        mock_client.models.generate_content.side_effect = Exception("API Error")

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            matcher = MedicalFingerprintMatcher(api_key="test_key")
            matcher.client = mock_client

            r1 = create_patient_record(
                "R001", "John", "Smith", date(1980, 1, 1),
                conditions=["Diabetes"],
            )
            r2 = create_patient_record(
                "R002", "John", "Smith", date(1980, 1, 1),
                conditions=["Diabetes"],
            )

            score, reasoning = matcher.compare_medical_histories(r1, r2)

            # Should return 0.0 with error message
            assert score == 0.0
            assert "error" in reasoning.lower() or "API" in reasoning


# ============================================================================
# Live API Tests (marked for optional skipping)
# ============================================================================

@pytest.mark.api
class TestLiveAPIComparison:
    """
    Live API tests that require a valid GOOGLE_AI_API_KEY.

    Run with: pytest -m api tests/test_medical_fingerprint.py
    Skip with: pytest -m "not api"
    """

    @pytest.fixture
    def live_matcher(self):
        """Create matcher with real API (if key available)."""
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_AI_API_KEY not set")

        return MedicalFingerprintMatcher(api_rate_limit=0)

    def test_ai_understands_abbreviations(self, live_matcher):
        """Test that AI correctly matches T2DM = Type 2 Diabetes."""
        r1 = create_patient_record(
            "R001", "John", "Smith", date(1980, 1, 1),
            conditions=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 500mg", "Lisinopril 10mg"],
        )
        r2 = create_patient_record(
            "R002", "J", "Smith", date(1980, 1, 1),
            conditions=["T2DM", "HTN"],
            medications=["Metformin 500mg BID", "Lisinopril 10mg daily"],
        )

        score, reasoning = live_matcher.compare_medical_histories(r1, r2)

        # Should recognize these are equivalent
        assert score >= 0.7, f"Expected high similarity, got {score}: {reasoning}"

    def test_ai_recognizes_medication_condition_link(self, live_matcher):
        """Test that AI links medications to conditions."""
        # Record 1: Explicit condition
        r1 = create_patient_record(
            "R001", "Jane", "Doe", date(1975, 5, 15),
            conditions=["Hypertension"],
            medications=["Amlodipine 5mg"],
        )
        # Record 2: Only medication (implies HTN)
        r2 = create_patient_record(
            "R002", "Jane", "Doe", date(1975, 5, 15),
            conditions=[],
            medications=["Amlodipine 5mg daily", "HCTZ 12.5mg"],
        )

        score, reasoning = live_matcher.compare_medical_histories(r1, r2)

        # Should recognize medication implies same condition
        assert score >= 0.5, f"Expected moderate similarity, got {score}: {reasoning}"

    def test_ai_different_profiles_low_score(self, live_matcher):
        """Test that truly different medical profiles get low scores."""
        r1 = create_patient_record(
            "R001", "Bob", "Johnson", date(1990, 8, 20),
            conditions=["Asthma", "Allergic Rhinitis"],
            medications=["Albuterol inhaler PRN", "Fluticasone nasal spray"],
        )
        r2 = create_patient_record(
            "R002", "Robert", "Johnson", date(1990, 8, 20),
            conditions=["Type 1 Diabetes", "Hypothyroidism"],
            medications=["Insulin Lantus 20 units", "Levothyroxine 75mcg"],
        )

        score, reasoning = live_matcher.compare_medical_histories(r1, r2)

        # Should recognize these are different profiles
        assert score <= 0.5, f"Expected low similarity, got {score}: {reasoning}"


# ============================================================================
# Integration with PatientMatcher Tests (Mocked)
# ============================================================================

class TestMatcherIntegration:
    """Tests for integration with PatientMatcher."""

    def test_matcher_with_ai_disabled(self):
        """Test matcher works with AI disabled."""
        from medmatch.matching import PatientMatcher

        matcher = PatientMatcher(
            use_blocking=False,
            use_rules=True,
            use_scoring=True,
            use_ai=False,
        )

        r1 = create_patient_record("R001", "John", "Smith", date(1980, 1, 1))
        r2 = create_patient_record("R002", "John", "Smith", date(1980, 1, 1))

        result = matcher.match_pair(r1, r2)

        # Should work without AI
        assert result.stage in ["rules", "scoring"]

    def test_matcher_with_ai_enabled(self, mock_genai):
        """Test matcher initializes with AI enabled."""
        mock_module, mock_client = mock_genai

        # Mock the API response for when AI is used
        mock_response = MagicMock()
        mock_response.text = """SIMILARITY_SCORE: 0.85
REASONING: Similar medical profiles."""
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict('sys.modules', {'google.genai': mock_module}):
            with patch.dict(os.environ, {'GOOGLE_AI_API_KEY': 'test_key'}):
                from medmatch.matching import PatientMatcher

                matcher = PatientMatcher(
                    use_blocking=False,
                    use_rules=True,
                    use_scoring=True,
                    use_ai=True,
                )

                # Should have initialized medical_matcher
                assert hasattr(matcher, 'medical_matcher')


# ============================================================================
# Accuracy on Hard/Ambiguous Cases (Live API)
# ============================================================================

@pytest.mark.api
@pytest.mark.slow
class TestAccuracyOnHardCases:
    """
    Test accuracy on hard/ambiguous difficulty cases.

    Target: 70%+ accuracy on hard/ambiguous cases.

    These tests require API access and are slow.
    Run with: pytest -m "api and slow" tests/test_medical_fingerprint.py
    """

    @pytest.fixture
    def test_records_and_ground_truth(self):
        """Load test data for accuracy evaluation."""
        import pandas as pd
        import json

        # Load demographics
        demo_path = "data/synthetic/synthetic_demographics.csv"
        if not os.path.exists(demo_path):
            pytest.skip("Synthetic data not generated")

        demo_df = pd.read_csv(demo_path)

        # Load medical records
        med_path = "data/synthetic/synthetic_medical_records.json"
        if os.path.exists(med_path):
            with open(med_path) as f:
                med_records = {r['record_id']: r for r in json.load(f)}
        else:
            med_records = {}

        # Load ground truth
        gt_path = "data/synthetic/ground_truth.csv"
        gt_df = pd.read_csv(gt_path)

        # Filter to hard/ambiguous cases
        hard_cases = gt_df[gt_df['difficulty'].isin(['hard', 'ambiguous'])]

        return demo_df, med_records, hard_cases

    def test_ai_accuracy_on_hard_cases(self, test_records_and_ground_truth):
        """Test AI achieves 70%+ accuracy on hard/ambiguous cases."""
        from medmatch.matching import PatientMatcher, PatientRecord
        from medmatch.data.models.patient import (
            Demographics, MedicalRecord, MedicalHistory, MedicalCondition, Address
        )

        demo_df, med_records, hard_cases = test_records_and_ground_truth

        if len(hard_cases) == 0:
            pytest.skip("No hard/ambiguous cases in test data")

        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_AI_API_KEY not set")

        # Create matcher with AI
        matcher = PatientMatcher(
            use_blocking=False,
            use_rules=True,
            use_scoring=True,
            use_ai=True,
            api_rate_limit=0,  # No rate limiting
        )

        # Convert demographics to PatientRecord
        def demo_to_record(row):
            addr = None
            if pd.notna(row.get('address_street')):
                addr = Address(
                    street=row.get('address_street', ''),
                    city=row.get('address_city', ''),
                    state=row.get('address_state', ''),
                    zip_code=row.get('address_zip', ''),
                )

            demo = Demographics(
                record_id=row['record_id'],
                patient_id=row['patient_id'],
                name_first=row['name_first'],
                name_middle=row.get('name_middle') if pd.notna(row.get('name_middle')) else None,
                name_last=row['name_last'],
                name_suffix=row.get('name_suffix') if pd.notna(row.get('name_suffix')) else None,
                date_of_birth=pd.to_datetime(row['date_of_birth']).date(),
                gender=row['gender'],
                mrn=row.get('mrn', ''),
                ssn_last4=row.get('ssn_last4') if pd.notna(row.get('ssn_last4')) else None,
                phone=row.get('phone') if pd.notna(row.get('phone')) else None,
                email=row.get('email') if pd.notna(row.get('email')) else None,
                address=addr,
                record_source=row.get('record_source', 'test'),
                record_date=date.today(),
                data_quality_flag=row.get('data_quality_flag'),
            )

            # Get medical record if available
            medical = None
            record_id = row['record_id']
            if record_id in med_records:
                mr = med_records[record_id]
                mh = mr.get('medical_history', {})
                conditions = [
                    MedicalCondition(name=c['name'], onset_year=c.get('onset_year'))
                    for c in mh.get('conditions', [])
                ]
                medical = MedicalRecord(
                    record_id=record_id,
                    patient_id=row['patient_id'],
                    medical_history=MedicalHistory(
                        conditions=conditions,
                        medications=mh.get('medications', []),
                        surgeries=[],
                        allergies=mh.get('allergies', []),
                    ),
                )

            return PatientRecord.from_demographics(demo, medical)

        # Build record lookup
        records = {}
        for _, row in demo_df.iterrows():
            records[row['record_id']] = demo_to_record(row)

        # Evaluate on hard cases (limit to 20 for speed)
        sample_cases = hard_cases.head(20)
        correct = 0
        total = 0

        for _, case in sample_cases.iterrows():
            r1_id = case['record_1_id']
            r2_id = case['record_2_id']

            if r1_id not in records or r2_id not in records:
                continue

            result = matcher.match_pair(records[r1_id], records[r2_id])
            expected_match = case['is_match']

            if result.is_match == expected_match:
                correct += 1
            total += 1

        if total == 0:
            pytest.skip("No valid test pairs found")

        accuracy = correct / total
        print(f"\nAI accuracy on hard/ambiguous cases: {accuracy:.2%} ({correct}/{total})")

        # Target: 70%
        assert accuracy >= 0.70, f"AI accuracy {accuracy:.2%} below 70% target"
