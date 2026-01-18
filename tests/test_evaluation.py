"""
Tests for the evaluation module.

Tests for:
- EvaluationMetrics calculation
- MatchEvaluator ground truth loading
- Accuracy evaluation against ground truth
- Error detection and analysis
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from medmatch.evaluation.metrics import (
    EvaluationMetrics,
    MatchEvaluator,
    ErrorCase,
)
from medmatch.matching.core import MatchResult


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ground_truth(tmp_path):
    """Create a sample ground truth CSV for testing."""
    gt_content = """record_id,patient_id,match_group,notes,is_common_name,is_name_variation,is_twin,is_family_member,has_data_error,difficulty
R001,P001,G001,clean,False,False,False,False,False,easy
R002,P001,G001,variation,False,True,False,False,False,easy
R003,P002,G002,clean,False,False,False,False,False,medium
R004,P002,G002,typo,False,False,False,False,True,medium
R005,P003,G003,clean,False,False,False,False,False,hard
R006,P004,G004,clean,False,False,False,False,False,hard
R007,P005,G005,clean,False,False,True,False,False,ambiguous
R008,P005,G005,twin,False,False,True,False,False,ambiguous
"""
    gt_file = tmp_path / "ground_truth.csv"
    gt_file.write_text(gt_content)
    return str(gt_file)


@pytest.fixture
def evaluator(sample_ground_truth):
    """Create evaluator with sample ground truth."""
    return MatchEvaluator(sample_ground_truth)


def create_match_result(
    r1: str,
    r2: str,
    is_match: bool,
    confidence: float = 0.8,
    match_type: str = "probable",
    stage: str = "scoring",
) -> MatchResult:
    """Helper to create MatchResult objects."""
    return MatchResult(
        record_1_id=r1,
        record_2_id=r2,
        is_match=is_match,
        confidence=confidence,
        match_type=match_type if is_match else "no_match",
        evidence={},
        stage=stage,
    )


# ============================================================================
# EvaluationMetrics Tests
# ============================================================================

class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_precision_calculation(self):
        """Test precision = TP / (TP + FP)."""
        metrics = EvaluationMetrics(
            true_positives=80,
            true_negatives=100,
            false_positives=20,
            false_negatives=10,
            total_pairs=210,
        )
        # precision = 80 / (80 + 20) = 0.8
        assert metrics.precision == 0.8

    def test_recall_calculation(self):
        """Test recall = TP / (TP + FN)."""
        metrics = EvaluationMetrics(
            true_positives=80,
            true_negatives=100,
            false_positives=20,
            false_negatives=10,
            total_pairs=210,
        )
        # recall = 80 / (80 + 10) = 0.888...
        assert abs(metrics.recall - 0.8889) < 0.001

    def test_f1_score_calculation(self):
        """Test F1 = 2 * (P * R) / (P + R)."""
        metrics = EvaluationMetrics(
            true_positives=80,
            true_negatives=100,
            false_positives=20,
            false_negatives=10,
            total_pairs=210,
        )
        p = 0.8
        r = 80 / 90
        expected_f1 = 2 * (p * r) / (p + r)
        assert abs(metrics.f1_score - expected_f1) < 0.001

    def test_accuracy_calculation(self):
        """Test accuracy = (TP + TN) / total."""
        metrics = EvaluationMetrics(
            true_positives=80,
            true_negatives=100,
            false_positives=20,
            false_negatives=10,
            total_pairs=210,
        )
        # accuracy = (80 + 100) / 210 = 0.857...
        expected = 180 / 210
        assert abs(metrics.accuracy - expected) < 0.001

    def test_specificity_calculation(self):
        """Test specificity = TN / (TN + FP)."""
        metrics = EvaluationMetrics(
            true_positives=80,
            true_negatives=100,
            false_positives=20,
            false_negatives=10,
            total_pairs=210,
        )
        # specificity = 100 / (100 + 20) = 0.833...
        expected = 100 / 120
        assert abs(metrics.specificity - expected) < 0.001

    def test_perfect_metrics(self):
        """Test perfect classification."""
        metrics = EvaluationMetrics(
            true_positives=50,
            true_negatives=50,
            false_positives=0,
            false_negatives=0,
            total_pairs=100,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.specificity == 1.0

    def test_zero_division_handling(self):
        """Test handling when denominators would be zero."""
        # No positive predictions
        metrics = EvaluationMetrics(
            true_positives=0,
            true_negatives=100,
            false_positives=0,
            false_negatives=10,
            total_pairs=110,
        )
        assert metrics.precision == 1.0  # Convention: no FP means perfect precision
        assert metrics.recall == 0.0  # No TP but have FN

        # No actual positives
        metrics2 = EvaluationMetrics(
            true_positives=0,
            true_negatives=100,
            false_positives=10,
            false_negatives=0,
            total_pairs=110,
        )
        assert metrics2.recall == 1.0  # Convention: no FN means perfect recall

    def test_to_dict(self):
        """Test dictionary export."""
        metrics = EvaluationMetrics(
            true_positives=8,
            true_negatives=10,
            false_positives=2,
            false_negatives=1,
            total_pairs=21,
            difficulty="easy",
        )
        d = metrics.to_dict()
        assert d['difficulty'] == "easy"
        assert d['total_pairs'] == 21
        assert d['true_positives'] == 8
        assert 'precision' in d
        assert 'accuracy' in d


# ============================================================================
# MatchEvaluator Loading Tests
# ============================================================================

class TestMatchEvaluatorLoading:
    """Tests for ground truth loading."""

    def test_load_ground_truth(self, evaluator):
        """Test ground truth file loads correctly."""
        assert len(evaluator.record_data) == 8
        assert 'R001' in evaluator.record_data
        assert 'R008' in evaluator.record_data

    def test_match_groups_indexed(self, evaluator):
        """Test match groups are correctly indexed."""
        # G001 has R001 and R002 (same patient P001)
        assert 'R001' in evaluator.match_groups['G001']
        assert 'R002' in evaluator.match_groups['G001']

    def test_patient_records_indexed(self, evaluator):
        """Test patient records are correctly indexed."""
        # P001 has R001 and R002
        assert 'R001' in evaluator.patient_records['P001']
        assert 'R002' in evaluator.patient_records['P001']

    def test_difficulties_loaded(self, evaluator):
        """Test difficulty levels are loaded."""
        assert evaluator.difficulties['R001'] == 'easy'
        assert evaluator.difficulties['R003'] == 'medium'
        assert evaluator.difficulties['R005'] == 'hard'
        assert evaluator.difficulties['R007'] == 'ambiguous'

    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            MatchEvaluator("/nonexistent/path.csv")


# ============================================================================
# MatchEvaluator should_match Tests
# ============================================================================

class TestShouldMatch:
    """Tests for should_match lookup."""

    def test_should_match_same_patient(self, evaluator):
        """Test records from same patient should match."""
        assert evaluator.should_match('R001', 'R002') is True
        assert evaluator.should_match('R003', 'R004') is True
        assert evaluator.should_match('R007', 'R008') is True

    def test_should_not_match_different_patients(self, evaluator):
        """Test records from different patients should not match."""
        assert evaluator.should_match('R001', 'R003') is False
        assert evaluator.should_match('R002', 'R005') is False
        assert evaluator.should_match('R005', 'R006') is False

    def test_should_match_order_independent(self, evaluator):
        """Test should_match is order-independent."""
        assert evaluator.should_match('R001', 'R002') == evaluator.should_match('R002', 'R001')

    def test_should_match_unknown_record(self, evaluator):
        """Test error on unknown record ID."""
        with pytest.raises(KeyError):
            evaluator.should_match('R001', 'R999')


# ============================================================================
# MatchEvaluator evaluate Tests
# ============================================================================

class TestEvaluate:
    """Tests for evaluation against ground truth."""

    def test_evaluate_all_correct(self, evaluator):
        """Test perfect predictions."""
        results = [
            create_match_result('R001', 'R002', is_match=True),   # TP
            create_match_result('R001', 'R003', is_match=False),  # TN
            create_match_result('R003', 'R004', is_match=True),   # TP
            create_match_result('R005', 'R006', is_match=False),  # TN
        ]
        metrics = evaluator.evaluate(results)
        assert metrics.accuracy == 1.0
        assert metrics.true_positives == 2
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_evaluate_with_errors(self, evaluator):
        """Test evaluation with some errors."""
        results = [
            create_match_result('R001', 'R002', is_match=True),   # TP (correct)
            create_match_result('R001', 'R003', is_match=True),   # FP (wrong - different patients)
            create_match_result('R003', 'R004', is_match=False),  # FN (wrong - same patient)
            create_match_result('R005', 'R006', is_match=False),  # TN (correct)
        ]
        metrics = evaluator.evaluate(results)
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.accuracy == 0.5

    def test_evaluate_empty_results(self, evaluator):
        """Test evaluation with no results."""
        metrics = evaluator.evaluate([])
        assert metrics.total_pairs == 0
        assert metrics.accuracy == 0.0


# ============================================================================
# MatchEvaluator evaluate_by_difficulty Tests
# ============================================================================

class TestEvaluateByDifficulty:
    """Tests for difficulty-based evaluation."""

    def test_evaluate_by_difficulty(self, evaluator):
        """Test breakdown by difficulty level."""
        results = [
            create_match_result('R001', 'R002', is_match=True),   # easy, TP
            create_match_result('R003', 'R004', is_match=True),   # medium, TP
            create_match_result('R005', 'R006', is_match=False),  # hard, TN
            create_match_result('R007', 'R008', is_match=True),   # ambiguous, TP
        ]
        by_diff = evaluator.evaluate_by_difficulty(results)

        assert 'easy' in by_diff
        assert 'medium' in by_diff
        assert 'hard' in by_diff
        assert 'ambiguous' in by_diff

        assert by_diff['easy'].total_pairs == 1
        assert by_diff['easy'].accuracy == 1.0

    def test_get_pair_difficulty(self, evaluator):
        """Test difficulty lookup for pairs."""
        # Both easy
        assert evaluator.get_pair_difficulty('R001', 'R002') == 'easy'

        # Both medium
        assert evaluator.get_pair_difficulty('R003', 'R004') == 'medium'

        # Mix: easy + medium -> medium (harder wins)
        difficulty = evaluator.get_pair_difficulty('R001', 'R003')
        assert difficulty in ['easy', 'medium']  # Either based on implementation


# ============================================================================
# MatchEvaluator find_errors Tests
# ============================================================================

class TestFindErrors:
    """Tests for error detection."""

    def test_find_false_positives(self, evaluator):
        """Test detection of false positives."""
        results = [
            create_match_result('R001', 'R003', is_match=True),  # FP - different patients
        ]
        errors = evaluator.find_errors(results)
        assert len(errors) == 1
        assert errors[0].error_type == 'false_positive'
        assert errors[0].record_1_id == 'R001'
        assert errors[0].record_2_id == 'R003'

    def test_find_false_negatives(self, evaluator):
        """Test detection of false negatives."""
        results = [
            create_match_result('R001', 'R002', is_match=False),  # FN - same patient
        ]
        errors = evaluator.find_errors(results)
        assert len(errors) == 1
        assert errors[0].error_type == 'false_negative'

    def test_no_errors_on_correct(self, evaluator):
        """Test no errors returned for correct predictions."""
        results = [
            create_match_result('R001', 'R002', is_match=True),   # TP
            create_match_result('R001', 'R003', is_match=False),  # TN
        ]
        errors = evaluator.find_errors(results)
        assert len(errors) == 0

    def test_error_case_details(self, evaluator):
        """Test ErrorCase contains all details."""
        results = [
            create_match_result(
                'R001', 'R003',
                is_match=True,
                confidence=0.75,
                stage='scoring',
            ),
        ]
        errors = evaluator.find_errors(results)
        assert len(errors) == 1

        error = errors[0]
        assert error.confidence == 0.75
        assert error.stage == 'scoring'
        assert error.predicted is True
        assert error.actual is False


# ============================================================================
# MatchEvaluator get_all_true_match_pairs Tests
# ============================================================================

class TestGetAllTrueMatchPairs:
    """Tests for true match pair extraction."""

    def test_get_all_true_match_pairs(self, evaluator):
        """Test extraction of all true match pairs."""
        pairs = evaluator.get_all_true_match_pairs()

        # P001: R001, R002 -> (R001, R002)
        # P002: R003, R004 -> (R003, R004)
        # P005: R007, R008 -> (R007, R008)
        # P003 and P004 have single records, no pairs

        assert len(pairs) == 3
        assert ('R001', 'R002') in pairs
        assert ('R003', 'R004') in pairs
        assert ('R007', 'R008') in pairs


# ============================================================================
# Report Generation Tests
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report(self, evaluator):
        """Test report generation produces output."""
        results = [
            create_match_result('R001', 'R002', is_match=True),
            create_match_result('R003', 'R004', is_match=True),
        ]
        report = evaluator.generate_report(results)

        assert "EVALUATION REPORT" in report
        assert "OVERALL METRICS" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "Accuracy" in report

    def test_generate_report_verbose(self, evaluator):
        """Test verbose report includes errors."""
        results = [
            create_match_result('R001', 'R003', is_match=True),  # FP
        ]
        report = evaluator.generate_report(results, verbose=True)

        assert "ERROR ANALYSIS" in report
        assert "False Positives" in report

    def test_get_summary_stats(self, evaluator):
        """Test summary statistics dictionary."""
        results = [
            create_match_result('R001', 'R002', is_match=True),
            create_match_result('R003', 'R004', is_match=True),
        ]
        stats = evaluator.get_summary_stats(results)

        assert 'overall' in stats
        assert 'by_difficulty' in stats
        assert 'by_stage' in stats
        assert 'error_count' in stats
        assert 'targets' in stats

        assert stats['overall']['total_pairs'] == 2
        assert stats['overall']['accuracy'] == 1.0


# ============================================================================
# Integration with Real Ground Truth
# ============================================================================

class TestWithRealGroundTruth:
    """Tests using the actual project ground truth file."""

    @pytest.fixture
    def real_evaluator(self):
        """Load real ground truth if available."""
        gt_path = Path(__file__).parent.parent / "data" / "synthetic" / "ground_truth.csv"
        if not gt_path.exists():
            pytest.skip("Real ground truth file not available")
        return MatchEvaluator(str(gt_path))

    def test_real_ground_truth_loads(self, real_evaluator):
        """Test real ground truth file loads."""
        assert len(real_evaluator.record_data) > 0

    def test_real_match_groups(self, real_evaluator):
        """Test real match groups are indexed."""
        assert len(real_evaluator.match_groups) > 0

    def test_real_true_match_pairs(self, real_evaluator):
        """Test extraction of real true match pairs."""
        pairs = real_evaluator.get_all_true_match_pairs()
        assert len(pairs) > 0
        # Each pair should be a tuple of two record IDs
        for p in list(pairs)[:5]:
            assert len(p) == 2
            assert p[0] < p[1]  # Ordered
