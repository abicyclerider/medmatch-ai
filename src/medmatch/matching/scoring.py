"""
Weighted confidence scoring for patient matching.

This module calculates weighted confidence scores from feature vectors and
classifies matches using configurable thresholds. Designed for medium difficulty
cases where rules don't fire but demographic evidence is strong.

Key Classes:
    ScoringWeights: Configurable weights for each feature (sum to 1.0)
    ConfidenceScorer: Calculates confidence scores and classifies matches
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from .features import FeatureVector


@dataclass
class ScoringWeights:
    """
    Configurable weights for confidence scoring.

    All weights must sum to 1.0 for proper confidence calculation.
    Default weights are based on empirical importance for patient matching.

    Attributes:
        name_first: Weight for first name similarity (default: 0.15)
        name_last: Weight for last name similarity (default: 0.20)
        name_middle: Weight for middle name similarity (default: 0.05)
        dob: Weight for date of birth match (default: 0.30, highest single field)
        phone: Weight for phone number match (default: 0.08)
        email: Weight for email match (default: 0.07)
        address: Weight for address similarity (default: 0.05)
        mrn: Weight for MRN exact match (default: 0.05)
        ssn: Weight for SSN match (default: 0.05)
    """

    # Name weights (total: 0.40)
    name_first: float = 0.15
    name_last: float = 0.20
    name_middle: float = 0.05

    # Date weight (highest single field)
    dob: float = 0.30

    # Contact weights (total: 0.20)
    phone: float = 0.08
    email: float = 0.07
    address: float = 0.05

    # Identifier weights (total: 0.10)
    mrn: float = 0.05
    ssn: float = 0.05

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.name_first + self.name_last + self.name_middle +
            self.dob +
            self.phone + self.email + self.address +
            self.mrn + self.ssn
        )

        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"Breakdown: name={self.name_first + self.name_last + self.name_middle:.2f}, "
                f"dob={self.dob:.2f}, contact={self.phone + self.email + self.address:.2f}, "
                f"identifiers={self.mrn + self.ssn:.2f}"
            )

    def to_dict(self) -> dict:
        """
        Convert weights to dictionary.

        Returns:
            Dictionary with weight names and values

        Example:
            >>> weights = ScoringWeights()
            >>> weights.to_dict()
            {'name_first': 0.15, 'name_last': 0.20, ...}
        """
        return {
            'name_first': self.name_first,
            'name_last': self.name_last,
            'name_middle': self.name_middle,
            'dob': self.dob,
            'phone': self.phone,
            'email': self.email,
            'address': self.address,
            'mrn': self.mrn,
            'ssn': self.ssn,
        }


class ConfidenceScorer:
    """
    Calculate confidence scores from feature vectors.

    Uses weighted scoring with configurable thresholds to classify matches.
    Handles missing features gracefully by redistributing weights.

    Example:
        >>> scorer = ConfidenceScorer()
        >>> features = FeatureVector(
        ...     name_first_score=0.95,
        ...     name_last_score=1.0,
        ...     dob_score=1.0,
        ...     phone_score=1.0,
        ... )
        >>> score, breakdown = scorer.score(features)
        >>> is_match, match_type = scorer.classify(score)
        >>> print(f"{match_type}: {score:.2f}")
        definite_match: 0.94
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        threshold_definite: float = 0.90,
        threshold_probable: float = 0.80,
        threshold_possible: float = 0.65,
    ):
        """
        Initialize confidence scorer.

        Args:
            weights: Custom scoring weights (uses defaults if None)
            threshold_definite: Minimum score for definite match (default: 0.90)
            threshold_probable: Minimum score for probable match (default: 0.80)
            threshold_possible: Minimum score for possible match (default: 0.65)

        Example:
            >>> # Use default weights and thresholds
            >>> scorer = ConfidenceScorer()
            >>>
            >>> # Custom thresholds (more conservative)
            >>> scorer = ConfidenceScorer(threshold_definite=0.95, threshold_probable=0.85)
            >>>
            >>> # Custom weights
            >>> custom_weights = ScoringWeights(dob=0.40, name_last=0.25)
            >>> scorer = ConfidenceScorer(weights=custom_weights)
        """
        self.weights = weights or ScoringWeights()
        self.threshold_definite = threshold_definite
        self.threshold_probable = threshold_probable
        self.threshold_possible = threshold_possible

        # Validate thresholds
        if not (0.0 <= threshold_possible < threshold_probable < threshold_definite <= 1.0):
            raise ValueError(
                f"Thresholds must be ordered: 0 <= possible < probable < definite <= 1. "
                f"Got: possible={threshold_possible}, probable={threshold_probable}, definite={threshold_definite}"
            )

    def score(
        self,
        features: FeatureVector,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted confidence score from features.

        Handles missing features by redistributing their weights proportionally
        among available features. This ensures scores remain in [0.0, 1.0] range.

        Args:
            features: Extracted feature vector

        Returns:
            Tuple of (total_score, breakdown_dict)
            - total_score: Weighted confidence in [0.0, 1.0]
            - breakdown_dict: Individual feature contributions for explainability

        Example:
            >>> features = FeatureVector(
            ...     name_first_score=0.95,
            ...     name_last_score=1.0,
            ...     dob_score=1.0,
            ...     phone_score=None,  # Missing
            ... )
            >>> score, breakdown = scorer.score(features)
            >>> score
            0.943
            >>> breakdown['name_last']
            0.20
            >>> breakdown['phone']
            0.0  # Missing, weight redistributed
        """
        breakdown = {}
        total_score = 0.0
        total_weight_available = 0.0
        total_weight_missing = 0.0

        # Calculate contributions from available features
        feature_weight_pairs = [
            (features.name_first_score, self.weights.name_first, 'name_first'),
            (features.name_last_score, self.weights.name_last, 'name_last'),
            (features.name_middle_score, self.weights.name_middle, 'name_middle'),
            (features.dob_score, self.weights.dob, 'dob'),
            (features.phone_score, self.weights.phone, 'phone'),
            (features.email_score, self.weights.email, 'email'),
            (features.address_score, self.weights.address, 'address'),
        ]

        # Handle boolean features (MRN, SSN) - convert to 1.0 or 0.0
        if features.mrn_match:
            feature_weight_pairs.append((1.0, self.weights.mrn, 'mrn'))
        else:
            feature_weight_pairs.append((0.0, self.weights.mrn, 'mrn'))

        if features.ssn_match:
            feature_weight_pairs.append((1.0, self.weights.ssn, 'ssn'))
        else:
            feature_weight_pairs.append((0.0, self.weights.ssn, 'ssn'))

        # First pass: identify available vs missing features
        for score, weight, name in feature_weight_pairs:
            if score is not None:
                total_weight_available += weight
            else:
                total_weight_missing += weight

        # Second pass: calculate weighted scores
        # If feature is missing, redistribute its weight among available features
        for score, weight, name in feature_weight_pairs:
            if score is not None:
                # Adjust weight if some features are missing
                if total_weight_missing > 0 and total_weight_available > 0:
                    # Redistribute missing weight proportionally
                    adjusted_weight = weight * (1.0 / total_weight_available)
                else:
                    adjusted_weight = weight

                contribution = score * adjusted_weight
                breakdown[name] = contribution
                total_score += contribution
            else:
                # Missing feature contributes 0
                breakdown[name] = 0.0

        return total_score, breakdown

    def classify(
        self,
        score: float,
    ) -> Tuple[bool, str]:
        """
        Classify match based on confidence score.

        Args:
            score: Confidence score in [0.0, 1.0]

        Returns:
            Tuple of (is_match, match_type)
            - is_match: Boolean match decision
            - match_type: Match classification string

        Match types:
            - "definite_match": score >= threshold_definite (default: 0.90)
            - "probable_match": score >= threshold_probable (default: 0.80)
            - "possible_match": score >= threshold_possible (default: 0.65)
            - "unlikely_match": score < threshold_possible
            - "no_match": score < 0.50 (clear non-match)

        Example:
            >>> scorer.classify(0.95)
            (True, 'definite_match')
            >>> scorer.classify(0.85)
            (True, 'probable_match')
            >>> scorer.classify(0.70)
            (True, 'possible_match')
            >>> scorer.classify(0.60)
            (False, 'unlikely_match')
            >>> scorer.classify(0.30)
            (False, 'no_match')
        """
        if score >= self.threshold_definite:
            return True, "definite_match"
        elif score >= self.threshold_probable:
            return True, "probable_match"
        elif score >= self.threshold_possible:
            return True, "possible_match"
        elif score >= 0.50:
            return False, "unlikely_match"
        else:
            return False, "no_match"

    def explain_score(
        self,
        score: float,
        breakdown: Dict[str, float],
        features: FeatureVector,
    ) -> str:
        """
        Generate human-readable explanation of score.

        Args:
            score: Total confidence score
            breakdown: Feature contribution breakdown
            features: Original feature vector (for methods)

        Returns:
            Formatted explanation string

        Example:
            >>> explanation = scorer.explain_score(0.94, breakdown, features)
            >>> print(explanation)
            Confidence Score: 0.94 (definite_match)

            Top Contributing Features:
            - dob: 0.30 (exact_match)
            - name_last: 0.20 (exact_match)
            - name_first: 0.14 (known_variation)
            - phone: 0.08 (normalized_match)

            Missing Features:
            - email: not available
        """
        is_match, match_type = self.classify(score)

        lines = [
            f"Confidence Score: {score:.2f} ({match_type})",
            "",
            "Top Contributing Features:",
        ]

        # Sort by contribution (highest first)
        sorted_features = sorted(breakdown.items(), key=lambda x: -x[1])

        # Show top 5 contributing features
        for name, contribution in sorted_features[:5]:
            if contribution > 0:
                # Get method if available
                method = None
                if name == 'name_first':
                    method = features.name_first_method
                elif name == 'name_last':
                    method = features.name_last_method
                elif name == 'dob':
                    method = features.dob_method
                elif name == 'address':
                    method = features.address_method

                if method:
                    lines.append(f"- {name}: {contribution:.2f} ({method})")
                else:
                    lines.append(f"- {name}: {contribution:.2f}")

        # Show missing features
        missing = [name for name, contrib in breakdown.items() if contrib == 0]
        if missing:
            lines.append("")
            lines.append("Missing Features:")
            for name in missing[:3]:  # Show up to 3 missing
                lines.append(f"- {name}: not available")

        return "\n".join(lines)
