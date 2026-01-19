"""
Main patient matching orchestrator.

This module provides the PatientMatcher class, which orchestrates the
complete matching pipeline:

Phase 2.2: Blocking + Rules
Phase 2.3: Add Feature Scoring
Phase 2.4: Add AI Medical Fingerprinting

The matcher applies stages progressively:
1. Blocking: Reduce O(n²) comparisons to candidate pairs
2. Rules: Deterministic matching for clear cases
3. Scoring: Weighted confidence for medium difficulty (Phase 2.3)
4. AI: Medical fingerprinting for hard cases (Phase 2.4)

Each stage only runs if previous stages didn't make a decision.
"""

from typing import List, Optional
from tqdm import tqdm

from .core import PatientRecord, MatchResult
from .blocking import (
    MultiBlocker,
    SoundexYearGenderBlocker,
    NamePrefixDOBBlocker,
    PhoneBlocker,
    SSNYearGenderBlocker,
    MRNBlocker,
)
from .rules import RuleEngine
from .features import FeatureExtractor
from .scoring import ConfidenceScorer, ScoringWeights
from .medical_fingerprint import MedicalFingerprintMatcher


class PatientMatcher:
    """
    Main patient matching orchestrator.

    Coordinates the matching pipeline across all stages:
    - Blocking: Reduces candidate pairs
    - Rules: Fast deterministic matching
    - Scoring: Weighted confidence (Phase 2.3)
    - AI: Medical fingerprinting (Phase 2.4)

    Example:
        >>> matcher = PatientMatcher(
        ...     use_blocking=True,
        ...     use_rules=True,
        ...     confidence_threshold=0.85,
        ... )
        >>> records = load_patient_records()
        >>> results = matcher.match_datasets(records)
        >>> print(f"Found {len(results)} candidate pairs")
        >>> print(f"Matches: {sum(r.is_match for r in results)}")
    """

    def __init__(
        self,
        use_blocking: bool = True,
        use_rules: bool = True,
        use_scoring: bool = False,  # Phase 2.3
        use_ai: bool = False,       # Phase 2.4
        confidence_threshold: float = 0.85,
        scoring_weights: Optional[any] = None,  # Phase 2.3
        scoring_thresholds: Optional[dict] = None,  # Phase 2.3
        ai_backend: str = "gemini",  # Phase 2.4: Backend ("gemini" or "ollama")
        ai_client: Optional[any] = None,  # Phase 2.4: Pre-configured AI client
        api_rate_limit: int = 0,  # Phase 2.4 (0 = no rate limiting with billing)
        **ai_kwargs,  # Phase 2.4: Passed to AI client (e.g., model, device)
    ):
        """
        Initialize patient matcher.

        Args:
            use_blocking: Enable blocking to reduce candidate pairs
            use_rules: Enable deterministic matching rules
            use_scoring: Enable feature-based scoring (Phase 2.3)
            use_ai: Enable AI medical fingerprinting (Phase 2.4)
            confidence_threshold: Minimum confidence for match decision (0.0-1.0)
            scoring_weights: Custom scoring weights (Phase 2.3)
            scoring_thresholds: Custom scoring thresholds (Phase 2.3)
            ai_backend: AI backend ("gemini" or "ollama", default: "gemini")
                - "gemini": Cloud API (fast, requires internet)
                - "ollama": Local MedGemma (HIPAA-compliant, recommended for production)
            ai_client: Pre-configured AI client (if None, creates one)
            api_rate_limit: API requests per minute (0=unlimited, Phase 2.4)
            **ai_kwargs: Passed to AI client (e.g., model, temperature, timeout)

        Example:
            >>> # Use Gemini (default, for development/testing)
            >>> matcher = PatientMatcher(use_ai=True)

            >>> # Use Ollama with MedGemma (recommended for production)
            >>> matcher = PatientMatcher(use_ai=True, ai_backend="ollama")

            >>> # Ollama with custom config
            >>> matcher = PatientMatcher(
            ...     use_ai=True,
            ...     ai_backend="ollama",
            ...     model="medgemma:1.5-4b",
            ...     temperature=0.3,
            ... )
        """
        self.use_blocking = use_blocking
        self.use_rules = use_rules
        self.use_scoring = use_scoring
        self.use_ai = use_ai
        self.confidence_threshold = confidence_threshold

        # Initialize blocking (Phase 2.2)
        if use_blocking:
            self.blocker = MultiBlocker([
                SoundexYearGenderBlocker(),
                NamePrefixDOBBlocker(),
                PhoneBlocker(),
                SSNYearGenderBlocker(),
                MRNBlocker(),
            ])

        # Initialize rules (Phase 2.2)
        if use_rules:
            self.rule_engine = RuleEngine()

        # Initialize scoring (Phase 2.3)
        if use_scoring:
            self.feature_extractor = FeatureExtractor()
            # Use custom weights if provided, otherwise defaults
            weights = ScoringWeights(**scoring_weights) if scoring_weights else None
            # Use custom thresholds if provided
            threshold_kwargs = {}
            if scoring_thresholds:
                if 'definite' in scoring_thresholds:
                    threshold_kwargs['threshold_definite'] = scoring_thresholds['definite']
                if 'probable' in scoring_thresholds:
                    threshold_kwargs['threshold_probable'] = scoring_thresholds['probable']
                if 'possible' in scoring_thresholds:
                    threshold_kwargs['threshold_possible'] = scoring_thresholds['possible']
            self.scorer = ConfidenceScorer(weights=weights, **threshold_kwargs)

        # Initialize AI (Phase 2.4)
        self.api_rate_limit = api_rate_limit
        if use_ai:
            self.medical_matcher = MedicalFingerprintMatcher(
                ai_client=ai_client,
                ai_backend=ai_backend,
                api_rate_limit=api_rate_limit,
                **ai_kwargs,
            )

    def match_datasets(
        self,
        records: List[PatientRecord],
        show_progress: bool = True,
    ) -> List[MatchResult]:
        """
        Match all records in a dataset.

        Args:
            records: List of PatientRecord objects to match
            show_progress: Show progress bar during matching

        Returns:
            List of MatchResult for all candidate pairs

        Example:
            >>> records = load_records()  # 261 records
            >>> results = matcher.match_datasets(records)
            >>> print(f"Generated {len(results)} results")
            >>> print(f"Matches: {sum(r.is_match for r in results)}")
            >>> print(f"Non-matches: {sum(not r.is_match for r in results)}")
        """
        results = []

        # 1. Generate candidate pairs (blocking)
        if self.use_blocking:
            pairs = self.blocker.generate_candidate_pairs(records)
            if show_progress:
                print(f"Blocking: {len(pairs)} candidate pairs from {len(records)} records "
                      f"({len(records)*(len(records)-1)//2} total possible)")
        else:
            # All pairs (O(n²) - only for small datasets or testing)
            pairs = [
                (records[i], records[j])
                for i in range(len(records))
                for j in range(i+1, len(records))
            ]
            if show_progress:
                print(f"No blocking: {len(pairs)} total pairs")

        # 2. Evaluate each pair
        if show_progress:
            pairs_iter = tqdm(pairs, desc="Matching pairs")
        else:
            pairs_iter = pairs

        for record1, record2 in pairs_iter:
            result = self.match_pair(record1, record2)
            results.append(result)

        return results

    def match_pair(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> MatchResult:
        """
        Match two patient records.

        Phase 2.2: Uses blocking + rules only
        Phase 2.3: Adds scoring layer
        Phase 2.4: Adds AI medical fingerprinting

        The pipeline applies stages progressively:
        1. Try rules (Phase 2.2) - if decided, return
        2. Try scoring (Phase 2.3) - if confident, return
        3. Try AI (Phase 2.4) - for ambiguous cases
        4. Return uncertain if no stage decides

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            MatchResult with decision, confidence, evidence, explanation

        Example:
            >>> r1 = PatientRecord(...)  # John Smith
            >>> r2 = PatientRecord(...)  # John Smith (same person)
            >>> result = matcher.match_pair(r1, r2)
            >>> result.is_match
            True
            >>> result.confidence
            0.99
            >>> result.stage
            'rules'
            >>> result.rules_triggered
            ['ExactMatchRule']
        """
        # Phase 2.2: Try rules
        if self.use_rules:
            result = self.rule_engine.evaluate(record1, record2)
            if result is not None:
                # Rule fired - return decision
                return result

        # Phase 2.3: Try scoring
        if self.use_scoring:
            # Extract features
            features = self.feature_extractor.extract(record1, record2)

            # Calculate confidence score
            score, breakdown = self.scorer.score(features)

            # Classify match
            is_match, match_type = self.scorer.classify(score)

            # Build evidence dictionary with feature breakdown
            evidence = {
                "confidence_score": score,
                "feature_breakdown": breakdown,
                "features": features.to_dict(),
            }

            # Add match methods for explainability
            if features.name_first_method:
                evidence["name_first_method"] = features.name_first_method
            if features.name_last_method:
                evidence["name_last_method"] = features.name_last_method
            if features.dob_method:
                evidence["dob_method"] = features.dob_method
            if features.address_method:
                evidence["address_method"] = features.address_method

            # Generate explanation
            explanation = self.scorer.explain_score(score, breakdown, features)

            # Check if we should pass to AI for ambiguous cases
            # Ambiguous = score between 0.50 and 0.90 (not clearly match or non-match)
            is_ambiguous = 0.50 <= score <= 0.90

            # If AI is enabled and score is ambiguous, pass to AI stage
            # Otherwise return the scoring result
            if not (self.use_ai and is_ambiguous):
                return MatchResult(
                    record_1_id=record1.record_id,
                    record_2_id=record2.record_id,
                    is_match=is_match,
                    confidence=score,
                    match_type=match_type,
                    evidence=evidence,
                    stage="scoring",
                    explanation=explanation,
                )

            # Fall through to AI stage with pre-computed demographic score
            demo_score = score
            demo_breakdown = breakdown

        # Phase 2.4: Try AI for ambiguous cases
        if self.use_ai:
            # Get demographic score if not already computed by scoring
            if not self.use_scoring:
                # If scoring disabled, compute features just for AI
                self.feature_extractor = FeatureExtractor()
                self.scorer = ConfidenceScorer()
                features = self.feature_extractor.extract(record1, record2)
                demo_score, demo_breakdown = self.scorer.score(features)

            # Only use AI for ambiguous demographic scores (0.50-0.90)
            # Clear matches (>0.90) and clear non-matches (<0.50) don't need AI
            if 0.50 <= demo_score <= 0.90:
                # Get medical similarity from AI
                medical_score, ai_reasoning = self.medical_matcher.compare_medical_histories(
                    record1, record2
                )

                # Combine scores: 60% demographic + 40% medical
                combined_score = (0.6 * demo_score) + (0.4 * medical_score)

                # Determine match based on combined score
                if combined_score >= 0.80:
                    is_match = True
                    match_type = "probable" if combined_score < 0.90 else "definite"
                elif combined_score >= 0.65:
                    is_match = True
                    match_type = "possible"
                else:
                    is_match = False
                    match_type = "no_match"

                # Build evidence
                evidence = {
                    "demographic_score": round(demo_score, 3),
                    "medical_score": round(medical_score, 3),
                    "combined_score": round(combined_score, 3),
                    "weight_demographic": 0.6,
                    "weight_medical": 0.4,
                }

                # Build explanation
                explanation = (
                    f"AI Medical Fingerprinting: Combined score {combined_score:.2f} "
                    f"(demographic: {demo_score:.2f}, medical: {medical_score:.2f}). "
                    f"{ai_reasoning}"
                )

                return MatchResult(
                    record_1_id=record1.record_id,
                    record_2_id=record2.record_id,
                    is_match=is_match,
                    confidence=combined_score,
                    match_type=match_type,
                    evidence=evidence,
                    stage="ai",
                    medical_similarity=medical_score,
                    ai_reasoning=ai_reasoning,
                    explanation=explanation,
                )

        # No stage made a decision - return uncertain result
        # This represents pairs that need human review or more sophisticated analysis
        return MatchResult(
            record_1_id=record1.record_id,
            record_2_id=record2.record_id,
            is_match=False,  # Conservative: default to no-match
            confidence=0.5,  # Maximum uncertainty
            match_type="uncertain",
            evidence={
                "reason": "No deterministic rule fired and scoring disabled",
            },
            stage="none",
            explanation="No deterministic rule fired and scoring disabled. "
                       "Enable scoring (use_scoring=True) or AI (use_ai=True) for better coverage.",
        )

    def get_stats(self, results: List[MatchResult]) -> dict:
        """
        Get statistics about matching results.

        Args:
            results: List of MatchResult objects

        Returns:
            Dictionary with matching statistics

        Example:
            >>> results = matcher.match_datasets(records)
            >>> stats = matcher.get_stats(results)
            >>> print(stats)
            {
                'total_pairs': 1023,
                'matches': 245,
                'no_matches': 778,
                'by_stage': {'rules': 1023},
                'by_match_type': {'exact': 150, 'probable': 95, 'no_match': 778},
                'avg_confidence': 0.73,
            }
        """
        total = len(results)
        matches = sum(1 for r in results if r.is_match)
        no_matches = total - matches

        # Count by stage
        by_stage = {}
        for result in results:
            by_stage[result.stage] = by_stage.get(result.stage, 0) + 1

        # Count by match type
        by_match_type = {}
        for result in results:
            by_match_type[result.match_type] = by_match_type.get(result.match_type, 0) + 1

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0.0

        return {
            'total_pairs': total,
            'matches': matches,
            'no_matches': no_matches,
            'by_stage': by_stage,
            'by_match_type': by_match_type,
            'avg_confidence': round(avg_confidence, 3),
        }
