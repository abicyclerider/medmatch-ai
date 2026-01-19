"""
AI-powered medical history comparison with pluggable backends.

This module provides the MedicalFingerprintMatcher class for comparing
patient medical histories using AI. Supports two backends:
- Gemini API (cloud-based, for development/testing)
- Ollama/MedGemma (local, HIPAA-compliant, for production)

Phase 2.4 of the entity resolution system.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

from .core import PatientRecord
from .ai_client import BaseMedicalAIClient, MedicalAIClient


@dataclass
class RateLimiter:
    """
    Simple rate limiter for API calls.

    Tracks request timestamps and enforces a maximum requests-per-minute limit.
    Set requests_per_minute=0 to disable rate limiting (recommended for
    development with billing enabled).

    Attributes:
        requests_per_minute: Maximum requests per minute (0=unlimited)
        last_request_time: Timestamp of last API call
    """

    requests_per_minute: int = 0  # 0 = unlimited (billing enabled)
    last_request_time: float = 0.0

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limit.

        Does nothing if requests_per_minute is 0 (unlimited).
        Otherwise, enforces minimum interval between requests.
        """
        if self.requests_per_minute <= 0:
            return  # No rate limiting

        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self.last_request_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class MedicalFingerprintMatcher:
    """
    AI-powered medical history comparison with pluggable backends.

    Compares two patient records' medical histories to determine if they
    likely refer to the same person. Supports two AI backends:
    - Gemini API: Cloud-based, requires API key (for development/testing)
    - Ollama: Local MedGemma inference, privacy-preserving, no API costs (for production)

    Uses AI to understand:
    - Medical abbreviations (T2DM, HTN, MI, etc.)
    - Medication equivalents (Lisinopril for hypertension)
    - Condition progressions and relationships
    - Temporal consistency

    Example:
        >>> # Use Ollama (default, recommended for production)
        >>> matcher = MedicalFingerprintMatcher()
        >>> score, reason = matcher.compare_medical_histories(record1, record2)

        >>> # Use Gemini API (development/testing only)
        >>> matcher = MedicalFingerprintMatcher(ai_backend="gemini")
        >>> score, reason = matcher.compare_medical_histories(record1, record2)

    Attributes:
        ai_client: Backend AI client (Gemini or MedGemma)
        rate_limiter: Optional rate limiter for API calls
    """

    def __init__(
        self,
        ai_client: Optional[BaseMedicalAIClient] = None,
        ai_backend: str = "ollama",
        api_rate_limit: int = 0,  # 0 = no rate limiting (billing enabled)
        **ai_kwargs,
    ):
        """
        Initialize the medical fingerprint matcher.

        Args:
            ai_client: Pre-configured AI client (if None, creates one)
            ai_backend: Backend to use if ai_client not provided ("ollama" or "gemini", default: "ollama")
            api_rate_limit: Requests per minute (0=unlimited, recommended for dev)
            **ai_kwargs: Passed to AI client factory (e.g., model, temperature, api_key)

        Example:
            >>> # Use Ollama (default, recommended for production)
            >>> matcher = MedicalFingerprintMatcher()

            >>> # Use Gemini API (development/testing only)
            >>> matcher = MedicalFingerprintMatcher(ai_backend="gemini")

            >>> # Custom Gemini model
            >>> matcher = MedicalFingerprintMatcher(
            ...     ai_backend="gemini",
            ...     model="gemini-pro",
            ... )

            >>> # Inject pre-configured client (advanced)
            >>> client = MedicalAIClient.create(backend="ollama")
            >>> matcher = MedicalFingerprintMatcher(ai_client=client)
        """
        # Accept pre-configured client or create new one
        if ai_client is not None:
            self.ai_client = ai_client
        else:
            self.ai_client = MedicalAIClient.create(
                backend=ai_backend,
                **ai_kwargs,
            )

        # Rate limiting (works with any backend)
        if api_rate_limit > 0:
            self.rate_limiter = RateLimiter(requests_per_minute=api_rate_limit)
        else:
            self.rate_limiter = None

    def compare_medical_histories(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> Tuple[float, str]:
        """
        Compare medical histories of two patient records using AI.

        Uses configured backend (Gemini or MedGemma) to analyze medical
        signatures and determine similarity. Returns a score and reasoning
        that can be used to augment demographic matching for hard/ambiguous cases.

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            Tuple of (similarity_score, reasoning):
            - similarity_score: 0.0 to 1.0 (1.0 = identical histories)
            - reasoning: AI-generated explanation of the comparison

        Example:
            >>> score, reason = matcher.compare_medical_histories(r1, r2)
            >>> if score > 0.7:
            ...     print("Medical histories suggest same patient")
        """
        # Get medical signatures
        med_sig_1 = record1.medical_signature
        med_sig_2 = record2.medical_signature

        # Handle cases with no medical history
        if med_sig_1 == "No medical history available" and med_sig_2 == "No medical history available":
            return 0.5, "Neither record has medical history available - cannot compare."

        if med_sig_1 == "No medical history available" or med_sig_2 == "No medical history available":
            return 0.5, "One record has no medical history - cannot make comparison."

        # Wait for rate limiter if needed (works with any backend)
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Delegate to AI client (handles prompt building, generation, parsing)
        score, reasoning = self.ai_client.compare_medical_histories(
            medical_history_1=med_sig_1,
            medical_history_2=med_sig_2,
            patient_name_1=record1.full_name,
            patient_name_2=record2.full_name,
        )

        return score, reasoning

    def compare_batch(
        self,
        pairs: list,
        show_progress: bool = True,
    ) -> list:
        """
        Compare multiple record pairs.

        Args:
            pairs: List of (PatientRecord, PatientRecord) tuples
            show_progress: Show progress bar

        Returns:
            List of (score, reasoning) tuples
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(pairs, desc="AI comparisons") if show_progress else pairs

        for record1, record2 in iterator:
            score, reasoning = self.compare_medical_histories(record1, record2)
            results.append((score, reasoning))

        return results
