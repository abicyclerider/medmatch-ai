"""
AI-powered medical history comparison using Gemini API.

This module provides the MedicalFingerprintMatcher class for comparing
patient medical histories using AI. It's designed for hard/ambiguous cases
where demographic comparison alone is insufficient.

Phase 2.4 of the entity resolution system.
"""

import os
import time
import re
from typing import Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from .core import PatientRecord


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
    AI-powered medical history comparison using Gemini API.

    Compares two patient records' medical histories to determine if they
    likely refer to the same person. Uses AI to understand:
    - Medical abbreviations (T2DM, HTN, MI, etc.)
    - Medication equivalents (Lisinopril for hypertension)
    - Condition progressions and relationships
    - Temporal consistency

    Example:
        >>> matcher = MedicalFingerprintMatcher()
        >>> score, reasoning = matcher.compare_medical_histories(record1, record2)
        >>> print(f"Medical similarity: {score:.2f}")
        >>> print(f"Reasoning: {reasoning}")

    Attributes:
        model: Name of the Gemini model to use
        rate_limiter: Optional rate limiter for API calls
        client: Google AI client instance
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_rate_limit: int = 0,  # 0 = no rate limiting (billing enabled)
        api_key: Optional[str] = None,
    ):
        """
        Initialize the medical fingerprint matcher.

        Args:
            model: Gemini model name (default: gemini-2.5-flash)
            api_rate_limit: Requests per minute (0=unlimited, recommended for dev)
            api_key: Optional API key (defaults to GOOGLE_AI_API_KEY env var)

        Raises:
            ValueError: If API key is not found
            ImportError: If google-genai package is not installed
        """
        self.model = model
        self.rate_limiter = RateLimiter(requests_per_minute=api_rate_limit)

        # Load API key
        load_dotenv()
        # Use provided key, or fall back to env var
        # Empty string should be treated as missing
        self.api_key = api_key if api_key else os.getenv('GOOGLE_AI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "GOOGLE_AI_API_KEY not found. "
                "Set it in .env file or pass api_key parameter."
            )

        # Initialize Google AI client
        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
            self._genai = genai  # Keep reference for types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

    def compare_medical_histories(
        self,
        record1: PatientRecord,
        record2: PatientRecord,
    ) -> Tuple[float, str]:
        """
        Compare medical histories of two patient records using AI.

        Uses Gemini to analyze medical signatures and determine similarity.
        Returns a score and reasoning that can be used to augment demographic
        matching for hard/ambiguous cases.

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

        # Build structured prompt
        prompt = self._build_comparison_prompt(
            record1.full_name, med_sig_1,
            record2.full_name, med_sig_2,
        )

        # Wait for rate limiter if needed
        self.rate_limiter.wait_if_needed()

        try:
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            # Parse response
            score, reasoning = self._parse_response(response.text)
            return score, reasoning

        except Exception as e:
            # Graceful fallback on API error
            error_msg = f"API error: {str(e)}"
            return 0.0, error_msg

    def _build_comparison_prompt(
        self,
        name1: str,
        med_sig_1: str,
        name2: str,
        med_sig_2: str,
    ) -> str:
        """
        Build structured prompt for medical history comparison.

        Args:
            name1: Name from first record
            med_sig_1: Medical signature from first record
            name2: Name from second record
            med_sig_2: Medical signature from second record

        Returns:
            Formatted prompt string
        """
        return f"""You are a medical entity resolution expert. Your task is to compare two medical histories and determine how similar they are.

**IMPORTANT CONSIDERATIONS:**
1. Medical abbreviations are equivalent to full names:
   - T2DM = Type 2 Diabetes Mellitus = Diabetes Type 2
   - HTN = Hypertension = High Blood Pressure
   - MI = Myocardial Infarction = Heart Attack
   - CAD = Coronary Artery Disease
   - CHF = Congestive Heart Failure
   - COPD = Chronic Obstructive Pulmonary Disease
   - GERD = Gastroesophageal Reflux Disease
   - DM = Diabetes Mellitus

2. Medications often indicate conditions:
   - Metformin → Diabetes
   - Lisinopril, Losartan, Amlodipine → Hypertension
   - Atorvastatin, Simvastatin, Rosuvastatin → Hyperlipidemia
   - Metoprolol, Carvedilol → Heart conditions
   - Sertraline, Fluoxetine → Depression/Anxiety
   - Albuterol, Fluticasone → Asthma/COPD

3. Consider:
   - Overlapping conditions (even with different wording)
   - Medication overlap (same drug class)
   - Temporal consistency (conditions don't disappear)
   - Disease progressions (diabetes + kidney disease makes sense)

**PATIENT 1:** {name1}
**Medical History:** {med_sig_1}

**PATIENT 2:** {name2}
**Medical History:** {med_sig_2}

**RESPOND IN THIS EXACT FORMAT:**
SIMILARITY_SCORE: [a decimal between 0.0 and 1.0]
REASONING: [Your explanation in 1-3 sentences]

**SCORING GUIDE:**
- 1.0: Identical or near-identical histories (same conditions, same meds)
- 0.8-0.9: Very similar (equivalent abbreviations/synonyms, matching medications)
- 0.6-0.7: Similar (some overlap in conditions or related medications)
- 0.4-0.5: Some overlap (1-2 shared conditions or medication classes)
- 0.2-0.3: Minimal overlap (possibly one shared common condition)
- 0.0-0.1: Different medical profiles (no meaningful overlap)

Provide your analysis:"""

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """
        Parse AI response to extract score and reasoning.

        Handles variations in AI response format robustly.

        Args:
            response_text: Raw text response from Gemini

        Returns:
            Tuple of (score, reasoning)
        """
        # Default values if parsing fails
        score = 0.5
        reasoning = "Unable to parse AI response"

        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Extract score
            if line.upper().startswith('SIMILARITY_SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    # Handle various formats: "0.8", "0.80", "80%", etc.
                    score_str = score_str.replace('%', '').strip()
                    parsed_score = float(score_str)
                    # Convert percentage to decimal if needed
                    if parsed_score > 1.0:
                        parsed_score = parsed_score / 100.0
                    # Clamp to valid range
                    score = max(0.0, min(1.0, parsed_score))
                except (ValueError, IndexError):
                    pass

            # Extract reasoning
            elif line.upper().startswith('REASONING:'):
                try:
                    reasoning = line.split(':', 1)[1].strip()
                except IndexError:
                    pass

        # If reasoning spans multiple lines (AI sometimes does this)
        if reasoning == "Unable to parse AI response":
            # Try to find any substantial text after SIMILARITY_SCORE
            found_score = False
            reasoning_parts = []
            for line in lines:
                if 'SIMILARITY_SCORE' in line.upper():
                    found_score = True
                    continue
                if found_score and line.strip() and not line.upper().startswith('REASONING:'):
                    reasoning_parts.append(line.strip())
                elif line.upper().startswith('REASONING:'):
                    reasoning_parts.append(line.split(':', 1)[1].strip())
            if reasoning_parts:
                reasoning = ' '.join(reasoning_parts)

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
