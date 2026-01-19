"""
AI backend abstraction layer for medical history comparison.

This module provides a framework-agnostic interface for medical AI comparison,
supporting both cloud API (Gemini) and local (Ollama/MedGemma) backends. The
abstraction layer enables easy switching between backends without code changes.

Architecture:
    BaseMedicalAIClient (ABC)
    ├── GeminiAIClient      - Google Gemini API (cloud, requires internet)
    └── OllamaClient        - Local MedGemma via Ollama server (HIPAA-compliant)

Factory:
    MedicalAIClient.create(backend="gemini" | "ollama")

Example:
    >>> # Use Gemini API (development/testing)
    >>> client = MedicalAIClient.create(backend="gemini")
    >>> score, reasoning = client.compare_medical_histories(hist1, hist2)

    >>> # Use MedGemma via Ollama (recommended for production)
    >>> client = MedicalAIClient.create(backend="ollama")
    >>> score, reasoning = client.compare_medical_histories(hist1, hist2)
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dotenv import load_dotenv


class BaseMedicalAIClient(ABC):
    """
    Abstract base class for medical AI comparison backends.

    All implementations must support comparing two medical history strings
    and returning a similarity score with reasoning. The template method
    pattern is used: subclasses implement only initialize_model() and
    generate_response(), while prompt building and response parsing are shared.

    Template Method Pattern:
        compare_medical_histories() orchestrates the flow:
        1. Build prompt (_build_comparison_prompt)
        2. Generate response (generate_response - implemented by subclass)
        3. Parse response (_parse_response)

    Attributes:
        Subclass-specific (Gemini: api_key/model, MedGemma: model_id/device)
    """

    @abstractmethod
    def initialize_model(self, **kwargs) -> None:
        """
        Initialize the underlying model or API client.

        This is called during __init__. Subclasses should:
        - API backends: Initialize client with credentials
        - Local backends: Load model weights and processor

        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If model initialization fails
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text response from prompt using underlying model/API.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (advisory for local models)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Return human-readable backend name for logging/debugging.

        Examples:
            "Gemini API (gemini-2.5-flash)"
            "MedGemma Local (google/medgemma-1.5-4b-it, mps)"
        """
        pass

    def compare_medical_histories(
        self,
        medical_history_1: str,
        medical_history_2: str,
        patient_name_1: Optional[str] = None,
        patient_name_2: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Compare two medical histories for similarity.

        This is the main public API. Uses template method pattern:
        prompt building and parsing are shared, generation is delegated
        to the subclass implementation.

        Args:
            medical_history_1: First patient's medical signature
            medical_history_2: Second patient's medical signature
            patient_name_1: Optional name for context (e.g., "John Smith")
            patient_name_2: Optional name for context

        Returns:
            Tuple of (similarity_score, reasoning):
            - similarity_score: Float in [0.0, 1.0], where 1.0 = identical
            - reasoning: Human-readable explanation (1-3 sentences)

        Example:
            >>> score, reason = client.compare_medical_histories(
            ...     "Type 2 Diabetes, Hypertension, on Metformin",
            ...     "T2DM, HTN, medications: Metformin",
            ...     "John Smith",
            ...     "Smith, John"
            ... )
            >>> print(f"Similarity: {score:.2f} - {reason}")
        """
        # Build prompt (shared implementation)
        prompt = self._build_comparison_prompt(
            medical_history_1,
            medical_history_2,
            patient_name_1 or "Patient 1",
            patient_name_2 or "Patient 2",
        )

        # Generate response (delegated to subclass)
        # Use 1024 tokens to accommodate MedGemma's thought process
        try:
            response_text = self.generate_response(prompt, max_tokens=1024)
        except Exception as e:
            # Graceful fallback on generation errors
            return 0.5, f"Error during AI generation: {str(e)}"

        # Parse response (shared implementation)
        score, reasoning = self._parse_response(response_text)
        return score, reasoning

    def _build_comparison_prompt(
        self,
        med_sig_1: str,
        med_sig_2: str,
        name1: str,
        name2: str,
    ) -> str:
        """
        Build structured prompt for medical history comparison.

        This prompt has been tested and validated at 99.4% accuracy.
        Do not modify unless benchmarks show improvement!

        Args:
            med_sig_1: Medical signature from first record
            med_sig_2: Medical signature from second record
            name1: Name from first record (for context)
            name2: Name from second record (for context)

        Returns:
            Formatted prompt string with instructions and scoring guide
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
        Parse AI response to extract similarity score and reasoning.

        Handles variations in AI response format robustly:
        - Exact format: "SIMILARITY_SCORE: 0.8\nREASONING: ..."
        - Mixed case: "similarity_score: 0.8"
        - Percentage format: "SIMILARITY_SCORE: 80%"
        - Multi-line reasoning
        - Missing fields

        Args:
            response_text: Raw text response from AI model/API

        Returns:
            Tuple of (score, reasoning):
            - score: Float in [0.0, 1.0] (default: 0.5 if parsing fails)
            - reasoning: String explanation (default: error message if parsing fails)
        """
        # Default values if parsing fails
        score = 0.5
        reasoning = "Unable to parse AI response"

        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Extract score (case-insensitive)
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
                    pass  # Keep default score

            # Extract reasoning (case-insensitive)
            elif line.upper().startswith('REASONING:'):
                try:
                    reasoning = line.split(':', 1)[1].strip()
                except IndexError:
                    pass  # Keep default reasoning

        # Fallback: if reasoning spans multiple lines (AI sometimes does this)
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


class GeminiAIClient(BaseMedicalAIClient):
    """
    Google Gemini API implementation of medical AI comparison.

    Uses the google-genai SDK to call Gemini models via API. Requires
    a valid Google AI API key.

    Example:
        >>> client = GeminiAIClient(model="gemini-2.5-flash")
        >>> score, reason = client.compare_medical_histories(hist1, hist2)

    Attributes:
        model: Gemini model name (default: "gemini-2.5-flash")
        api_key: Google AI API key (from env or parameter)
        client: Google AI client instance
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini API client.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash", "gemini-pro")
            api_key: Google AI API key (defaults to GOOGLE_AI_API_KEY env var)

        Raises:
            ValueError: If API key is not found
            ImportError: If google-genai package is not installed
        """
        self.model = model

        # Load API key from env or parameter
        load_dotenv()
        self.api_key = api_key if api_key else os.getenv('GOOGLE_AI_API_KEY')

        self.client = None
        self._genai = None
        self.initialize_model()

    def initialize_model(self, **kwargs) -> None:
        """
        Initialize Gemini API client.

        Raises:
            ValueError: If GOOGLE_AI_API_KEY not found
            ImportError: If google-genai not installed
        """
        if not self.api_key:
            raise ValueError(
                "GOOGLE_AI_API_KEY not found. "
                "Set it in .env file or pass api_key parameter."
            )

        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
            self._genai = genai  # Keep reference for types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate response using Gemini API.

        Args:
            prompt: Input prompt text
            max_tokens: Ignored (Gemini controls this internally)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}")

    @property
    def backend_name(self) -> str:
        """Return descriptive backend name for logging."""
        return f"Gemini API ({self.model})"


class OllamaClient(BaseMedicalAIClient):
    """
    Local MedGemma implementation via Ollama inference server.

    Connects to a running Ollama server to use MedGemma without loading
    the model weights in Python. Ollama handles model management, GPU
    acceleration, and optimization automatically.

    Prerequisites:
        1. Ollama installed: brew install ollama
        2. Ollama service running: brew services start ollama
        3. MedGemma imported in Ollama: ollama list | grep medgemma

    Example:
        >>> # Default configuration (localhost:11434, medgemma:1.5-4b)
        >>> client = OllamaClient()
        >>> score, reason = client.compare_medical_histories(hist1, hist2)

        >>> # Custom configuration
        >>> client = OllamaClient(
        ...     model="medgemma:1.5-4b",
        ...     base_url="http://localhost:11434",
        ...     temperature=0.3,
        ... )

    Attributes:
        model: Ollama model name (default: "medgemma:1.5-4b")
        base_url: Ollama server URL (default: "http://localhost:11434")
        temperature: Sampling temperature (default: 0.3 for factual responses)
        timeout: Request timeout in seconds (default: 60)
    """

    def __init__(
        self,
        model: str = "medgemma:1.5-4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout: int = 60,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (e.g., "medgemma:1.5-4b")
            base_url: Ollama server URL (default: localhost:11434)
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            timeout: Request timeout in seconds

        Raises:
            ImportError: If requests package not installed
            RuntimeError: If Ollama server is not accessible
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.timeout = timeout
        self.initialize_model()

    def initialize_model(self, **kwargs) -> None:
        """
        Verify Ollama server is accessible and model is available.

        Raises:
            ImportError: If requests package not installed
            RuntimeError: If Ollama server is not running or model not found
        """
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests package required for Ollama client. "
                "Install with: pip install requests"
            )

        # Verify server is running
        try:
            response = self._requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
        except self._requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running: brew services start ollama"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error connecting to Ollama server: {str(e)}"
            )

        # Verify model is available
        try:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if not any(self.model in name for name in model_names):
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {model_names}\n"
                    f"Import MedGemma with: ollama create {self.model} -f /tmp/medgemma-modelfile\n"
                    f"See docs/ollama_setup.md for details."
                )
        except KeyError:
            # If we can't verify, just warn and continue
            print(f"⚠ Could not verify model '{self.model}' availability")

    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate response using Ollama API.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (Ollama parameter: num_predict)
                       Default is 1024 to accommodate MedGemma's thought process
                       which uses <unused94>thought...<unused95> format

        Returns:
            Generated text response, with MedGemma thought tokens stripped

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self._requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "")

            # MedGemma uses <unused94>thought...<unused95> format
            # The actual response comes after <unused95>
            if "<unused95>" in response_text:
                response_text = response_text.split("<unused95>", 1)[1].strip()

            return response_text

        except self._requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try increasing timeout or check server load."
            )
        except self._requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API call failed: {str(e)}")

    @property
    def backend_name(self) -> str:
        """Return descriptive backend name for logging."""
        return f"Ollama ({self.model})"


class MedicalAIClient:
    """
    Factory for creating medical AI clients.

    Provides a simple interface to create backend-specific clients without
    knowing implementation details. Supports fallback mechanism for robustness.

    Usage:
        >>> # Simple creation
        >>> client = MedicalAIClient.create(backend="gemini")
        >>> client = MedicalAIClient.create(backend="ollama")

        >>> # With configuration
        >>> client = MedicalAIClient.create(
        ...     backend="ollama",
        ...     model="medgemma:1.5-4b",
        ...     temperature=0.3,
        ... )
    """

    @staticmethod
    def create(backend: str = "gemini", **kwargs) -> BaseMedicalAIClient:
        """
        Create an AI client instance.

        Args:
            backend: Backend type ("gemini" or "ollama")
            **kwargs: Backend-specific configuration
                Gemini: model, api_key
                Ollama: model, base_url, temperature, timeout

        Returns:
            Initialized AI client (GeminiAIClient or OllamaClient)

        Raises:
            ValueError: If backend is unknown

        Example:
            >>> # Gemini (cloud API)
            >>> client = MedicalAIClient.create(
            ...     backend="gemini",
            ...     model="gemini-pro",
            ... )

            >>> # Ollama (recommended for local/production)
            >>> client = MedicalAIClient.create(
            ...     backend="ollama",
            ...     model="medgemma:1.5-4b",
            ... )
        """
        backend = backend.lower()

        if backend == "gemini":
            return GeminiAIClient(**kwargs)
        elif backend == "ollama":
            return OllamaClient(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose 'gemini' or 'ollama'."
            )

    @staticmethod
    def create_with_fallback(
        preferred: str = "ollama",
        fallback: str = "gemini",
        **kwargs,
    ) -> BaseMedicalAIClient:
        """
        Create client with automatic fallback on failure.

        Tries preferred backend first, falls back if initialization fails.
        Useful for development environments where Ollama may not be running.

        WARNING: For production with real patient data, fallback to Gemini
        is a HIPAA violation (sends data to Google's API). Only use ollama
        in production, or ensure fallback is never triggered.

        Args:
            preferred: Preferred backend to try first (default: "ollama")
            fallback: Fallback backend if preferred fails (default: "gemini")
            **kwargs: Configuration for preferred backend

        Returns:
            Initialized AI client (preferred or fallback)

        Example:
            >>> # Development: Try Ollama, fallback to Gemini
            >>> client = MedicalAIClient.create_with_fallback(
            ...     preferred="ollama",
            ...     fallback="gemini",
            ... )

            >>> # WARNING: NEVER use fallback with real patient data!
            >>> # Production should only use ollama without fallback.
        """
        try:
            return MedicalAIClient.create(backend=preferred, **kwargs)
        except Exception as e:
            print(f"⚠ Failed to initialize {preferred}: {e}")
            print(f"Falling back to {fallback}...")
            return MedicalAIClient.create(backend=fallback)
