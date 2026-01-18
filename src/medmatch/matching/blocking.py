"""
Blocking strategies for patient entity resolution.

This module implements blocking strategies to reduce the O(n²) comparison
space to a manageable set of candidate pairs. Blocking groups potentially
matching records together using cheap-to-compute keys (e.g., phonetic name,
birth year, identifiers).

Key concept: Sacrifice precision for recall - we want to catch all true matches
while filtering out as many obvious non-matches as possible.

Target performance:
- Input: 261 records = 33,930 possible pairs
- Output: ~800-1,200 candidate pairs (2-4% of total)
- Recall: 100% (no true matches missed)
- Runtime: <1 second
"""

from typing import List, Tuple, Set, Protocol, Dict
from collections import defaultdict
import jellyfish

from .core import PatientRecord


class BlockingStrategy(Protocol):
    """Protocol for blocking strategies."""

    def get_block_key(self, record: PatientRecord) -> str:
        """
        Generate blocking key for a record.

        Args:
            record: PatientRecord to generate key for

        Returns:
            Blocking key string. Records with same key are candidate pairs.
            Returns empty string or "MISSING_{FIELD}" for missing data.
        """
        ...


class SoundexYearGenderBlocker:
    """
    Block by Soundex(last_name) + birth_year + gender.

    Rationale:
    - Soundex handles name misspellings and phonetic variations
    - Birth year is stable (rarely changes)
    - Gender is stable

    Example:
        "Smith" (S530) + "1980" + "M" → "S530_1980_M"
        "Smyth" (S530) + "1980" + "M" → "S530_1980_M" (same block!)
    """

    def get_block_key(self, record: PatientRecord) -> str:
        """Generate Soundex(last_name) + birth_year + gender key."""
        if not record.name_last:
            return "MISSING_NAME"

        try:
            soundex_code = jellyfish.soundex(record.name_last)
        except Exception:
            # Fallback: use first 4 chars if soundex fails
            soundex_code = record.name_last[:4].upper()

        birth_year = record.date_of_birth.year
        gender = record.gender or "U"  # Unknown gender

        return f"{soundex_code}_{birth_year}_{gender}"


class NamePrefixDOBBlocker:
    """
    Block by first 3 chars of last_name + full DOB.

    Rationale:
    - First 3 chars catch most name variations
    - Full DOB is highly discriminative
    - More precise than Soundex (fewer false candidates)

    Example:
        "Smith" → "SMI" + "1980-03-15" → "SMI_1980-03-15"
        "Smithson" → "SMI" + "1980-03-15" → "SMI_1980-03-15" (same block!)
    """

    def get_block_key(self, record: PatientRecord) -> str:
        """Generate first_3_chars(last_name) + DOB key."""
        if not record.name_last:
            return "MISSING_NAME"

        # Take first 3 chars, uppercase, pad if needed
        name_prefix = record.name_last[:3].upper().ljust(3, "_")
        dob = record.date_of_birth.isoformat()

        return f"{name_prefix}_{dob}"


class PhoneBlocker:
    """
    Block by normalized phone number.

    Rationale:
    - Phone numbers are often unique identifiers
    - Normalization handles format variations
    - Missing phones return distinct key (don't block with each other)

    Example:
        "(617) 555-1234" → "6175551234"
        "617-555-1234" → "6175551234" (same block!)
    """

    def get_block_key(self, record: PatientRecord) -> str:
        """Generate normalized phone number key."""
        if not record.phone:
            # Each missing phone gets unique key (don't block together)
            return f"MISSING_PHONE_{record.record_id}"

        # Normalize: keep only digits
        normalized = ''.join(c for c in record.phone if c.isdigit())

        if not normalized:
            return f"MISSING_PHONE_{record.record_id}"

        return normalized


class SSNYearGenderBlocker:
    """
    Block by SSN_last4 + birth_year + gender.

    Rationale:
    - SSN last 4 is semi-unique
    - Combined with birth_year + gender is highly discriminative
    - Many records may be missing SSN

    Example:
        "1234" + "1980" + "M" → "1234_1980_M"
    """

    def get_block_key(self, record: PatientRecord) -> str:
        """Generate SSN_last4 + birth_year + gender key."""
        if not record.ssn_last4:
            # Each missing SSN gets unique key
            return f"MISSING_SSN_{record.record_id}"

        birth_year = record.date_of_birth.year
        gender = record.gender or "U"

        return f"{record.ssn_last4}_{birth_year}_{gender}"


class MRNBlocker:
    """
    Block by exact MRN match.

    Rationale:
    - MRNs should be unique within a system
    - However, MRNs are system-specific (different hospitals = different MRNs)
    - Use conservatively - mainly catches exact system matches

    Note: This is the most conservative blocker. It will only catch records
    from the same source system that used the same MRN.
    """

    def get_block_key(self, record: PatientRecord) -> str:
        """Generate MRN key."""
        if not record.mrn:
            # Each missing MRN gets unique key
            return f"MISSING_MRN_{record.record_id}"

        return record.mrn


class MultiBlocker:
    """
    Combine multiple blocking strategies.

    Uses a union approach: if records share a key in ANY strategy,
    they become candidate pairs. This maximizes recall while still
    reducing the search space significantly.

    Example:
        Strategy 1: {A, B} {C, D}
        Strategy 2: {A, C} {B, D}
        Candidate pairs: (A,B), (C,D), (A,C), (B,D)
    """

    def __init__(self, strategies: List[BlockingStrategy]):
        """
        Initialize with list of blocking strategies.

        Args:
            strategies: List of blocking strategy instances
        """
        self.strategies = strategies

    def generate_candidate_pairs(
        self,
        records: List[PatientRecord],
    ) -> List[Tuple[PatientRecord, PatientRecord]]:
        """
        Generate candidate pairs using all blocking strategies.

        Algorithm:
        1. For each strategy, build block_key → [records] mapping
        2. Within each block, generate all pairs
        3. Union all pairs (deduplicate)
        4. Return unique pairs

        Performance: O(n*k) where k=number of strategies

        Args:
            records: List of PatientRecord objects

        Returns:
            List of (record1, record2) tuples representing candidate pairs.
            Each pair appears exactly once (no duplicates, no reversed pairs).

        Example:
            >>> blocker = MultiBlocker([
            ...     SoundexYearGenderBlocker(),
            ...     NamePrefixDOBBlocker(),
            ... ])
            >>> records = load_records()  # 261 records
            >>> pairs = blocker.generate_candidate_pairs(records)
            >>> len(pairs)  # ~800-1200 pairs
            1023
        """
        # Use set to track unique pairs
        # Store as (min_id, max_id) to avoid duplicates like (A,B) and (B,A)
        candidate_pairs_set: Set[Tuple[str, str]] = set()

        # For each blocking strategy
        for strategy in self.strategies:
            # Build block_key → [records] mapping
            blocks: Dict[str, List[PatientRecord]] = defaultdict(list)

            for record in records:
                block_key = strategy.get_block_key(record)
                blocks[block_key].append(record)

            # Generate pairs within each block
            for block_key, block_records in blocks.items():
                # Skip blocks with only 1 record
                if len(block_records) < 2:
                    continue

                # Skip blocks with "MISSING_" keys (each has unique key)
                if block_key.startswith("MISSING_"):
                    continue

                # Generate all pairs within this block
                for i, record1 in enumerate(block_records):
                    for record2 in block_records[i+1:]:
                        # Store as (min_id, max_id) to avoid duplicates
                        id1, id2 = record1.record_id, record2.record_id
                        pair_key = (min(id1, id2), max(id1, id2))
                        candidate_pairs_set.add(pair_key)

        # Convert back to list of (PatientRecord, PatientRecord) tuples
        # Build id → record lookup for efficiency
        record_lookup = {r.record_id: r for r in records}

        candidate_pairs = [
            (record_lookup[id1], record_lookup[id2])
            for id1, id2 in candidate_pairs_set
        ]

        return candidate_pairs

    def get_blocking_stats(
        self,
        records: List[PatientRecord],
    ) -> Dict[str, any]:
        """
        Get blocking statistics for analysis.

        Returns:
            Dictionary with:
            - total_records: Number of input records
            - total_possible_pairs: n*(n-1)/2
            - candidate_pairs: Number of candidate pairs generated
            - reduction_rate: Percentage of pairs filtered out
            - pairs_per_strategy: Number of pairs from each strategy
        """
        n = len(records)
        total_possible = n * (n - 1) // 2

        candidate_pairs = self.generate_candidate_pairs(records)
        num_candidates = len(candidate_pairs)

        reduction_rate = (1 - num_candidates / total_possible) * 100

        # Get pairs per strategy
        pairs_per_strategy = {}
        for strategy in self.strategies:
            strategy_blocker = MultiBlocker([strategy])
            strategy_pairs = strategy_blocker.generate_candidate_pairs(records)
            strategy_name = strategy.__class__.__name__
            pairs_per_strategy[strategy_name] = len(strategy_pairs)

        return {
            'total_records': n,
            'total_possible_pairs': total_possible,
            'candidate_pairs': num_candidates,
            'reduction_rate': f"{reduction_rate:.1f}%",
            'pairs_per_strategy': pairs_per_strategy,
        }
