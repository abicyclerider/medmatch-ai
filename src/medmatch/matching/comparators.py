"""
Field comparison functions for patient record matching.

This module provides specialized comparators for different field types:
- NameComparator: Compare names with fuzzy matching and variation detection
- DateComparator: Compare dates with error pattern detection
- AddressComparator: Compare addresses at different granularities

All comparators leverage existing utilities from src/medmatch/data/utils/
"""

from typing import Tuple, Optional
from datetime import date
import jellyfish

from ..data.utils import name_utils
from ..data.models.patient import Address


class NameComparator:
    """
    Compare name fields with fuzzy matching.

    Uses multiple matching strategies to handle variations:
    - Exact match
    - Known nickname variations (William → Bill, etc.)
    - Typographical errors (Levenshtein distance)
    - Phonetic similarity (Soundex)
    - First initial match

    Leverages existing NAME_VARIATIONS dict from name_utils.
    """

    def compare(self, name1: str, name2: str) -> Tuple[float, str]:
        """
        Compare two names and return similarity score.

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            Tuple of (score, method) where:
            - score: 0.0-1.0 similarity score
            - method: String describing matching method used

        Score scale:
        - 1.0: Exact match
        - 0.95: Known nickname variation
        - 0.85: Levenshtein distance ≤ 2 (typo)
        - 0.75: Soundex match (phonetic)
        - 0.5: First initial match
        - 0.0: No similarity

        Example:
            >>> comp = NameComparator()
            >>> score, method = comp.compare("William", "Bill")
            >>> score
            0.95
            >>> method
            'known_variation'
        """
        if not name1 or not name2:
            return 0.0, "missing_name"

        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        # Exact match
        if name1 == name2:
            return 1.0, "exact_match"

        # Known variation (uses existing NAME_VARIATIONS dict)
        if self._is_known_variation(name1, name2):
            return 0.95, "known_variation"

        # Levenshtein distance (typos)
        lev_dist = jellyfish.levenshtein_distance(name1, name2)
        if lev_dist <= 1:
            return 0.9, "typo_1char"
        elif lev_dist == 2:
            return 0.85, "typo_2char"

        # Soundex (phonetic match)
        try:
            if jellyfish.soundex(name1) == jellyfish.soundex(name2):
                return 0.75, "soundex_match"
        except Exception:
            # Soundex can fail on very short or non-ASCII names
            pass

        # First initial match
        if name1[0] == name2[0]:
            return 0.5, "first_initial_match"

        return 0.0, "no_match"

    @staticmethod
    def _is_known_variation(name1: str, name2: str) -> bool:
        """
        Check if names are known variations using existing name_utils.

        Args:
            name1: First name (lowercase)
            name2: Second name (lowercase)

        Returns:
            True if names are known variations of each other
        """
        # Get variations for both names
        name1_capitalized = name1.capitalize()
        name2_capitalized = name2.capitalize()

        # Check if either name appears in NAME_VARIATIONS for the other
        for base_name, variations in name_utils.NAME_VARIATIONS.items():
            base_lower = base_name.lower()
            variations_lower = [v.lower() for v in variations]

            # Check if name1 is base and name2 is variant (or vice versa)
            if (name1 == base_lower and name2 in variations_lower) or \
               (name2 == base_lower and name1 in variations_lower):
                return True

            # Check if both are variants of the same base
            if name1 in variations_lower and name2 in variations_lower:
                return True

        return False


class DateComparator:
    """
    Compare dates with error pattern detection.

    Handles common data entry errors:
    - Transposed digits (15 vs 51)
    - Month/day swap (MM/DD vs DD/MM confusion)
    - Off-by-one errors
    - Year typos

    Leverages existing date_utils for error detection.
    """

    def compare(self, dob1: date, dob2: date) -> Tuple[float, str]:
        """
        Compare two dates and return similarity score.

        Args:
            dob1: First date of birth
            dob2: Second date of birth

        Returns:
            Tuple of (score, method) where:
            - score: 0.0-1.0 similarity score
            - method: String describing matching method used

        Score scale:
        - 1.0: Exact match
        - 0.95: Within 2 days (twins/triplets)
        - 0.9: Known typo pattern (transposed digits, month/day swap)
        - 0.8: Off by 1 year (typo in year)
        - 0.5: Same month/day, different year
        - 0.0: Significantly different

        Example:
            >>> comp = DateComparator()
            >>> date1 = date(1980, 3, 15)
            >>> date2 = date(1980, 3, 15)
            >>> score, method = comp.compare(date1, date2)
            >>> score
            1.0
        """
        if dob1 == dob2:
            return 1.0, "exact_match"

        days_diff = abs((dob1 - dob2).days)

        # Twins (within 2 days)
        if days_diff <= 2:
            return 0.95, "twins_possible"

        # Check for known error patterns
        if self._is_transposed_digits(dob1, dob2):
            return 0.9, "transposed_digits"

        if self._is_month_day_swap(dob1, dob2):
            return 0.9, "month_day_swap"

        # Off by one day (data entry error)
        if days_diff == 1:
            return 0.9, "off_by_one_day"

        # Off by approximately 1 month (30-31 days)
        if 28 <= days_diff <= 31 and dob1.year == dob2.year:
            return 0.7, "off_by_one_month"

        # Off by approximately 1 year
        years_diff = abs(dob1.year - dob2.year)
        if years_diff == 1 and dob1.month == dob2.month and dob1.day == dob2.day:
            return 0.8, "year_typo"

        # Same month/day, different year (less likely to be same person)
        if dob1.month == dob2.month and dob1.day == dob2.day:
            return 0.5, "same_month_day"

        # Significantly different
        return 0.0, "no_match"

    @staticmethod
    def _is_transposed_digits(d1: date, d2: date) -> bool:
        """
        Check if dates differ by transposed digits.

        Examples:
        - Day: 15 vs 51 (invalid but pattern detectable)
        - Month: 01 vs 10
        - Day within month: 12 vs 21

        Args:
            d1: First date
            d2: Second date

        Returns:
            True if transposed digit pattern detected
        """
        # Check day transposition
        d1_day_str = str(d1.day).zfill(2)
        d2_day_str = str(d2.day).zfill(2)

        # Check month transposition
        d1_month_str = str(d1.month).zfill(2)
        d2_month_str = str(d2.month).zfill(2)

        # Same year, and either day or month transposed
        if d1.year == d2.year:
            day_transposed = (d1_day_str == d2_day_str[::-1]) and d1.month == d2.month
            month_transposed = (d1_month_str == d2_month_str[::-1]) and d1.day == d2.day

            if day_transposed or month_transposed:
                return True

        return False

    @staticmethod
    def _is_month_day_swap(d1: date, d2: date) -> bool:
        """
        Check if month and day are swapped (MM/DD vs DD/MM confusion).

        Example:
        - 03/15/1980 vs 15/03/1980

        Args:
            d1: First date
            d2: Second date

        Returns:
            True if month/day swap detected
        """
        return (
            d1.month == d2.day and
            d1.day == d2.month and
            d1.year == d2.year
        )


class AddressComparator:
    """
    Compare address fields at different granularities.

    Provides flexible matching at multiple levels:
    - Exact match (all components)
    - Same street + city (zip might change)
    - Same city + state
    - Same zip code only
    """

    def compare(
        self,
        addr1: Optional[Address],
        addr2: Optional[Address]
    ) -> Tuple[float, str]:
        """
        Compare two addresses and return similarity score.

        Args:
            addr1: First address (optional)
            addr2: Second address (optional)

        Returns:
            Tuple of (score, method) where:
            - score: 0.0-1.0 similarity score
            - method: String describing matching method used

        Score scale:
        - 1.0: Exact match (all components)
        - 0.8: Same street + city (zip might change over time)
        - 0.6: Same city + state
        - 0.4: Same zip code only
        - 0.0: Different or missing

        Example:
            >>> comp = AddressComparator()
            >>> addr1 = Address(street="123 Main St", city="Boston", state="MA", zip_code="02108")
            >>> addr2 = Address(street="123 Main St", city="Boston", state="MA", zip_code="02108")
            >>> score, method = comp.compare(addr1, addr2)
            >>> score
            1.0
        """
        # Handle missing addresses
        if addr1 is None or addr2 is None:
            return 0.0, "missing_address"

        # Exact match
        if (addr1.street == addr2.street and
            addr1.city == addr2.city and
            addr1.state == addr2.state and
            addr1.zip_code == addr2.zip_code):
            return 1.0, "exact_match"

        # Same street + city (zip might change with redistricting)
        if addr1.street == addr2.street and addr1.city == addr2.city:
            return 0.8, "same_street_city"

        # Same city + state (person moved within city)
        if addr1.city == addr2.city and addr1.state == addr2.state:
            return 0.6, "same_city_state"

        # Same zip code (nearby, possibly same person)
        if addr1.zip_code == addr2.zip_code:
            return 0.4, "same_zip"

        # Different
        return 0.0, "no_match"


class PhoneComparator:
    """
    Compare phone numbers with normalization.

    Handles different phone number formats by normalizing to digits only.
    """

    def compare(
        self,
        phone1: Optional[str],
        phone2: Optional[str]
    ) -> Tuple[float, str]:
        """
        Compare two phone numbers.

        Args:
            phone1: First phone number (optional)
            phone2: Second phone number (optional)

        Returns:
            Tuple of (score, method):
            - 1.0: Exact match (normalized)
            - 0.0: Different or missing

        Example:
            >>> comp = PhoneComparator()
            >>> score, method = comp.compare("(617) 555-1234", "617-555-1234")
            >>> score
            1.0
        """
        if not phone1 or not phone2:
            return 0.0, "missing_phone"

        # Normalize: remove all non-digits
        phone1_normalized = ''.join(c for c in phone1 if c.isdigit())
        phone2_normalized = ''.join(c for c in phone2 if c.isdigit())

        if phone1_normalized == phone2_normalized:
            return 1.0, "exact_match"

        return 0.0, "no_match"


class EmailComparator:
    """
    Compare email addresses.

    Simple exact match comparison (case-insensitive).
    """

    def compare(
        self,
        email1: Optional[str],
        email2: Optional[str]
    ) -> Tuple[float, str]:
        """
        Compare two email addresses.

        Args:
            email1: First email address (optional)
            email2: Second email address (optional)

        Returns:
            Tuple of (score, method):
            - 1.0: Exact match (case-insensitive)
            - 0.0: Different or missing

        Example:
            >>> comp = EmailComparator()
            >>> score, method = comp.compare("john@example.com", "JOHN@example.com")
            >>> score
            1.0
        """
        if not email1 or not email2:
            return 0.0, "missing_email"

        if email1.lower() == email2.lower():
            return 1.0, "exact_match"

        return 0.0, "no_match"
