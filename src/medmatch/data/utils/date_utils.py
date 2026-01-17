"""
Date formatting and variation utilities for synthetic data generation.

Handles different date formats and introduces realistic date entry errors.
"""

import random
from datetime import date, datetime, timedelta
from typing import Optional, Tuple


class DateFormatter:
    """Utility class for formatting dates in various styles."""

    @staticmethod
    def format_date(d: date, format_style: str = "standard") -> str:
        """
        Format a date in various common styles.

        Args:
            d: The date to format
            format_style: One of:
                - "standard": MM/DD/YYYY
                - "iso": YYYY-MM-DD
                - "short_year": MM/DD/YY
                - "month_name": Month DD, YYYY
                - "dots": MM.DD.YYYY
                - "no_zero": M/D/YYYY (no leading zeros)
                - "slash_reverse": DD/MM/YYYY

        Returns:
            Formatted date string
        """
        if format_style == "standard":
            return d.strftime("%m/%d/%Y")
        elif format_style == "iso":
            return d.strftime("%Y-%m-%d")
        elif format_style == "short_year":
            return d.strftime("%m/%d/%y")
        elif format_style == "month_name":
            return d.strftime("%B %d, %Y")
        elif format_style == "dots":
            return d.strftime("%m.%d.%Y")
        elif format_style == "no_zero":
            return f"{d.month}/{d.day}/{d.year}"
        elif format_style == "slash_reverse":
            return d.strftime("%d/%m/%Y")
        else:
            return d.strftime("%m/%d/%Y")

    @staticmethod
    def random_format(d: date) -> str:
        """
        Format a date in a randomly selected style.

        Args:
            d: The date to format

        Returns:
            Randomly formatted date string
        """
        formats = ["standard", "iso", "short_year", "month_name", "dots", "no_zero"]
        return DateFormatter.format_date(d, random.choice(formats))


class DateErrorGenerator:
    """Generate realistic date entry errors."""

    @staticmethod
    def transpose_digits(d: date) -> date:
        """
        Transpose adjacent digits in day or month.

        Args:
            d: Original date

        Returns:
            Date with transposed digits (might be invalid, needs checking)
        """
        # Decide whether to transpose day or month
        if random.choice([True, False]) and d.day >= 10:
            # Transpose day digits
            day_str = str(d.day)
            new_day = int(day_str[1] + day_str[0])
        else:
            new_day = d.day

        if random.choice([True, False]) and d.month >= 10:
            # Transpose month digits
            month_str = str(d.month)
            new_month = int(month_str[1] + month_str[0])
        else:
            new_month = d.month

        # Validate the result
        try:
            return date(d.year, new_month, new_day)
        except ValueError:
            # If invalid, just return original
            return d

    @staticmethod
    def swap_month_day(d: date) -> date:
        """
        Swap month and day (common with MM/DD vs DD/MM confusion).

        Args:
            d: Original date

        Returns:
            Date with month and day swapped (if valid)
        """
        try:
            return date(d.year, d.day, d.month)
        except ValueError:
            # If swap creates invalid date, return original
            return d

    @staticmethod
    def transpose_year_digits(d: date) -> date:
        """
        Transpose two adjacent digits in the year.

        Args:
            d: Original date

        Returns:
            Date with year digits transposed
        """
        year_str = str(d.year)
        # Choose position to transpose (0-2 for 4-digit year)
        pos = random.randint(0, 2)
        new_year = list(year_str)
        new_year[pos], new_year[pos + 1] = new_year[pos + 1], new_year[pos]

        try:
            return date(int(''.join(new_year)), d.month, d.day)
        except ValueError:
            return d

    @staticmethod
    def off_by_one_day(d: date) -> date:
        """
        Change date by ±1 day (simple data entry error).

        Args:
            d: Original date

        Returns:
            Date one day before or after
        """
        delta = timedelta(days=random.choice([-1, 1]))
        return d + delta

    @staticmethod
    def off_by_one_month(d: date) -> date:
        """
        Change date by ±1 month (data entry error).

        Args:
            d: Original date

        Returns:
            Date one month before or after (approximately)
        """
        # Simple approximation - just change month number
        new_month = d.month + random.choice([-1, 1])

        # Wrap around
        if new_month < 1:
            new_month = 12
            new_year = d.year - 1
        elif new_month > 12:
            new_month = 1
            new_year = d.year + 1
        else:
            new_year = d.year

        # Handle day overflow (e.g., Jan 31 -> Feb 31 invalid)
        new_day = min(d.day, 28)  # Safe day that exists in all months

        try:
            return date(new_year, new_month, new_day)
        except ValueError:
            return d

    @staticmethod
    def apply_random_error(d: date, error_type: Optional[str] = None) -> date:
        """
        Apply a random date error.

        Args:
            d: Original date
            error_type: Specific error type, or None for random selection
                       ("transpose_digits", "swap_month_day", "transpose_year",
                        "off_by_one_day", "off_by_one_month")

        Returns:
            Date with error applied
        """
        if error_type is None:
            error_type = random.choice([
                "transpose_digits",
                "swap_month_day",
                "transpose_year",
                "off_by_one_day",
                "off_by_one_month"
            ])

        if error_type == "transpose_digits":
            return DateErrorGenerator.transpose_digits(d)
        elif error_type == "swap_month_day":
            return DateErrorGenerator.swap_month_day(d)
        elif error_type == "transpose_year":
            return DateErrorGenerator.transpose_year_digits(d)
        elif error_type == "off_by_one_day":
            return DateErrorGenerator.off_by_one_day(d)
        elif error_type == "off_by_one_month":
            return DateErrorGenerator.off_by_one_month(d)
        else:
            return d


class DateGenerator:
    """Generate dates for synthetic patients."""

    @staticmethod
    def generate_dob(min_age: int = 18, max_age: int = 90) -> date:
        """
        Generate a random date of birth.

        Args:
            min_age: Minimum age in years
            max_age: Maximum age in years

        Returns:
            Random date of birth
        """
        today = date.today()
        # Calculate date range
        max_date = today - timedelta(days=min_age * 365)
        min_date = today - timedelta(days=max_age * 365)

        # Random date in range
        days_diff = (max_date - min_date).days
        random_days = random.randint(0, days_diff)
        return min_date + timedelta(days=random_days)

    @staticmethod
    def generate_twin_dob(reference_dob: date) -> date:
        """
        Generate a DOB for a twin (same day, or very close for fraternal).

        Args:
            reference_dob: DOB of the first twin

        Returns:
            DOB for the second twin
        """
        # 80% identical (same day), 20% fraternal (within 2 days)
        if random.random() < 0.8:
            return reference_dob
        else:
            # Fraternal twins born 1-2 days apart (rare but happens)
            delta = timedelta(days=random.randint(1, 2))
            return reference_dob + delta

    @staticmethod
    def generate_sibling_dob(reference_dob: date, min_gap_years: int = 1,
                            max_gap_years: int = 8) -> date:
        """
        Generate DOB for a sibling.

        Args:
            reference_dob: DOB of reference sibling
            min_gap_years: Minimum age gap
            max_gap_years: Maximum age gap

        Returns:
            DOB for sibling
        """
        # Random gap in years
        gap_years = random.randint(min_gap_years, max_gap_years)
        # Decide if older or younger
        if random.choice([True, False]):
            gap_years = -gap_years

        # Approximate year gap (ignore leap years for simplicity)
        sibling_dob = reference_dob + timedelta(days=gap_years * 365)

        return sibling_dob

    @staticmethod
    def generate_record_date(dob: date, min_age_at_record: int = 18) -> date:
        """
        Generate a record date (when medical record was created).

        Args:
            dob: Patient's date of birth
            min_age_at_record: Minimum age when record was created

        Returns:
            Record date
        """
        today = date.today()
        earliest_date = dob + timedelta(days=min_age_at_record * 365)

        if earliest_date > today:
            earliest_date = dob + timedelta(days=365)  # At least 1 year old

        days_range = (today - earliest_date).days
        if days_range <= 0:
            return today

        random_days = random.randint(0, days_range)
        return earliest_date + timedelta(days=random_days)
