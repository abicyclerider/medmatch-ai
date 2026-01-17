#!/usr/bin/env python3
"""
Test script for field comparators.

This script validates that the NameComparator, DateComparator, and
AddressComparator are working correctly with various test cases.
"""

from datetime import date
from src.medmatch.matching.comparators import (
    NameComparator,
    DateComparator,
    AddressComparator,
    PhoneComparator,
    EmailComparator,
)
from src.medmatch.data.models.patient import Address


def test_name_comparator():
    """Test name comparison with various scenarios."""
    print("=" * 60)
    print("Testing NameComparator")
    print("=" * 60)

    comp = NameComparator()

    test_cases = [
        ("John", "John", "Exact match"),
        ("William", "Bill", "Known variation (nickname)"),
        ("Smith", "Smyth", "Typo (1 char)"),
        ("Johnson", "Jonson", "Typo (1 char)"),
        ("Smith", "Smith", "Soundex match"),
        ("Catherine", "Katherine", "Soundex match"),
        ("John", "Jane", "First initial match"),
        ("Alice", "Bob", "No match"),
    ]

    for name1, name2, description in test_cases:
        score, method = comp.compare(name1, name2)
        print(f"\n{description}:")
        print(f"  '{name1}' vs '{name2}'")
        print(f"  Score: {score:.2f}, Method: {method}")


def test_date_comparator():
    """Test date comparison with various scenarios."""
    print("\n" + "=" * 60)
    print("Testing DateComparator")
    print("=" * 60)

    comp = DateComparator()

    test_cases = [
        (date(1980, 3, 15), date(1980, 3, 15), "Exact match"),
        (date(1980, 3, 15), date(1980, 3, 16), "Twins (1 day apart)"),
        (date(1980, 3, 15), date(1980, 12, 3), "Month/day swap (3/15 vs 12/3)"),
        (date(1980, 3, 12), date(1980, 3, 21), "Transposed digits (12 vs 21)"),
        (date(1980, 3, 15), date(1981, 3, 15), "Year typo (off by 1 year)"),
        (date(1980, 3, 15), date(1985, 3, 15), "Same month/day, different year"),
        (date(1980, 3, 15), date(1980, 8, 22), "No match"),
    ]

    for dob1, dob2, description in test_cases:
        try:
            score, method = comp.compare(dob1, dob2)
            print(f"\n{description}:")
            print(f"  {dob1} vs {dob2}")
            print(f"  Score: {score:.2f}, Method: {method}")
        except Exception as e:
            print(f"\n{description}:")
            print(f"  {dob1} vs {dob2}")
            print(f"  Error: {e}")


def test_address_comparator():
    """Test address comparison with various scenarios."""
    print("\n" + "=" * 60)
    print("Testing AddressComparator")
    print("=" * 60)

    comp = AddressComparator()

    addr1 = Address(street="123 Main St", city="Boston", state="MA", zip_code="02108")
    addr2_exact = Address(street="123 Main St", city="Boston", state="MA", zip_code="02108")
    addr2_zip = Address(street="123 Main St", city="Boston", state="MA", zip_code="02109")
    addr2_city = Address(street="456 Elm St", city="Boston", state="MA", zip_code="02115")
    addr2_state = Address(street="789 Oak Ave", city="Cambridge", state="MA", zip_code="02138")
    addr2_different = Address(street="999 Park Blvd", city="New York", state="NY", zip_code="10001")

    test_cases = [
        (addr1, addr2_exact, "Exact match"),
        (addr1, addr2_zip, "Same street+city, different zip"),
        (addr1, addr2_city, "Same city+state, different street"),
        (addr1, addr2_state, "Same state, different city"),
        (addr1, addr2_different, "Completely different"),
        (addr1, None, "Missing address"),
    ]

    for a1, a2, description in test_cases:
        score, method = comp.compare(a1, a2)
        print(f"\n{description}:")
        if a2:
            print(f"  {a1}")
            print(f"  {a2}")
        else:
            print(f"  {a1}")
            print(f"  None")
        print(f"  Score: {score:.2f}, Method: {method}")


def test_phone_comparator():
    """Test phone number comparison."""
    print("\n" + "=" * 60)
    print("Testing PhoneComparator")
    print("=" * 60)

    comp = PhoneComparator()

    test_cases = [
        ("(617) 555-1234", "617-555-1234", "Same phone, different format"),
        ("6175551234", "617.555.1234", "Same phone, different format"),
        ("(617) 555-1234", "(617) 555-5678", "Different phones"),
        (None, "617-555-1234", "Missing phone"),
    ]

    for phone1, phone2, description in test_cases:
        score, method = comp.compare(phone1, phone2)
        print(f"\n{description}:")
        print(f"  '{phone1}' vs '{phone2}'")
        print(f"  Score: {score:.2f}, Method: {method}")


def test_email_comparator():
    """Test email comparison."""
    print("\n" + "=" * 60)
    print("Testing EmailComparator")
    print("=" * 60)

    comp = EmailComparator()

    test_cases = [
        ("john@example.com", "john@example.com", "Exact match"),
        ("john@example.com", "JOHN@example.com", "Case insensitive match"),
        ("john@example.com", "jane@example.com", "Different emails"),
        (None, "john@example.com", "Missing email"),
    ]

    for email1, email2, description in test_cases:
        score, method = comp.compare(email1, email2)
        print(f"\n{description}:")
        print(f"  '{email1}' vs '{email2}'")
        print(f"  Score: {score:.2f}, Method: {method}")


def main():
    """Run all comparator tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Field Comparator Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    test_name_comparator()
    test_date_comparator()
    test_address_comparator()
    test_phone_comparator()
    test_email_comparator()

    print("\n" + "=" * 60)
    print("✓ All comparator tests completed!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
