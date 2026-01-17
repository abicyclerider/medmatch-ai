"""
Name variation utilities for synthetic data generation.

Handles name variations, nicknames, cultural patterns, and common names.
"""

import random
from typing import List, Dict, Tuple, Optional


# Common first name variations (formal -> informal/nicknames)
NAME_VARIATIONS: Dict[str, List[str]] = {
    # Male names
    "William": ["Bill", "Billy", "Will", "Willie", "Liam"],
    "Robert": ["Rob", "Bob", "Bobby", "Robbie"],
    "Richard": ["Rick", "Dick", "Rich", "Ricky"],
    "James": ["Jim", "Jimmy", "Jamie"],
    "Michael": ["Mike", "Mickey", "Mikey"],
    "Thomas": ["Tom", "Tommy"],
    "Charles": ["Charlie", "Chuck", "Chas"],
    "Christopher": ["Chris", "Topher"],
    "Daniel": ["Dan", "Danny"],
    "Matthew": ["Matt", "Matty"],
    "Joseph": ["Joe", "Joey"],
    "David": ["Dave", "Davey"],
    "Anthony": ["Tony"],
    "Donald": ["Don", "Donnie"],
    "Steven": ["Steve", "Stevie"],
    "Andrew": ["Andy", "Drew"],
    "Joshua": ["Josh"],
    "Nicholas": ["Nick", "Nicky"],
    "Alexander": ["Alex", "Xander"],
    "Benjamin": ["Ben", "Benny", "Benji"],
    "Samuel": ["Sam", "Sammy"],
    "Jonathan": ["Jon", "Johnny"],
    "Timothy": ["Tim", "Timmy"],
    "Kenneth": ["Ken", "Kenny"],

    # Female names
    "Elizabeth": ["Beth", "Liz", "Lizzy", "Betty", "Eliza"],
    "Margaret": ["Maggie", "Meg", "Peggy", "Marge"],
    "Catherine": ["Cathy", "Kate", "Katie", "Kat"],
    "Jennifer": ["Jen", "Jenny", "Jenn"],
    "Jessica": ["Jess", "Jessie"],
    "Patricia": ["Pat", "Patty", "Trish"],
    "Barbara": ["Barb", "Barbie", "Babs"],
    "Susan": ["Sue", "Susie", "Suzy"],
    "Sarah": ["Sally", "Sara"],
    "Nancy": ["Nan"],
    "Dorothy": ["Dot", "Dottie", "Dory"],
    "Rebecca": ["Becky", "Becca"],
    "Michelle": ["Shelly", "Mickey"],
    "Kimberly": ["Kim", "Kimmy"],
    "Amanda": ["Mandy", "Amy"],
    "Samantha": ["Sam", "Sammy"],
    "Victoria": ["Vicky", "Tori"],
    "Christina": ["Chris", "Tina", "Christie"],
    "Deborah": ["Debbie", "Deb"],
    "Katherine": ["Kathy", "Kate", "Katie", "Kat"],
}

# Very common names that appear frequently
COMMON_MALE_NAMES = [
    "John", "James", "Robert", "Michael", "William",
    "David", "Richard", "Joseph", "Thomas", "Charles"
]

COMMON_FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara",
    "Elizabeth", "Susan", "Jessica", "Sarah", "Karen"
]

COMMON_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott",
    "Torres", "Nguyen", "Hill", "Flores", "Green"
]

# Names with accent variations
ACCENT_VARIATIONS: Dict[str, str] = {
    "Jose": "José",
    "Maria": "María",
    "Angel": "Ángel",
    "Raul": "Raúl",
    "Sofia": "Sofía",
    "Rene": "René",
}

# Common misspellings for data entry errors
COMMON_MISSPELLINGS: Dict[str, List[str]] = {
    "Smith": ["Smyth", "Smythe", "Schmidt"],
    "Johnson": ["Jonson", "Johnsen"],
    "Brown": ["Browne"],
    "Martinez": ["Martines", "Martínez"],
    "Garcia": ["Garciá", "García"],
    "Anderson": ["Andersen", "Andrson"],
}


def get_name_variations(name: str) -> List[str]:
    """
    Get all variations of a given name.

    Args:
        name: The formal/full name

    Returns:
        List of name variations (including the original)
    """
    if name in NAME_VARIATIONS:
        return [name] + NAME_VARIATIONS[name]
    return [name]


def get_random_variation(name: str) -> str:
    """
    Get a random variation of a name.

    Args:
        name: The formal/full name

    Returns:
        A random variation (might be the same if no variations exist)
    """
    variations = get_name_variations(name)
    return random.choice(variations)


def apply_accent_variation(name: str, remove_accent: bool = None) -> str:
    """
    Apply or remove accent marks from names.

    Args:
        name: The name to modify
        remove_accent: If True, removes accents; if False, adds them;
                      if None, randomly decides

    Returns:
        Name with accents modified
    """
    if remove_accent is None:
        remove_accent = random.choice([True, False])

    if remove_accent:
        # Try to remove accent
        for plain, accented in ACCENT_VARIATIONS.items():
            if name == accented:
                return plain
    else:
        # Try to add accent
        if name in ACCENT_VARIATIONS:
            return ACCENT_VARIATIONS[name]

    return name


def apply_misspelling(name: str) -> str:
    """
    Apply a common misspelling to a name.

    Args:
        name: The correct name

    Returns:
        Misspelled version or original if no common misspellings
    """
    if name in COMMON_MISSPELLINGS:
        return random.choice(COMMON_MISSPELLINGS[name])
    return name


def generate_common_name_pair(gender: str) -> Tuple[str, str]:
    """
    Generate a common first/last name combination.

    Args:
        gender: "M" or "F"

    Returns:
        Tuple of (first_name, last_name)
    """
    if gender == "M":
        first = random.choice(COMMON_MALE_NAMES)
    else:
        first = random.choice(COMMON_FEMALE_NAMES)

    last = random.choice(COMMON_LAST_NAMES)
    return first, last


def format_name_variation(first: str, middle: Optional[str], last: str,
                         style: str = "standard") -> Tuple[str, Optional[str], str]:
    """
    Format a name in different styles.

    Args:
        first: First name
        middle: Middle name (optional)
        last: Last name
        style: "standard", "last_first", "initial_only", "no_middle"

    Returns:
        Tuple of (first, middle, last) in the requested format
    """
    if style == "standard":
        return first, middle, last
    elif style == "last_first":
        # Note: This is for display only, actual structure stays same
        return first, middle, last
    elif style == "initial_only":
        # Middle name becomes initial
        if middle:
            middle = middle[0] + "."
        return first, middle, last
    elif style == "no_middle":
        return first, None, last
    else:
        return first, middle, last


def transpose_characters(text: str, num_errors: int = 1) -> str:
    """
    Introduce typos by transposing adjacent characters.

    Args:
        text: The text to modify
        num_errors: Number of transpositions to make

    Returns:
        Text with transposed characters
    """
    if len(text) < 2:
        return text

    result = list(text)
    for _ in range(num_errors):
        # Find valid positions (not last character)
        valid_positions = list(range(len(result) - 1))
        if not valid_positions:
            break

        pos = random.choice(valid_positions)
        # Swap with next character
        result[pos], result[pos + 1] = result[pos + 1], result[pos]

    return ''.join(result)


def generate_married_name_change(original_last: str, new_last: str = None) -> str:
    """
    Generate a married/maiden name change.

    Args:
        original_last: Original last name
        new_last: New last name (if None, generates a random common one)

    Returns:
        New last name
    """
    if new_last is None:
        new_last = random.choice(COMMON_LAST_NAMES)
    return new_last


class NameGenerator:
    """Helper class for generating and varying names."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize name generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate_common_name(self, gender: str) -> Tuple[str, str, str]:
        """
        Generate a common name that might appear multiple times.

        Returns:
            Tuple of (first, middle_initial, last)
        """
        first, last = generate_common_name_pair(gender)
        middle = random.choice(["A", "B", "C", "D", "E", "J", "L", "M", "R", "S"])
        return first, middle, last

    def apply_variation(self, first: str, middle: Optional[str], last: str,
                       variation_type: str) -> Tuple[str, Optional[str], str]:
        """
        Apply a specific type of variation to a name.

        Args:
            first: First name
            middle: Middle name
            last: Last name
            variation_type: Type of variation to apply
                          ("nickname", "no_middle", "middle_initial",
                           "accent", "misspelling", "typo")

        Returns:
            Modified (first, middle, last) tuple
        """
        if variation_type == "nickname":
            first = get_random_variation(first)
        elif variation_type == "no_middle":
            middle = None
        elif variation_type == "middle_initial":
            if middle and len(middle) > 1:
                middle = middle[0]
        elif variation_type == "accent":
            first = apply_accent_variation(first)
        elif variation_type == "misspelling":
            last = apply_misspelling(last)
        elif variation_type == "typo":
            # Random typo in last name
            last = transpose_characters(last)

        return first, middle, last
