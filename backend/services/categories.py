"""Canonical transaction categories used by receipt review and edits."""

import re


_CATEGORY_ALIASES = {
    "grocery": "Groceries",
    "groceries": "Groceries",
    "supermarket": "Groceries",
    "supermarkets": "Groceries",
    "restaurant": "Dining",
    "restaurants": "Dining",
    "dining": "Dining",
    "food dining": "Dining",
    "takeout": "Dining",
    "transport": "Transportation",
    "transportation": "Transportation",
    "gas": "Transportation",
    "fuel": "Transportation",
    "rideshare": "Transportation",
    "shopping": "Shopping",
    "retail": "Shopping",
    "entertainment": "Entertainment",
    "utilities": "Utilities",
    "utility": "Utilities",
    "healthcare": "Healthcare",
    "medical": "Healthcare",
    "travel": "Travel",
    "personal care": "Personal Care",
    "uncategorized": "Uncategorized",
}


def normalize_category(value: str | None) -> str:
    """Map spelling/case/singular variants to one stable category label."""
    raw = (value or "").strip()
    if not raw:
        return "Uncategorized"
    key = re.sub(r"[^a-z]+", " ", raw.lower()).strip()
    return _CATEGORY_ALIASES.get(key, raw.title())
