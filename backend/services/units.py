"""Package sizes and unit prices.

A price without a size is not comparable to anything. "$4.99" beats "$6.49"
only if the jars are the same size; at 12oz vs 20oz the cheaper jar is the worse
deal. Every comparison this app shows therefore runs through here first, and the
module is deliberately willing to answer "I don't know" — a refused comparison
is recoverable, a confidently wrong one is not.

Two traps this handles explicitly:

* **"oz" is ambiguous.** 12 oz of cereal is a weight; 12 fl oz of soda is a
  volume. They are not interchangeable, so "fl oz" parses to the volume family
  and bare "oz" to mass, and the two never compare.
* **Multipacks.** "6 x 12 oz" is 72 oz, not 6 and not 12. Getting this wrong
  understates unit price by the pack count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Unit families. Sizes only compare within a family — there is no meaningful
# conversion from grams to millilitres without knowing the substance.
MASS = "mass"
VOLUME = "volume"
COUNT = "count"

# unit -> (family, how many base units it is worth).
# Base units: grams, millilitres, and items.
_UNITS: dict[str, tuple[str, float]] = {
    # mass
    "g": (MASS, 1.0),
    "gram": (MASS, 1.0),
    "grams": (MASS, 1.0),
    "kg": (MASS, 1000.0),
    "kilogram": (MASS, 1000.0),
    "oz": (MASS, 28.349523125),
    "ounce": (MASS, 28.349523125),
    "ounces": (MASS, 28.349523125),
    "lb": (MASS, 453.59237),
    "lbs": (MASS, 453.59237),
    "pound": (MASS, 453.59237),
    "pounds": (MASS, 453.59237),
    # volume
    "ml": (VOLUME, 1.0),
    "milliliter": (VOLUME, 1.0),
    "millilitre": (VOLUME, 1.0),
    "l": (VOLUME, 1000.0),
    "liter": (VOLUME, 1000.0),
    "litre": (VOLUME, 1000.0),
    "fl oz": (VOLUME, 29.5735295625),
    "floz": (VOLUME, 29.5735295625),
    "fluid ounce": (VOLUME, 29.5735295625),
    "pt": (VOLUME, 473.176473),
    "pint": (VOLUME, 473.176473),
    "qt": (VOLUME, 946.352946),
    "quart": (VOLUME, 946.352946),
    "gal": (VOLUME, 3785.411784),
    "gallon": (VOLUME, 3785.411784),
    # count
    "ct": (COUNT, 1.0),
    "count": (COUNT, 1.0),
    "pk": (COUNT, 1.0),
    "pack": (COUNT, 1.0),
    "pack of": (COUNT, 1.0),
    "ea": (COUNT, 1.0),
    "each": (COUNT, 1.0),
    "roll": (COUNT, 1.0),
    "rolls": (COUNT, 1.0),
    "dozen": (COUNT, 12.0),
}

# Longest first, so "fl oz" is matched before "oz" would swallow the "oz" in it.
# Public because price_service strips the same unit vocabulary out of item names
# — two copies of this list would drift and break matching on the ones that
# disagreed.
UNIT_PATTERN = "|".join(
    re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)
)

_NUMBER = r"\d+(?:[.,]\d+)?"

# "6 x 12 oz", "6x12oz", "6 ct x 500 ml" — a pack count times a per-unit size.
_MULTIPACK_RE = re.compile(
    rf"(?P<packs>{_NUMBER})\s*(?:x|×|\*)\s*(?P<size>{_NUMBER})\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

# "12 oz", "1.5L", "500g", "12ct"
_SIZE_RE = re.compile(
    rf"(?P<size>{_NUMBER})\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

# "6 pack", "pack of 6", "dozen" — a bare count with no per-item size.
_BARE_PACK_RE = re.compile(
    rf"(?:pack\s+of\s+(?P<n1>{_NUMBER}))|(?:(?P<n2>{_NUMBER})\s*-?\s*(?:pack|pk|ct|count)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Size:
    """A parsed package size, normalised to one canonical unit label."""

    value: float
    unit: str

    @property
    def family(self) -> Optional[str]:
        entry = _UNITS.get(self.unit)
        return entry[0] if entry else None

    def to_base(self) -> Optional[float]:
        """Size expressed in grams / millilitres / items, or None if unknown."""
        entry = _UNITS.get(self.unit)
        if entry is None:
            return None
        return self.value * entry[1]


def _to_float(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _canonical_unit(raw: str) -> str:
    """Collapse spelling variants onto the label used in _UNITS."""
    unit = re.sub(r"\s+", " ", raw.strip().lower())
    aliases = {
        "floz": "fl oz",
        "fluid ounce": "fl oz",
        "ounce": "oz", "ounces": "oz",
        "gram": "g", "grams": "g",
        "kilogram": "kg",
        "pound": "lb", "pounds": "lb", "lbs": "lb",
        "milliliter": "ml", "millilitre": "ml",
        "liter": "l", "litre": "l",
        "pint": "pt", "quart": "qt", "gallon": "gal",
        "count": "ct", "pack": "ct", "pk": "ct", "each": "ct", "ea": "ct",
        "roll": "ct", "rolls": "ct",
    }
    return aliases.get(unit, unit)


def parse_size(text: Optional[str]) -> Optional[Size]:
    """Pull a package size out of free text, or None when there isn't one.

    Returning None is a normal outcome, not an error: plenty of tags show no
    size at all, and the caller degrades to same-item-only comparison.
    """
    if not text:
        return None
    haystack = str(text)

    # Multipack first — "6 x 12 oz" also matches the plain size pattern as
    # "12 oz", which would understate the package by the pack count.
    multipack = _MULTIPACK_RE.search(haystack)
    if multipack:
        packs = _to_float(multipack.group("packs"))
        size = _to_float(multipack.group("size"))
        unit = _canonical_unit(multipack.group("unit"))
        if packs and size and unit in _UNITS:
            return Size(round(packs * size, 4), unit)

    match = _SIZE_RE.search(haystack)
    if match:
        value = _to_float(match.group("size"))
        unit = _canonical_unit(match.group("unit"))
        if value and unit in _UNITS:
            return Size(round(value, 4), unit)

    bare = _BARE_PACK_RE.search(haystack)
    if bare:
        value = _to_float(bare.group("n1") or bare.group("n2") or "")
        if value:
            return Size(round(value, 4), "ct")

    if re.search(r"\bdozen\b", haystack, re.IGNORECASE):
        return Size(12.0, "ct")

    return None


def comparable(a: Optional[Size], b: Optional[Size]) -> bool:
    """True only when two sizes are in the same unit family.

    An unknown size compares to nothing — the caller must then fall back to
    matching the identical item rather than pretending the prices line up.
    """
    if a is None or b is None:
        return False
    family_a, family_b = a.family, b.family
    return family_a is not None and family_a == family_b


def unit_price(price: Optional[float], size: Optional[Size]) -> Optional[float]:
    """Price per base unit (per gram / per mL / per item), or None.

    This is the only honest number to compare two differently-sized packages
    with, so it is stored rather than derived at read time.
    """
    if price is None or size is None:
        return None
    base = size.to_base()
    if not base:
        return None
    try:
        return round(float(price) / base, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def format_unit_price(price: Optional[float], size: Optional[Size]) -> Optional[str]:
    """Human-facing unit price, shown in the unit the shopper actually saw.

    Per-gram figures are unreadable ($0.0127/g), so this scales back to the
    package's own unit: "$0.36/oz".
    """
    per_base = unit_price(price, size)
    if per_base is None or size is None:
        return None
    entry = _UNITS.get(size.unit)
    if entry is None:
        return None
    per_display_unit = per_base * entry[1]
    return f"${per_display_unit:.2f}/{size.unit}"


def scale_from_base(per_base: Optional[float], unit: Optional[str]) -> Optional[float]:
    """Turn a per-gram / per-mL / per-item price back into the shopper's unit.

    Per-base figures are the only honest thing to compare across sizes, and the
    only unreadable thing to show: $0.0044/g means nothing to anyone standing in
    a shop. Stored and compared in base units, rendered in theirs.
    """
    if per_base is None or not unit:
        return None
    entry = _UNITS.get(unit)
    if entry is None:
        return None
    return per_base * entry[1]
