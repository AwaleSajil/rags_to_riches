"""Package sizes and unit prices.

Every price comparison the app shows runs through this module. A bug here does
not produce an obvious error — it produces a confident, wrong recommendation
("this is 20% cheaper!") which is worse than showing nothing, so the refusal
cases matter as much as the conversions.
"""

import pytest

from backend.services.units import (
    COUNT,
    MASS,
    VOLUME,
    Size,
    comparable,
    format_unit_price,
    parse_size,
    unit_price,
)


# --- parsing -----------------------------------------------------------------

@pytest.mark.parametrize("text,value,unit", [
    ("12 oz", 12.0, "oz"),
    ("12oz", 12.0, "oz"),
    ("12 OZ", 12.0, "oz"),
    ("1.5 L", 1.5, "l"),
    ("500g", 500.0, "g"),
    ("2 lb", 2.0, "lb"),
    ("2 lbs", 2.0, "lb"),
    ("1 gal", 1.0, "gal"),
    ("Cheerios Toasted Whole Grain 12 oz", 12.0, "oz"),
    ("Milk, 1 Gallon Jug", 1.0, "gal"),
])
def test_parses_common_sizes(text, value, unit):
    size = parse_size(text)
    assert size == Size(value, unit)


def test_fluid_ounces_are_volume_not_mass():
    """12 oz of cereal and 12 fl oz of soda are different quantities. Collapsing
    them would silently compare a weight against a volume."""
    assert parse_size("12 fl oz").family == VOLUME
    assert parse_size("12 oz").family == MASS
    assert not comparable(parse_size("12 fl oz"), parse_size("12 oz"))


def test_fl_oz_is_not_swallowed_by_the_oz_pattern():
    size = parse_size("Soda 12 fl oz")
    assert size == Size(12.0, "fl oz")


@pytest.mark.parametrize("text,expected", [
    ("6 x 12 oz", Size(72.0, "oz")),
    ("6x12oz", Size(72.0, "oz")),
    ("4 × 500 ml", Size(2000.0, "ml")),
])
def test_multipacks_multiply_out(text, expected):
    """"6 x 12 oz" is 72 oz. Reading it as 12 oz understates unit price 6-fold —
    the difference between a good deal and a bad one."""
    assert parse_size(text) == expected


@pytest.mark.parametrize("text,count", [
    ("6 pack", 6.0),
    ("pack of 6", 6.0),
    ("12 ct", 12.0),
    ("12-pack", 12.0),
    ("dozen", 12.0),
])
def test_bare_counts(text, count):
    size = parse_size(text)
    assert size == Size(count, "ct")
    assert size.family == COUNT


@pytest.mark.parametrize("text", ["", None, "Cheerios", "great value", "$4.99"])
def test_no_size_returns_none(text):
    """A tag with no size is normal, not an error."""
    assert parse_size(text) is None


# --- conversion --------------------------------------------------------------

@pytest.mark.parametrize("size,grams", [
    (Size(1, "g"), 1.0),
    (Size(1, "kg"), 1000.0),
    (Size(1, "oz"), 28.349523125),
    (Size(1, "lb"), 453.59237),
])
def test_mass_to_base(size, grams):
    assert size.to_base() == pytest.approx(grams)


@pytest.mark.parametrize("size,ml", [
    (Size(1, "ml"), 1.0),
    (Size(1, "l"), 1000.0),
    (Size(1, "fl oz"), 29.5735295625),
    (Size(1, "gal"), 3785.411784),
])
def test_volume_to_base(size, ml):
    assert size.to_base() == pytest.approx(ml)


def test_unknown_unit_has_no_base_or_family():
    unknown = Size(5, "widgets")
    assert unknown.to_base() is None
    assert unknown.family is None


# --- comparability -----------------------------------------------------------

def test_same_family_is_comparable():
    assert comparable(parse_size("12 oz"), parse_size("500 g"))
    assert comparable(parse_size("1 L"), parse_size("12 fl oz"))


@pytest.mark.parametrize("a,b", [
    ("12 oz", "12 fl oz"),   # mass vs volume
    ("12 oz", "6 ct"),       # mass vs count
    ("1 L", "6 pack"),       # volume vs count
])
def test_cross_family_is_refused(a, b):
    assert not comparable(parse_size(a), parse_size(b))


def test_unknown_size_compares_to_nothing():
    """Falling back to same-item-only comparison is the caller's job; this must
    not quietly say yes."""
    assert not comparable(None, parse_size("12 oz"))
    assert not comparable(parse_size("12 oz"), None)
    assert not comparable(None, None)


# --- unit price --------------------------------------------------------------

def test_unit_price_is_per_base_unit():
    # $4.99 for 12 oz -> per gram
    assert unit_price(4.99, Size(12, "oz")) == pytest.approx(4.99 / 340.194, rel=1e-4)


def test_bigger_pack_can_be_the_better_deal():
    """The case that motivates the whole module: the cheaper sticker price is
    the worse value."""
    small = unit_price(4.99, parse_size("12 oz"))
    large = unit_price(6.49, parse_size("20 oz"))
    assert large < small


def test_unit_price_needs_both_inputs():
    assert unit_price(None, Size(12, "oz")) is None
    assert unit_price(4.99, None) is None
    assert unit_price(4.99, Size(5, "widgets")) is None


def test_zero_size_does_not_divide_by_zero():
    assert unit_price(4.99, Size(0, "oz")) is None


def test_free_item_is_zero_not_none():
    assert unit_price(0, Size(12, "oz")) == 0


# --- display -----------------------------------------------------------------

def test_display_uses_the_unit_the_shopper_saw():
    """Per-gram figures are unreadable; show $/oz for an item sold in oz."""
    assert format_unit_price(4.29, parse_size("12 oz")) == "$0.36/oz"


def test_display_for_count_items():
    assert format_unit_price(3.00, parse_size("12 ct")) == "$0.25/ct"


def test_display_none_without_a_size():
    assert format_unit_price(4.29, None) is None


def test_fluid_ounces_and_ounces_are_different_families():
    """'oz' is a weight, 'fl oz' is a volume, and conflating them is what filed a
    gallon of milk as 128 ounces of mass — after which it could never be
    compared against another gallon. The vision prompt offers both so the model
    can say which it saw; only it can tell, since it is looking at the product."""
    from backend.services.units import comparable, parse_size

    fluid = parse_size("128 fl oz")
    gallon = parse_size("1 gal")
    weight = parse_size("128 oz")

    assert comparable(fluid, gallon) is True
    assert comparable(weight, gallon) is False
    # And when read correctly, 128 fl oz IS a gallon.
    assert round(fluid.to_base() / gallon.to_base(), 3) == 1.0
