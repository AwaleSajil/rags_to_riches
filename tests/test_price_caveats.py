"""What a past purchase price really means.

Two things on a receipt are called "savings" and behave oppositely:

  * `item_savings` is a markdown already netted out of the unit price —
    informational, never subtracted again;
  * `discount_total` is an order coupon already subtracted from the amount.

Subtracting either a second time is the mistake these tests exist to catch, and
it matters more now than it used to: comparison is the agent's job, and an LLL
handed `unit_quantity_subtotal 1.00, item_savings 2.71` will reach for the
subtraction. These helpers hand it pre-labelled facts instead.

The observation-side tests that used to live here (is_baseline_quality,
price_caveats) were removed in migration 028. PriceObservation no longer stores
is_promotional / promo_text / promo_ends_on / expires_on as structured columns —
they are one free-text `observed_context` field in the tag's own words, and
whether a price should be read at face value is now the agent's judgement rather
than a rule over flags a vision model guessed at.
"""

import pytest

from backend.services.price_service import (
    purchase_caveats,
    purchase_regular_unit_price,
    purchase_was_discounted,
)


# --- the past purchase it is compared against --------------------------------

def test_a_full_price_purchase_is_not_discounted():
    assert not purchase_was_discounted({"item_savings": 0, "unit_quantity_subtotal": 3.49})


def test_a_marked_down_purchase_is_detected():
    assert purchase_was_discounted({"item_savings": 1.50, "unit_quantity_subtotal": 3.49})


def test_an_order_coupon_marks_the_whole_trip():
    assert purchase_was_discounted(
        {"item_savings": 0}, {"discount_total": 5.00}
    )


def test_regular_price_adds_back_the_markdown_per_unit():
    """item_savings is the markdown for the whole LINE, so a 2-quantity line
    with $1.00 saved was 50c off each — not $1.00 off each."""
    assert purchase_regular_unit_price(
        {"unit_quantity_subtotal": 3.00, "item_savings": 1.00, "item_quantity": 2}
    ) == 3.50


def test_regular_price_of_an_undiscounted_line_is_what_was_paid():
    assert purchase_regular_unit_price(
        {"unit_quantity_subtotal": 3.49, "item_savings": 0, "item_quantity": 1}
    ) == 3.49


def test_regular_price_defaults_quantity_to_one():
    assert purchase_regular_unit_price(
        {"unit_quantity_subtotal": 3.00, "item_savings": 0.50}
    ) == 3.50


@pytest.mark.parametrize("detail", [
    {},
    {"unit_quantity_subtotal": 0},
    {"unit_quantity_subtotal": "oops"},
])
def test_regular_price_gives_up_cleanly(detail):
    assert purchase_regular_unit_price(detail) is None


def test_purchase_caveat_reports_the_markdown_and_the_regular_price():
    caveats = purchase_caveats(
        {"unit_quantity_subtotal": 3.00, "item_savings": 1.00, "item_quantity": 2}
    )
    assert caveats[0]["code"] == "purchase_marked_down"
    assert "$3.50" in caveats[0]["message"]


def test_order_coupon_is_reported_but_not_split_across_items():
    """Pro-rating "$5 off $50" over line items would invent a per-item price
    that was never on any shelf."""
    caveats = purchase_caveats({"item_savings": 0}, {"discount_total": 5.00})
    assert [c["code"] for c in caveats] == ["order_coupon"]
    assert "$5.00" in caveats[0]["message"]


def test_a_plain_purchase_has_no_caveats():
    assert purchase_caveats({"unit_quantity_subtotal": 3.49, "item_savings": 0}, {}) == []
