"""Receipt totals.

Two different things on a receipt are called "savings" and they behave
oppositely, which is exactly why this needs tests:

  * item markdowns (`item_savings`) are already netted out of the unit price —
    informational, never subtracted again;
  * order-level coupons (`discount_total`) reduce the whole basket and are the
    only savings that come off the amount.

Subtracting the savings recap is the mistake these tests exist to catch.
"""

import pytest

from backend.services.categories import normalize_category
from backend.services.transaction_service import (
    _header_totals_from_details,
    _prepare_detail_rows,
)


def detail(subtotal, tax=0.0, savings=0.0, rate=0.0):
    return {
        "item_subtotal_price": subtotal,
        "tax_amount": tax,
        "item_savings": savings,
        "tax_rate": rate,
    }


# --- header rollup -----------------------------------------------------------

def test_totals_sum_line_items():
    subtotal, tax_total, amount, breakdown, savings = _header_totals_from_details(
        [detail(10.00, 0.80, rate=8.0), detail(5.00, 0.40, rate=8.0)], None
    )
    assert subtotal == 15.00
    assert tax_total == 1.20
    assert amount == 16.20
    assert savings == 0.0
    assert breakdown == [{"label": "8.0% tax", "rate": 8.0, "amount": 1.20}]


def test_item_markdowns_are_reported_but_not_subtracted():
    """The unit price is already the marked-down one, so amount must ignore it."""
    subtotal, _, amount, _, savings = _header_totals_from_details(
        [detail(10.00, savings=3.00), detail(5.00, savings=1.50)], None
    )
    assert subtotal == 15.00
    assert amount == 15.00, "item markdowns must not come off the total"
    assert savings == 4.50, "but they should still be reported to the shopper"


def test_order_coupon_is_subtracted_and_folded_into_savings():
    _, _, amount, _, savings = _header_totals_from_details(
        [detail(50.00)], None, discount_total=5.00
    )
    assert amount == 45.00
    assert savings == 5.00


def test_markdowns_and_coupon_together():
    """The coupon comes off once; the markdown only shows up in the recap."""
    _, _, amount, _, savings = _header_totals_from_details(
        [detail(20.00, 1.60, savings=4.00, rate=8.0)], None, discount_total=5.00
    )
    assert amount == pytest.approx(16.60)   # 20.00 + 1.60 - 5.00
    assert savings == pytest.approx(9.00)   # 4.00 markdown + 5.00 coupon


def test_multiple_tax_rates_grouped_separately():
    _, tax_total, _, breakdown, _ = _header_totals_from_details(
        [detail(10.00, 0.50, rate=5.0), detail(10.00, 1.30, rate=13.0),
         detail(10.00, 0.50, rate=5.0)],
        None,
    )
    assert tax_total == 2.30
    assert breakdown == [
        {"label": "5.0% tax", "rate": 5.0, "amount": 1.00},
        {"label": "13.0% tax", "rate": 13.0, "amount": 1.30},
    ]


def test_existing_rate_labels_are_preserved():
    """A receipt saying "GST" should keep saying GST after an edit."""
    _, _, _, breakdown, _ = _header_totals_from_details(
        [detail(10.00, 0.50, rate=5.0)],
        [{"label": "GST", "rate": 5.0, "amount": 0.50}],
    )
    assert breakdown == [{"label": "GST", "rate": 5.0, "amount": 0.50}]


def test_tax_exempt_items_leave_no_breakdown():
    _, tax_total, amount, breakdown, _ = _header_totals_from_details(
        [detail(10.00, 0.0, rate=0.0)], None
    )
    assert tax_total == 0.0
    assert amount == 10.00
    assert breakdown is None


def test_non_numeric_values_do_not_crash_the_rollup():
    subtotal, tax_total, amount, _, _ = _header_totals_from_details(
        [{"item_subtotal_price": None, "tax_amount": "oops", "tax_rate": None}], None
    )
    assert (subtotal, tax_total, amount) == (0.0, 0.0, 0.0)


def test_empty_receipt():
    assert _header_totals_from_details([], None) == (0.0, 0.0, 0.0, None, 0.0)


# --- line-item derivation ----------------------------------------------------

def test_derived_fields_filled_in_when_omitted():
    rows = _prepare_detail_rows(
        "user-1", "tx-1", "bill-1",
        [{"item_description": "Milk", "item_quantity": 2, "item_unit_subtotal_price": 3.00,
          "tax_rate": 10.0}],
    )
    row = rows[0]
    assert row["item_subtotal_price"] == 6.00      # 2 x 3.00
    assert row["tax_amount"] == 0.60              # 10% of 6.00
    assert row["item_total_price"] == 6.60        # post-tax
    assert row["taxable"] is True


def test_zero_rate_marks_item_non_taxable():
    rows = _prepare_detail_rows(
        "user-1", "tx-1", None,
        [{"item_description": "Bread", "item_quantity": 1,
          "item_unit_subtotal_price": 2.50, "tax_rate": 0}],
    )
    assert rows[0]["taxable"] is False
    assert rows[0]["tax_amount"] == 0.0
    assert rows[0]["item_total_price"] == 2.50


def test_explicit_values_are_not_recomputed():
    """A receipt's own printed totals win over our arithmetic."""
    rows = _prepare_detail_rows(
        "user-1", "tx-1", None,
        [{"item_description": "Odd", "item_quantity": 3, "item_unit_subtotal_price": 1.00,
          "item_subtotal_price": 2.99, "tax_amount": 0.11, "item_total_price": 3.10}],
    )
    assert rows[0]["item_subtotal_price"] == 2.99
    assert rows[0]["item_total_price"] == 3.10


def test_missing_quantity_defaults_to_one():
    rows = _prepare_detail_rows(
        "user-1", "tx-1", None,
        [{"item_description": "Single", "item_unit_subtotal_price": 4.25}],
    )
    assert rows[0]["item_subtotal_price"] == 4.25


def test_rows_carry_ownership_columns():
    rows = _prepare_detail_rows("user-1", "tx-1", "bill-1", [{"item_description": "X"}])
    assert rows[0]["user_id"] == "user-1"
    assert rows[0]["transaction_id"] == "tx-1"
    assert rows[0]["bill_file_id"] == "bill-1"


# --- category normalization --------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("grocery", "Groceries"),
    ("GROCERIES", "Groceries"),
    ("Supermarket", "Groceries"),
    ("restaurants", "Dining"),
    ("food & dining", "Dining"),
    ("gas", "Transportation"),
    ("", "Uncategorized"),
    (None, "Uncategorized"),
    ("  ", "Uncategorized"),
    ("something novel", "Something Novel"),
])
def test_category_normalization(raw, expected):
    assert normalize_category(raw) == expected
