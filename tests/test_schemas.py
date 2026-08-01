"""Request-body validation.

These fields feed the dedup content hash and the ledger display, so a blank or
malformed value is worth rejecting at the edge rather than storing.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from backend.schemas.transactions import ReceiptReviewInput, TransactionCreate


def make(**overrides):
    body = {"description": "Coffee", "amount": 4.50, "trans_date": date(2026, 1, 1)}
    body.update(overrides)
    return TransactionCreate(**body)


def test_minimal_valid_transaction():
    tx = make()
    assert tx.description == "Coffee"
    assert tx.category == "Uncategorized"


@pytest.mark.parametrize("description", ["", "   ", "\t\n"])
def test_blank_description_rejected(description):
    """A whitespace-only description used to reach `.split()[0]` and raise
    IndexError, surfacing as a 500."""
    with pytest.raises(ValidationError):
        make(description=description)


def test_description_is_stripped():
    assert make(description="  Coffee  ").description == "Coffee"


@pytest.mark.parametrize("amount", [0, -1, -0.01])
def test_non_positive_amount_rejected(amount):
    with pytest.raises(ValidationError):
        make(amount=amount)


# --- receipt review ----------------------------------------------------------

def review(**overrides):
    body = {"date": date(2026, 1, 1), "merchant_name": "Store"}
    body.update(overrides)
    return ReceiptReviewInput(**body)


def test_merchant_name_required():
    with pytest.raises(ValidationError):
        review(merchant_name="")


@pytest.mark.parametrize("value", ["14:30", "00:00", "23:59"])
def test_valid_times_accepted(value):
    assert review(time=value).time == value


@pytest.mark.parametrize("value", ["24:00", "12:60", "2:30pm", "1430", "noon"])
def test_invalid_times_rejected(value):
    with pytest.raises(ValidationError):
        review(time=value)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_time_normalises_to_none(value):
    assert review(time=value).time is None


def test_negative_discount_rejected():
    with pytest.raises(ValidationError):
        review(discount_total=-5)


def test_item_savings_cannot_be_negative():
    with pytest.raises(ValidationError):
        review(line_items=[{"item_description": "X", "item_savings": -1}])
