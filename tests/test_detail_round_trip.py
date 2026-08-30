"""A line item must survive the round trip through the API unchanged.

Pydantic drops what a model does not declare, in BOTH directions: an undeclared
field is stripped off the request, and `response_model` filters it back off the
reply. So a column can exist in the database, be written by the service layer,
be read by the screen, and still be silently erased — with nothing anywhere
raising.

That is what happened to `item_quantity_unit`, `size_value` and `size_unit`.
`_prepare_detail_rows` carries them deliberately (its comment says omitting one
"wiped the unit off every line the moment a user touched an unrelated field"),
the editor sends them, the TypeScript types name them — and the schema in
between named none of the three, so every save nulled them and every read came
back blank. `_verify_receipt_row` reads `item.get("size_value")` off the review
payload for the same reason and never once saw one, which is why
TransactionDetail.size_value was NULL on every receipt the app has verified.

These fields are what a per-unit price comparison rests on. Without them
`purchase_unit_size` falls back to parsing the abbreviated description, which is
the guess migration 034 added the columns to replace ("+RED POTA 5L US#" is five
POUNDS, and reading it as litres turns a $4.99 bag into $1.00/l).
"""

import pytest

from backend.schemas.transactions import (
    ReceiptReviewLineItem,
    TransactionDetailInput,
    TransactionDetailItem,
)
from backend.services.transaction_service import _prepare_detail_rows

# The three that were dropped. Named once so a future field cannot be added to
# one direction and forgotten in the other.
CARRIED = ("item_quantity_unit", "size_value", "size_unit")


@pytest.mark.parametrize("field", CARRIED)
def test_the_request_schema_keeps_what_the_editor_sends(field):
    sent = {
        "item_description": "GV LF 2 GAL",
        "item_quantity": 1,
        "item_quantity_unit": "each",
        "size_value": 1,
        "size_unit": "gal",
        "unit_quantity_subtotal": 3.49,
    }
    assert TransactionDetailInput(**sent).model_dump()[field] == sent[field]


@pytest.mark.parametrize("field", CARRIED)
def test_the_response_schema_returns_what_the_row_holds(field):
    stored = {
        "id": "d1",
        "item_description": "GV LF 2 GAL",
        "item_quantity_unit": "each",
        "size_value": 1,
        "size_unit": "gal",
    }
    assert TransactionDetailItem(**stored).model_dump()[field] == stored[field]


@pytest.mark.parametrize("field", ("item_quantity_unit", "size_value", "size_unit"))
def test_the_receipt_review_schema_keeps_the_size_the_photo_showed(field):
    """The vision pass reads these off the label; verifying must not discard them."""
    read = {
        "item_description": "+RED POTA 5L US#",
        "item_quantity": 1,
        "item_quantity_unit": "each",
        "size_value": 5,
        "size_unit": "lb",
        "item_unit_price": 4.99,
    }
    assert ReceiptReviewLineItem(**read).model_dump()[field] == read[field]


def test_a_replace_preserves_the_unit_and_size_end_to_end():
    """The schema and the row builder agree, which is the pairing that broke."""
    body = TransactionDetailInput(
        item_description="+RED POTA 5L US#",
        item_quantity=1,
        item_quantity_unit="each",
        size_value=5,
        size_unit="LB",
        unit_quantity_subtotal=4.99,
    )
    row = _prepare_detail_rows("u1", "t1", "b1", [body.model_dump()])[0]

    assert row["item_quantity_unit"] == "each"
    assert row["size_value"] == 5
    # Units are lowercased on the way in so 'LB' and 'lb' are one unit.
    assert row["size_unit"] == "lb"


def test_an_absent_size_still_means_unknown():
    """Null is a real answer here — the receipt genuinely may not say."""
    row = _prepare_detail_rows(
        "u1", "t1", None,
        [TransactionDetailInput(item_description="BANANAS", item_quantity=2.25).model_dump()],
    )[0]
    assert row["size_value"] is None
    assert row["size_unit"] is None
    assert row["item_quantity_unit"] is None


def test_a_negative_size_is_refused():
    with pytest.raises(ValueError):
        TransactionDetailInput(item_description="X", size_value=-1)
