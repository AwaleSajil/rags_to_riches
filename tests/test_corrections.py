"""What the agent may correct, and what it may never touch.

Two real errors motivated this — a 5 lb bag stored as five litres, and 2% milk
stored as a two-gallon jug — both spotted while reading an answer and both
needing SQL by hand to fix.

The allowlist is the safety, so these tests are about what is REFUSED. The
question behind every column: can getting this wrong change what the user is
recorded as having spent?
"""

import pytest

from backend.services import correction_service as cs


# --- the door is narrow ------------------------------------------------------

def test_only_three_tables_are_correctable():
    assert set(cs.CORRECTABLE) == {"Transaction", "TransactionDetail", "PriceObservation"}


@pytest.mark.parametrize("table", ["User", "AccountConfig", "BillFile", "auth.users", "Transactions"])
def test_every_other_table_is_refused(table):
    """AccountConfig holds the API key; User is identity. Neither is reachable,
    and nor is anything simply misspelled."""
    with pytest.raises(ValueError):
        cs.validate(table, {"note": "x"})


@pytest.mark.parametrize("column", [
    "amount", "subtotal", "tax_total", "discount_total", "savings_total",
    "content_hash", "user_id", "id", "source", "embedding",
])
def test_transaction_money_and_plumbing_are_refused(column):
    """A total has to agree with its line items and its receipt. Correcting one
    of them alone silently desyncs the rest, so money is fixed on the review
    screen that recomputes all of it."""
    with pytest.raises(ValueError):
        cs.validate("Transaction", {column: 1})


@pytest.mark.parametrize("column", [
    "unit_quantity_subtotal", "item_subtotal_price", "item_savings",
    "tax_amount", "item_total_price", "transaction_id", "user_id",
])
def test_line_item_money_is_refused(column):
    with pytest.raises(ValueError):
        cs.validate("TransactionDetail", {column: 1})


def test_a_shelf_price_IS_correctable():
    """The exception, and the reason for it: nothing is derived from an
    observation and it never reaches a spending total."""
    cleaned = cs.validate("PriceObservation", {"item_subtotal_price": "3.49"})
    assert cleaned["item_subtotal_price"] == 3.49


@pytest.mark.parametrize("column", ["user_id", "bill_file_id", "embedding", "tag_index", "id"])
def test_observation_plumbing_is_refused(column):
    with pytest.raises(ValueError):
        cs.validate("PriceObservation", {column: "x"})


# --- the errors this was built for -------------------------------------------

def test_the_size_errors_that_started_this_are_fixable():
    """5 lb read as five litres; 2% milk read as two gallons."""
    assert cs.validate("TransactionDetail", {"size_value": 5, "size_unit": "LB"}) == {
        "size_value": 5.0, "size_unit": "lb",
    }
    assert cs.validate("TransactionDetail", {"size_value": "1", "size_unit": "gal"}) == {
        "size_value": 1.0, "size_unit": "gal",
    }


def test_units_are_lowercased_so_OZ_and_oz_are_one_unit():
    assert cs.validate("PriceObservation", {"size_unit": "FL OZ"})["size_unit"] == "fl oz"


# --- validation --------------------------------------------------------------

def test_a_number_column_rejects_text():
    with pytest.raises(ValueError):
        cs.validate("TransactionDetail", {"size_value": "two"})


def test_negative_quantities_are_refused():
    with pytest.raises(ValueError):
        cs.validate("TransactionDetail", {"item_quantity": -1})


def test_an_empty_change_set_is_refused():
    with pytest.raises(ValueError):
        cs.validate("Transaction", {})


def test_clearing_is_allowed_only_where_empty_means_something():
    """A note can be removed. An item cannot be nameless."""
    assert cs.validate("Transaction", {"note": ""}) == {"note": None}
    with pytest.raises(ValueError):
        cs.validate("TransactionDetail", {"item_description": ""})


def test_one_bad_column_rejects_the_whole_correction():
    """Applying the safe half of a rejected change would be worse than refusing:
    the user confirmed a card describing all of it."""
    with pytest.raises(ValueError):
        cs.validate("Transaction", {"note": "fine", "amount": 999})


# --- re-embedding ------------------------------------------------------------

def test_changing_searchable_text_triggers_a_reembed():
    """The vector describes the old wording until it is rebuilt."""
    assert cs.needs_reembedding("TransactionDetail", {"item_description": "x"}) is True
    assert cs.needs_reembedding("PriceObservation", {"brand_name": "x"}) is True


def test_correcting_a_size_also_invalidates_the_description():
    """This assertion used to say the opposite, and that was the bug.

    A size is not itself embedded, but the ENRICHMENT describes it and the
    enrichment IS embedded. Fixing "GV LF 2 GAL" to one gallon left the stored
    description reading "A two-gallon container of Great Value low-fat milk" —
    which the agent read and repeated to the user as fact, while the row it came
    from said one gallon."""
    assert cs.invalidates_description({"size_value": 1}) is True
    assert cs.invalidates_description({"item_quantity_unit": "lb"}) is True
    assert cs.needs_reembedding("TransactionDetail", {"size_value": 1}) is True


def test_a_note_does_not_invalidate_the_description():
    """A note is the user's own words about a purchase; it says nothing about
    what the product is, so the description still holds."""
    assert cs.invalidates_description({"note": "for the party"}) is False
