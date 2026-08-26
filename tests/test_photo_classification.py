"""Splitting the vision model's classify-and-extract response.

One photo, one call, three possible answers. The rule that matters throughout:
never guess between a receipt and a price tag. A receipt wrongly read as a price
tag loses a purchase; a price tag wrongly read as a receipt invents spending
that never happened. "unknown" routes to a one-tap question instead.
"""

import pytest

from backend.services.naming import photo_extension, slugify
from money_rag import MoneyRAG

split = MoneyRAG._split_classified_photo
filename = MoneyRAG._photo_filename

RECEIPT_BODY = {"merchant_name": "Stop & Shop", "total_amount": 42.10, "line_items": []}
# Keys renamed in migration 029 to mirror TransactionDetail, so a price tag
# and a receipt line compare without a translation layer.
TAG_BODY = {"item_description": "Cheerios", "item_subtotal_price": 4.29}


# --- the well-formed envelope ------------------------------------------------

def test_receipt_envelope():
    assert split({"kind": "receipt", "receipt": RECEIPT_BODY, "price_tag": None}) == (
        "receipt", RECEIPT_BODY
    )


def test_price_tag_envelope():
    """A shelf photo holds a tag per product, so the payload is a list."""
    assert split({"kind": "price_tag", "price_tags": [TAG_BODY], "receipt": None}) == (
        "price_tag", {"tags": [TAG_BODY]}
    )


def test_every_tag_in_the_photo_is_kept():
    """Returning only the first would silently discard the rest of the shelf,
    and the user would have no way to tell which one survived."""
    second = {"item_description": "2% Milk", "item_subtotal_price": 2.29}
    kind, draft = split({"kind": "price_tag", "price_tags": [TAG_BODY, second]})
    assert kind == "price_tag"
    assert draft["tags"] == [TAG_BODY, second]


def test_lone_tag_object_is_still_accepted():
    """A model that ignores the array instruction has still read one real tag."""
    assert split({"kind": "price_tag", "price_tag": TAG_BODY}) == (
        "price_tag", {"tags": [TAG_BODY]}
    )


@pytest.mark.parametrize("payload", [[], [{}], [None], "nonsense"])
def test_price_tag_with_no_usable_tags_is_undecided(payload):
    """An empty list gives the card nothing to show — ask instead of opening a
    blank form and inviting a confirmed-but-empty record."""
    assert split({"kind": "price_tag", "price_tags": payload}) == ("unknown", {})


def test_unknown_envelope():
    kind, draft = split({"kind": "unknown", "receipt": None, "price_tag": None})
    assert kind == "unknown"
    assert draft == {}


# --- malformed responses must never become a confident wrong answer ----------

@pytest.mark.parametrize("payload", [
    {"kind": "receipt", "receipt": None},
    {"kind": "price_tag", "price_tag": {}},
])
def test_declared_kind_with_empty_body_is_undecided(payload):
    """An empty draft gives the review screen nothing to show. Opening a blank
    form invites a confirmed-but-empty record, so ask instead."""
    assert split(payload) == ("unknown", {})


@pytest.mark.parametrize("payload", [{"foo": "bar"}, {}, "nonsense", None, []])
def test_unrecognisable_responses_are_undecided(payload):
    kind, _ = split(payload)
    assert kind == "unknown"


def test_unexpected_kind_value_falls_back_to_inference():
    """A model answering "bill" instead of "receipt" should not lose the data."""
    kind, draft = split({"kind": "bill", "line_items": [], "total_amount": 9.99})
    assert kind == "receipt"
    assert draft["total_amount"] == 9.99


# --- backwards compatibility -------------------------------------------------

def test_bare_receipt_still_reads_as_a_receipt():
    """The previous prompt returned a flat receipt object with no envelope. A
    model that ignores the new wrapper must not break existing receipt uploads."""
    assert split(RECEIPT_BODY) == ("receipt", RECEIPT_BODY)


def test_bare_price_tag_is_inferred_from_its_keys():
    assert split(TAG_BODY) == ("price_tag", {"tags": [TAG_BODY]})


# --- display names -----------------------------------------------------------

def test_receipt_filename_keeps_merchant_date_time():
    name = filename("IMG_1234.jpg", "receipt",
                    {"merchant_name": "Stop & Shop", "date": "2026-08-01", "time": "14:30"})
    assert name == "Stop_Shop_20260801_1430.jpg"


def test_price_tag_filename_leads_with_the_item():
    """A tag has no date of its own, so the capture date is used."""
    name = filename("IMG_1234.jpg", "price_tag",
                    {"item_description": "Cheerios 12oz", "merchant_name": "Walmart"})
    assert name.startswith("pricetag_Cheerios_12oz_Walmart_")
    assert name.endswith(".jpg")


def test_filename_survives_missing_fields():
    assert filename("IMG_1234.jpg", "price_tag", {}).startswith("pricetag_item_")
    assert filename("IMG_1234.jpg", "receipt", {}).startswith("receipt_nodate")


def test_filename_keeps_the_original_extension():
    assert filename("scan.png", "receipt", {"merchant_name": "X"}).endswith(".png")


# --- the shared naming primitives --------------------------------------------
# Used both here and by the rename that runs after the user corrects a receipt
# at review, so the same merchant has to slug identically down both paths.

@pytest.mark.parametrize("value,expected", [
    ("Stop & Shop", "Stop_Shop"),
    ("  Trader Joe's  ", "Trader_Joe_s"),
    ("7-Eleven", "7_Eleven"),
    ("&Co", "Co"),               # no leading underscore
    ("Co&", "Co"),               # no trailing underscore
    ("A  ---  B", "A_B"),        # a run collapses to one separator
])
def test_slugify_reduces_free_text(value, expected):
    assert slugify(value) == expected


@pytest.mark.parametrize("value", ["", "   ", None, "&&&", "---"])
def test_slugify_falls_back_when_nothing_survives(value):
    """A model that returned null, or a name written entirely in punctuation."""
    assert slugify(value, "receipt") == "receipt"
    assert slugify(value) == ""


def test_photo_extension_preserves_case():
    """The display name has to keep pointing at the stored key."""
    assert photo_extension("IMG_0001.JPG") == ".JPG"


def test_photo_extension_falls_back_when_there_is_none():
    assert photo_extension("noextension") == ".jpg"
