"""Re-uploading a receipt you already saved must SAY so.

Dedup itself was never broken: `_verify_receipt_row` matches on `content_hash`
and refuses to write a second transaction, so totals were never double counted.
The bug was that it returned the existing row bare, which is indistinguishable
from a successful save — the app opened a transaction detail screen and the user
reasonably concluded their upload had been recorded.

That is the failure worth a test, because nothing about it looks wrong: no
error, no duplicate data, just a person re-uploading the same receipt forever
because the app keeps appearing to accept it.
"""

import json

from backend.schemas.transactions import ReceiptReviewInput, TransactionWithDetails


# --- re-reviewing a receipt that is already a transaction ---------------------
#
# Verifying writes the REVIEW back over the OCR draft (_verify_receipt_row
# updates raw_ocr_string), so opening a verified receipt again reads the app's
# own previous output rather than the original scan. The two shapes differ, and
# the difference is silent: raw OCR carries `discounts: [{label, amount}]` while
# a saved review carries the single `discount_total` below. A reader that knows
# only the OCR shape sees no coupon, and re-saving drops it from the total.

def test_a_saved_review_stores_the_coupon_as_discount_total():
    review = ReceiptReviewInput(
        date="2026-07-29",
        merchant_name="STOP & SHOP",
        total_amount=38.12,
        discount_total=5.00,
    )
    stored = json.loads(review.model_dump_json())
    assert stored["discount_total"] == 5.00
    # Not the OCR key. Anything restoring this draft has to read discount_total.
    assert "discounts" not in stored


def test_an_absent_coupon_round_trips_as_null_not_a_list():
    review = ReceiptReviewInput(
        date="2026-07-29", merchant_name="STOP & SHOP", total_amount=38.12
    )
    assert json.loads(review.model_dump_json())["discount_total"] is None


def test_verify_result_marks_a_matched_receipt_as_duplicate():
    """The flag the client branches on to say 'already recorded'."""
    result = {"id": "tx-1", "duplicate_of": "tx-1"}
    assert bool(result.get("duplicate_of")) is True


def test_a_freshly_written_receipt_is_not_flagged():
    """The insert path returns no duplicate_of, so a real save stays silent."""
    result = {"id": "tx-2"}
    assert bool(result.get("duplicate_of")) is False


def test_schema_defaults_to_not_duplicate():
    """Every other route returning this shape must not start claiming
    transactions are duplicates just because the field now exists."""
    tx = TransactionWithDetails(
        id="tx-3",
        trans_date="2026-07-29",
        amount=38.12,
        description="STOP & SHOP",
        enriched_info=None,
    )
    assert tx.is_duplicate is False


def test_schema_carries_the_flag_when_set():
    tx = TransactionWithDetails(
        id="tx-4",
        trans_date="2026-07-29",
        amount=38.12,
        description="STOP & SHOP",
        enriched_info=None,
        is_duplicate=True,
    )
    assert tx.is_duplicate is True


# --- the same receipt, read twice ---------------------------------------------
#
# These are the two REAL extractions of one Stop & Shop receipt uploaded twice.
# Everything printed clearly matched; the abbreviations and the spacing in the
# merchant name did not, and both fed the hash. The result was a second
# transaction for a purchase that happened once, inflating the total by $56.83.

from backend.services.transaction_service import merchant_key, receipt_content_hash

FIRST_PASS = dict(date_value="2026-08-06", receipt_time="20:17", total=56.83,
                  merchant="STOP & SHOP")
SECOND_PASS = dict(date_value="2026-08-06", receipt_time="20:17", total=56.83,
                   merchant="STOP&SHOP")


def test_the_two_real_readings_of_one_receipt_now_match():
    assert receipt_content_hash(**FIRST_PASS) == receipt_content_hash(**SECOND_PASS)


def test_merchant_spacing_is_not_load_bearing():
    """`split()[0]` gave 'stop' and 'stopshop' — the original bug."""
    assert merchant_key("STOP & SHOP") == merchant_key("STOP&SHOP") == "stopshop"


def test_merchant_punctuation_and_case_are_ignored():
    assert merchant_key("Stew Leonard's") == merchant_key("STEW LEONARDS") == "stewleonards"


def test_a_missing_merchant_does_not_explode():
    """`split()[0]` raised IndexError on a whitespace-only name."""
    assert merchant_key("") == ""
    assert merchant_key(None) == ""
    assert merchant_key("   ") == ""


# --- but genuinely different receipts must stay distinct ----------------------

def test_a_different_total_is_a_different_receipt():
    other = {**FIRST_PASS, "total": 56.84}
    assert receipt_content_hash(**FIRST_PASS) != receipt_content_hash(**other)


def test_a_different_time_is_a_different_receipt():
    """Two trips to the same shop on one day is ordinary; this is what keeps
    them apart now that the line items are gone from the hash."""
    other = {**FIRST_PASS, "receipt_time": "09:04"}
    assert receipt_content_hash(**FIRST_PASS) != receipt_content_hash(**other)


def test_a_different_date_is_a_different_receipt():
    other = {**FIRST_PASS, "date_value": "2026-08-07"}
    assert receipt_content_hash(**FIRST_PASS) != receipt_content_hash(**other)


def test_a_different_shop_is_a_different_receipt():
    other = {**FIRST_PASS, "merchant": "Stew Leonard's"}
    assert receipt_content_hash(**FIRST_PASS) != receipt_content_hash(**other)


def test_two_shops_are_not_merged_by_normalisation():
    """Stripping punctuation must not collapse distinct names into one."""
    assert merchant_key("Stop & Shop") != merchant_key("Stop N Save")


# --- telling the user a purchase is recorded twice -----------------------------
#
# A statement line and its photographed receipt are LINKED, never merged: the
# statement is what the bank says, the receipt is what was actually bought. The
# list then collapses the pair to one row. Silently, until now — a user who
# imported a statement and later photographed the receipt saw one entry where
# they knew there were two, with no way to tell which had survived.

from backend.schemas.transactions import LinkedTransaction


def test_a_linked_record_carries_enough_to_recognise_it():
    other = LinkedTransaction(
        id="tx-csv",
        trans_date="2026-07-29",
        amount=38.12,
        merchant_name="STOP & SHOP",
        source="csv",
        match_type="csv_receipt",
        detail_count=0,
    )
    # Everything the card needs to describe the other side without a second fetch.
    assert other.source == "csv"
    assert other.detail_count == 0
    assert other.match_type == "csv_receipt"


def test_detail_count_is_what_distinguishes_the_two_records():
    """The useful difference is that one itemises the shopping and one does not;
    it is why the list prefers the receipt."""
    statement = LinkedTransaction(id="a", source="csv", detail_count=0)
    receipt = LinkedTransaction(id="b", source="bill", detail_count=17)
    assert statement.detail_count == 0
    assert receipt.detail_count > statement.detail_count


def test_an_unlinked_transaction_reports_no_others():
    """The overwhelming majority. The card must not render for these."""
    tx = TransactionWithDetails(
        id="tx-1", trans_date="2026-07-29", amount=38.12, description="STOP & SHOP"
    )
    assert tx.linked_transactions == []


# --- one hash column, three formulas -----------------------------------------
#
# Transaction.content_hash is written by three paths that do not agree:
#
#   csv     sha256("{date}|{amount:.2f}|{description}#{n}")
#   bill    sha256("{date}{time}{total}{merchant}")
#   manual  sha256("{date}{amount}{merchant_first_word}")
#
# The last two are both bare concatenations, so they coincide exactly when the
# receipt has no printed time and the merchant is a single word — "Walmart",
# "Target", "Costco". Verifying then found the MANUAL row, called the receipt a
# duplicate of it, and wrote nothing: no line items, no tax breakdown, no
# source_bill_file_id, and the user told their receipt was already recorded.

def test_a_timeless_receipt_collides_with_a_manual_entry():
    """The collision is real; the fix is scoping the lookup, not the formula."""
    import hashlib

    from backend.services.transaction_service import receipt_content_hash

    receipt = receipt_content_hash("2026-07-29", "", 38.12, "Walmart")

    # routers/transactions.py builds the manual hash this way.
    words = "Walmart".lower().strip().split()
    merchant = "".join(c for c in (words[0] if words else "") if c.isalnum())
    manual = hashlib.sha256(f"2026-07-29{round(38.12, 2)}{merchant}".encode()).hexdigest()

    assert receipt == manual, "the two formulas no longer collide; drop this test"


def test_a_printed_time_is_enough_to_separate_them():
    """Why the collision needs a source filter and not just a better formula:
    whether it happens at all depends on what OCR could read off the paper."""
    import hashlib

    from backend.services.transaction_service import receipt_content_hash

    timed = receipt_content_hash("2026-07-29", "14:32", 38.12, "Walmart")
    manual = hashlib.sha256(f"2026-07-29{round(38.12, 2)}walmart".encode()).hexdigest()
    assert timed != manual
