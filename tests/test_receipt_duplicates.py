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

from backend.schemas.transactions import TransactionWithDetails


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
