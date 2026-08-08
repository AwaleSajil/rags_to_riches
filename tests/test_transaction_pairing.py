"""Pairing rows that describe the same real-world purchase.

Two CSV exports overlapping a date range, or a statement and the receipt for
one of its lines, describe purchases that happened once. They are LINKED, not
merged — neither record is safe to discard — and the transactions list then
collapses each linked group into a single row.

That collapse is what makes the pairing rule load-bearing. Every extra link
pulls another transaction into the group, and the group displays as ONE row, so
a careless match does not just mislabel something: it removes money from the
month's total.
"""

import pytest

from money_rag import pair_same_purchase

NEW = "csv-new"
OLD = "csv-old"


def row(rid, date, amount, merchant, csv_id=OLD, source="csv"):
    return {
        "id": rid,
        "trans_date": date,
        "amount": amount,
        "merchant_name": merchant,
        "description": merchant,
        "source": source,
        "source_csv_id": csv_id,
        "enriched_info": None,
    }


def ids(pairs):
    return {(str(a["id"]), str(b["id"])) for a, b in pairs}


# --- the failure this rule exists to prevent ----------------------------------

def test_recurring_identical_purchases_do_not_chain():
    """Three coffees on consecutive days, re-imported.

    Every new row matches up to three old ones inside the ±1 day window. Linked
    without restraint they form one connected group, the list shows a single
    row, and two of the three purchases vanish from the month.
    """
    old = [
        row("o1", "2026-08-03", 4.50, "STARBUCKS"),
        row("o2", "2026-08-04", 4.50, "STARBUCKS"),
        row("o3", "2026-08-05", 4.50, "STARBUCKS"),
    ]
    new = [
        row("n1", "2026-08-03", 4.50, "STARBUCKS", csv_id=NEW),
        row("n2", "2026-08-04", 4.50, "STARBUCKS", csv_id=NEW),
        row("n3", "2026-08-05", 4.50, "STARBUCKS", csv_id=NEW),
    ]
    pairs = pair_same_purchase(old + new, NEW)

    # Exactly three pairs, and each row appears in exactly one of them.
    assert len(pairs) == 3
    assert ids(pairs) == {("n1", "o1"), ("n2", "o2"), ("n3", "o3")}


def test_every_row_is_claimed_at_most_once():
    """The invariant behind the test above, stated directly."""
    old = [row(f"o{i}", "2026-08-04", 9.99, "TESCO") for i in range(4)]
    new = [row(f"n{i}", "2026-08-04", 9.99, "TESCO", csv_id=NEW) for i in range(4)]
    pairs = pair_same_purchase(old + new, NEW)

    seen = [str(r["id"]) for pair in pairs for r in pair]
    assert len(seen) == len(set(seen)), "a row was linked twice"
    assert len(pairs) == 4


def test_the_closest_match_wins_over_file_order():
    """A same-day match must beat a next-day one even when the next-day row
    comes first — the reason pairings are scored globally, not row by row."""
    old = [
        row("o-next-day", "2026-08-05", 20.00, "WAITROSE"),
        row("o-same-day", "2026-08-04", 20.00, "WAITROSE"),
    ]
    new = [row("n1", "2026-08-04", 20.00, "WAITROSE", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "o-same-day")}


# --- and it must still find the overlap it is there for -----------------------

def test_a_plain_overlapping_export_is_matched():
    old = [
        row("o1", "2026-08-01", 12.30, "TESCO"),
        row("o2", "2026-08-09", 55.00, "SHELL"),
    ]
    new = [
        row("n1", "2026-08-09", 55.00, "SHELL", csv_id=NEW),
        row("n2", "2026-08-14", 7.20, "GREGGS", csv_id=NEW),
    ]
    pairs = pair_same_purchase(old + new, NEW)
    # Only the overlapping day pairs; the new purchase stands alone.
    assert ids(pairs) == {("n1", "o2")}


def test_a_receipt_is_matched_to_its_statement_line():
    """OCR reads the shopfront name; the statement adds a branch number. Both
    normalise to the same key once punctuation and digits are stripped."""
    old = [row("receipt", "2026-08-04", 38.12, "STOP & SHOP", csv_id=None, source="bill")]
    new = [row("n1", "2026-08-04", 38.12, "STOP & SHOP #1234", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "receipt")}


def test_a_spelled_out_word_defeats_the_match():
    """A known limitation, pinned rather than papered over.

    Containment is substring-based, so "STOP & SHOP" (stopshop) does not match
    "STOP AND SHOP" (stopandshop). Failing to link is the SAFE direction — both
    records survive and the purchase is shown twice — where a wrong link would
    quietly delete money from the month's total. Worth revisiting only with a
    token-based comparison, not by loosening containment.
    """
    old = [row("receipt", "2026-08-04", 38.12, "STOP & SHOP", csv_id=None, source="bill")]
    new = [row("n1", "2026-08-04", 38.12, "STOP AND SHOP", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_a_posting_delay_of_one_day_still_matches():
    old = [row("o1", "2026-08-04", 38.12, "TESCO")]
    new = [row("n1", "2026-08-05", 38.12, "TESCO", csv_id=NEW)]
    assert len(pair_same_purchase(old + new, NEW)) == 1


# --- and must not invent matches ----------------------------------------------

def test_rows_inside_the_same_csv_are_never_paired():
    """Two identical purchases in ONE statement really did happen twice."""
    new = [
        row("n1", "2026-08-04", 4.50, "STARBUCKS", csv_id=NEW),
        row("n2", "2026-08-04", 4.50, "STARBUCKS", csv_id=NEW),
    ]
    assert pair_same_purchase(new, NEW) == []


@pytest.mark.parametrize(
    "date,amount,merchant",
    [
        ("2026-08-07", 38.12, "TESCO"),    # three days apart
        ("2026-08-04", 39.99, "TESCO"),    # £1.87 apart
        ("2026-08-04", 38.12, "SAINSBURYS"),  # different shop
    ],
    ids=["date too far", "amount too far", "different merchant"],
)
def test_near_misses_are_not_paired(date, amount, merchant):
    old = [row("o1", "2026-08-04", 38.12, "TESCO")]
    new = [row("n1", date, amount, merchant, csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_short_merchant_names_are_left_alone():
    """"bp" and "atm" identify nothing; matching on them would link unrelated
    purchases that happen to share a day and an amount."""
    old = [row("o1", "2026-08-04", 40.00, "BP")]
    new = [row("n1", "2026-08-04", 40.00, "BP", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_unparseable_rows_are_skipped_rather_than_raising():
    old = [row("o1", "not-a-date", 38.12, "TESCO"), row("o2", "2026-08-04", None, "TESCO")]
    new = [row("n1", "2026-08-04", 38.12, "TESCO", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_pairing_is_deterministic():
    """Same input, same links — otherwise re-running ingestion churns the table."""
    old = [row(f"o{i}", "2026-08-04", 5.00, "COSTA") for i in range(3)]
    new = [row(f"n{i}", "2026-08-04", 5.00, "COSTA", csv_id=NEW) for i in range(3)]
    first = ids(pair_same_purchase(old + new, NEW))
    second = ids(pair_same_purchase(old + new, NEW))
    assert first == second
