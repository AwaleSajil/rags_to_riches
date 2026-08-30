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

from money_rag import MoneyRAG, pair_same_purchase

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


def test_a_spelled_out_word_no_longer_defeats_the_match():
    """The limitation this file used to pin, now closed.

    Substring containment could not bridge "STOP & SHOP" (stopshop) and "STOP
    AND SHOP" (stopandshop), and the note here said it was worth revisiting only
    with a token-based comparison rather than by loosening containment. That is
    what merchant_match now does: it splits on punctuation BEFORE discarding it,
    so both sides yield {stop, shop} and the filler word drops out as a
    stopword. Containment is unchanged and still runs, for the opposite case
    ("TJ Maxx" against "TJMAXX") where there are no boundaries to split on.
    """
    old = [row("receipt", "2026-08-04", 38.12, "STOP & SHOP", csv_id=None, source="bill")]
    new = [row("n1", "2026-08-04", 38.12, "STOP AND SHOP", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "receipt")}


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


# --- the LLM disagreeing with itself ------------------------------------------
#
# `merchant_name` is generated per upload by the enrichment model, so the same
# statement line comes back named differently in two exports. Real pair, from
# two overlapping exports of one account:
#
#   "INTERNET PAYMENT - THANK YOU" -> "Credit Card Payment" / "Online Payment"
#   "CASHBACK BONUS REDEMPTION..." -> "Discover Financial Services" / "Discover Card"
#
# Neither name contains the other, so matching on the name alone lost both
# links. The bank's own description was byte-identical in both files.

def bank_row(rid, date, amount, description, merchant, csv_id=OLD):
    """A row whose clean name differs from the raw statement text."""
    return {
        "id": rid,
        "trans_date": date,
        "amount": amount,
        "merchant_name": merchant,
        "description": description,
        "source": "csv",
        "source_csv_id": csv_id,
        "enriched_info": None,
    }


@pytest.mark.parametrize(
    "description,old_name,new_name",
    [
        ("INTERNET PAYMENT - THANK YOU", "Credit Card Payment", "Online Payment"),
        ("CASHBACK BONUS REDEMPTION PYMT/STMT CRDT", "Discover Financial Services", "Discover Card"),
        ("SPI*EVERSOURCE BERLIN CT", "Eversource Energy", "Eversource"),
    ],
)
def test_same_statement_line_links_however_the_model_named_it(description, old_name, new_name):
    old = [bank_row("o1", "2026-07-05", -45.73, description, old_name)]
    new = [bank_row("n1", "2026-07-05", -45.73, description, new_name, csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "o1")}


def test_matching_descriptions_do_not_override_the_amount_and_date_windows():
    """A stable key is not a licence to link two genuinely separate purchases."""
    description = "INTERNET PAYMENT - THANK YOU"
    old = [
        bank_row("o-far", "2026-07-01", -45.73, description, "Credit Card Payment"),
        bank_row("o-costly", "2026-07-05", -900.00, description, "Credit Card Payment"),
    ]
    new = [bank_row("n1", "2026-07-05", -45.73, description, "Online Payment", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_identical_descriptions_win_over_a_merely_contained_name():
    """Two candidates, both allowable — the exact statement text is the surer one."""
    old = [
        bank_row("o-contained", "2026-07-05", 10.43, "SQ *EVERSOURCE PMT", "Eversource"),
        bank_row("o-exact", "2026-07-05", 10.43, "SPI*EVERSOURCE BERLIN CT", "Eversource Energy"),
    ]
    new = [bank_row("n1", "2026-07-05", 10.43, "SPI*EVERSOURCE BERLIN CT", "Eversource", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "o-exact")}


def test_a_receipt_with_no_bank_description_still_matches_on_its_name():
    """The description key must not become a requirement — receipts have none."""
    old = [{
        "id": "receipt",
        "trans_date": "2026-07-01",
        "amount": 46.21,
        "merchant_name": "Walmart",
        "description": None,
        "source": "bill",
        "source_csv_id": None,
        "enriched_info": None,
    }]
    new = [bank_row("n1", "2026-07-01", 46.21, "WALMART.COM 800-925-6278 AR", "Walmart", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "receipt")}


def test_pairing_is_deterministic():
    """Same input, same links — otherwise re-running ingestion churns the table."""
    old = [row(f"o{i}", "2026-08-04", 5.00, "COSTA") for i in range(3)]
    new = [row(f"n{i}", "2026-08-04", 5.00, "COSTA", csv_id=NEW) for i in range(3)]
    first = ids(pair_same_purchase(old + new, NEW))
    second = ids(pair_same_purchase(old + new, NEW))
    assert first == second


# --- what the upload screen is told about the links ---------------------------
#
# The links themselves are invisible: the transactions list collapses each
# linked pair into one row, so a statement overlapping the previous one looks
# like it imported fewer rows than it held. `_link_new_csv_transactions` reports
# what it linked so the upload can say so.

def linker(rows):
    """A MoneyRAG with only the database calls linking uses stubbed out."""
    rag = object.__new__(MoneyRAG)
    rag.user_id = "user-1"
    rag.upserted = []
    rag._db_select = lambda *a, **kw: rows
    rag._db_upsert = lambda table, records, **kw: rag.upserted.append((table, records))
    rag._db_update = lambda *a, **kw: None
    return rag


def test_links_are_reported_with_the_new_row_and_its_match_type():
    """One summary per link, describing the row the user just uploaded."""
    old = [
        row("o1", "2026-08-04", 38.12, "TESCO"),
        row("o2", "2026-08-05", 12.00, "BOOTS", source="bill", csv_id=None),
    ]
    new = [
        row("n1", "2026-08-04", 38.12, "TESCO", csv_id=NEW),
        row("n2", "2026-08-05", 12.00, "BOOTS", csv_id=NEW),
    ]
    rag = linker(old + new)

    summaries = rag._link_new_csv_transactions(NEW)

    assert len(summaries) == 2
    by_merchant = {s["merchant"]: s for s in summaries}
    assert by_merchant["TESCO"] == {
        "date": "2026-08-04",
        "merchant": "TESCO",
        "amount": 38.12,
        "match_type": "csv_csv",
    }
    # A photographed receipt is a different story to tell than an overlapping
    # export, so the two must not arrive indistinguishable.
    assert by_merchant["BOOTS"]["match_type"] == "csv_receipt"


def test_nothing_matched_reports_nothing():
    """No links, no notification — an ordinary import stays quiet."""
    old = [row("o1", "2026-07-01", 38.12, "TESCO")]
    new = [row("n1", "2026-08-04", 4.50, "STARBUCKS", csv_id=NEW)]
    rag = linker(old + new)

    assert rag._link_new_csv_transactions(NEW) == []
    assert rag.upserted == []


def test_summary_count_matches_the_rows_written():
    """The number shown to the user is the number of links actually recorded."""
    old = [row(f"o{i}", "2026-08-04", 5.00 + i, "COSTA") for i in range(3)]
    new = [row(f"n{i}", "2026-08-04", 5.00 + i, "COSTA", csv_id=NEW) for i in range(3)]
    rag = linker(old + new)

    summaries = rag._link_new_csv_transactions(NEW)

    assert len(summaries) == 3
    assert len(rag.upserted[0][1]) == len(summaries)


# --- keeping the model from renaming the same line twice -----------------------
#
# The links above only had to be repaired because `merchant_name` drifted. The
# drift is fixed at the source: a description this user has uploaded before
# keeps the name it already has, and never reaches the model a second time.

def namer(rows):
    """A MoneyRAG with only the lookup `_known_merchant_names` makes stubbed."""
    rag = object.__new__(MoneyRAG)
    rag.user_id = "user-1"
    rag.queried = []
    def _select_in(table, columns, field, values_list, filters=None):
        rag.queried.append((field, list(values_list), filters))
        return [r for r in rows if r["description"] in values_list]
    rag._db_select_in = _select_in
    return rag


def named(description, merchant, info=""):
    return {"description": description, "merchant_name": merchant, "enriched_info": info}


def test_a_description_keeps_the_name_it_already_has():
    rag = namer([named("INTERNET PAYMENT - THANK YOU", "Credit Card Payment", "A payment.")])

    known = rag._known_merchant_names(["INTERNET PAYMENT - THANK YOU", "NEW MERCHANT LLC"])

    assert known == {
        "INTERNET PAYMENT - THANK YOU": {
            "merchant_name": "Credit Card Payment",
            "enriched_info": "A payment.",
        }
    }
    # The unseen description is left out, so it still reaches the model.
    assert "NEW MERCHANT LLC" not in known
    # Scoped to the user: merchant names are not shared between accounts.
    assert rag.queried[0][2] == {"user_id": "user-1"}


def test_the_name_already_used_most_wins():
    """Rows predating this lookup carry both names. One of them has to win."""
    rag = namer([
        named("SPI*EVERSOURCE BERLIN CT", "Eversource"),
        named("SPI*EVERSOURCE BERLIN CT", "Eversource Energy"),
        named("SPI*EVERSOURCE BERLIN CT", "Eversource"),
    ])

    known = rag._known_merchant_names(["SPI*EVERSOURCE BERLIN CT"])

    assert known["SPI*EVERSOURCE BERLIN CT"]["merchant_name"] == "Eversource"


def test_an_even_split_still_resolves_the_same_way_every_run():
    """A tie must not be settled by database row order, or the drift comes back."""
    pair = [
        named("CASHBACK BONUS REDEMPTION", "Discover Card"),
        named("CASHBACK BONUS REDEMPTION", "Discover Financial Services"),
    ]
    forwards = namer(pair)._known_merchant_names(["CASHBACK BONUS REDEMPTION"])
    backwards = namer(list(reversed(pair)))._known_merchant_names(["CASHBACK BONUS REDEMPTION"])

    assert forwards == backwards


def test_a_failed_lookup_falls_back_to_enriching_everything():
    """Consistency is worth an upload; it is not worth failing one."""
    rag = object.__new__(MoneyRAG)
    rag.user_id = "user-1"
    def _boom(*a, **kw):
        raise RuntimeError("no such column")
    rag._db_select_in = _boom

    assert rag._known_merchant_names(["ANYTHING"]) == {}


def test_no_descriptions_makes_no_query():
    rag = namer([])
    assert rag._known_merchant_names([]) == {}
    assert rag.queried == []


# --- the name rule: what a bank and a receipt call the same shop --------------
#
# These are the pairs the rule exists for, and the regression that prompted it:
# dropping the enrichment LLM's `merchant_name` from the key removed the only
# thing bridging "Stop & Shop" and "STOP AND SHOP", because squashing the
# punctuation out of both leaves "stopshop" and "stopandshop" — neither of which
# contains the other.

RECEIPT_AND_STATEMENT = [
    ("Walmart", "WALMART.COM 800-925-6278 AR"),
    ("Target", "TARGET T-1234 NORWALK CT"),
    ("Trader Joe's", "TRADER JOE S #519 QPS"),
    ("Costco", "COSTCO WHSE #0357"),
    # The bank spells out a word the receipt prints as "&".
    ("Stop & Shop", "STOP AND SHOP #1234"),
    # Three letters, but with word boundaries around them.
    ("CVS", "CVS/PHARMACY #04567"),
    # The reverse: boundaries the statement does not have at all.
    ("TJ Maxx", "TJMAXX 1234"),
    ("Shell", "SHELL OIL 57444286104"),
]

DIFFERENT_SHOPS = [
    ("Shell", "SHEETZ 123"),
    ("Costco", "COSTA COFFEE LONDON"),
    # The one a prefix rule would wrongly accept. A false link is worse than a
    # missed one: linked rows collapse to a single row in the list, so it
    # REMOVES money from the month rather than showing a duplicate.
    ("Whole Foods", "WHOLESALE CLUB #22"),
    # Squashed containment matches across word boundaries; tokens must not.
    ("Arts", "TARGET T-1234"),
]


@pytest.mark.parametrize("merchant,description", RECEIPT_AND_STATEMENT)
def test_a_receipt_links_to_its_statement_line(merchant, description):
    old = [{
        "id": "receipt", "trans_date": "2026-07-01", "amount": 46.21,
        "merchant_name": merchant, "description": None,
        "source": "bill", "source_csv_id": None, "enriched_info": None,
    }]
    new = [bank_row("n1", "2026-07-01", 46.21, description, "(enrichment, ignored)", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "receipt")}


@pytest.mark.parametrize("merchant,description", DIFFERENT_SHOPS)
def test_two_different_shops_are_never_linked(merchant, description):
    """Same day, same amount to the penny — only the name keeps them apart."""
    old = [{
        "id": "receipt", "trans_date": "2026-07-01", "amount": 46.21,
        "merchant_name": merchant, "description": None,
        "source": "bill", "source_csv_id": None, "enriched_info": None,
    }]
    new = [bank_row("n1", "2026-07-01", 46.21, description, "(enrichment, ignored)", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


def test_the_enrichment_name_cannot_create_a_link():
    """The whole point: identity may not depend on what the model said today.

    Two unrelated shops that the enrichment call happened to give the same
    name. Before the key was moved onto the bank's own text, this linked.
    """
    old = [bank_row("o1", "2026-07-01", 46.21, "SHEETZ 123", "Fuel Stop")]
    new = [bank_row("n1", "2026-07-01", 46.21, "SHELL OIL 5744", "Fuel Stop", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []


# --- manual entries are purchases too ----------------------------------------
#
# "I spent $50 at Target today", confirmed in chat, is as real a record as a
# statement line — and when the bank exports that same purchase weeks later,
# they are the same money. Excluded from linking, that pair could never be
# reconciled by anything: no link exists, so the deduped view has nothing to act
# on and the cost is counted twice permanently.

def manual_row(rid, date, amount, description):
    return {
        "id": rid, "trans_date": date, "amount": amount,
        "merchant_name": description, "description": description,
        "source": "manual", "source_csv_id": None, "enriched_info": None,
    }


def test_a_confirmed_manual_entry_links_to_its_statement_line():
    old = [manual_row("m1", "2026-08-04", 50.00, "Target")]
    new = [bank_row("n1", "2026-08-04", 50.00, "TARGET T-1234 NORWALK CT",
                    "(enrichment, ignored)", csv_id=NEW)]
    assert ids(pair_same_purchase(old + new, NEW)) == {("n1", "m1")}


def test_a_manual_entry_still_claims_only_one_partner():
    """Cash payments recur. The one-to-one rule has to hold for them too."""
    old = [manual_row(f"m{i}", "2026-08-04", 20.00, "Simran") for i in range(3)]
    new = [bank_row(f"n{i}", "2026-08-04", 20.00, "SIMRAN TRANSFER",
                    "(ignored)", csv_id=NEW) for i in range(3)]
    pairs = pair_same_purchase(old + new, NEW)
    seen = [str(r["id"]) for pair in pairs for r in pair]
    assert len(pairs) == 3
    assert len(seen) == len(set(seen)), "a row was linked twice"


def test_a_cash_manual_entry_matches_nothing_and_is_left_alone():
    """The common case: money that never touches a statement."""
    old = [manual_row("m1", "2026-08-04", 100.00, "Simran")]
    new = [bank_row("n1", "2026-08-04", 42.00, "TESCO STORES 3421",
                    "(ignored)", csv_id=NEW)]
    assert pair_same_purchase(old + new, NEW) == []
