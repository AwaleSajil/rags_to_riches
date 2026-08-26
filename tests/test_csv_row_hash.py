"""The CSV transaction hash, pinned.

Three things reproduce this formula: ingestion, the backfill script that
rewrites rows written before it changed, and any future reader trying to work
out whether two rows are the same purchase. They agree only because they call
the same function — and the backfill in particular has to reproduce, byte for
byte, what ingestion wrote months earlier. A digest test is the only thing that
notices when an innocuous-looking edit changes that.
"""

from backend.services.purchase_match import csv_row_hash, csv_row_signature


def test_the_signature_is_exactly_this():
    """Change this string and every stored hash silently stops matching."""
    assert (
        csv_row_signature("2026-07-29", 38.12, "WALMART.COM 800-925-6278 AR")
        == "2026-07-29|38.12|walmartcom8009256278ar"
    )


def test_the_digest_is_stable():
    assert csv_row_hash("2026-07-29", 38.12, "WALMART.COM 800-925-6278 AR", 0) == (
        "e9741daad890d5fcbb3772767f6b859379986b26f6e0dbd22ec6bae3ad8906b8"
    )


# --- the normalisations that have to hold ------------------------------------

def test_the_amount_is_fixed_width():
    """"5.0" and "5.00" are the same money and must be the same hash."""
    assert csv_row_signature("2026-07-29", 5, "X") == csv_row_signature("2026-07-29", 5.00, "X")
    assert "|5.00|" in csv_row_signature("2026-07-29", 5, "X")


def test_negative_zero_is_zero():
    """A sign convention must not change identity."""
    assert csv_row_signature("2026-07-29", -0.0, "X") == csv_row_signature("2026-07-29", 0.0, "X")


def test_punctuation_in_the_description_is_ignored():
    assert (
        csv_row_signature("2026-07-29", 1.0, "TRADER JOE'S #519")
        == csv_row_signature("2026-07-29", 1.0, "trader joes 519")
    )


def test_digits_are_kept():
    """Unlike the MATCHING keys, which strip store numbers. This is an exact
    comparison against the bank's own text, where a reference number is signal."""
    assert csv_row_signature("2026-07-29", 1.0, "TARGET 1234") != csv_row_signature(
        "2026-07-29", 1.0, "TARGET 5678"
    )


def test_a_missing_description_does_not_explode():
    assert csv_row_signature("2026-07-29", 1.0, None).endswith("|")


def test_the_occurrence_index_separates_identical_rows():
    """Two $4.50 coffees on one day are two purchases; the bank listing both
    in one export is the authority on that."""
    args = ("2026-08-03", 4.50, "SBUX 1234")
    assert csv_row_hash(*args, 0) != csv_row_hash(*args, 1)


def test_no_file_identity_survives_in_the_signature():
    """The regression guard for the change this whole formula exists for.

    The old signature led with the CSV's own file id — new on every upload — so
    the same purchase hashed differently in every export and cross-file dedup was
    impossible. Three fields, not four: date, amount, description.
    """
    parts = csv_row_signature("2026-08-03", 4.50, "SBUX 1234").split("|")
    assert len(parts) == 3, f"a fourth component crept back in: {parts}"
    assert parts[0] == "2026-08-03"
