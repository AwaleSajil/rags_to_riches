"""Deciding whether two records describe the same real-world purchase.

Both directions of receipt/statement reconciliation have to answer this, and
they used to answer it differently. `pair_same_purchase` in money_rag.py runs
when the bank CSV arrives last; `_link_verified_bill_transaction` in
transaction_service.py runs when the receipt is verified last. The second says
in its own docstring that it keeps "the same conservative merchant/date/amount
rules" as the first, and it did not: it kept the windows and reimplemented the
name test by hand. Whether a purchase was reconciled came to depend on which
half of it you happened to file second.

Living here, the rule is one rule. The callers still differ in what they DO with
a match — one scores every candidate and settles them one-to-one, the other
links what it finds — but they can no longer disagree about what a match is.

The date and amount windows live here for the same reason the name test does.
transaction_service.py had them written out as bare `> 1` and `> 0.10` beside
money_rag.py's named constants — two copies of one judgement, which is already
the thing that let the two directions disagree.

Nothing here consults `merchant_name` on a bank row. That field is written by
the enrichment LLM on every ingest, so one statement line comes back as "Credit
Card Payment" this upload and "Online Payment" the next, and a link that rests
on it appears or vanishes with the model's mood. A receipt is the exception, and
only because it has no bank text to use instead: its merchant was read off
printed paper and confirmed by the user on the review screen.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, FrozenSet, Optional, Tuple

# Bank posting dates slip by a day, and a receipt total can round against the
# statement by a penny or two. Wider than this starts matching purchases that
# genuinely happened separately.
MAX_LINK_DAY_GAP = 1
MAX_LINK_AMOUNT_GAP = 0.10

# A squashed name shorter than this identifies nothing: "bp", "atm" and "h&m"
# would match half a statement between them.
MIN_MERCHANT_KEY = 4

# A single TOKEN may be shorter, because a token has word boundaries around it
# that a squashed string does not. "cvs" is a real merchant and only ever
# matches another "cvs"; the same three letters inside a squashed string would
# also match "cvsomething".
MIN_TOKEN_LENGTH = 3

# Tokens that appear across unrelated statement lines and so cannot contribute
# to identity. Without these, "STOP AND SHOP" and "PAY AND GO" share a token and
# the subset test below would have one more way to go wrong.
_GENERIC_TOKENS = frozenset({
    "and", "the", "for", "com", "www", "inc", "llc", "ltd",
    "pos", "ach", "atm", "fee", "ref", "txn", "pmt", "chk", "dda",
})

_SPLIT = re.compile(r"[^a-z0-9]+")


def squash(value: Any) -> str:
    """Letters only, lowercased — the whole name run together.

    Digits go with the punctuation: a store number is the most common thing to
    differ between the receipt and the statement for one purchase.
    """
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def tokens(value: Any) -> FrozenSet[str]:
    """The identifying words in a name, as a set.

    Splitting BEFORE the punctuation is stripped is the whole point. Squashing
    first turns "STOP AND SHOP" into "stopandshop" and "Stop & Shop" into
    "stopshop", and neither string contains the other — one bank spelling out a
    word the receipt prints as "&" was enough to lose the link.
    """
    parts = _SPLIT.split(str(value or "").lower())
    return frozenset(
        p for p in parts
        if len(p) >= MIN_TOKEN_LENGTH and not p.isdigit() and p not in _GENERIC_TOKENS
    )


def row_keys(row: Dict[str, Any]) -> Tuple[str, FrozenSet[str]]:
    """(squashed name, token set) for one transaction row.

    WHERE THE ROW CAME FROM picks the field, not whichever one is populated:

        csv / manual -> the bank's own description, identical in every export
        bill         -> the merchant read off the receipt and confirmed

    Both forms of the same name are returned because they fail differently, and
    `name_match` needs both to cover the two ways a bank and a receipt disagree.
    """
    value = row.get("merchant_name") if row.get("source") == "bill" else row.get("description")
    squashed = squash(value)
    return (squashed if len(squashed) >= MIN_MERCHANT_KEY else "", tokens(value))


# How convincingly two rows carry the same name. Lower is stronger, and it only
# breaks ties between candidates already inside the date and amount windows.
MATCH_EXACT = 0
MATCH_TOKENS = 1
MATCH_CONTAINED = 2


def name_match(
    left: Tuple[str, FrozenSet[str]], right: Tuple[str, FrozenSet[str]]
) -> Optional[int]:
    """How well two rows' names agree, or None if they do not agree at all.

    Two weaker tests below the exact one, kept SEPARATE because each catches
    what the other cannot:

      tokens     "Stop & Shop" vs "STOP AND SHOP #1234". Word boundaries survive
                 a filler word the other side spells out. Ranked above
                 containment because it cannot match across a boundary — the
                 squashed test happily reports "art" inside "target".

      contained  "TJ Maxx" vs "TJMAXX 1234". Boundaries the other side does not
                 have at all, which is exactly where the token test fails.

    Together they are strictly more permissive than either alone, which is the
    intent: a name here is breaking a tie between candidates that already agree
    on date to within a day and on amount to within ten cents.
    """
    left_key, left_tokens = left
    right_key, right_tokens = right

    if left_key and left_key == right_key:
        return MATCH_EXACT
    if left_tokens and right_tokens and (
        left_tokens <= right_tokens or right_tokens <= left_tokens
    ):
        return MATCH_TOKENS
    if left_key and right_key and (left_key in right_key or right_key in left_key):
        return MATCH_CONTAINED
    return None


def has_key(keys: Tuple[str, FrozenSet[str]]) -> bool:
    """Whether this row can be matched on its name at all."""
    return bool(keys[0] or keys[1])


def csv_row_signature(trans_date: Any, amount: Any, description: Any) -> str:
    """The identity of one bank-statement row, before the occurrence index.

    Every input is something the BANK wrote, normalised the same way on every
    upload. Nothing the LLM produced appears here — `merchant_name` used to, and
    it is regenerated by the enrichment call on each ingest, so the hash for one
    statement line changed between two uploads of the same statement and the
    duplicate it existed to catch went unnoticed.
    """
    # Fixed two-decimal form, not str(round(...)): that returns the float's
    # repr, so one amount prints as "5.0" or "5.00" depending on how the CSV
    # wrote it. -0.0 folds into 0.00 for the same reason — a sign convention
    # must not change a hash.
    value = round(float(amount), 2)
    amount_str = f"{value:.2f}" if value else "0.00"
    # Digits kept, unlike the matching keys above: this is an exact comparison
    # against the bank's verbatim text, where a reference number is signal
    # rather than the store-number noise that matching has to strip.
    desc = re.sub(r"[^a-z0-9]", "", str(description or "").lower())
    return f"{str(trans_date).strip()}|{amount_str}|{desc}"


def csv_row_hash(trans_date: Any, amount: Any, description: Any, occurrence: int) -> str:
    """content_hash for one CSV transaction, including its occurrence index.

    `occurrence` is what lets a hash express "never within a file, only across
    files". Two identical rows in one export are two purchases — the bank
    listing both is the authority on that — but the unique constraint is
    (user_id, content_hash) with no notion of which file a row came from, so two
    rows that must both survive need two different hashes.

    Counting per file also reconciles across them: N rows here against M already
    stored resolves to max(0, N - M) inserts, which is the right answer whether
    the next export repeats both, one, or adds a third. That holds because the
    signature pins an exact DATE and bank exports are cut on date boundaries —
    any file covering a date carries every row for it, never a fragment.

    Shared with scripts/backfill_transaction_content_hash.py, which has to
    reproduce this byte for byte over rows written before the formula changed.
    """
    return hashlib.sha256(
        f"{csv_row_signature(trans_date, amount, description)}#{occurrence}".encode()
    ).hexdigest()
