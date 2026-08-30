"""Fixing a value the app got wrong, from chat, with the user's confirmation.

Two real errors motivated this and neither could be fixed from inside the app:
a 5 lb bag of potatoes stored as five litres, and 2% milk stored as a two-gallon
jug. Both were spotted while reading an answer, and both needed someone to run
SQL by hand.

The rules the agent operates under do not change. It still cannot write: every
correction is a proposal that the USER confirms, and the write then happens on a
normal authenticated request under the caller's own RLS. sql_guard still rejects
every INSERT/UPDATE/DELETE the model could author.

What this adds is a narrow, named door:

  * three tables, not the schema;
  * a fixed list of columns per table, not whatever was asked for;
  * UPDATE only — there is no delete path here at all, by design. Removing a
    purchase changes what was spent, and that belongs on a screen where the
    consequence is visible;
  * rows the caller owns, enforced by RLS rather than by a filter this module
    has to remember.

The columns are chosen on one question: can getting this wrong change what the
user is recorded as having SPENT? Item names, units and sizes cannot — they
describe the thing, and they are exactly what has been going wrong. Money on a
receipt can, because a line total, a subtotal and the transaction amount all
have to agree; that is what the receipt review screen is for, and it recomputes
the whole receipt rather than one field of it.

A shelf price is the exception that proves the rule: nothing is derived from a
PriceObservation and it never touches a spending total, so its price IS
correctable here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("moneyrag.services.correction")

# table -> {column: human label}. The label is what the confirmation card shows,
# so it has to read like something the user recognises.
CORRECTABLE: Dict[str, Dict[str, str]] = {
    "Transaction": {
        "merchant_name": "Merchant",
        "description": "Description",
        "category": "Category",
        "location": "Location",
        "trans_date": "Date",
        "note": "Note",
        # Deliberately absent: amount, subtotal, tax_total, discount_total,
        # savings_total. Those must stay consistent with each other and with the
        # line items, so they are corrected on the receipt review screen, which
        # recomputes all of them together.
    },
    "TransactionDetail": {
        "item_description": "Item",
        "item_quantity": "Quantity bought",
        "item_quantity_unit": "Quantity unit",
        "size_value": "Package size",
        "size_unit": "Size unit",
        # Deliberately absent: unit_quantity_subtotal, item_subtotal_price,
        # item_savings, tax_amount, item_total_price. Changing one in isolation
        # desyncs the line from its receipt and its transaction.
    },
    "PriceObservation": {
        "item_description": "Item",
        "brand_name": "Brand",
        "size_value": "Package size",
        "size_unit": "Size unit",
        # Prices ARE correctable here: an observation is a note about a shelf,
        # nothing is derived from it, and it never reaches a spending total.
        "item_subtotal_price": "Shelf price",
        "unit_quantity_subtotal": "Price per unit",
        "merchant_name": "Shop",
        "location": "Where",
        "item_qualitative_description": "What the tag said",
        "note": "Note",
    },
}

# Changing any of these changes the text a row is embedded from, so its vector
# has to be rebuilt or search keeps matching the old wording.
_VECTOR_FIELDS = {"item_description", "brand_name", "item_qualitative_description", "note"}

# Correcting a size also invalidates the DESCRIPTION of the row, because the
# description states the size: fixing "GV LF 2 GAL" to one gallon left
# enriched_info reading "A two-gallon container of Great Value low-fat milk",
# which the agent then quoted back as fact. The row contradicted itself and the
# wrong half was the half in the embedded text.
_SIZE_FIELDS = {"size_value", "size_unit", "item_quantity", "item_quantity_unit"}


def invalidates_description(changes: Dict[str, Any]) -> bool:
    """True when a stored description now says something the row denies."""
    return bool(_SIZE_FIELDS & set(changes))

# Free text is trimmed and an empty string means "clear it"; these are the
# columns where that is a legitimate thing to want.
_NULLABLE_TEXT = {
    "note", "location", "brand_name", "item_qualitative_description",
    "item_quantity_unit", "size_unit",
}

_NUMERIC = {
    "item_quantity", "size_value", "item_subtotal_price", "unit_quantity_subtotal",
}


def describe_correctable() -> str:
    """What may be corrected, for the tool description the agent reads."""
    return "\n".join(
        f"  {table}: " + ", ".join(sorted(columns))
        for table, columns in CORRECTABLE.items()
    )


def validate(table: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Check a proposed correction and return it cleaned.

    Raises ValueError with something worth showing the user. Rejecting is the
    normal outcome for anything outside the allowlist — the point of the list is
    that it is short, not that it is negotiable.
    """
    allowed = CORRECTABLE.get(table)
    if allowed is None:
        raise ValueError(
            f"'{table}' cannot be corrected here. Allowed: {', '.join(CORRECTABLE)}."
        )
    if not changes:
        raise ValueError("No changes were given.")

    rejected = [c for c in changes if c not in allowed]
    if rejected:
        raise ValueError(
            f"{', '.join(sorted(rejected))} cannot be changed here. "
            f"On {table} you may change: {', '.join(sorted(allowed))}."
        )

    cleaned: Dict[str, Any] = {}
    for column, value in changes.items():
        if column in _NUMERIC:
            if value is None or value == "":
                cleaned[column] = None
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{allowed[column]} must be a number, got {value!r}.")
            if number < 0:
                raise ValueError(f"{allowed[column]} cannot be negative.")
            cleaned[column] = number
            continue

        text = "" if value is None else str(value).strip()
        if not text:
            if column not in _NULLABLE_TEXT:
                raise ValueError(f"{allowed[column]} cannot be emptied.")
            cleaned[column] = None
            continue
        # Units are stored lowercase everywhere so 'OZ' and 'oz' are one unit.
        cleaned[column] = text.lower() if column.endswith("_unit") else text

    return cleaned


def needs_reembedding(table: str, changes: Dict[str, Any]) -> bool:
    """True when the change alters the text this row is searched by.

    Includes size changes: the enrichment describes the size, so correcting one
    makes the stored description wrong, and the description is embedded.
    """
    return bool((_VECTOR_FIELDS | _SIZE_FIELDS) & set(changes))


def apply_correction(
    client, table: str, row_id: str, changes: Dict[str, Any], user_id: str
) -> Optional[dict]:
    """Write a validated correction. UPDATE only — never an insert or a delete.

    Scoped by user_id AND by the table's RLS policy on the caller's token, so a
    row belonging to someone else does not update rather than raising: the two
    together mean a wrong id is a no-op, not a leak.
    """
    cleaned = validate(table, changes)
    result = (
        client.table(table).update(cleaned)
        .eq("id", row_id).eq("user_id", user_id).execute()
    )
    if not result.data:
        return None
    logger.info(
        "Corrected %s %s for user_id=%s: %s",
        table, row_id, user_id, ", ".join(sorted(cleaned)),
    )
    return result.data[0]
