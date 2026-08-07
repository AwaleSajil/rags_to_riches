"""Fill in what existing receipt lines never recorded about their size.

Two separate backfills, because they rest on different kinds of evidence and
deserve to be judged separately:

  --units   item_quantity_unit = 'lb' on rows whose quantity is FRACTIONAL.
            An INFERENCE. On a US grocery receipt a line reading "2.25 @ $0.50"
            is 2.25 pounds; buying 2.25 *of* a banana is not a thing. Strong,
            but still a guess about a column that drives price comparisons.

  --sizes   size_value/size_unit parsed out of item_description ("SC 20OZ" ->
            20 oz). Weaker than it looks, and CHECK WHAT IT WRITES: a parser
            cannot tell a size from a product VARIANT that happens to sit beside
            a unit. "GV LF 2 GAL" is Great Value low-fat 2% milk in a ONE gallon
            jug, and reading it as two gallons halved the unit price and turned
            an ordinary shelf price into a 107% rip-off. The same receipt lists
            "GV FF GAL" with no number at all, which is the giveaway a human
            sees and a regex does not.

            Ambiguous readings are skipped rather than guessed — a bare "L"
            glued to an integer is five POUNDS on a produce bag and five litres
            on a soda bottle — but "skipped" only covers the cases a rule can
            recognise. Run --dry-run and read the implied unit price on every
            line before writing.

Neither touches a price, a quantity, or a total. Rows that already carry a value
are left alone, so this is safe to re-run. Use --dry-run first.
"""

from __future__ import annotations

import argparse
import os
import sys

# Runnable as `python scripts/...` from the repo root, which does not put the
# project on sys.path the way `python -m` would.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.services.units import parse_size
from backend.vector_db_client import _get_engine


# A grocery item costing more than this per base unit means the size was misread,
# not that the shopper bought saffron. $0.50/g is $500/kg; $0.50/ml is $500/litre.
_ABSURD_PER_BASE = 0.50


def _ambiguous(size, paid: float | None) -> str | None:
    """Why this reading cannot be trusted, or None if it can.

    Two traps, both from single letters glued to numbers on receipt text:

      "L"  "+RED POTA 5L US#" is a five-POUND bag of potatoes, read as five
           litres — which made a $4.99 bag look like $1.00/l and dropped it out
           of every potato comparison.
      "G"  "MS 13.2G STP" at $34.46 is a 13.2-GALLON step-on bin, read as 13.2
           grams, which prices it at $2.61 per gram.

    The second is caught by arithmetic rather than by pattern: if a reading
    implies a price no grocery has, the reading is wrong. That test is worth
    more than a list of suspicious spellings, because it does not need to know
    which abbreviations a particular chain uses.
    """
    if size.unit == "l" and float(size.value) == round(float(size.value)):
        return "bare 'L' — pounds or litres?"
    base = size.to_base()
    if paid and base and (paid / base) > _ABSURD_PER_BASE:
        return f"implies ${paid / base:.2f} per base unit"
    return None


def backfill_units(conn, user_id: str | None, dry_run: bool) -> int:
    rows = conn.execute(text(
        'SELECT id, item_description, item_quantity, unit_quantity_subtotal '
        'FROM public."TransactionDetail" '
        'WHERE item_quantity_unit IS NULL AND item_quantity IS NOT NULL '
        '  AND item_quantity <> round(item_quantity) '
        '  AND (CAST(:uid AS uuid) IS NULL OR user_id = CAST(:uid AS uuid))'
    ), {"uid": user_id}).mappings().all()

    for row in rows:
        print(f"   lb   {row['item_description'][:28]:28} "
              f"qty {float(row['item_quantity']):g} @ ${row['unit_quantity_subtotal']}")
        if not dry_run:
            conn.execute(
                text('UPDATE public."TransactionDetail" SET item_quantity_unit = :u WHERE id = :i'),
                {"u": "lb", "i": row["id"]},
            )
    return len(rows)


def backfill_sizes(conn, user_id: str | None, dry_run: bool) -> tuple[int, int]:
    rows = conn.execute(text(
        'SELECT id, item_description, unit_quantity_subtotal FROM public."TransactionDetail" '
        'WHERE size_value IS NULL AND item_description IS NOT NULL '
        '  AND (CAST(:uid AS uuid) IS NULL OR user_id = CAST(:uid AS uuid))'
    ), {"uid": user_id}).mappings().all()

    written = skipped = 0
    for row in rows:
        size = parse_size(row["item_description"])
        if size is None:
            continue
        try:
            paid = float(row["unit_quantity_subtotal"]) if row["unit_quantity_subtotal"] else None
        except (TypeError, ValueError):
            paid = None
        reason = _ambiguous(size, paid)
        if reason:
            print(f"   SKIP {row['item_description'][:28]:28} -> {size.value:g} {size.unit}  ({reason})")
            skipped += 1
            continue
        print(f"   size {row['item_description'][:28]:28} -> {size.value:g} {size.unit}")
        if not dry_run:
            conn.execute(
                text('UPDATE public."TransactionDetail" '
                     'SET size_value = :v, size_unit = :u WHERE id = :i'),
                {"v": float(size.value), "u": size.unit, "i": row["id"]},
            )
        written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", action="store_true", help="infer item_quantity_unit='lb' for weighed lines")
    parser.add_argument("--sizes", action="store_true", help="parse size_value/size_unit from item_description")
    parser.add_argument("--user-id", default=None, help="limit to one user")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.units or args.sizes):
        parser.error("pick at least one of --units / --sizes")

    engine = _get_engine()
    with engine.begin() as conn:
        if args.sizes:
            print("-- sizes parsed from item_description --")
            written, skipped = backfill_sizes(conn, args.user_id, args.dry_run)
            print(f"   {written} written, {skipped} skipped as ambiguous")
        if args.units:
            print("-- units inferred for weighed lines --")
            count = backfill_units(conn, args.user_id, args.dry_run)
            print(f"   {count} rows")
        if args.dry_run:
            print("\nDRY RUN — nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
