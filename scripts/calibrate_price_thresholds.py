"""Measure where the semantic floor should sit — do not guess it.

The floor decides whether "is this a good price?" compares against the right
product or a different one. Guessing has been wrong three times here (0.75,
0.83, 0.94), each looking reasonable until measured.

## What this measures, and why the direction matters

Matching is **query-to-row**: a shelf tag's phrasing ("Great Value Lactose Free
Milk 2 Gallon") against a stored receipt line ("GV LF 2 GAL" plus its
enrichment). That is a systematically lower-scoring direction than row-to-row,
because the two sides are written by different authors for different purposes.
An earlier calibration measured row-to-row and produced 0.94, which rejects
genuine matches in the direction that actually gets used.

Ground truth is hand-labelled below rather than derived from normalized_name —
that column no longer exists, and matching is semantic-only.

## Reading the result

A usable floor sits strictly above the highest DIFFERENT pair and at or below
the lowest SAME pair. If those overlap there is no safe floor: the space cannot
separate these products, and semantic-only matching is not viable.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/calibrate_price_thresholds.py --email you@example.com
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
import psycopg.rows
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.price_service import embedding_text  # noqa: E402

# (shelf-tag phrasing, receipt line as stored, same product?)
#
# Receipt lines are real `item_description` values from this database; their
# enrichment is looked up at run time so the pair reflects exactly what gets
# embedded. The DIFFERENT pairs are deliberately near misses — same aisle, same
# category, same brand family — because those are what a floor has to exclude.
# An unrelated pair proves nothing.
PAIRS = [
    # --- same product, written as a shelf tag would write it ------------------
    ("Bananas",                                  "BANANAS",               True),
    ("Tomatoes on the Vine",                     "WT TOMATO ON THE VINE", True),
    ("Fresh Cilantro",                           "FRESH CILANTRO",        True),
    ("D'Anjou Pears",                            "DANJOU PEARS",          True),
    ("Red Potatoes",                             "RED POTATO",            True),
    ("Broccoli Crowns",                          "WT BROCCOLI CROWNS PC", True),
    ("Head & Shoulders Classic Clean Shampoo 8.5 oz", "HS SH CLS8.5",     True),
    ("Great Value Trash Bags",                   "GV TRASHBAG",           True),
    ("Great Value Lactose Free Milk 2 Gallon",   "GV LF 2 GAL",           True),
    # --- different products, chosen to be hard -------------------------------
    ("Organic Bananas",        "BANANAS",               False),  # variant, costs more
    ("Bananas",                "DANJOU PEARS",          False),  # both fruit
    ("Fresh Cilantro",         "KALE GREENS",           False),  # both leafy green
    ("Broccoli Crowns",        "KALE GREENS",           False),  # both green vegetable
    ("Tomatoes on the Vine",   "RED POTATO",            False),  # both vegetable
    ("Great Value Trash Bags", "ZIPLOC KITCHEN",        False),  # both household storage
    ("Whole Milk 1 Gallon",    "GV LF 2 GAL",           False),  # different milk
    ("Lentils",                "KALE GREENS",           False),
    ("Head & Shoulders Shampoo", "GV TRASHBAG",         False),
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument(
        "--no-enrichment",
        action="store_true",
        help="Embed the bare description, to show what enrichment is worth",
    )
    args = parser.parse_args()
    load_dotenv()

    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row) as conn:
        config = conn.execute(
            '''SELECT c.llm_provider, c.api_key, c.embedding_model
               FROM public."AccountConfig" c JOIN public."User" u ON u.id = c.user_id
               WHERE u.email = %s''',
            (args.email,),
        ).fetchone()
        if not config or not config.get("api_key"):
            raise SystemExit(f"No AccountConfig with an API key for {args.email}")

        descriptions = sorted({row for _, row, _ in PAIRS})
        enrichment = {
            r["item_description"]: r["enriched_info"]
            for r in conn.execute(
                '''SELECT DISTINCT ON (item_description) item_description, enriched_info
                   FROM public."TransactionDetail"
                   WHERE item_description = ANY(%s) AND enriched_info IS NOT NULL''',
                (descriptions,),
            ).fetchall()
        }

    from backend.services.transaction_service import _build_embeddings

    model = _build_embeddings(config)

    # One call per distinct text — several pairs share a side and embedding
    # quota is the scarce resource here.
    def row_text(description: str) -> str:
        enr = None if args.no_enrichment else enrichment.get(description)
        return embedding_text(description, None, None, enr)

    texts = {q for q, _, _ in PAIRS} | {row_text(r) for _, r, _ in PAIRS}
    vectors = {t: model.embed_query(t) for t in sorted(texts)}

    rows = [
        (same, query, row, cosine(vectors[query], vectors[row_text(row)]))
        for query, row, same in PAIRS
    ]

    missing = [d for d in descriptions if d not in enrichment]
    print(f"\nmodel: {config.get('embedding_model')}   calls: {len(texts)}")
    print(f"enrichment: {'DISABLED' if args.no_enrichment else f'{len(enrichment)}/{len(descriptions)} rows have it'}")
    if missing and not args.no_enrichment:
        print(f"  without enrichment: {', '.join(missing)}")
    print(f"\n{'':5} {'shelf tag':42} {'receipt line':24} {'cosine':>7}")
    for same, query, row, sem in sorted(rows, key=lambda r: (-r[0], -r[3])):
        print(f"{'SAME' if same else 'DIFF':5} {query[:42]:42} {row[:24]:24} {sem:7.3f}")

    same_scores = [r[3] for r in rows if r[0]]
    diff_scores = [r[3] for r in rows if not r[0]]
    lo_same, hi_diff = min(same_scores), max(diff_scores)

    print(f"\nSAME  min={lo_same:.3f}  max={max(same_scores):.3f}")
    print(f"DIFF  min={min(diff_scores):.3f}  max={hi_diff:.3f}")

    if lo_same > hi_diff:
        print(f"\nSEPARATED by {lo_same - hi_diff:.3f}")
        print(f"  -> floor {(lo_same + hi_diff) / 2:.2f}  (every SAME pair kept, no DIFF admitted)")
        return

    # Overlapping is not automatically fatal: a strict floor may still admit no
    # wrong product, at the cost of missing the weakest true matches. That is
    # the right trade here — a miss says "no history", a false match quotes the
    # price of something you did not buy.
    print(f"\nOVERLAP of {hi_diff - lo_same:.3f} — no floor keeps every SAME pair.")
    safe = [v for v in same_scores if v > hi_diff]
    if not safe:
        print("STOP: no floor admits any SAME pair without also admitting a DIFF one.")
        print("Semantic-only matching is not viable on this data.")
        return
    print(f"  zero-false-positive floor: > {hi_diff:.3f} (worst DIFF)")
    print(f"  lowest SAME above it: {min(safe):.3f} — margin {min(safe) - hi_diff:.3f}")
    print(f"  cost: {len(same_scores) - len(safe)} of {len(same_scores)} true matches missed")
    worst = sorted((v, q, r) for (s, q, r, v) in rows if s and v <= hi_diff)
    for v, q, r in worst:
        print(f"    missed {v:.3f}  {q}  ->  {r}")


if __name__ == "__main__":
    main()
