"""Re-describe every receipt line item, then rebuild its vector.

Worth doing because the descriptions already stored were written WITHOUT the
merchant. A receipt line is one chain's private shorthand, and read alone it can
come back as an entirely different product: "GV LF 2 GAL" searched on its own
was described as Great Value Lemon Fresh BLEACH, and as Great Value low-fat milk
once Walmart was named. That description is part of the text each row is embedded
from, so a wrong one does not merely read badly — it puts the row in the wrong
place in vector space and quietly breaks every later product match.

Also repairs descriptions that contradict their own row. A size corrected after
the fact left text saying "a two-gallon container" on a row recorded as one
gallon, which the agent then quoted back as fact.

Two phases, because they have different costs:

  describe  one LLM call per row, run a few at a time. The slow part.
  embed     batched, 50 documents per request. The cheap part.

Safe to RE-RUN — every row is simply overwritten — but NOT safe to interrupt
during the describe phase: descriptions are collected in full and written in one
transaction, so a run stopped early loses that work and its API spend. The
embedding phase does commit per batch, so an interruption there keeps whatever
had been written.

Use --dry-run first, and --limit to read a sample before committing to the whole
table: the descriptions are guesses about what an abbreviation means, and a wrong
one goes into the embedded text and is quoted back to the user as fact.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.services.price_service import describe_product, size_text_of, to_vector_literal
from backend.services.transaction_service import _build_embeddings
from backend.vector_db_client import _get_engine, detail_document

# Enough to keep the API busy without tripping a rate limit; the receipt-
# enrichment path uses the same number for the same reason.
DESCRIBE_WORKERS = 4
EMBED_BATCH = 50


def load_config(conn, user_id: str) -> dict | None:
    row = conn.execute(
        text('SELECT * FROM public."AccountConfig" WHERE user_id = CAST(:u AS uuid) LIMIT 1'),
        {"u": user_id},
    ).mappings().first()
    return dict(row) if row else None


def load_rows(conn, user_id: str, only_missing: bool, limit: int | None):
    query = '''
        SELECT d.id, d.item_description, d.size_value, d.size_unit,
               d.enriched_info, t.merchant_name
        FROM public."TransactionDetail" d
        JOIN public."Transaction" t ON t.id = d.transaction_id
        WHERE d.user_id = CAST(:u AS uuid)
          AND d.item_description IS NOT NULL
    '''
    if only_missing:
        query += " AND (d.enriched_info IS NULL OR d.enriched_info = '')"
    query += " ORDER BY d.item_description"
    if limit:
        query += f" LIMIT {int(limit)}"
    return conn.execute(text(query), {"u": user_id}).mappings().all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--only-missing", action="store_true",
                        help="skip rows that already have a description")
    parser.add_argument("--limit", type=int, default=None, help="sample this many rows")
    parser.add_argument("--dry-run", action="store_true",
                        help="describe and print, write nothing")
    args = parser.parse_args()

    engine = _get_engine()
    with engine.connect() as conn:
        config = load_config(conn, args.user_id)
        if not config:
            print("No AccountConfig for that user — nothing to describe with.")
            return 1
        rows = [dict(r) for r in load_rows(conn, args.user_id, args.only_missing, args.limit)]

    if not rows:
        print("Nothing to do.")
        return 0
    print(f"{len(rows)} line item(s) to describe\n")

    # --- describe ----------------------------------------------------------
    def describe(row):
        size = size_text_of(row.get("size_value"), row.get("size_unit"))
        return row, describe_product(
            config, row["item_description"], shop=row.get("merchant_name"), size=size,
        )

    described: list[tuple[dict, str]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        for row, description in pool.map(describe, rows):
            if not description:
                # Better none than a wrong one: a description is quoted to the
                # user as fact and is embedded as if it were true.
                failed += 1
                print(f"  ---- {row['item_description'][:26]:26} (no description)")
                continue
            described.append((row, description))
            print(f"  ok   {row['item_description'][:26]:26} {description[:78]}")

    print(f"\n{len(described)} described, {failed} left alone")
    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0
    if not described:
        return 0

    # --- write, then embed in batches --------------------------------------
    with engine.begin() as conn:
        for row, description in described:
            conn.execute(
                text('UPDATE public."TransactionDetail" SET enriched_info = :e WHERE id = :i'),
                {"e": description, "i": row["id"]},
            )
    print("descriptions saved")

    embeddings = _build_embeddings(config)
    model_name = config.get("embedding_model") or "unknown"
    written = 0
    for start in range(0, len(described), EMBED_BATCH):
        chunk = described[start : start + EMBED_BATCH]
        documents = [
            detail_document({**row, "enriched_info": description})
            for row, description in chunk
        ]
        try:
            vectors = embeddings.embed_documents(documents)
        except Exception as e:  # noqa: BLE001
            # The descriptions are already saved; a vector can be rebuilt later.
            print(f"  embedding batch failed ({len(chunk)} rows): {e}")
            continue
        with engine.begin() as conn:
            for (row, _), vector in zip(chunk, vectors):
                conn.execute(
                    text('UPDATE public."TransactionDetail" '
                         'SET embedding = CAST(:v AS extensions.vector), embedding_model = :m '
                         'WHERE id = :i'),
                    {"v": to_vector_literal(vector), "m": model_name, "i": row["id"]},
                )
                written += 1
        print(f"  embedded {written}/{len(described)}")

    print(f"\ndone — {len(described)} described, {written} re-embedded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
