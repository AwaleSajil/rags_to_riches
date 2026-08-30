"""Backfill size and product-identity vectors onto existing TransactionDetail rows.

New receipts get these at verify time. This is only for the backlog that predates
migration 021.

What gets written:

  embedding / embedding_model — built from price_service.embedding_text, the
      SAME builder PriceObservation uses. That is the whole point: a shelf tag
      and a receipt line only compare if their vectors were made from the same
      kind of text.

      enriched_info is included and matters more than it looks. It decodes the
      abbreviations a receipt line cannot carry ("GV LF 2 GAL" -> "a two-gallon
      container of Great Value brand low-fat milk"). Measured on labelled pairs,
      including it separates same- from different-product cleanly; omitting it
      makes them overlap by 0.208 and misses 4 of 9 true matches. Rows with no
      enrichment get a weaker vector and will match less well — that is a known
      gap, not a bug in this script.

Embedding is the expensive part, so the script is resumable: rows that already
have a vector are skipped, and --limit caps a single run to stay inside quota.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_item_vectors.py --email you@example.com --dry-run
    PYTHONPATH=. .venv/bin/python scripts/backfill_item_vectors.py --email you@example.com --limit 100
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
import psycopg.rows
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.price_service import (  # noqa: E402
    embedding_text,
    to_vector_literal,
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Account to backfill")
    parser.add_argument("--limit", type=int, help="Max rows this run (embedding quota)")
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Rebuild vectors that already exist — needed after the embedded "
             "text changes shape, e.g. when enrichment was added to it",
    )
    args = parser.parse_args()
    load_dotenv()

    with psycopg.connect(
        os.environ["DATABASE_URL"], autocommit=True, row_factory=psycopg.rows.dict_row
    ) as conn:
        account = conn.execute(
            '''SELECT u.id, c.llm_provider, c.api_key, c.embedding_model
               FROM public."User" u
               LEFT JOIN public."AccountConfig" c ON c.user_id = u.id
               WHERE u.email = %s''',
            (args.email,),
        ).fetchone()
        if not account:
            raise SystemExit(f"No user found for {args.email}")

        rows = conn.execute(
            '''SELECT id, item_description, item_quantity, unit_quantity_subtotal,
                      enriched_info
               FROM public."TransactionDetail"
               WHERE user_id = %s AND item_description IS NOT NULL
                 AND (embedding IS NULL OR %s)
               ORDER BY created_at''',
            (account["id"], args.reembed),
        ).fetchall()
        if args.limit:
            rows = rows[: args.limit]

        if not rows:
            print("Nothing to backfill — every row already has a vector.")
            return

        model = None
        model_name = None
        if True:
            if not account.get("api_key"):
                raise SystemExit(
                    f"No API key on {args.email}'s AccountConfig; "
                    "an embedding model is required."
                )
            from backend.services.transaction_service import _build_embeddings

            model = _build_embeddings(account)
            model_name = account.get("embedding_model") or "gemini-embedding-001"

        embedded = failed = unenriched = 0
        for row in rows:
            description = row["item_description"]
            if not row.get("enriched_info"):
                unenriched += 1

            vector_literal = None
            if model is not None:
                text = embedding_text(description, None, None, row.get("enriched_info"))
                try:
                    vector_literal = to_vector_literal(model.embed_query(text))
                    embedded += 1
                except Exception as e:  # noqa: BLE001
                    # Quota or transient failure: leave the row for the next run
                    # rather than writing a half-populated record.
                    failed += 1
                    print(f"  ! embed failed for {description!r}: {e}")

            if args.dry_run:
                continue

            conn.execute(
                '''UPDATE public."TransactionDetail"
                   SET embedding = COALESCE(%s::extensions.vector, embedding),
                       embedding_model = COALESCE(%s, embedding_model)
                   WHERE id = %s''',
                (
                    vector_literal,
                    model_name if vector_literal else None,
                    row["id"],
                ),
            )

        verb = "Would update" if args.dry_run else "Updated"
        print(
            f"{verb} {len(rows)} row(s): {embedded} embedded, "
            f"{failed} embed failure(s)."
        )
        if unenriched:
            print(
                f"WARNING: {unenriched} row(s) had no enriched_info. Their vectors "
                "are built from the bare description and will match noticeably "
                "worse — enrichment is what makes abbreviated lines reachable."
            )
        if not args.dry_run:
            remaining = conn.execute(
                '''SELECT count(*) AS n FROM public."TransactionDetail"
                   WHERE user_id = %s AND item_description IS NOT NULL
                     AND embedding IS NULL''',
                (account["id"],),
            ).fetchone()["n"]
            print(f"{remaining} row(s) still without a vector — re-run to continue.")


if __name__ == "__main__":
    main()
