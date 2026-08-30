"""Backfill high-confidence CSV/receipt and cross-CSV transaction links.

This preserves every transaction. It only creates rows in TransactionLink,
which supports a transaction matching any number of other source records.
"""

import argparse
import os
import re
from datetime import date

import psycopg
from dotenv import load_dotenv


def merchant_key(row: dict) -> str:
    value = str(row.get("merchant_name") or row.get("description") or "").lower()
    return re.sub(r"[^a-z]", "", value)


def is_match(left: dict, right: dict) -> bool:
    left_merchant, right_merchant = merchant_key(left), merchant_key(right)
    if len(left_merchant) < 4 or len(right_merchant) < 4:
        return False
    if left_merchant not in right_merchant and right_merchant not in left_merchant:
        return False
    try:
        left_date = date.fromisoformat(str(left["trans_date"]))
        right_date = date.fromisoformat(str(right["trans_date"]))
        return abs((left_date - right_date).days) <= 1 and abs(float(left["amount"]) - float(right["amount"])) <= 0.10
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Account email to backfill")
    args = parser.parse_args()
    load_dotenv()

    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        user = conn.execute('SELECT id FROM public."User" WHERE email = %s', (args.email,)).fetchone()
        if not user:
            raise SystemExit(f"No user found for {args.email}")
        rows = conn.execute(
            '''SELECT id, trans_date, amount, merchant_name, description, source, source_csv_id
               FROM public."Transaction"
               WHERE user_id = %s AND source IN ('csv', 'bill')''',
            (user["id"],),
        ).fetchall()

        links: set[tuple[str, str, str]] = set()
        for index, row in enumerate(rows):
            for candidate in rows[index + 1 :]:
                # Never link entries from the same CSV file; those may be
                # independent purchases with identical values.
                if row.get("source_csv_id") and row.get("source_csv_id") == candidate.get("source_csv_id"):
                    continue
                if not is_match(row, candidate):
                    continue
                left_id, right_id = sorted((str(row["id"]), str(candidate["id"])))
                match_type = "csv_receipt" if {row["source"], candidate["source"]} == {"csv", "bill"} else "csv_csv"
                links.add((left_id, right_id, match_type))

        for left_id, right_id, match_type in links:
            conn.execute(
                '''INSERT INTO public."TransactionLink"
                   (user_id, transaction_id, linked_transaction_id, match_type, confidence)
                   VALUES (%s, %s, %s, %s, 1)
                   ON CONFLICT (user_id, transaction_id, linked_transaction_id) DO NOTHING''',
                (user["id"], left_id, right_id, match_type),
            )
        print(f"Created or retained {len(links)} transaction link(s) for {args.email}.")


if __name__ == "__main__":
    main()
