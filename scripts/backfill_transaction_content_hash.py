"""Re-fingerprint CSV transactions written before the hash stopped being per-file.

The old signature began with the CSV's own file id, which is new on every
upload. Two exports covering the same week therefore produced entirely
different hashes for the same purchase, matched nothing, and were both written
— deliberately, at the time: overlapping rows were meant to survive as separate
durable records and be reconciled by a TransactionLink instead.

The signature is now the bank's own (date, amount, description) plus an
occurrence index, so a purchase already stored is recognised on the next upload
and skipped before it costs an enrichment call and an embedding. Rows written
under the old scheme cannot match it, which means without this script the change
only protects uploads made from here on and your existing history stays invisible
to it.

Nothing is deleted and no row moves. Only content_hash is rewritten.

WHY THE INDEX IS ASSIGNED ACROSS ALL HISTORY, not per file: ingestion counts
occurrences within one export because that export carries every row for the
dates it covers. A backfill has no files to count within — it sees the union —
so it orders each (date, amount, description) group by created_at and hands out
0, 1, 2 in that order. That is stable, and it is also what stops the rewrite
colliding with itself: two rows that the new scheme WOULD have deduplicated are
already both here, and giving them one hash would violate
UNIQUE (user_id, content_hash) halfway through the run.

Those cross-file groups are the interesting output. A group whose rows come from
more than one CSVFile is very likely one purchase imported twice — the exact
thing the old per-file hash could not see. They are REPORTED, never merged:
which copy to keep, and whether what hangs off it has since been edited or
linked, is not a judgement a script should make.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_transaction_content_hash.py --email you@example.com
    PYTHONPATH=. .venv/bin/python scripts/backfill_transaction_content_hash.py --email you@example.com --apply
"""

import argparse
import getpass
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

from backend.services.purchase_match import csv_row_hash, csv_row_signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="the account whose rows to re-hash")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the hashes. Without this, nothing is changed anywhere.",
    )
    args = parser.parse_args()

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("SUPABASE_URL and SUPABASE_KEY must be set (see .env).")
        return 1

    password = getpass.getpass(f"Password for {args.email}: ")
    client = create_client(url, key)
    session = client.auth.sign_in_with_password(
        {"email": args.email, "password": password}
    )
    user_id = session.user.id

    rows = (
        client.table("Transaction")
        .select("id,trans_date,amount,description,source_csv_id,content_hash,created_at")
        .eq("user_id", user_id)
        .eq("source", "csv")
        .order("created_at")
        .execute()
        .data
        or []
    )
    print(f"\n{len(rows)} CSV transaction(s) for {args.email}.\n")
    if not rows:
        print("Nothing to do.")
        return 0

    # created_at then id: created_at alone ties for rows written in one batch,
    # and a tie broken by insertion order would hand out different indices on a
    # second run over identical data.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (str(r.get("created_at") or ""), str(r["id"]))):
        groups[csv_row_signature(row["trans_date"], row["amount"], row.get("description"))].append(row)

    to_write: list[tuple[dict, str]] = []
    cross_file: list[list[dict]] = []
    for signature, members in groups.items():
        if len({str(m.get("source_csv_id")) for m in members}) > 1:
            cross_file.append(members)
        for index, row in enumerate(members):
            digest = csv_row_hash(
                row["trans_date"], row["amount"], row.get("description"), index
            )
            if digest != row.get("content_hash"):
                to_write.append((row, digest))

    print(f"{len(to_write)} row(s) need a new hash; {len(rows) - len(to_write)} already current.\n")

    if cross_file:
        print(f"{len(cross_file)} purchase(s) appear in more than one CSV export:\n")
        for members in cross_file[:20]:
            first = members[0]
            print(
                f"  {str(first['trans_date'])[:10]}  {float(first['amount']):>9.2f}  "
                f"{str(first.get('description') or '')[:44]:<46} x{len(members)}"
            )
        if len(cross_file) > 20:
            print(f"  ...and {len(cross_file) - 20} more")
        print(
            "\n  Each is one purchase written more than once, which the old\n"
            "  per-file hash could not detect. Every total over these rows is\n"
            "  inflated by the extra copies. They are kept as they are: check\n"
            "  whether a TransactionLink already reconciles the pair (the\n"
            "  transactions list and TransactionDeduped both collapse linked\n"
            "  rows) before deleting anything by hand.\n"
        )

    if not args.apply:
        print(f"Dry run. {len(to_write)} hash(es) would be rewritten. Re-run with --apply.")
        return 0

    written = 0
    for row, digest in to_write:
        client.table("Transaction").update({"content_hash": digest}).eq(
            "id", row["id"]
        ).eq("user_id", user_id).execute()
        written += 1
    print(
        f"Rewrote {written} hash(es). Re-uploading a statement that overlaps "
        "these dates now skips the rows already here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
