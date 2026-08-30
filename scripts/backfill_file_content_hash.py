"""Fingerprint files uploaded before `content_hash` existed.

New uploads are hashed on arrival and a byte-identical re-upload is refused
(see backend/services/file_service.py). The backlog has no hash, so the FIRST
re-upload of an old file still slips through — it matches nothing and gets
accepted. For a CSV that doubles every transaction in it; for a photo it pays
for a second vision extraction and leaves a stray receipt in the Files tab.
This closes that window, for either table.

It also answers the question the hash makes askable for the first time: are any
of the files already in the bucket duplicates of each other? Two CSV rows
sharing a hash means one statement was imported twice and every total computed
from those transactions is inflated; two BillFile rows means a photo was
uploaded twice, so a vision extraction was paid for and a stray receipt is
sitting in the Files tab. They are REPORTED, never deleted — which of two
copies to keep, and whether what sits under them has since been edited, is not
a judgement a script should make.

The unique index means both rows of a duplicate pair cannot carry the same hash.
The earliest keeps it; the later ones are left unhashed and listed for you.

The bucket is private with RLS scoped to auth.uid(), so this signs in as the
account whose files it is touching, exactly like backfill_compress_images. It
can therefore never reach anyone else's files, which is why it wants a password
rather than a service-role key.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_file_content_hash.py --email you@example.com
    PYTHONPATH=. .venv/bin/python scripts/backfill_file_content_hash.py --email you@example.com --apply
    PYTHONPATH=. .venv/bin/python scripts/backfill_file_content_hash.py --email you@example.com --table BillFile --apply
"""

import argparse
import getpass
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

BUCKET = "money-rag-files"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="the account whose files to hash")
    parser.add_argument(
        "--table",
        choices=("CSVFile", "BillFile"),
        default="CSVFile",
        help="which backlog to fingerprint (default: CSVFile)",
    )
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
        client.table(args.table)
        .select("id,filename,s3_key,upload_date,content_hash")
        .eq("user_id", user_id)
        .order("upload_date")
        .execute()
        .data
        or []
    )
    pending = [r for r in rows if not r.get("content_hash")]

    print(f"\n{len(rows)} {args.table} row(s); {len(pending)} without a hash.\n")
    if not pending:
        print("Nothing to do.")
        return 0

    store = client.storage.from_(BUCKET)
    # hash -> the first row that claimed it, in upload order.
    claimed: dict[str, dict] = {r["content_hash"]: r for r in rows if r.get("content_hash")}
    to_write: list[tuple[dict, str]] = []
    duplicates: list[tuple[dict, dict]] = []
    failures = 0

    for row in pending:
        try:
            raw = store.download(row["s3_key"])
        except Exception as e:  # noqa: BLE001 — one missing object must not stop the run
            print(f"  {row['filename'][:48]:<50} could not read: {e}")
            failures += 1
            continue

        digest = hashlib.sha256(raw).hexdigest()
        first = claimed.get(digest)
        if first:
            duplicates.append((row, first))
            print(f"  {row['filename'][:48]:<50} DUPLICATE of {first['filename']}")
            continue

        claimed[digest] = row
        to_write.append((row, digest))
        print(f"  {row['filename'][:48]:<50} {digest[:16]}…  ({len(raw):,} bytes)")

    print()
    if duplicates:
        print(f"{len(duplicates)} file(s) are byte-identical to one you already had:")
        for later, first in duplicates:
            print(
                f"  '{later['filename']}' ({str(later['upload_date'])[:10]})"
                f"  ==  '{first['filename']}' ({str(first['upload_date'])[:10]})"
            )
        if args.table == "CSVFile":
            consequence = (
                "  One statement imported twice. Its transactions are written\n"
                "  twice, so any SQL the agent runs over them double-counts."
            )
        else:
            consequence = (
                "  The same photo uploaded twice. A second vision extraction was\n"
                "  paid for, and there is a stray receipt in the Files tab. If BOTH\n"
                "  were verified there may be a duplicate transaction too, though\n"
                "  receipt_content_hash normally refuses the second."
            )
        print(
            "\n" + consequence + "\n"
            "  Delete the later one from the Files tab if you want it gone — this\n"
            "  script will not, because what sits under it may have been edited\n"
            "  since.\n"
        )

    if failures:
        print(f"{failures} file(s) could not be read from storage.\n")

    if not args.apply:
        print(f"Dry run. {len(to_write)} hash(es) would be written. Re-run with --apply.")
        return 0

    written = 0
    for row, digest in to_write:
        client.table(args.table).update({"content_hash": digest}).eq("id", row["id"]).execute()
        written += 1
    print(f"Wrote {written} hash(es). Re-uploading any of these files is now refused.")
    if duplicates:
        print(
            f"{len(duplicates)} duplicate(s) left unhashed on purpose — the unique "
            "index allows only one row per hash."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
