"""Encrypt AccountConfig.api_key rows written before encryption existed.

Keys used to be stored in plaintext. backend/crypto.py now encrypts them, and
reads tolerate the old format — a legacy row still works, and gets encrypted the
next time that user saves their config. This script converts the backlog now
rather than waiting for every user to visit Settings, which some never will.

Idempotent: a row already carrying the "enc:v1:" prefix is skipped, so running
it twice is harmless.

Usage:

  1. (default) Dry run — reports what would change, writes nothing.

       APP_ENCRYPTION_KEY=... DATABASE_URL=... python scripts/encrypt_existing_api_keys.py

  2. --apply — performs the update.

       APP_ENCRYPTION_KEY=... DATABASE_URL=... python scripts/encrypt_existing_api_keys.py --apply

APP_ENCRYPTION_KEY must be the SAME key the API runs with. Encrypting with a
different one leaves rows the API cannot read, and the affected users would have
to re-enter their keys.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
import psycopg.rows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.crypto import _PREFIX, encrypt_secret  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the encrypted values. Without this, only reports.",
    )
    args = parser.parse_args()

    if not os.environ.get("APP_ENCRYPTION_KEY"):
        sys.exit("APP_ENCRYPTION_KEY is not set — refusing to run.")

    with psycopg.connect(
        os.environ["DATABASE_URL"], autocommit=True, row_factory=psycopg.rows.dict_row
    ) as conn:
        rows = conn.execute(
            'SELECT id, user_id, api_key FROM public."AccountConfig" '
            "WHERE api_key IS NOT NULL AND api_key <> ''"
        ).fetchall()

        legacy = [r for r in rows if not r["api_key"].startswith(_PREFIX)]
        print(f"{len(rows)} config row(s); {len(legacy)} still plaintext.")

        if not legacy:
            print("Nothing to do.")
            return

        if not args.apply:
            for row in legacy:
                print(f"  would encrypt user_id={row['user_id']}")
            print("\nDry run — nothing written. Re-run with --apply.")
            return

        for row in legacy:
            conn.execute(
                'UPDATE public."AccountConfig" SET api_key = %s WHERE id = %s',
                (encrypt_secret(row["api_key"]), row["id"]),
            )
            print(f"  encrypted user_id={row['user_id']}")

        print(f"\nDone — {len(legacy)} row(s) encrypted.")


if __name__ == "__main__":
    main()
