"""Re-encode receipt photos already in Supabase storage.

New uploads are shrunk on the phone (frontend/src/lib/compressImage.ts). This is
only for the backlog that predates it — photos uploaded straight off the camera
sensor at ~2MB each.

The settings deliberately match the client, so a backfilled photo and a freshly
uploaded one are indistinguishable:

    long edge <= 2048px, JPEG quality 70

2048 is not a guess. The vision call sends the photo to OpenAI with no `detail`
parameter, so it defaults to high detail, which rescales anything larger to fit
inside 2048x2048 before the model sees it. Everything above that line is already
being discarded server-side on every extraction; this only stops paying to store
and upload it. Extraction sees the same pixels it saw before.

Three modes, in the order you should use them:

  1. (default) Dry run. Downloads each photo, compresses it in memory, and
     reports what it would save. Writes nothing, anywhere.

  2. --sample DIR. Same, but writes original/compressed pairs to DIR so you can
     open them side by side and judge the quality yourself before committing.
     This is the point of running it twice.

  3. --apply --backup DIR. Overwrites the stored object. The backup directory is
     required, not optional: the original photo bytes are otherwise gone, and a
     receipt you can no longer read is worse than a receipt that is large.

The bucket is private with RLS scoped to auth.uid(), so this signs in as the
account whose photos it is touching. It can therefore never reach anyone else's
files, which is why it wants a password rather than a service-role key.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_compress_images.py --email you@example.com
    PYTHONPATH=. .venv/bin/python scripts/backfill_compress_images.py --email you@example.com --sample /tmp/quality --limit 5
    PYTHONPATH=. .venv/bin/python scripts/backfill_compress_images.py --email you@example.com --apply --backup ~/receipt-backup
"""

import argparse
import getpass
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BUCKET = "money-rag-files"
MAX_EDGE = 2048
QUALITY = 70


def human(size: float) -> str:
    for unit in ("B", "kB", "MB"):
        if abs(size) < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} MB"


def compress(raw: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return (jpeg_bytes, original_size, new_size, already_done).

    `already_done` marks a photo that is already within MAX_EDGE — which, after
    a first pass, means every photo this script has touched. JPEG is lossy in
    both directions: re-encoding an already-compressed receipt at quality 70
    throws away a second helping of the same detail, and nothing about the
    result would tell you it had happened. Running --apply twice by accident has
    to be a no-op, so the caller skips these rather than uploading them again.
    """
    image = Image.open(io.BytesIO(raw))
    # Phones record rotation in EXIF rather than rotating the pixels. Re-saving
    # without applying it first would leave every photo taken sideways stored
    # sideways, since the tag does not survive the round trip.
    image = ImageOps.exif_transpose(image)
    before = image.size

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    longest = max(image.size)
    already_done = longest <= MAX_EDGE
    if not already_done:
        scale = MAX_EDGE / longest
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    # optimize rebuilds the Huffman tables — a few percent, for free. EXIF is
    # dropped by not passing it through, which also takes the GPS tag off any
    # photo that carried one.
    image.save(buffer, format="JPEG", quality=QUALITY, optimize=True, progressive=True)
    return buffer.getvalue(), before, image.size, already_done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Account whose photos to re-encode")
    parser.add_argument("--limit", type=int, help="Stop after this many photos")
    parser.add_argument("--sample", help="Write original/compressed pairs here for review")
    parser.add_argument("--apply", action="store_true", help="Overwrite the stored objects")
    parser.add_argument("--backup", help="Save originals here first. Required with --apply")
    args = parser.parse_args()

    if args.apply and not args.backup:
        raise SystemExit(
            "--apply overwrites the stored photo and the original is not recoverable.\n"
            "Pass --backup DIR so the originals land on disk first."
        )

    load_dotenv()
    from supabase import create_client  # noqa: E402  (after load_dotenv)

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    password = os.environ.get("BACKFILL_PASSWORD") or getpass.getpass(
        f"Password for {args.email}: "
    )
    session = client.auth.sign_in_with_password({"email": args.email, "password": password})
    user_id = session.user.id

    store = client.storage.from_(BUCKET)
    folder = f"{user_id}/bills"
    # list() pages at 100 by default, which would silently ignore the backlog
    # this script exists for.
    entries = store.list(folder, {"limit": 1000})

    photos = [e for e in entries if e["name"].lower().endswith((".jpg", ".jpeg"))]
    # There are no PNGs in the bucket today. Handling one properly means writing
    # JPEG bytes to a new key and repointing BillFile.s3_key at it, and untested
    # rename logic is a worse trade than a loud skip.
    skipped = [e["name"] for e in entries if e not in photos]
    photos.sort(key=lambda e: e["name"])
    if args.limit:
        photos = photos[: args.limit]

    if not photos:
        print(f"No JPEGs under {folder}.")
        return

    for directory in (args.sample, args.backup):
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)

    mode = "APPLYING" if args.apply else "dry run — nothing will be written"
    print(f"{len(photos)} photo(s) under {folder}  [{mode}]\n")
    print(f"{'photo':<44} {'before':>10} {'after':>10} {'saved':>8}  dimensions")

    total_before = total_after = 0
    failures = skipped_count = 0

    for entry in photos:
        key = f"{folder}/{entry['name']}"
        try:
            raw = store.download(key)
            small, before_dim, after_dim, already_done = compress(raw)
        except Exception as e:  # noqa: BLE001 — one bad photo must not stop the run
            print(f"{entry['name'][:43]:<44} failed: {e}")
            failures += 1
            continue

        # Left alone when it is already within MAX_EDGE (a second --apply would
        # otherwise compress it again), or when re-encoding made it bigger. The
        # totals count the original in both cases, so a dry run never promises a
        # saving that --apply would not deliver.
        kept = already_done or len(small) >= len(raw)
        skipped_count += already_done
        total_before += len(raw)
        total_after += len(raw) if kept else len(small)
        percent = 0 if kept else (1 - len(small) / len(raw)) * 100

        if already_done:
            note = "  (already within 2048px — left alone)"
        elif kept:
            note = "  (kept original — re-encoding made it larger)"
        else:
            note = ""
        if kept:
            after_dim = before_dim
        print(
            f"{entry['name'][:43]:<44} {human(len(raw)):>10} "
            f"{human(len(raw) if kept else len(small)):>10} "
            f"{percent:>7.0f}%  {before_dim[0]}x{before_dim[1]} -> "
            f"{after_dim[0]}x{after_dim[1]}{note}"
        )

        # Nothing is written for a photo that is being left alone: there is no
        # pair to compare, and no original to restore.
        if kept:
            continue

        stem = Path(entry["name"]).stem
        if args.sample:
            (Path(args.sample) / f"{stem}.original.jpg").write_bytes(raw)
            (Path(args.sample) / f"{stem}.compressed.jpg").write_bytes(small)
        if args.backup:
            (Path(args.backup) / entry["name"]).write_bytes(raw)

        if args.apply:
            store.upload(
                file=small,
                path=key,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

    saved = total_before - total_after
    print(f"\n{'total':<44} {human(total_before):>10} {human(total_after):>10} "
          f"{(saved / total_before * 100 if total_before else 0):>7.0f}%")
    print(f"Would free {human(saved)}." if not args.apply else f"Freed {human(saved)}.")

    if skipped_count:
        print(
            f"{skipped_count} photo(s) were already within {MAX_EDGE}px and were "
            "left untouched — re-encoding them would only lose detail."
        )
    if skipped:
        print(f"\nSkipped {len(skipped)} non-JPEG object(s): {', '.join(skipped[:5])}")
    if failures:
        print(f"{failures} photo(s) failed and were left untouched.")
    if args.sample:
        print(f"\nPairs written to {args.sample} — compare before using --apply.")


if __name__ == "__main__":
    main()
