"""Shared validation for anything a client uploads.

Both the batch `/files/upload` route and the single-photo `/captures` route take
a client-supplied filename and stream bytes to disk, so the same two guards have
to apply to both: the name must not escape the temp directory, and the body must
not be unbounded.

Extracted from routers/files.py rather than duplicated — a second copy of the
extension allowlist would drift, and drift here means a .heic quietly reaching
the CSV parser again.
"""

from __future__ import annotations

import os

from fastapi import UploadFile

# Anything not recognised as an image is handed to the CSV parser, so the
# allowlist is what keeps a .pdf or an iPhone .heic from being silently parsed
# as a spreadsheet. Matches the picker's filter in frontend/app/(tabs)/ingest.tsx.
ALLOWED_EXTENSIONS = frozenset({".csv", ".png", ".jpg", ".jpeg"})

# A capture is a single photo, so images are the only thing /captures accepts —
# sending a CSV there is a client bug, not a user mistake to be accommodated.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

# Receipts and bank exports are small; this is generous for both and keeps a
# single request from filling the disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def safe_filename(raw: str | None, allowed: frozenset[str] = ALLOWED_EXTENSIONS) -> str:
    """Reduce a client-supplied filename to a bare, allowlisted basename.

    The name reaches both `os.path.join(temp_dir, ...)` and the storage key, so
    a value like `../../etc/passwd` would otherwise escape the temp directory
    and land somewhere it shouldn't. Strips any directory component (both
    separators, since the client may be on Windows) plus null bytes, then
    requires a known extension.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].replace("\x00", "").strip()
    # Leading dots would make the file hidden, or resolve to "." / ".." outright.
    name = name.lstrip(".")
    if not name:
        raise ValueError("A file was uploaded without a usable filename")

    extension = os.path.splitext(name)[1].lower()
    if extension not in allowed:
        if allowed is IMAGE_EXTENSIONS:
            raise ValueError(f"'{name}' is not an image. Capture a PNG or JPG photo.")
        raise ValueError(
            f"'{name}' has an unsupported type. Upload a CSV or a PNG/JPG image."
        )
    return name


async def save_within_limit(upload: UploadFile, destination: str) -> int:
    """Stream one upload to disk, aborting if it exceeds MAX_UPLOAD_BYTES.

    Written in chunks rather than a single `.read()` so a large file is never
    held in memory in full.
    """
    total = 0
    with open(destination, "wb") as fh:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"'{upload.filename}' is larger than the "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit"
                )
            fh.write(chunk)
    if total == 0:
        raise ValueError(f"'{upload.filename}' is empty")
    return total


def file_sha256(path: str) -> str:
    """Fingerprint a saved upload by its bytes.

    Streamed in chunks rather than read whole: an upload is already capped at
    MAX_UPLOAD_BYTES, but there is no reason to hold all of it in memory again
    just to hash it.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
