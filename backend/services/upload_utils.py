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

# One extension → content type mapping for every path that writes a file to
# storage or hands one to the vision model.
#
# It used to be spelled out at each of those call sites, and the copies had
# already drifted apart: the capture path defaulted an unrecognised extension to
# image/jpeg while the vision path defaulted the same name to image/png, so one
# file could be labelled two different things depending on how it arrived.
CONTENT_TYPES = {
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Unreachable for anything that came through safe_filename, which admits only
# the extensions above. Deliberately not an image type: a file that reached here
# by some other route should be refused by whatever receives it rather than
# mislabelled as a picture, which is exactly what the old per-site defaults did.
DEFAULT_CONTENT_TYPE = "application/octet-stream"

# Receipts and bank exports are small; this is generous for both and keeps a
# single request from filling the disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def extension_of(filename: str) -> str:
    """The lowercased extension including its dot, or "" when there is none.

    Lowercased because every comparison in this module is against a lowercase
    allowlist, and an iPhone hands back `IMG_0001.JPG` often enough that the
    per-call-site `.lower().endswith(...)` chains this replaces all had to
    remember it individually.
    """
    return os.path.splitext(filename)[1].lower()


def is_image(filename: str) -> bool:
    """Whether this name routes to the image pipeline instead of the CSV parser.

    The routing decision itself, in one place: storage folder, destination
    table, and content type all follow from it, and a name that answers False
    here is handed to the CSV parser.
    """
    return extension_of(filename) in IMAGE_EXTENSIONS


def content_type_for(filename: str) -> str:
    """The content type to store or transmit this file under."""
    return CONTENT_TYPES.get(extension_of(filename), DEFAULT_CONTENT_TYPE)


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

    extension = extension_of(name)
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


def storage_key(user_id: str, folder: str, filename: str, content_hash: str) -> str:
    """Where this file's bytes live in the bucket.

    The content hash is in the KEY, not just the row, because the display name
    is not unique and was being used as the key on its own. Two different files
    sharing a name — `IMG_0001.jpg` picked from a gallery twice, a bank that
    always exports `activity.csv` — collided, and the upload uses upsert, so the
    second one overwrote the first one's BYTES while creating a second row
    pointing at the same object. The first receipt then showed the second
    receipt's photo, and deleting either row removed the object both rows named.

    Byte-identical uploads are turned away before this is reached
    (`file_by_content_hash`), so an identical hash here cannot be a collision
    between two different files — it is the same file, and sharing one object is
    then correct.

    Rows written before this keep whatever key they were given: every read goes
    through the stored `s3_key`, so old and new coexist without a migration.
    """
    return f"{user_id}/{folder}/{content_hash[:16]}_{filename}"


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
