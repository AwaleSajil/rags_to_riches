"""Display names for stored photos.

A photo is stored under an opaque key but shown to the user by filename, so both
paths that produce one — the vision pass in money_rag.py, which names a photo
from what it read, and transaction_service.py, which renames it again after the
user corrects the merchant or date at review — have to turn free text into
something safe to put in a filename.

They had a copy of that each. The copies agreed, but the rule is fiddly enough
to drift: the character class, the `strip("_")` that keeps a leading merchant
like "& Co" from producing a leading underscore, and the fallback for when the
model returned nothing at all are three separate decisions, and a name that
differs between the two paths means the same receipt is called one thing on
upload and another after review.

The FORMAT is deliberately not shared — the two paths build different names on
purpose (the vision pass appends a receipt time, the review rename does not).
Only the primitives live here.
"""

from __future__ import annotations

import os
import re

# Anything outside this collapses to a single underscore. Deliberately strict
# rather than an "unsafe characters" denylist: this text is model output, it
# lands in both a filename and a storage key, and the set of characters that are
# awkward in one of those is longer than it looks — quotes, colons, slashes,
# and anything non-ASCII a merchant name may carry.
_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def slugify(value: object, fallback: str = "") -> str:
    """Reduce free text to underscore-separated alphanumerics.

    Returns `fallback` when there is nothing left — a model that returned null,
    an empty string, or a merchant written entirely in punctuation all collapse
    to "" here, and an empty run would otherwise leave a filename with a double
    underscore or a leading one.
    """
    return _UNSAFE.sub("_", str(value or "")).strip("_") or fallback


def photo_extension(original: str, fallback: str = ".jpg") -> str:
    """The extension to keep when renaming a photo.

    Case is preserved rather than normalised: this only ever re-labels a file
    that already exists in storage under its original name, so lowercasing here
    would make the display name disagree with the key it points at.
    """
    return os.path.splitext(original)[1] or fallback
