"""Single-photo capture from the chat tab.

The batch upload path (`/files/upload`) hands work to a subprocess and the client
polls for it. That shape is right for a folder of bank statements and wrong here:
the user is standing in a shop holding up a phone, so this route uploads,
classifies and extracts inline and answers with the draft. One round trip, one
vision call, no polling.

Nothing is committed to the ledger from this, and — for a shelf price — nothing
is stored at all until the user confirms. A photo of a tag is a working artifact:
its value is the observation it becomes, not the image, and a tag that was read
and then abandoned used to leave a `BillFile` row cluttering the Files tab
forever.

So a capture is held in memory with its bytes in a temp directory, and the row is
written only when it is confirmed. A RECEIPT is the exception and materialises as
soon as it is recognised: it is a financial record, the review screen needs a
durable handle to come back to, and losing one costs far more than a stray file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from typing import Optional

from backend.dependencies import get_supabase
from backend.services import background, config_service

logger = logging.getLogger("moneyrag.services.capture")

VALID_KINDS = ("receipt", "price_tag", "unknown")

# Captures read but not yet confirmed: capture_id -> details, with the photo's
# bytes on disk beside it. In memory for the same reason file_service tracks
# ingestion progress that way — it is short-lived state about an in-flight
# request, not a record. A restart drops in-flight captures, which is why a
# receipt is written to the database the moment it is recognised.
_pending: dict[str, dict] = {}

# Long enough that coming back to a card after lunch still works, short enough
# that a forgotten capture does not keep its bytes on disk indefinitely. Any TTL
# leaves a window where the card is still on screen and the capture is gone, so
# the length only moves that boundary — materialise() is what has to fail
# gracefully when it is crossed.
PENDING_TTL_SECONDS = 6 * 60 * 60


def _sweep_pending(now: Optional[float] = None) -> int:
    """Drop captures nobody came back for, and their temp files."""
    now = now if now is not None else time.time()
    stale = [
        key for key, entry in _pending.items()
        if now - entry["created_at"] > PENDING_TTL_SECONDS
    ]
    for key in stale:
        entry = _pending.pop(key, None)
        if entry and not entry.get("file_id"):
            shutil.rmtree(os.path.dirname(entry["local_path"]), ignore_errors=True)
    if stale:
        logger.info("Swept %d abandoned capture(s)", len(stale))
    return len(stale)


def forget_pending(capture_id: str, user_id: str) -> bool:
    """Discard an unconfirmed capture. Nothing was stored, so nothing is left."""
    entry = _pending.get(capture_id)
    if not entry or entry["user_id"] != user_id:
        return False
    if entry.get("file_id"):
        # It was confirmed, so there IS a row and a stored photo. Say so, and let
        # the caller delete them properly rather than silently orphaning both.
        return False
    _pending.pop(capture_id, None)
    shutil.rmtree(os.path.dirname(entry["local_path"]), ignore_errors=True)
    logger.info("Discarded unconfirmed capture %s", capture_id)
    return True


async def materialise(user: dict, capture_id: str) -> Optional[str]:
    """Write a pending capture to storage and give it a BillFile row.

    Called when the user commits to it — confirming a shelf price, or answering
    "which is this?" with 'receipt'. Returns the new file_id, or the id itself if
    it is already a real one (so callers can pass either kind of handle).

    Storing happens ONCE. One photo can hold a tag per product and the card
    confirms them one at a time with the same handle, so the resolved file_id is
    remembered against the capture: dropping the entry outright made every tag
    after the first hand back the capture id, which is not a BillFile id and
    would fail the observation's foreign key.
    """
    entry = _pending.get(capture_id)
    if entry is None:
        # Two very different situations, and treating them alike is what turned a
        # stale card into a foreign-key violation shown to the user: the id is
        # either a photo already stored, or a capture that has been swept. Only
        # the first is safe to hand back.
        if await asyncio.to_thread(_billfile_exists_sync, user, capture_id):
            return capture_id
        raise ValueError(
            "That photo is no longer available — it is only held while you review "
            "it, and this one has expired. Take it again and it will save."
        )
    if entry["user_id"] != user["id"]:
        raise ValueError("Photo not found")
    if entry.get("file_id"):
        return entry["file_id"]

    file_id = await asyncio.to_thread(
        _upload_photo_sync, user, entry["local_path"], entry["filename"],
    )
    await asyncio.to_thread(
        _write_draft_sync, user, file_id, entry["kind"], entry["draft"], entry["filename"],
    )
    # The bytes are in storage now, so the temp copy goes; the entry stays as the
    # capture-id -> file-id mapping until the sweep retires it.
    entry["file_id"] = file_id
    shutil.rmtree(os.path.dirname(entry["local_path"]), ignore_errors=True)
    logger.info("Materialised capture %s as BillFile %s", capture_id, file_id)
    return file_id


def _billfile_exists_sync(user: dict, file_id: str) -> bool:
    """Is this a photo that was already stored, rather than a stale capture id?"""
    try:
        rows = (
            get_supabase(user["access_token"]).table("BillFile").select("id")
            .eq("id", file_id).eq("user_id", user["id"]).limit(1).execute().data or []
        )
    except Exception:  # noqa: BLE001 - a malformed id is simply not one of ours
        return False
    return bool(rows)


def _write_draft_sync(user: dict, file_id: str, kind: str, draft: dict, filename: str) -> None:
    """Store what the vision pass read, now that the row exists."""
    get_supabase(user["access_token"]).table("BillFile").update({
        "kind": kind or "unknown",
        "raw_ocr_string": json.dumps(draft or {}),
        "filename": filename,
    }).eq("id", file_id).eq("user_id", user["id"]).execute()


def _upload_photo_sync(user: dict, local_path: str, filename: str) -> str:
    """Put the photo in storage and create its BillFile row. Returns file_id."""
    client = get_supabase(user["access_token"])
    s3_key = f"{user['id']}/bills/{filename}"

    content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    client.storage.from_("money-rag-files").upload(
        file=local_path,
        path=s3_key,
        file_options={"content-type": content_type, "upsert": "true"},
    )

    inserted = client.table("BillFile").insert({
        "user_id": user["id"],
        "filename": filename,
        "s3_key": s3_key,
        # Set once the vision model has looked at it; 'unknown' until then so a
        # crash mid-classification leaves a row that prompts rather than one
        # that silently claims to be a receipt.
        "kind": "unknown",
    }).execute()
    if not inserted.data:
        raise RuntimeError("Could not record the uploaded photo")
    return str(inserted.data[0]["id"])


def enrich_price_tag_draft(draft: dict) -> dict:
    """Fill in what the confirm card needs, for every tag in the photo.

    A shelf photo holds a tag per product, so the draft is {"tags": [...]}. A
    flat single tag is still accepted and wrapped: rows written before the photo
    could hold more than one still have to open.
    """
    if not isinstance(draft, dict):
        return {"tags": []}
    tags = draft.get("tags")
    if not isinstance(tags, list):
        # Pre-multi-tag shape, or a model that returned one bare object.
        tags = [draft] if draft else []
    return {"tags": [_enrich_one_tag(tag) for tag in tags if isinstance(tag, dict)]}


def _enrich_one_tag(draft: dict) -> dict:
    """Fill in what the confirm card needs that the tag did not print.

    Only the per-unit price is derived, and only when the tag prints none. The
    store's own figure always wins: it did the pack-size arithmetic for the
    exact package in front of the user, and a disagreement between its number
    and ours means the size was misread rather than something to average away.

    Nothing else is computed. Sizes and names are no longer parsed into columns
    — the row keeps what the tag said, and the agent reads it.
    """
    enriched = dict(draft or {})

    def number(key):
        try:
            value = enriched.get(key)
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    quantity = number("size_value")
    subtotal = number("item_subtotal_price")

    if enriched.get("unit_quantity_subtotal") is None and quantity and subtotal:
        enriched["unit_quantity_subtotal"] = round(subtotal / quantity, 4)

    package_unit = (enriched.get("size_unit") or "").strip().lower() or None
    per_unit = number("unit_quantity_subtotal")

    # What the printed figure is PER, which is regularly NOT the package unit:
    # a one-gallon milk jug at $3.49 is commonly tagged "$0.87 PER QUART".
    priced_per = (enriched.get("unit_price_unit") or "").strip().lower() or None
    if priced_per is None and per_unit is not None and quantity and subtotal:
        # The tag did not say, so check whether it CAN be the package unit:
        # printed-price x size should come back to the package price. Milk fails
        # this by a factor of four, which is exactly the quarts-in-a-gallon it
        # was really quoting. Guessing wrong here labels $0.87 as the price of a
        # gallon that costs $3.49, so an unlabelled figure is left unlabelled.
        consistent = abs(per_unit * float(quantity) - float(subtotal)) <= 0.02
        priced_per = package_unit if consistent else None

    enriched["unit_price_unit"] = priced_per
    enriched["unit_price_display"] = (
        f"${per_unit:.2f}/{priced_per}" if per_unit is not None and priced_per else None
    )
    # True when the tag gave no size at all, so the UI can say the comparison
    # will be same-item-only rather than implying it is per-ounce.
    #
    # Reads size_value/size_unit, the names migration 034 gave these. When this
    # still looked for item_quantity it found nothing and every tag claimed "no
    # size found" while the card, already renamed, displayed the size right above
    # the warning.
    enriched["size_unknown"] = not (quantity and package_unit)
    return enriched


async def classify_photo(user: dict, config: dict, capture_id: str) -> None:
    """Read one held photo and record what it turned out to be.

    Runs detached from the request; the pending entry's `kind` leaving
    'processing' is the completion signal the client polls for.

    A RECEIPT is written to the database here. Everything else stays in memory
    until the user confirms it, so a shelf price that is read and then abandoned
    leaves nothing behind.
    """
    from backend.services.rag_manager import rag_manager

    entry = _pending.get(capture_id)
    if entry is None:
        return

    try:
        rag = await rag_manager.get_or_create(user, config)
        # file_id=None: there is no row to write to yet, so the draft comes back
        # rather than being stored.
        draft, kind = await rag._ingest_bill(entry["local_path"], None)
        entry["kind"] = kind
        entry["draft"] = enrich_price_tag_draft(draft) if kind == "price_tag" else (draft or {})
        entry["filename"] = _photo_display_name(entry["filename"], kind, draft)
        logger.info("Capture %s read as %s", capture_id, kind)
    except Exception as e:
        # The bytes are still held, so the user can classify by hand rather than
        # losing the capture entirely.
        logger.error("Capture extraction failed for %s: %s", capture_id, e, exc_info=True)
        entry["kind"] = "unknown"
        entry["draft"] = {}
        entry["error"] = str(e)
        return

    # A receipt is a financial record and its review screen needs a durable
    # handle, so it does not wait for a confirmation that may never come.
    if entry["kind"] == "receipt":
        try:
            await materialise(user, capture_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not store receipt capture %s: %s", capture_id, e, exc_info=True)


def _photo_display_name(original: str, kind: str, draft: Optional[dict]) -> str:
    """A recognisable filename, chosen once the photo has been read."""
    from money_rag import MoneyRAG

    try:
        return MoneyRAG._photo_filename(original, kind, draft or {})
    except Exception:  # noqa: BLE001 - a name is never worth failing a capture over
        return original


async def capture_photo(
    user: dict,
    local_path: str,
    filename: str,
    location: Optional[str] = None,
) -> dict:
    """Store one photo and start reading it. Returns as soon as it is saved.

    The vision call takes ~15 seconds, and this used to hold the HTTP response
    open for all of it. A phone will not keep a request alive that long: the
    upload succeeded, the server finished its work, and the reply arrived to
    nobody — so the app sat on a typing indicator forever and no card ever
    appeared. Whether a mobile client tolerates a 15-second response is not
    something this feature should depend on.

    So the response is now just "stored, id is X". The client polls
    GET /captures/{file_id} until `kind` settles.
    """
    config = await config_service.get_config(user)
    if not config:
        raise ValueError("Account config required. Please add your API key in Settings.")

    _sweep_pending()
    capture_id = str(uuid.uuid4())
    _pending[capture_id] = {
        "user_id": user["id"],
        "local_path": local_path,
        "filename": filename,
        "kind": "processing",
        "draft": {},
        "error": None,
        # A resolved place name, never coordinates. Held with the capture so it
        # survives the poll and a later re-open, not just the upload response.
        "location": location,
        # Set once confirmed; also marks the capture as no longer discardable.
        "file_id": None,
        "created_at": time.time(),
    }
    logger.info("Capture %s held for user_id=%s — reading in background", capture_id, user["id"])

    background.spawn(
        classify_photo(user, config, capture_id),
        name=f"classify:{capture_id}",
    )

    return {
        # An opaque handle. It becomes a real BillFile id once the photo is
        # confirmed — or for a receipt, as soon as it is recognised.
        "file_id": capture_id,
        # Not yet known. The client shows "reading…" and polls.
        "kind": "processing",
        "draft": {},
        # An already-resolved place name such as "Main St, Norwalk". The device
        # does the GPS fix and the reverse geocode; only the answer is sent, so
        # no coordinate ever reaches the server or the database.
        "location": location,
    }


def _read_draft_sync(user: dict, file_id: str) -> dict:
    """Read back what the vision pass stored, scoped to the owner."""
    client = get_supabase(user["access_token"])
    rows = (
        client.table("BillFile")
        .select("raw_ocr_string")
        .eq("id", file_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows or not rows[0].get("raw_ocr_string"):
        return {}
    try:
        return json.loads(rows[0]["raw_ocr_string"])
    except (TypeError, ValueError):
        return {}


def _read_capture_sync(user: dict, file_id: str) -> Optional[dict]:
    """Read back a stored photo's kind, scoped to the owner."""
    client = get_supabase(user["access_token"])
    rows = (
        client.table("BillFile")
        .select("id,kind,raw_ocr_string")
        .eq("id", file_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0] if rows else None


def _pending_result(capture_id: str, entry: dict) -> dict:
    return {
        "file_id": capture_id,
        "kind": entry["kind"],
        "draft": entry["draft"],
        "location": entry.get("location"),
        **({"error": entry["error"]} if entry.get("error") else {}),
    }


async def get_capture(user: dict, file_id: str) -> dict:
    """Re-open a photo that was read earlier, in the shape capture_photo returns.

    The batch path reads a photo in a subprocess and hands the app nothing but a
    file id, so a price tag uploaded from the Files tab had no way to reach the
    card that confirms it — the app routed to chat and chat had nothing to show.
    This is that missing read.
    """
    # Held captures answer from memory — there is deliberately no row for them
    # until the user confirms, so a shelf price they walk away from leaves
    # nothing in the Files tab.
    entry = _pending.get(file_id)
    if entry is not None and entry["user_id"] == user["id"]:
        return _pending_result(file_id, entry)

    row = await asyncio.to_thread(_read_capture_sync, user, file_id)
    if not row:
        raise ValueError("Photo not found")

    kind = row.get("kind") or "unknown"
    # 'unknown' means two different things, and the client has to tell them
    # apart: the vision pass has not run yet, or it ran and could not decide.
    # raw_ocr_string is what separates them — it is written by the same update
    # that sets a real kind, so its absence means "still reading".
    if kind == "unknown" and not row.get("raw_ocr_string"):
        return {"file_id": file_id, "kind": "processing", "draft": {}}

    draft = await asyncio.to_thread(_read_draft_sync, user, file_id)
    if kind == "price_tag":
        # Applied on read rather than stored: it is derived display data, and the
        # batch path never ran it. Doing it here means both entry points show the
        # same card regardless of which one read the photo.
        draft = enrich_price_tag_draft(draft)
    return {"file_id": file_id, "kind": kind, "draft": draft}


def _set_kind_sync(user: dict, file_id: str, kind: str) -> dict:
    client = get_supabase(user["access_token"])
    updated = (
        client.table("BillFile")
        .update({"kind": kind})
        .eq("id", file_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not updated.data:
        raise ValueError("Photo not found")
    return updated.data[0]


async def set_kind(user: dict, file_id: str, kind: str) -> dict:
    """Record the user's answer when the model could not tell what the photo was."""
    if kind not in ("receipt", "price_tag"):
        raise ValueError("Choose either 'receipt' or 'price_tag'")

    entry = _pending.get(file_id)
    if entry is not None and entry["user_id"] == user["id"]:
        entry["kind"] = kind
        if kind == "price_tag":
            entry["draft"] = enrich_price_tag_draft(entry["draft"])
            return _pending_result(file_id, entry)
        # Calling it a receipt is a commitment, so it gets stored now — same
        # rule as one the model recognised on its own.
        new_id = await materialise(user, file_id)
        draft = await asyncio.to_thread(_read_draft_sync, user, new_id)
        return {"file_id": new_id, "kind": kind, "draft": draft}

    await asyncio.to_thread(_set_kind_sync, user, file_id, kind)
    draft = await asyncio.to_thread(_read_draft_sync, user, file_id)
    if kind == "price_tag":
        draft = enrich_price_tag_draft(draft)
    return {"file_id": file_id, "kind": kind, "draft": draft}
