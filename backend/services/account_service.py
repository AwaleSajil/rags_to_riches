"""Deleting an account, for real.

Both stores require this and both mean the same thing by it: the account itself
goes, not just its contents. Apple's guideline 5.1.1(v) and Play's deletion
policy are both explicit that offering only a "clear my data" button, or only a
support email, is not enough.

So the operation that matters is removing the row in `auth.users`. Everything
this app stores hangs off it — "User".id references it ON DELETE CASCADE, and
AccountConfig, CSVFile, BillFile, Transaction, TransactionDetail,
TransactionLink, PriceObservation, Conversation and Message all cascade from
"User" in turn. One delete takes the lot, which is worth far more than a
hand-written list of tables that would silently miss whichever table gets added
next.

Three things do NOT cascade, and each is handled here:

  * OBJECT STORAGE. Uploaded statements and receipt photos live in a bucket with
    no foreign key to anything. They are the most sensitive thing the app holds
    — a photograph of a receipt, a full bank export — so leaving them behind
    would make the deletion a lie.
  * IN-MEMORY STATE. A cached RAG instance owns a temp directory; a pending
    capture holds photo bytes on local disk. Both outlive the database row.
  * THE ENCRYPTED API KEY. It goes with AccountConfig via the cascade, which is
    the point of encrypting it at rest in the first place.

Storage is cleared BEFORE the auth record, deliberately. If storage fails the
account still exists and the user can try again, which is recoverable; the other
order would delete someone's account and leave their receipts in a bucket with
nothing left pointing at them, which is not.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from backend.dependencies import admin_client, client_for

logger = logging.getLogger("moneyrag.services.account")

BUCKET = "money-rag-files"

# Every prefix the app writes under, as built by upload_utils.storage_key.
# Listed explicitly rather than discovered: a bucket listing that silently
# returned nothing would look exactly like a successful cleanup.
USER_FOLDERS = ("bills", "csvs")


def _stored_keys_sync(user: dict) -> List[str]:
    """Every object belonging to this user, from the rows AND from the bucket.

    Both sources, because they fail differently. The rows are authoritative for
    what the app knows about; the bucket listing catches anything orphaned by an
    earlier partial failure — an upload that succeeded moments before its
    INSERT did not. A deletion is exactly the wrong moment to trust bookkeeping
    over the actual contents.
    """
    client = client_for(user)
    user_id = user["id"]
    keys: set[str] = set()

    for table in ("CSVFile", "BillFile"):
        try:
            rows = (
                client.table(table).select("s3_key")
                .eq("user_id", user_id).execute().data or []
            )
            keys.update(r["s3_key"] for r in rows if r.get("s3_key"))
        except Exception as e:  # noqa: BLE001
            # Not fatal on its own — the bucket listing below may still find
            # them — but it must be visible, because the consequence is a file
            # surviving a deletion.
            logger.warning("Could not list %s rows for deletion: %s", table, e)

    for folder in USER_FOLDERS:
        prefix = f"{user_id}/{folder}"
        try:
            for entry in client.storage.from_(BUCKET).list(prefix) or []:
                name = entry.get("name")
                # A folder placeholder has no id; only real objects do.
                if name and entry.get("id"):
                    keys.add(f"{prefix}/{name}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not list storage under %s: %s", prefix, e)

    return sorted(keys)


def _delete_storage_sync(user: dict, keys: List[str]) -> None:
    """Remove the user's objects. Raises if the bucket refuses, so the caller
    can stop before the account itself is removed."""
    if not keys:
        logger.info("No stored objects for user_id=%s", user["id"])
        return
    client_for(user).storage.from_(BUCKET).remove(keys)
    logger.info("Removed %d stored object(s) for user_id=%s", len(keys), user["id"])


def _delete_auth_user_sync(user_id: str) -> None:
    """Remove the auth record. Everything in the database cascades from it."""
    admin_client().auth.admin.delete_user(user_id)
    logger.info("Deleted auth user %s — database rows cascade", user_id)


async def _forget_in_memory(user_id: str) -> None:
    """Drop what the process is holding for this user, on disk and in memory.

    None of this is in the database, so none of it cascades: a cached RAG
    instance owns a temp directory, and an unconfirmed capture holds the photo's
    bytes. Best-effort — the account is what has to go, and a stale cache entry
    for a user who no longer exists is harmless by comparison.
    """
    from backend.services import capture_service, file_service
    from backend.services.rag_manager import rag_manager

    try:
        await rag_manager.invalidate(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not release RAG instance for %s: %s", user_id, e)

    file_service.ingestion_status.pop(user_id, None)

    stale = [
        capture_id
        for capture_id, entry in capture_service._pending.items()
        if entry.get("user_id") == user_id
    ]
    for capture_id in stale:
        # Reuses the normal discard so the temp directory goes with the entry.
        # A capture already materialised into a BillFile has its bytes removed
        # by the storage sweep above instead.
        capture_service.forget_pending(capture_id, user_id)
        capture_service._pending.pop(capture_id, None)
    if stale:
        logger.info("Dropped %d in-flight capture(s) for %s", len(stale), user_id)


async def delete_account(user: dict) -> dict:
    """Delete this user's account and everything belonging to it.

    Returns a small summary for the log and the response. Raises RuntimeError
    when the service key is missing, which the route turns into a message the
    user can act on rather than a generic failure.
    """
    user_id = user["id"]
    logger.info("Account deletion requested for user_id=%s", user_id)

    # Fail before anything is removed if this deployment cannot finish the job.
    # Half-deleting an account and then reporting an error is worse than
    # refusing outright.
    admin_client()

    keys = await asyncio.to_thread(_stored_keys_sync, user)
    await asyncio.to_thread(_delete_storage_sync, user, keys)
    await _forget_in_memory(user_id)
    await asyncio.to_thread(_delete_auth_user_sync, user_id)

    logger.info(
        "Account deleted for user_id=%s — %d stored object(s) removed",
        user_id, len(keys),
    )
    return {"deleted": True, "objects_removed": len(keys)}
