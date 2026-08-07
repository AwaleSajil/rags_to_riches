import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.dependencies import get_supabase

logger = logging.getLogger("moneyrag.services.conversation")

MAX_HISTORY = 12  # recent turns passed to the agent for context


def _client(user: dict):
    return get_supabase(user["access_token"])


async def list_conversations(user: dict) -> List[Dict[str, Any]]:
    def _run():
        res = (
            _client(user)
            .table("Conversation")
            .select("id,title,created_at,updated_at")
            .eq("user_id", user["id"])
            .order("updated_at", desc=True)
            .execute()
        )
        return res.data or []
    return await asyncio.to_thread(_run)


async def create_conversation(user: dict, title: str = "New chat") -> Dict[str, Any]:
    def _run():
        res = (
            _client(user)
            .table("Conversation")
            .insert({"user_id": user["id"], "title": title})
            .execute()
        )
        return res.data[0]
    return await asyncio.to_thread(_run)


async def get_messages(user: dict, conversation_id: str) -> List[Dict[str, Any]]:
    """Load a conversation, with any photos it referred to resolved to URLs.

    A capture used to vanish on reload: the photo lived only as a local file URI
    in the running app and the turn was never written down. Anything the user
    CONFIRMED is stored, so it can come back — it just needs a fresh signed URL,
    because the one shown at the time has long expired.
    """
    def _run():
        client = _client(user)
        rows = (
            client.table("Message")
            .select("*")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user["id"])
            .order("created_at")
            .execute()
        ).data or []

        wanted = {
            str(file_id)
            for row in rows
            for file_id in (row.get("bill_file_ids") or [])
        }
        if not wanted:
            return rows

        keys = {
            str(f["id"]): f["s3_key"]
            for f in (
                client.table("BillFile").select("id,s3_key")
                .in_("id", list(wanted)).eq("user_id", user["id"]).execute().data or []
            )
        }
        for row in rows:
            urls = list(row.get("images") or [])
            for file_id in row.get("bill_file_ids") or []:
                key = keys.get(str(file_id))
                if not key:
                    # The photo was deleted since. Showing nothing is right; the
                    # turn stays, which is the honest record of what happened.
                    continue
                try:
                    signed = client.storage.from_("money-rag-files").create_signed_url(key, 3600)
                    url = (
                        signed.get("signedURL")
                        or signed.get("signedUrl")
                        or signed.get("signed_url")
                    )
                    if url:
                        urls.append(url)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not sign %s for replay: %s", key, e)
            if urls:
                row["images"] = urls
        return rows

    return await asyncio.to_thread(_run)


async def delete_conversation(user: dict, conversation_id: str) -> None:
    def _run():
        _client(user).table("Conversation").delete().eq("id", conversation_id).eq(
            "user_id", user["id"]
        ).execute()
    await asyncio.to_thread(_run)


async def add_message(
    user: dict,
    conversation_id: str,
    role: str,
    content: str,
    charts: Optional[list] = None,
    images: Optional[list] = None,
    pending_transactions: Optional[list] = None,
    bill_file_ids: Optional[list] = None,
) -> Dict[str, Any]:
    def _run():
        client = _client(user)
        record = {
            "conversation_id": conversation_id,
            "user_id": user["id"],
            "role": role,
            "content": content,
            "charts": charts,
            "images": images,
            "pending_transactions": pending_transactions,
            # Stored as ids, never as URLs: a signed URL lasts an hour and would
            # reload as a broken image. Resolved afresh on every read.
            "bill_file_ids": bill_file_ids,
        }
        res = client.table("Message").insert(record).execute()
        # Bump the conversation so it sorts to the top of the list.
        client.table("Conversation").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", conversation_id).eq("user_id", user["id"]).execute()
        return res.data[0]
    return await asyncio.to_thread(_run)


async def set_title_from_first_message(user: dict, conversation_id: str, message: str) -> None:
    """Set the title from the first user message — only while it's still the default."""
    title = " ".join((message or "").split())[:48].strip() or "New chat"
    if len(message or "") > 48:
        title += "…"

    def _run():
        _client(user).table("Conversation").update({"title": title}).eq(
            "id", conversation_id
        ).eq("user_id", user["id"]).eq("title", "New chat").execute()
    await asyncio.to_thread(_run)


def to_agent_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Map stored rows to the {role, content} the agent expects, capped to recent turns."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("content")
    ]
    return history[-MAX_HISTORY:]
