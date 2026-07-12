import asyncio
import logging
from typing import Any, Dict, List, Optional

from backend.dependencies import get_supabase

logger = logging.getLogger("moneyrag.services.transaction")

# Columns returned for the browser list (kept lean — no enriched_info blob).
_LIST_COLUMNS = (
    "id,trans_date,description,amount,category,merchant_name,location,"
    "subtotal,tax_total,tax_breakdown,source,created_at"
)


def _client(user: dict):
    return get_supabase(user["access_token"])


def _sanitize_or_term(q: str) -> str:
    """Strip characters that would break PostgREST's `or=(...)` filter grammar.

    Commas and parentheses delimit the logical-operator list, so a raw search
    term containing them could smuggle in extra filters. We drop them and use
    `*` (PostgREST's ilike wildcard) around what remains.
    """
    cleaned = "".join(c for c in q if c not in ",()")
    return cleaned.strip()


async def list_transactions(
    user: dict,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List the user's transactions, newest first, with optional filters."""

    def _run():
        query = (
            _client(user)
            .table("Transaction")
            .select(_LIST_COLUMNS)
            .eq("user_id", user["id"])
        )
        if category:
            query = query.eq("category", category)
        if start_date:
            query = query.gte("trans_date", start_date)
        if end_date:
            query = query.lte("trans_date", end_date)
        if q:
            term = _sanitize_or_term(q)
            if term:
                pattern = f"*{term}*"
                query = query.or_(
                    f"merchant_name.ilike.{pattern},description.ilike.{pattern}"
                )
        # Newest first; created_at breaks ties for same-day rows.
        query = query.order("trans_date", desc=True).order("created_at", desc=True)
        res = query.execute()
        return res.data or []

    return await asyncio.to_thread(_run)


async def get_transaction(user: dict, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one transaction plus its ordered line items, or None if not found."""

    def _run():
        client = _client(user)
        tx_res = (
            client.table("Transaction")
            .select("*")
            .eq("id", transaction_id)
            .eq("user_id", user["id"])
            .limit(1)
            .execute()
        )
        if not tx_res.data:
            return None
        tx = tx_res.data[0]
        details_res = (
            client.table("TransactionDetail")
            .select("*")
            .eq("transaction_id", transaction_id)
            .eq("user_id", user["id"])
            .order("created_at")
            .execute()
        )
        tx["details"] = details_res.data or []
        return tx

    return await asyncio.to_thread(_run)
