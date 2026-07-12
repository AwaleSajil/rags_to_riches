import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from backend.dependencies import get_supabase
from backend.services import config_service

logger = logging.getLogger("moneyrag.services.transaction")

# Fields whose value feeds the pgvector document text (merchant + category) —
# a change to any of them requires re-embedding the transaction's vector.
_VECTOR_TEXT_FIELDS = ("merchant_name", "category")

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


def _build_embeddings(config: dict):
    """Construct the user's embedding model from their AccountConfig."""
    provider = (config.get("llm_provider") or "").lower()
    api_key = config.get("api_key")
    model = config.get("embedding_model")
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        os.environ["GOOGLE_API_KEY"] = api_key
        return GoogleGenerativeAIEmbeddings(model=model or "gemini-embedding-001")
    from langchain_openai import OpenAIEmbeddings

    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAIEmbeddings(model=model or "text-embedding-3-small")


def _reembed_transaction(tx: Dict[str, Any], details: List[Dict[str, Any]], user_id: str, config: dict) -> None:
    """Re-embed one transaction's vector(s) so semantic search reflects the edit."""
    from backend.vector_db_client import get_vector_client

    embeddings = _build_embeddings(config)
    get_vector_client().sync_single_transaction(tx, details, user_id, embeddings)


def _apply_update(user: dict, transaction_id: str, changes: Dict[str, Any]):
    """Blocking: verify ownership, update, and re-read row + details.

    Returns (updated_tx, details, needs_reembed) or None if the row does not
    exist / is not the user's.
    """
    client = _client(user)
    existing = (
        client.table("Transaction")
        .select("*")
        .eq("id", transaction_id)
        .eq("user_id", user["id"])
        .limit(1)
        .execute()
    )
    if not existing.data:
        return None
    old = existing.data[0]

    if changes:
        upd = (
            client.table("Transaction")
            .update(changes)
            .eq("id", transaction_id)
            .eq("user_id", user["id"])
            .execute()
        )
        updated = upd.data[0] if upd.data else {**old, **changes}
    else:
        updated = old

    details = (
        client.table("TransactionDetail")
        .select("*")
        .eq("transaction_id", transaction_id)
        .eq("user_id", user["id"])
        .order("created_at")
        .execute()
        .data
        or []
    )
    needs_reembed = any(
        f in changes and changes[f] != old.get(f) for f in _VECTOR_TEXT_FIELDS
    )
    return updated, details, needs_reembed


async def update_transaction(
    user: dict, transaction_id: str, changes: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Update editable fields; re-embed the vector if merchant/category changed.

    A re-embed failure (e.g. LLM quota) is logged but does not fail the edit —
    the DB is the source of truth and a later full sync repairs the vector.
    """
    result = await asyncio.to_thread(_apply_update, user, transaction_id, changes)
    if result is None:
        return None
    updated, details, needs_reembed = result

    if needs_reembed:
        config = await config_service.get_config(user)
        if config:
            try:
                await asyncio.to_thread(
                    _reembed_transaction, updated, details, user["id"], config
                )
                logger.info("Re-embedded vector for transaction %s", transaction_id)
            except Exception as e:
                logger.warning(
                    "Vector re-embed failed for transaction %s (edit still saved): %s",
                    transaction_id,
                    e,
                )
        else:
            logger.warning(
                "No AccountConfig for user %s — skipping re-embed of %s",
                user["id"],
                transaction_id,
            )

    updated["details"] = details
    return updated


def _delete_row(user: dict, transaction_id: str) -> Optional[List[str]]:
    """Blocking: delete the transaction (details cascade). Returns the deleted
    line-item ids for vector cleanup, or None if nothing was deleted (404)."""
    client = _client(user)
    detail_ids = [
        d["id"]
        for d in (
            client.table("TransactionDetail")
            .select("id")
            .eq("transaction_id", transaction_id)
            .eq("user_id", user["id"])
            .execute()
            .data
            or []
        )
    ]
    deleted = (
        client.table("Transaction")
        .delete()
        .eq("id", transaction_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not deleted.data:
        return None
    return detail_ids


async def delete_transaction(user: dict, transaction_id: str) -> bool:
    """Delete a transaction and remove its vector(s). Returns False if not found."""
    detail_ids = await asyncio.to_thread(_delete_row, user, transaction_id)
    if detail_ids is None:
        return False

    # Vector cleanup is pure SQL (no LLM). Log but don't fail if it errors —
    # the row is already gone; an orphaned vector is filtered by the deleted id.
    try:
        from backend.vector_db_client import get_vector_client

        await asyncio.to_thread(
            get_vector_client().delete_transaction_vectors, transaction_id, detail_ids
        )
        logger.info("Deleted vectors for transaction %s", transaction_id)
    except Exception as e:
        logger.warning("Vector delete failed for transaction %s: %s", transaction_id, e)
    return True


def _prepare_detail_rows(
    user_id: str, transaction_id: str, bill_file_id, details_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fill in derived fields (subtotal, tax, post-tax total, taxable) for each
    incoming line item, matching the ingestion parser's math."""
    rows: List[Dict[str, Any]] = []
    for d in details_input:
        qty = d.get("item_quantity")
        unit = d.get("item_unit_subtotal_price")
        rate = d.get("tax_rate")

        subtotal = d.get("item_subtotal_price")
        if subtotal is None:
            qty_val = qty if qty is not None else 1
            unit_val = unit if unit is not None else 0
            subtotal = round(float(qty_val) * float(unit_val), 2)

        taxable = d.get("taxable")
        if taxable is None:
            taxable = rate is not None and rate > 0

        tax_amount = d.get("tax_amount")
        if tax_amount is None:
            tax_amount = round(subtotal * rate / 100.0, 2) if (taxable and rate) else 0.0

        total = d.get("item_total_price")
        if total is None:
            total = round(subtotal + tax_amount, 2)

        rows.append({
            "transaction_id": transaction_id,
            "user_id": user_id,
            "bill_file_id": bill_file_id,
            "item_description": d.get("item_description"),
            "item_quantity": qty,
            "item_unit_subtotal_price": unit,
            "item_subtotal_price": subtotal,
            "tax_amount": tax_amount,
            "tax_rate": rate,
            "taxable": taxable,
            "item_total_price": total,
            "enriched_info": d.get("enriched_info"),
        })
    return rows


def _replace_details_rows(
    user: dict, transaction_id: str, details_input: List[Dict[str, Any]]
):
    """Blocking: verify ownership, swap line items, return (tx_row, new_details)
    or None if the transaction is not found / not the user's."""
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
    tx_row = tx_res.data[0]

    new_rows = _prepare_detail_rows(
        user["id"], transaction_id, tx_row.get("source_bill_file_id"), details_input
    )

    # Replace: clear existing line items, then insert the new set.
    client.table("TransactionDetail").delete().eq(
        "transaction_id", transaction_id
    ).eq("user_id", user["id"]).execute()

    details: List[Dict[str, Any]] = []
    if new_rows:
        # insert() returns rows in insert order, preserving the user's ordering.
        details = client.table("TransactionDetail").insert(new_rows).execute().data or []

    return tx_row, details


async def replace_details(
    user: dict, transaction_id: str, details_input: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Replace a transaction's line items and resync its vectors.

    Vector resync removes the old parent + line-item vectors and re-embeds the
    new set. Failures are logged, not fatal — the DB stays the source of truth.
    """
    prepared = await asyncio.to_thread(
        _replace_details_rows, user, transaction_id, details_input
    )
    if prepared is None:
        return None
    tx_row, details = prepared

    config = await config_service.get_config(user)
    if config:
        try:
            from backend.vector_db_client import get_vector_client

            vc = get_vector_client()
            # Drop old vectors (parent + all prior line items), then re-embed.
            await asyncio.to_thread(vc.delete_transaction_vectors, transaction_id, None)
            await asyncio.to_thread(
                _reembed_transaction, tx_row, details, user["id"], config
            )
            logger.info("Resynced vectors after detail replace for %s", transaction_id)
        except Exception as e:
            logger.warning(
                "Vector resync failed after detail replace for %s: %s",
                transaction_id,
                e,
            )
    else:
        logger.warning(
            "No AccountConfig for user %s — skipping vector resync of %s",
            user["id"],
            transaction_id,
        )

    tx_row["details"] = details
    return tx_row
