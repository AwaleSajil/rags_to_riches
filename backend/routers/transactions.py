import hashlib
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.dependencies import get_current_user, get_supabase
from backend.schemas.transactions import (
    TransactionCreate,
    TransactionListItem,
    TransactionResponse,
    TransactionUpdate,
    TransactionWithDetails,
)
from backend.services import transaction_service

logger = logging.getLogger("moneyrag.routers.transactions")

router = APIRouter()


def _require_user(user: dict | None) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.get("", response_model=List[TransactionListItem])
async def list_transactions(
    category: Optional[str] = Query(None, description="Exact category match"),
    start_date: Optional[str] = Query(None, description="Inclusive lower bound (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Inclusive upper bound (YYYY-MM-DD)"),
    q: Optional[str] = Query(None, description="Text search on merchant/description"),
    user: dict = Depends(get_current_user),
):
    """List the current user's transactions, newest first, with optional filters."""
    user = _require_user(user)
    try:
        return await transaction_service.list_transactions(
            user, category=category, start_date=start_date, end_date=end_date, q=q
        )
    except Exception as e:
        logger.error("Failed to list transactions: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list transactions: {e}")


@router.get("/{transaction_id}", response_model=TransactionWithDetails)
async def get_transaction(
    transaction_id: str,
    user: dict = Depends(get_current_user),
):
    """Fetch one transaction plus its line items (ordered), including tax breakdown."""
    user = _require_user(user)
    try:
        tx = await transaction_service.get_transaction(user, transaction_id)
    except Exception as e:
        logger.error("Failed to get transaction %s: %s", transaction_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get transaction: {e}")
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return tx


@router.patch("/{transaction_id}", response_model=TransactionWithDetails)
async def update_transaction(
    transaction_id: str,
    body: TransactionUpdate,
    user: dict = Depends(get_current_user),
):
    """Update editable fields. Re-embeds the vector if merchant/category changed."""
    user = _require_user(user)
    changes = body.model_dump(exclude_unset=True)
    if "trans_date" in changes and changes["trans_date"] is not None:
        changes["trans_date"] = changes["trans_date"].isoformat()
    if not changes:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        tx = await transaction_service.update_transaction(user, transaction_id, changes)
    except Exception as e:
        logger.error("Failed to update transaction %s: %s", transaction_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update transaction: {e}")
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return tx


@router.delete("/{transaction_id}")
async def delete_transaction(
    transaction_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a transaction (line items cascade) and remove its vector(s)."""
    user = _require_user(user)
    try:
        deleted = await transaction_service.delete_transaction(user, transaction_id)
    except Exception as e:
        logger.error("Failed to delete transaction %s: %s", transaction_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete transaction: {e}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return {"message": "Transaction deleted"}


@router.post("", response_model=TransactionResponse)
async def create_transaction(
    body: TransactionCreate,
    user: dict = Depends(get_current_user),
):
    """Insert a manually-entered transaction. user_id always comes from JWT."""
    logger.info("Creating manual transaction for user_id=%s: %s", user["id"], body.description)

    user_id = user["id"]

    # Content hash (same algorithm as money_rag.py _ingest_csv)
    date_str = body.trans_date.isoformat()
    amount_str = str(round(body.amount, 2))
    merchant = (body.merchant_name or body.description).lower().strip().split()[0]
    merchant_clean = "".join(c for c in merchant if c.isalnum())
    hash_input = f"{date_str}{amount_str}{merchant_clean}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    record = {
        "user_id": user_id,
        "description": body.description,
        "amount": body.amount,
        "trans_date": date_str,
        "category": body.category,
        "merchant_name": body.merchant_name or body.description,
        "source": "manual",
        "content_hash": content_hash,
    }

    try:
        sb = get_supabase(access_token=user.get("access_token"))
        result = sb.table("Transaction").upsert([record], on_conflict="content_hash").execute()

        if result.data:
            row = result.data[0]
            return TransactionResponse(
                id=row["id"],
                description=row["description"],
                amount=float(row["amount"]),
                trans_date=str(row["trans_date"]),
                category=row.get("category", "Uncategorized"),
                merchant_name=row.get("merchant_name"),
            )

        # Fallback: fetch by hash if upsert didn't return data
        fetch = (
            sb.table("Transaction")
            .select("*")
            .eq("content_hash", content_hash)
            .eq("user_id", user_id)
            .execute()
        )
        if fetch.data:
            row = fetch.data[0]
            return TransactionResponse(
                id=row["id"],
                description=row["description"],
                amount=float(row["amount"]),
                trans_date=str(row["trans_date"]),
                category=row.get("category", "Uncategorized"),
                merchant_name=row.get("merchant_name"),
            )
        raise HTTPException(status_code=500, detail="Transaction insert returned no data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create transaction: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save transaction: {e}")
