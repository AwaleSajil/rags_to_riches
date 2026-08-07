import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.dependencies import get_current_user, get_supabase
from backend.services import config_service, correction_service

logger = logging.getLogger("moneyrag.routers.corrections")

router = APIRouter()


class CorrectionApply(BaseModel):
    """A fix the user confirmed on a card the agent proposed.

    Arrives on a normal authenticated request, not from the model: the agent
    only ever produces the proposal. There is no delete counterpart to this
    endpoint by design — removing a purchase changes what was spent, and that
    belongs somewhere the consequence is visible.
    """

    table: str = Field(min_length=1)
    row_id: str = Field(min_length=1)
    changes: Dict[str, Any]


@router.post("")
async def apply_correction(
    body: CorrectionApply,
    user: dict = Depends(get_current_user),
):
    """Apply a confirmed correction to one row the caller owns."""
    client = get_supabase(user.get("access_token"))

    try:
        updated = await asyncio.to_thread(
            correction_service.apply_correction,
            client, body.table, body.row_id, body.changes, user["id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Correction failed for %s %s: %s", body.table, body.row_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not apply that fix: {e}")

    if not updated:
        # Either the row is gone or it is not this user's. Same answer either
        # way: there is nothing here for you.
        raise HTTPException(status_code=404, detail="That row was not found")

    # The searchable text changed, so the stored vector now describes the old
    # wording. Rebuilt in the background — the fix itself is already saved, and
    # losing it to an embedding hiccup would be the worse trade.
    if correction_service.needs_reembedding(body.table, body.changes):
        try:
            config = await config_service.get_config(user)
            if config:
                from backend.services import background

                background.spawn(
                    _refresh_corrected_row(
                        user, config, body.table, body.row_id,
                        rewrite_description=correction_service.invalidates_description(body.changes),
                    ),
                    name=f"reembed:{body.row_id}",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Correction saved but re-embedding did not start: %s", e)

    return {"message": "Fixed", "row": updated}


async def _refresh_corrected_row(
    user: dict, config: dict, table: str, row_id: str, rewrite_description: bool = False
) -> None:
    """Bring a corrected row's description and vector back in step with it.

    A size correction leaves the stored description stating the OLD size —
    "A two-gallon container of Great Value low-fat milk" on a row now recorded as
    one gallon — and that text is both what the agent quotes and what the row is
    embedded from. So a size fix rewrites the description before re-embedding.
    """
    try:
        if rewrite_description:
            await asyncio.to_thread(_rewrite_description_sync, user, config, table, row_id)
        if table == "PriceObservation":
            from backend.services import price_service

            client = get_supabase(user.get("access_token"))
            rows = (
                client.table("PriceObservation").select("*")
                .eq("id", row_id).eq("user_id", user["id"]).limit(1).execute().data or []
            )
            if not rows:
                return
            row = rows[0]
            embedding, model = price_service.build_observation_embedding(
                row.get("item_description"), row.get("brand_name"),
                price_service.size_text_of(row.get("size_value"), row.get("size_unit")),
                config,
                observed_context=row.get("enriched_info")
                or row.get("item_qualitative_description"),
            )
            if embedding:
                client.table("PriceObservation").update(
                    {"embedding": embedding, "embedding_model": model}
                ).eq("id", row_id).eq("user_id", user["id"]).execute()
            return

        from backend.services import transaction_service

        # A line item's vector lives on its parent's re-embed, which rebuilds the
        # transaction and every child together.
        client = get_supabase(user.get("access_token"))
        transaction_id = row_id
        if table == "TransactionDetail":
            rows = (
                client.table("TransactionDetail").select("transaction_id")
                .eq("id", row_id).eq("user_id", user["id"]).limit(1).execute().data or []
            )
            if not rows:
                return
            transaction_id = rows[0]["transaction_id"]

        tx = await transaction_service.get_transaction(user, transaction_id)
        if tx:
            await asyncio.to_thread(
                transaction_service._reembed_transaction,
                tx, tx.get("details", []), user["id"], config,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not re-embed %s %s: %s", table, row_id, e)


def _rewrite_description_sync(user: dict, config: dict, table: str, row_id: str) -> None:
    """Regenerate enriched_info so it agrees with the corrected row."""
    from backend.services import price_service

    client = get_supabase(user.get("access_token"))
    rows = (
        client.table(table).select("*")
        .eq("id", row_id).eq("user_id", user["id"]).limit(1).execute().data or []
    )
    if not rows:
        return
    row = rows[0]

    merchant = row.get("merchant_name")
    if table == "TransactionDetail" and row.get("transaction_id"):
        parent = (
            client.table("Transaction").select("merchant_name")
            .eq("id", row["transaction_id"]).limit(1).execute().data or []
        )
        merchant = parent[0].get("merchant_name") if parent else None

    described = price_service.describe_product(
        config,
        row.get("item_description") or "",
        shop=merchant,
        brand=row.get("brand_name"),
        size=price_service.size_text_of(row.get("size_value"), row.get("size_unit")),
    )
    if not described:
        # Better to have none than one that contradicts the row: a stale
        # description gets quoted to the user as fact.
        described = None
    client.table(table).update({"enriched_info": described}).eq(
        "id", row_id
    ).eq("user_id", user["id"]).execute()
    logger.info("Rewrote description for %s %s", table, row_id)
