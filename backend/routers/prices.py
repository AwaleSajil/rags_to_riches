import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.dependencies import client_for, get_current_user, get_supabase
from backend.services import background, capture_service, config_service, price_service

logger = logging.getLogger("moneyrag.routers.prices")

router = APIRouter()


class PriceObservationCreate(BaseModel):
    """A shelf price the user confirmed.

    Deliberately not a transaction: seeing a price is not spending money, so
    this never reaches "Transaction" and never affects a spending total.

    Shaped to mirror TransactionDetail, because the two get compared against
    each other. A tag reading "$4.29 / 12 OZ" lands as quantity 12, unit 'oz',
    unit_quantity_subtotal 0.3575, item_subtotal_price 4.29 — the same shape a
    receipt line uses, so the comparison needs no translation layer.
    """

    item_description: str = Field(min_length=1)
    brand_name: Optional[str] = None

    # The tag's own arithmetic. unit_quantity_subtotal is the PRINTED per-unit
    # price where the tag shows one, never recomputed: the store already did the
    # pack-size division for the exact package on the shelf, and a disagreement
    # between its figure and ours means the size was misread.
    # The PACKAGE the tag prices — 12 for "$4.29 / 12 OZ". Named for what it
    # holds: nothing is bought from a shelf, so there was never a count here.
    size_value: Optional[float] = Field(default=None, gt=0)
    size_unit: Optional[str] = None
    unit_quantity_subtotal: Optional[float] = Field(default=None, ge=0)
    # What that printed figure is PER. Often not the package unit — a one-gallon
    # jug is commonly tagged per quart — so without this the number gets labelled
    # with the package unit and reads as the price of the whole package.
    unit_price_unit: Optional[str] = None
    item_subtotal_price: Optional[float] = Field(default=None, ge=0)

    merchant_name: Optional[str] = None
    # A human-readable place, not coordinates — "Main St, Norwalk". Resolved on
    # the device at capture time, so the raw fix never leaves the phone.
    location: Optional[str] = None

    # Everything the photo shows that is not a number, in the tag's own words:
    # "2 for $5 with card", "CLEARANCE", "Sale ends 8/15", "best before 08/05",
    # "dented box". Deliberately not parsed into flags and dates — a model
    # reading a shelf sign is guessing at structure, and a wrong end date
    # silently turns a limited offer into what the item normally costs. The
    # agent reads this alongside the price and reasons about it.
    item_qualitative_description: Optional[str] = None

    # What the USER said about this item in chat — "for the party", "mum likes
    # this one", "cheaper than last week". Distinct from
    # item_qualitative_description, which is what the photo showed.
    #
    # Deliberately NOT part of the embedded text. The vector represents product
    # identity, and an occasion or an opinion is not identity: embedding "for
    # the party" would pull an unrelated item toward every other party purchase
    # and weaken the match this table exists to make.
    note: Optional[str] = None

    bill_file_id: Optional[str] = None
    # Which tag in the source photo this is, 0-based. One shelf photo holds a tag
    # per product, and this is what keeps a re-confirm updating the right row
    # instead of adding another.
    tag_index: int = Field(default=0, ge=0)


@router.post("")
async def create_price_observation(
    body: PriceObservationCreate,
    user: dict = Depends(get_current_user),
):
    """Record a confirmed shelf price."""
    # Embedded on the way in so a later "have I seen this before?" can match.
    # The qualitative description joins the embedded text for the same reason
    # enriched_info does on the receipt side: a bare tag like "GV LF 2 GAL" is
    # close to unmatchable, and the surrounding words make it findable.
    #
    # Failure yields (None, None) rather than raising. Refusing to save because
    # an embedding call hit its quota would lose the photo the user is standing
    # in a shop to take; the row can be embedded later.
    config = await config_service.get_config(user)
    client = client_for(user)

    # This is the moment a shelf photo earns its place: until now it was held in
    # memory with its bytes in a temp dir and no row anywhere, so a tag that was
    # read and abandoned left nothing behind. Confirming it writes the photo to
    # storage and gives it the BillFile row the observation points at.
    bill_file_id = body.bill_file_id
    if bill_file_id:
        try:
            bill_file_id = await capture_service.materialise(user, bill_file_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    try:
        # Shared with the agent's check_price tool rather than duplicated. The
        # copy that lived here passed None where the size belongs, so a 12 oz jar
        # and a 64 oz jar embedded identically — the exact distinction a price
        # comparison rests on.
        saved = await asyncio.to_thread(
            price_service.record_observation,
            client, config, body.model_dump(), user["id"], bill_file_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Researched, then re-embedded, then compared — all inline. The vector
    # written at insert holds only the tag's own words; the receipt side is
    # matched on a product description, and the measured floor assumes both
    # carry the same kind of text.
    #
    # Inline rather than backgrounded because the button says "Compare" and the
    # user is waiting for an answer. Saving quietly and reporting nothing is what
    # made this feel like nothing had happened.
    comparison = None
    if config:
        try:
            saved = await asyncio.to_thread(
                price_service.enrich_observation, client, config, saved
            )
            # The enrichment is stored on the ROW but deliberately NOT used as
            # the query. MIN_SEMANTIC_SCORE was calibrated in one direction —
            # short tag phrasing against an enriched row — and enriching both
            # sides measurably makes it worse: BANANAS falls 0.772 -> 0.748,
            # under the floor, because a generic product description dilutes the
            # product identity the vector is supposed to carry.
            comparison = await asyncio.to_thread(
                price_service.compare_price,
                config, user["id"], saved.get("item_description"),
                saved.get("item_subtotal_price"),
                saved.get("item_quantity"), saved.get("item_quantity_unit"),
                saved.get("brand_name"),
                None,
                8,
                saved.get("id"),
            )
        except Exception as e:  # noqa: BLE001
            # The price is already saved. Losing the record because the
            # comparison failed would be the worse trade.
            logger.warning("Saved %s but could not compare it: %s", saved.get("id"), e)

    logger.info(
        "Price observation saved for user_id=%s: %s at %s (%d comparable purchases)",
        user["id"], saved.get("item_description"), saved.get("item_subtotal_price"),
        len(comparison["purchases"]) if comparison else 0,
    )
    # Evidence, not a verdict — the same contract the agent's tool has. The card
    # shows what was paid before and what is uncertain; it does not pronounce the
    # price good, because that depends on what the tag said.
    saved = {**saved, "comparison": comparison}
    return saved
