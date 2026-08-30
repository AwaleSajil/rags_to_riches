"""Price observations: naming, matching, and comparison.

The join this feature depends on is between two very differently written
descriptions of the same thing — a shelf tag ("CHEERIOS TSTD WHL GRN 12OZ") and
a receipt line ("Cheerios 12 oz").

Matching is **semantic only**: cosine distance over product-identity vectors,
with a measured floor (MIN_SEMANTIC_SCORE). There is no lexical tier —
TransactionDetail.normalized_name and its trigram index were dropped in
migration 023. What makes the abbreviated side reachable is `enriched_info`
being part of the embedded text; without it the same labelled pairs stop
separating at all. See scripts/calibrate_price_thresholds.py.

normalize_item_name survives for PriceObservation, which still carries a
normalized_name column.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from typing import Any, Optional

from backend.services.units import (
    UNIT_PATTERN, Size, comparable, format_unit_price, parse_size, scale_from_base,
    unit_price,
)

logger = logging.getLogger("moneyrag.services.price")

# Words that describe the *container* rather than the product, plus filler.
# Dropping these is what lets "Whole Milk 1 Gal Jug" and "WHOLE MILK" meet.
#
# Deliberately NOT dropped: anything that names a product variant with its own
# price. "organic", "large", "family" and friends look like marketing noise but
# organic milk and large eggs genuinely cost more than the conventional item —
# stripping them would quote the user a price for something they did not buy.
# Over-merging is the dangerous failure here; a miss just falls through to the
# trigram and embedding tiers.
_NOISE_WORDS = frozenset({
    # containers
    "pack", "packs", "pk", "ct", "count", "bag", "bags", "box", "boxes",
    "bottle", "bottles", "can", "cans", "jar", "jars", "jug", "carton",
    "container", "pouch", "tub", "tray", "case", "each", "ea",
    # pure filler
    "fresh", "new", "the", "a", "an", "of", "and", "with", "for",
})

# Receipt abbreviations are rampant and mostly conventional. Expanding the
# common ones before the noise filter meaningfully lifts the exact-match rate.
_ABBREVIATIONS = {
    "wht": "white", "whl": "whole", "wht.": "white",
    "chkn": "chicken", "chz": "cheese", "chse": "cheese",
    "brd": "bread", "mlk": "milk", "yog": "yogurt", "ygrt": "yogurt",
    "tstd": "toasted", "grn": "grain", "org": "organic",
    "swt": "sweet", "veg": "vegetable", "frzn": "frozen",
    "bnls": "boneless", "sknls": "skinless", "grnd": "ground",
    "choc": "chocolate", "van": "vanilla", "straw": "strawberry",
    "lg": "large", "sm": "small", "med": "medium",
}

# Size text is stripped from the name and captured separately — the size belongs
# in size_value/size_unit, not in the matching key, or a 12oz and a 20oz box of
# the same cereal would never match as "the same item".
#
# Both patterns borrow units.UNIT_PATTERN rather than restating the unit list.
# A local copy drifted once already: a loose `[a-z]+` matched only the "fl" of
# "fl oz" and left a stray "oz" in the key, so "Sprite 6 x 12 fl oz" and
# "Sprite" stopped matching.
_MULTIPACK_FRAGMENT_RE = re.compile(
    rf"\b\d+(?:[.,]\d+)?\s*(?:x|×|\*)\s*\d+(?:[.,]\d+)?\s*(?:{UNIT_PATTERN})\b",
    re.IGNORECASE,
)
_SIZE_FRAGMENT_RE = re.compile(
    rf"\b\d+(?:[.,]\d+)?\s*(?:{UNIT_PATTERN})\b", re.IGNORECASE
)


def normalize_item_name(text: Optional[str]) -> str:
    """Reduce an item description to a stable key for matching.

    Lowercases, expands common receipt abbreviations, removes size fragments and
    packaging noise, and sorts nothing — word order is preserved because
    "chicken stock" and "stock chicken" are the same item but "milk chocolate"
    and "chocolate milk" are not.
    """
    if not text:
        return ""

    value = str(text).lower()
    value = _MULTIPACK_FRAGMENT_RE.sub(" ", value)
    value = _SIZE_FRAGMENT_RE.sub(" ", value)
    # Keep letters and digits (digits matter: "2%" milk, "V8"), drop the rest.
    value = re.sub(r"[^a-z0-9%\s]", " ", value)

    words = []
    for word in value.split():
        word = _ABBREVIATIONS.get(word, word)
        if word in _NOISE_WORDS or not word:
            continue
        words.append(word)

    return " ".join(words).strip()


def observation_size(item_description: Optional[str], size_text: Optional[str]) -> Optional[Size]:
    """Best available size: an explicit size field, else parsed from the name."""
    return parse_size(size_text) or parse_size(item_description)


# --- why a price is what it is ----------------------------------------------

# A discount attached to a date this close is about clearing stock, not about
# the item's going rate.
NEAR_EXPIRY_DAYS = 3
# Grocery prices move enough that a season-old sighting is history, not a quote.
STALE_OBSERVATION_DAYS = 90


def _as_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def purchase_was_discounted(detail: dict, transaction: Optional[dict] = None) -> bool:
    """True if this past purchase was itself made on offer.

    A receipt line with `item_savings` was marked down, and a `discount_total`
    means a coupon covered the basket — so what was paid is below the item's
    ordinary price. Treating it as "what you usually pay" makes every normal
    shelf price look like a rip-off.
    """
    try:
        if float(detail.get("item_savings") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    if transaction:
        try:
            if float(transaction.get("discount_total") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def purchase_regular_unit_price(detail: dict) -> Optional[float]:
    """What one unit would have cost without the markdown.

    Receipt semantics (see transaction_service._verify_receipt_row):
    `unit_quantity_subtotal` is already the NET price paid per unit, and
    `item_savings` is the markdown for the WHOLE line — so the per-unit regular
    price adds back savings divided by quantity, not the raw savings figure.
    """
    try:
        net_unit = float(detail.get("unit_quantity_subtotal") or 0)
    except (TypeError, ValueError):
        return None
    if net_unit <= 0:
        return None

    try:
        savings = float(detail.get("item_savings") or 0)
        quantity = float(detail.get("item_quantity") or 1) or 1
    except (TypeError, ValueError):
        return net_unit
    if savings <= 0:
        return net_unit
    return round(net_unit + (savings / quantity), 2)


def purchase_caveats(detail: dict, transaction: Optional[dict] = None) -> list[dict]:
    """Why a past purchase price may not be the item's ordinary price."""
    caveats: list[dict] = []
    try:
        savings = float(detail.get("item_savings") or 0)
    except (TypeError, ValueError):
        savings = 0.0

    if savings > 0:
        regular = purchase_regular_unit_price(detail)
        detail_text = f" (normally about ${regular:.2f} each)" if regular else ""
        caveats.append({
            "code": "purchase_marked_down",
            "message": f"You bought that one on markdown, saving ${savings:.2f}{detail_text}.",
        })

    if transaction:
        try:
            order_discount = float(transaction.get("discount_total") or 0)
        except (TypeError, ValueError):
            order_discount = 0.0
        if order_discount > 0:
            # Deliberately not pro-rated across the basket: splitting a "$5 off
            # $50" coupon over line items invents a per-item price that was
            # never on any shelf. Flag it and let the number stand.
            caveats.append({
                "code": "order_coupon",
                "message": f"A ${order_discount:.2f} order coupon applied to that trip.",
            })

    return caveats


# Mean Earth radius. Haversine over a sphere is accurate to a few metres at the
# ~150m scale this is used at, so an ellipsoidal model would add dependency for
# no gain.
_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def round_coordinate(value: Optional[float]) -> Optional[float]:
    """Clamp stored precision to ~11m.

    Enough to tell one storefront from another, deliberately not enough to
    identify a dwelling. Applied before anything is persisted.
    """
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# Close enough to be the same storefront. Generous because a phone fix indoors
# drifts, and the cost of asking "which shop is this?" one extra time is far
# lower than silently attributing a price to the wrong store.
MERCHANT_MATCH_RADIUS_M = 150.0


def nearest_merchant(
    known_locations: list[dict],
    latitude: Optional[float],
    longitude: Optional[float],
    radius_m: float = MERCHANT_MATCH_RADIUS_M,
) -> Optional[dict]:
    """Closest previously-confirmed store within `radius_m`, else None.

    Runs in Python over the user's own (small) list rather than in SQL, which
    keeps PostGIS/earthdistance out of the deployment.
    """
    if latitude is None or longitude is None or not known_locations:
        return None

    best: Optional[dict] = None
    best_distance = radius_m
    for location in known_locations:
        try:
            distance = haversine_m(
                float(latitude), float(longitude),
                float(location["latitude"]), float(location["longitude"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        # Strictly-less keeps the first-seen store on an exact tie rather than
        # flapping between two rows at identical coordinates.
        if distance < best_distance:
            best, best_distance = location, distance

    if best is None:
        return None
    return {**best, "distance_m": round(best_distance, 1)}


# --- hybrid search ----------------------------------------------------------
#
# Two axes, because they fail in different places. Trigram fixes OCR mangling
# ("peanut buttr" -> "peanut butter") since that is a character problem;
# measured here against real receipts it scores 0.48 and works. It cannot fix
# receipt abbreviations ("gv ff gal" -> "Great Value Fat Free Gallon"), which is
# a meaning problem and scored 0.16 — indistinguishable from noise. Embeddings
# cover that axis.
#
# Ranking and both thresholds live in the match_price_observations SQL function
# (migration 020) so they are applied once, next to the indexes, and under the
# caller's RLS rather than depending on Python to filter by user.

# Matches the SQL functions' default. MEASURED by
# scripts/calibrate_price_thresholds.py, in the direction matching actually runs:
# a shelf tag's phrasing against a stored receipt line.
#
#   SAME pairs 0.776-0.925, DIFF pairs 0.535-0.733 — separated by 0.044.
#
# At 0.75 every true match is kept and every hard negative excluded, including
# the ones that matter: "Organic Bananas" vs BANANAS (0.733, a variant that
# costs more) and "Whole Milk 1 Gallon" vs GV LF 2 GAL (0.697).
#
# This holds ONLY because enriched_info is part of the embedded text. Without it
# the same pairs OVERLAP by 0.208 and no floor works — the best zero-false-
# positive floor misses 4 of 9 true matches, including every heavily abbreviated
# line (HS SH CLS8.5 0.639, GV LF 2 GAL 0.666). Enrichment is a correctness
# dependency of semantic-only matching, not a nice-to-have.
#
# There is no lexical floor: normalized_name was dropped, so trigram matching is
# gone and cosine is the only axis. Re-derive this number if the embedding model
# changes or the embedded text changes shape.
MIN_SEMANTIC_SCORE = 0.75


def to_vector_literal(values: Optional[list]) -> Optional[str]:
    """Render an embedding as the '[1,2,3]' text pgvector parses.

    Sent as text rather than relying on the client to encode a vector type,
    which PostgREST has no notion of.
    """
    if not values:
        return None
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def embedding_text(
    item_description: Optional[str],
    brand: Optional[str] = None,
    size_text: Optional[str] = None,
    enrichment: Optional[str] = None,
) -> str:
    """What gets embedded for an item, on either side of a price comparison.

    Brand and size are included because they are what distinguishes two
    otherwise identical names — "Great Value milk 1 gal" and "Organic Valley
    milk 12 oz" are not the same product, and a name-only vector would place
    them almost on top of each other.

    `enrichment` is the LLM description, and it carries the abbreviation
    decoding a receipt line cannot: "GV LF 2 GAL" alone reaches only 0.667
    against a shelf tag's wording, and 0.897 once "a two-gallon container of
    Great Value brand low-fat milk" is appended. That is the difference between
    a miss and a match on exactly the lines OCR mangles worst, and it is what
    carries the weight now that there is no lexical axis.

    Deliberately excludes merchant. Embedding "Line item from Walmart:" is what
    made every piece of produce land on top of every other — STRAWBERRIES scored
    0.937 against a BANANAS probe. Merchant is a column; it belongs in a WHERE
    clause, not in the meaning of the item.
    """
    parts = [
        p.strip()
        for p in (brand, item_description, size_text, enrichment)
        if p and p.strip()
    ]
    return " ".join(parts)


def build_observation_embedding(
    item_description: Optional[str],
    brand: Optional[str],
    size_text: Optional[str],
    config: Optional[dict],
    observed_context: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Embed one observation. Returns (vector_literal, model_name).

    Returns (None, None) on any failure — no config, no API key, quota
    exhausted. An observation that could not be embedded is still worth saving:
    the lexical axis keeps working, and the row can be embedded later. Losing
    the user's capture because an embedding call failed would be the worse
    trade, and they are standing in a shop.
    """
    # observed_context plays the role enriched_info plays on the receipt side:
    # it is the surrounding words that make a terse tag matchable at all.
    text = embedding_text(item_description, brand, size_text, observed_context)
    if not text or not config:
        return None, None
    try:
        from backend.services.transaction_service import _build_embeddings

        model_name = config.get("embedding_model") or "gemini-embedding-001"
        vector = _build_embeddings(config).embed_query(text)
        return to_vector_literal(vector), model_name
    except Exception as e:  # noqa: BLE001 - never block a capture on this
        logger.warning("Could not embed price observation %r: %s", text, e)
        return None, None


def search_observations(
    client,
    item_description: Optional[str],
    brand: Optional[str] = None,
    size_text: Optional[str] = None,
    config: Optional[dict] = None,
    limit: int = 10,
    min_score: float = MIN_SEMANTIC_SCORE,
) -> list[dict]:
    """Find previously seen prices for an item.

    `client` must be a Supabase client carrying the user's token: the SQL
    function is SECURITY INVOKER, so RLS is what scopes rows to the caller.
    There is deliberately no user_id argument — one that could be passed
    wrongly.
    """
    vector_literal, model_name = build_observation_embedding(
        item_description, brand, size_text, config
    )
    if not vector_literal:
        return []
    try:
        response = client.rpc(
            "match_price_observations",
            {
                # These names are the SQL function's parameters, and they have
                # been wrong: this passed a `p_query_text` that the function
                # never had, and omitted the floor entirely, so every call
                # raised. Nothing called it, which is why that went unnoticed.
                "p_query_embedding": vector_literal,
                "p_embedding_model": model_name,
                "p_limit": limit,
                "p_min_semantic": min_score,
            },
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.error("Price observation search failed for %r: %s", item_description, e)
        return []
    return response.data or []


# --- recency ----------------------------------------------------------------
#
# A price you paid two years ago is not evidence of today's price, but it is
# still the honest answer when it is the only one you have. So age is a WEIGHT
# applied when summarising, never a filter applied when matching.
#
# Half-life of 180 days: grocery prices drift enough over six months that a
# price from then should count roughly half as much as one from this week. On
# real data here, bananas went 0.50 -> 0.46 over 14 months while cilantro spanned
# 0.83 -> 2.99 within five, so the curve has to discount old evidence heavily
# without discarding it.
PRICE_HALF_LIFE_DAYS = 180

# Below this much summed weight there is not enough recent evidence to quote a
# figure. One purchase from two years ago weighs ~0.06, so it cannot on its own
# produce a confident "you usually pay X" — which is the point.
MIN_BASELINE_CONFIDENCE = 0.25


def price_weight(
    observed_on: Any,
    today: Optional[date] = None,
    half_life_days: int = PRICE_HALF_LIFE_DAYS,
) -> float:
    """How much a record from `observed_on` should count, in [0, 1].

    Exponential decay: 1.0 today, 0.5 at one half-life, 0.25 at two.

    A record with an unreadable or missing date weighs 0, which
    `weighted_baseline` treats as "leave it out". That is deliberate and the
    docstring used to claim the opposite: recency weighting is the whole
    mechanism here, and a record that cannot be placed in time cannot be
    discounted for age — including it at some invented floor would let an
    undated row shape "what you typically pay" while pretending to have been
    weighed. It is still SHOWN as evidence; it just does not vote.
    """
    when = _as_date(observed_on)
    if when is None:
        return 0.0
    reference = today or date.today()
    age_days = (reference - when).days
    if age_days <= 0:
        # A future-dated receipt is a typo, not a prophecy; treat it as current
        # rather than letting it out-weigh everything real.
        return 1.0
    return 2 ** (-age_days / float(half_life_days))


def weighted_baseline(
    records: list[dict],
    today: Optional[date] = None,
) -> Optional[dict]:
    """Summarise what an item usually costs, weighted toward recent evidence.

    `records` are dicts with a numeric `price` and a date under `observed_at` or
    `trans_date`. Callers filter out promotional and marked-down records first —
    those are shown to the user but must not shape the baseline.

    Returns a RANGE, not a point. On this data the same normalized item spans
    0.83 to 2.99, so a lone average would be a confident fiction; the spread is
    the honest part of the answer.

    Returns None when nothing usable survives, so callers say "not enough recent
    evidence" instead of quoting a number built from one ancient receipt.
    """
    priced: list[tuple[float, float]] = []
    for record in records or []:
        try:
            value = float(record.get("price"))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        weight = price_weight(
            record.get("observed_at") or record.get("trans_date"), today
        )
        if weight > 0:
            priced.append((value, weight))

    if not priced:
        return None

    confidence = sum(w for _, w in priced)
    if confidence < MIN_BASELINE_CONFIDENCE:
        return None

    # Weighted median rather than mean: one mis-OCR'd price or a bulk buy would
    # drag a mean well off the figure the user actually recognises.
    priced.sort(key=lambda pair: pair[0])
    half = confidence / 2
    running = 0.0
    median = priced[-1][0]
    for value, weight in priced:
        running += weight
        if running >= half:
            median = value
            break

    values = [value for value, _ in priced]
    return {
        # Six places, not two: these may be per-gram figures ($0.004416/g), and
        # rounding those to cents rounds them to nothing. Callers scale back to
        # the shopper's unit for display.
        "typical": round(median, 6),
        "low": round(min(values), 6),
        "high": round(max(values), 6),
        "count": len(priced),
        "confidence": round(confidence, 3),
    }


# --- storing, enriching and comparing ---------------------------------------
#
# One place owns the lifecycle of a shelf price, because two entry points reach
# it: the confirm card after a photo, and the chat agent when the user simply
# says "broccoli is $2.99 at Stop & Shop". When the card owned this alone, the
# agent had no way to record a price at all, and the router's own copy quietly
# dropped the size out of the embedded text.


def purchase_unit_size(
    quantity_unit: Optional[str],
    item_description: Optional[str],
    size_value: Any = None,
    size_unit: Optional[str] = None,
) -> Optional[Size]:
    """The size of ONE purchase unit — what a per-unit price is charged for.

    Not the same as the quantity bought, and conflating the two is a silent
    factor-of-N error. `unit_quantity_subtotal` is the price of one unit:
    bananas at 2.25 lb / $0.46 means $0.46 PER POUND, so the per-gram figure
    divides by one pound, not by 2.25 of them.

    Two cases:
      * the unit is a weight or volume ('lb', 'gal') -> one of those;
      * the unit counts things ('each', 'ct', or unrecorded) -> the price is per
        package, so the package size has to come out of the name
        ("+RED POTA 5L US#", "GV LF 2 GAL").
    """
    # A confirmed size beats any parse. "+RED POTA 5L US#" is a five POUND bag
    # and no regex can know that; a human or the vision pass reading the label
    # can, and this is where that answer lives.
    stored = size_from_quantity(size_value, size_unit, None)
    if stored is not None:
        return stored

    unit = (quantity_unit or "").strip().lower() or None
    if unit:
        one = parse_size(f"1 {unit}")
        if one is not None and one.family in ("mass", "volume"):
            return one
    # Last resort, and a guess: the size is being read out of an abbreviation.
    return parse_size(item_description)


def size_from_quantity(
    quantity: Any, unit: Optional[str], item_description: Optional[str] = None
) -> Optional[Size]:
    """A comparable Size from the quantity/unit columns, else from the name.

    Both sides of a comparison store quantity and unit apart, so this is what
    puts them back together into something units.py can convert.
    """
    if quantity is not None and unit:
        try:
            candidate = parse_size(f"{float(quantity)} {unit}")
        except (TypeError, ValueError):
            candidate = None
        if candidate:
            return candidate
    return parse_size(item_description)


def size_text_of(quantity: Any, unit: Optional[str]) -> Optional[str]:
    """"12 oz" from the two columns, for the embedded text."""
    if quantity is None or not unit:
        return None
    try:
        value = float(quantity)
    except (TypeError, ValueError):
        return None
    rendered = f"{value:g}"
    return f"{rendered} {str(unit).strip().lower()}"


def record_observation(
    client,
    config: Optional[dict],
    draft: dict,
    user_id: str,
    bill_file_id: Optional[str] = None,
) -> dict:
    """Store one confirmed shelf price and return the saved row.

    `client` must carry the user's token — RLS is what scopes the insert.

    The embedded text includes the size. It is what separates two otherwise
    identical names, and leaving it out made a 12 oz jar and a 64 oz jar embed
    identically — precisely the distinction a price comparison rests on.
    """
    description = (draft.get("item_description") or "").strip()
    if not description:
        raise ValueError("An item description is required to record a price")

    quantity = draft.get("size_value")
    unit = (draft.get("size_unit") or "").strip().lower() or None

    embedding, embedding_model = build_observation_embedding(
        description,
        draft.get("brand_name"),
        size_text_of(quantity, unit),
        config,
        observed_context=draft.get("item_qualitative_description"),
    )

    # Which tag in the photo this is. Positional rather than name-based: the
    # description is editable on the confirm card, so keying on it would move the
    # key out from under the row the moment a user corrected a misread name, and
    # the correction would insert a second observation instead of amending one.
    try:
        tag_index = int(draft.get("tag_index") or 0)
    except (TypeError, ValueError):
        tag_index = 0

    record = {
        "user_id": user_id,
        "tag_index": tag_index,
        "bill_file_id": bill_file_id or draft.get("bill_file_id"),
        "merchant_name": draft.get("merchant_name"),
        "location": draft.get("location"),
        "item_description": description,
        "size_value": quantity,
        "size_unit": unit,
        "unit_quantity_subtotal": draft.get("unit_quantity_subtotal"),
        "unit_price_unit": (draft.get("unit_price_unit") or "").strip().lower() or None,
        "item_subtotal_price": draft.get("item_subtotal_price"),
        "item_qualitative_description": draft.get("item_qualitative_description"),
        "brand_name": draft.get("brand_name"),
        "note": draft.get("note"),
        "embedding": embedding,
        "embedding_model": embedding_model,
    }

    # One photo is one sighting. Re-opening a confirmed tag and pressing Compare
    # again is not seeing the price a second time, and storing it twice makes one
    # photo read as two independent observations agreeing with each other.
    # Prices mentioned in chat carry no bill_file_id and each one IS a separate
    # sighting, so they still insert. Backed by a partial unique index
    # (migration 031) in case two confirms race.
    photo_id = record["bill_file_id"]
    if photo_id:
        updated = (
            client.table("PriceObservation").update(record)
            .eq("user_id", user_id).eq("bill_file_id", photo_id)
            .eq("tag_index", tag_index).execute()
        )
        if updated.data:
            saved = updated.data[0]
            logger.info("Updated existing observation for photo %s", photo_id)
            return saved

    inserted = client.table("PriceObservation").insert(record).execute()
    if not inserted.data:
        raise RuntimeError("Price observation insert returned no data")
    saved = inserted.data[0]
    logger.info(
        "Recorded price observation %s: %s at %s",
        saved.get("id"), description, saved.get("item_subtotal_price"),
    )
    return saved


def _message_text(content: Any) -> str:
    """Flatten an LLM message's content to plain text.

    Gemini returns a LIST of content blocks where OpenAI returns a string, so
    calling .strip() on it raised and enrichment silently gave up — which in turn
    left the observation embedded on the tag's words alone.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts).strip()
    return ""


def enrich_observation(client, config: Optional[dict], observation: dict) -> dict:
    """Research the product, store the description, and re-embed with it.

    Not cosmetic. MIN_SEMANTIC_SCORE only separates true from false matches
    because `enriched_info` is part of the embedded text on the receipt side;
    without the same treatment here the two sides carry different KINDS of text
    and the measured floor does not transfer. A tag's own words describe the
    OFFER ("2 for $5 with card"); enrichment describes the PRODUCT, which is
    what the receipt side is matched on.

    Failure is non-fatal: the row keeps its original vector.
    """
    observation_id = observation.get("id")
    description = observation.get("item_description")
    if not observation_id or not description or not config:
        return observation

    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.prompts import ChatPromptTemplate

        # "google_genai", not the config's own "Google" — langchain's provider
        # names are its own vocabulary, and passing the stored value straight
        # through raises "Unsupported provider". Same mapping as money_rag and
        # transaction_service.
        provider = (config.get("llm_provider") or "").lower()
        model = init_chat_model(
            config.get("decode_model") or "gemini-3-flash-preview",
            model_provider="google_genai" if provider == "google" else "openai",
            api_key=config.get("api_key"),
        )
        brand = observation.get("brand_name") or ""
        size = size_text_of(
            observation.get("size_value"), observation.get("size_unit")
        ) or ""
        prompt = ChatPromptTemplate.from_messages([(
            "human",
            "In one or two plain sentences, say what this grocery product IS: "
            "what kind of thing, and any variant that affects price (organic, "
            "low-fat, brand tier). Describe the PRODUCT only — say nothing about "
            "the price, the offer, or whether it is good value.\n"
            "The shop is given because tag and receipt text is often that chain's "
            "own shorthand, and it is the context that makes an abbreviation "
            "readable.\n\n"
            "Shop: {merchant}\nProduct: {brand} {description} {size}",
        )])
        enriched = (prompt | model).invoke({
            "brand": brand,
            "description": description,
            "size": size,
            "merchant": observation.get("merchant_name") or "unknown",
        })
        text_value = _message_text(getattr(enriched, "content", ""))
        if not text_value:
            return observation
    except Exception as e:  # noqa: BLE001 - enrichment is never worth failing over
        logger.warning("Could not enrich observation %s: %s", observation_id, e)
        return observation

    # Re-embed WITH the enrichment, so the vector matches what the receipt side
    # carries. Embedding at insert time cannot include this — it does not exist
    # until now — which is why this is a second write rather than one.
    embedding, embedding_model = build_observation_embedding(
        description,
        observation.get("brand_name"),
        size_text_of(observation.get("size_value"), observation.get("size_unit")),
        config,
        observed_context=text_value,
    )
    update = {"enriched_info": text_value}
    if embedding:
        update["embedding"] = embedding
        update["embedding_model"] = embedding_model
    try:
        client.table("PriceObservation").update(update).eq("id", observation_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not save enrichment for %s: %s", observation_id, e)
        return observation
    return {**observation, **update}


def compare_price(
    config: Optional[dict],
    user_id: str,
    item_description: str,
    shelf_price: Optional[float] = None,
    quantity: Any = None,
    quantity_unit: Optional[str] = None,
    brand_name: Optional[str] = None,
    enrichment: Optional[str] = None,
    limit: int = 8,
    exclude_observation_id: Optional[str] = None,
) -> dict:
    """Gather what is known about an item's price. Returns EVIDENCE, not a verdict.

    Deliberately stops short of saying "this is a good price". Whether a price is
    good depends on why it is that price — a clearance on something expiring
    Thursday, a multi-buy that requires taking three, a loyalty-card rate — and
    that reasoning belongs to the agent reading the tag's own words, not to a
    function that only sees numbers.

    What this DOES do is the part a language model does badly and inconsistently:
    retrieve above a measured similarity floor, convert both sides to a common
    base unit, and weight old evidence down.
    """
    from backend.services.transaction_service import _build_embeddings

    result: dict[str, Any] = {
        "item": item_description,
        "shelf_price": shelf_price,
        "purchases": [],
        "prior_observations": [],
        "baseline": None,
        "comparison": None,
        # The best like-for-like number available when there is no baseline —
        # set whenever ANY retrieved item could be put in the same units.
        "closest_comparable": None,
        "shelf_unit_price": None,
        "cautions": [],
    }

    shelf_size = size_from_quantity(quantity, quantity_unit, item_description)
    result["size"] = f"{shelf_size.value:g} {shelf_size.unit}" if shelf_size else None
    if shelf_size is None:
        result["cautions"].append(
            "No size for this item, so prices can only be compared package to "
            "package, not per unit."
        )

    # The shelf price per gram / mL / item — what every candidate is measured
    # against below.
    shelf_per_base = unit_price(shelf_price, shelf_size)
    result["shelf_unit_price"] = format_unit_price(shelf_price, shelf_size)

    if not config:
        result["cautions"].append("No account config, so nothing could be searched.")
        return result

    try:
        embeddings = _build_embeddings(config)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not build embeddings for comparison: %s", e)
        result["cautions"].append("Search is unavailable right now.")
        return result

    query = embedding_text(
        item_description, brand_name, size_text_of(quantity, quantity_unit), enrichment
    )

    from backend.vector_db_client import get_vector_client

    vdb = get_vector_client()
    try:
        # Two searches, each with its own budget, rather than one scope="all".
        # "all" also ranks whole TRANSACTIONS, which this function has no use for
        # — and because semantic_search merges every corpus and truncates to
        # top_k, a well-scoring "Stop & Shop (Groceries)" was taking slots away
        # from the line items the comparison is actually built from.
        # No floor on RETRIEVAL. A hard cut answered "nothing to compare" while
        # the user's own milk purchases sat at 0.737 and 0.653, which is a worse
        # answer than showing them and saying how close they are. The score
        # travels with every match instead, so the reader decides.
        #
        # The floor still governs what may be ASSERTED — see `confident` below.
        # The two jobs are different: a weak match is useful evidence to look at
        # and a terrible thing to compute an average from.
        matches = vdb.semantic_search(
            query, user_id=user_id, top_k=limit, embeddings_model=embeddings,
            scope="line_items", min_score=0.0,
        ) + vdb.semantic_search(
            query, user_id=user_id, top_k=limit, embeddings_model=embeddings,
            scope="price_observations", min_score=0.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Price comparison search failed for %r: %s", item_description, e)
        result["cautions"].append("Search failed, so there is no history to compare against.")
        return result

    baseline_records: list[dict] = []
    for match in matches:
        meta = match["metadata"]
        kind = meta.get("vector_type")
        if kind == "line_item":
            detail = {
                "item_savings": meta.get("item_savings"),
                "item_quantity": meta.get("quantity"),
                "unit_quantity_subtotal": meta.get("unit_price"),
            }
            transaction = {"discount_total": meta.get("discount_total")}
            discounted = purchase_was_discounted(detail, transaction)
            unit_size = purchase_unit_size(
                meta.get("quantity_unit"), meta.get("item_description"),
                meta.get("size_value"), meta.get("size_unit"),
            )
            size = unit_size
            paid_unit = meta.get("unit_price")
            score = meta.get("score", 0.0)
            # Above the measured floor this is safe to build a number from;
            # below it, it is something to look at and reason about.
            confident = score >= MIN_SEMANTIC_SCORE
            entry = {
                "confident": confident,
                "description": meta.get("item_description") or match["page_content"],
                "date": str(meta.get("transaction_date"))[:10],
                "merchant": meta.get("merchant_name"),
                # Carried so a comparison can say "that was a different city".
                # Prices are local: the same jug is not the same price 900 miles
                # away, and recency weighting says nothing about distance.
                "location": meta.get("location"),
                "bill_file_id": meta.get("bill_file_id"),
                "paid_per_unit": paid_unit,
                "quantity": meta.get("quantity"),
                "quantity_unit": meta.get("quantity_unit"),
                "unit_price_display": format_unit_price(paid_unit, size) if size else None,
                # The actual comparison, per candidate. Only when both sides
                # carry a size in the SAME unit family: a 2-gallon jug at $3.38
                # is $1.69/gal, which is the only figure a $3.49 gallon can
                # honestly be held against. Grams against millilitres, or a
                # package price against a per-pound price, is not a comparison.
                "vs_shelf_percent": _percent_difference(
                    shelf_per_base, unit_price(paid_unit, size)
                ) if comparable(shelf_size, size) else None,
                "was_on_offer": discounted,
                "caveats": [c["message"] for c in purchase_caveats(detail, transaction)],
                "score": round(score, 3),
            }
            result["purchases"].append(entry)
            # Three independent reasons a match is shown but excluded from the
            # baseline: bought on offer (not the ordinary price), only a loose
            # match (not necessarily the same product), or measured in units that
            # cannot be put beside the shelf item.
            #
            # That last one used to be missing, and it produced confident
            # nonsense: 1.49 LB of loose potatoes at $1.49/lb averaged with a
            # 5-lb BAG at $4.99 became "you typically pay $4.99 per unit", and a
            # $1.99/lb shelf tag was reported as 60% below it. Per-base units
            # only, so every number in the average measures the same thing.
            # Divide by ONE unit's size, never by how many were bought.
            paid_per_base = unit_price(paid_unit, unit_size)
            if (
                confident and not discounted and paid_per_base
                and comparable(shelf_size, size)
            ):
                baseline_records.append(
                    {"price": paid_per_base, "trans_date": meta.get("transaction_date")}
                )
        elif kind == "price_observation":
            # The sighting being asked about is saved before this runs, so it is
            # sitting in the corpus and matches itself perfectly. Left in, it
            # reported "0% dearer than the closest price you've seen — $1.99/lb
            # vs $1.99/lb", which is the row comparing itself to itself.
            if exclude_observation_id and str(meta.get("id")) == str(exclude_observation_id):
                continue
            # The id is only known to the caller that saved it. The agent asks
            # about the same price moments later with no id at all, and the row
            # it is asking ABOUT comes back as evidence FOR it — "you have seen
            # them at this same price earlier today", about a sighting seconds
            # old. So a sighting is also recognised by what it is.
            if _is_the_same_sighting(meta, item_description, shelf_price):
                continue
            # Compared, not just listed. A price you photographed last week at
            # another shop is directly comparable evidence for "is this a good
            # price" — often better than a loose purchase match, because it is
            # the same question asked twice. It was being retrieved and printed
            # without ever being put beside the shelf price.
            #
            # An observation's quantity/unit describe the PACKAGE, exactly like
            # the shelf side, so the size is read the same way.
            seen_size = size_from_quantity(
                meta.get("size_value"), meta.get("size_unit"),
                meta.get("description") or match["page_content"],
            )
            seen_price = meta.get("shelf_price")
            seen_per_base = unit_price(seen_price, seen_size)
            result["prior_observations"].append({
                "confident": meta.get("score", 0.0) >= MIN_SEMANTIC_SCORE,
                "description": match["page_content"],
                "seen": str(meta.get("observed_on"))[:10],
                "merchant": meta.get("merchant_name"),
                "location": meta.get("location"),
                "price": seen_price,
                "per_unit": meta.get("unit_price"),
                "quantity_unit": meta.get("quantity_unit"),
                "unit_price_display": (
                    format_unit_price(seen_price, seen_size) if seen_size else None
                ),
                "vs_shelf_percent": _percent_difference(shelf_per_base, seen_per_base)
                if comparable(shelf_size, seen_size) else None,
                "tag_says": meta.get("tag_says"),
                "score": round(meta.get("score", 0.0), 3),
            })

    result["baseline"] = weighted_baseline(baseline_records)

    # Rank whatever CAN be compared, best evidence first: a confident match
    # beats a loose one, and among equals the closer match wins. This is the
    # answer to "is this a good price" when no baseline is available — showing
    # the retrieval and stopping is not.
    # Both corpora are candidates. What the user PAID is the stronger evidence
    # and is preferred at equal confidence, but a price they merely SAW is still
    # a real like-for-like number and beats having none at all.
    comparable_rows = [
        {**p, "kind": "paid", "when": p["date"]}
        for p in result["purchases"]
        if p["vs_shelf_percent"] is not None and not p["was_on_offer"]
    ] + [
        {**o, "kind": "seen", "when": o["seen"]}
        for o in result["prior_observations"]
        if o["vs_shelf_percent"] is not None
    ]
    comparable_rows.sort(
        key=lambda p: (not p["confident"], p["kind"] != "paid", -p["score"])
    )
    if comparable_rows:
        best = comparable_rows[0]
        result["closest_comparable"] = {
            "description": best["description"],
            "date": best["when"],
            "merchant": best["merchant"],
            "their_unit_price": best["unit_price_display"],
            "location": best.get("location"),
            "percent": best["vs_shelf_percent"],
            "confident": best["confident"],
            # "paid" = a purchase of yours; "seen" = a shelf price you
            # photographed. The card and the agent must not call the second one
            # something you paid.
            "kind": best["kind"],
        }

    confident = [p for p in result["purchases"] if p["confident"]]
    if not result["purchases"]:
        result["cautions"].append(
            "You have no purchase history at all that resembles this item. Say so "
            "rather than guessing whether the price is good."
        )
    elif not confident:
        best = max(p["score"] for p in result["purchases"])
        result["cautions"].append(
            f"Nothing matched confidently - the closest thing you have bought scored "
            f"{best} against a {MIN_SEMANTIC_SCORE} bar. The items below are the "
            "nearest candidates, NOT confirmed matches: check whether they are "
            "really the same product (watch for a different variant, size or fat "
            "content) before drawing any conclusion from them."
        )
    elif result["baseline"] is None:
        result["cautions"].append(
            "Every confident match was bought on offer, or the evidence is too old "
            "to quote a typical price from."
        )

    # Only compared when both sides carry a size in the SAME unit family. Grams
    # against millilitres is not a comparison, and a package price against a
    # per-pound price is worse than no answer.
    baseline = result["baseline"]
    if baseline and shelf_per_base and baseline["typical"]:
        typical = baseline["typical"]
        unit = shelf_size.unit if shelf_size else None
        # Rendered back in the shopper's own unit: "$1.66/lb", not "$0.0037/g".
        show = lambda v: scale_from_base(v, unit)
        result["comparison"] = {
            "shelf_per_unit": round(scale_from_base(shelf_per_base, unit) or 0, 4),
            "typical_paid_per_unit": round(show(typical) or 0, 4),
            "difference": round((show(shelf_per_base) or 0) - (show(typical) or 0), 4),
            "percent": round(100 * (shelf_per_base - typical) / typical, 1),
            "range_paid": [round(show(baseline["low"]) or 0, 4),
                           round(show(baseline["high"]) or 0, 4)],
            "unit": unit,
            "based_on": baseline["count"],
            "confidence": baseline["confidence"],
        }
        if baseline["high"] > baseline["low"] * 1.5:
            result["cautions"].append(
                "What you have paid for this ranges "
                f"${result['comparison']['range_paid'][0]:.2f}–"
                f"${result['comparison']['range_paid'][1]:.2f} per "
                f"{shelf_size.unit if shelf_size else 'unit'}, so a single typical "
                "figure is weak evidence here."
            )

    return result


def _percent_difference(shelf_per_base: Optional[float], paid_per_base: Optional[float]) -> Optional[float]:
    """How much more (+) or less (-) the shelf price is, per base unit.

    None when either side has no usable per-unit figure — an absent answer is
    recoverable, a fabricated one is not.
    """
    if not shelf_per_base or not paid_per_base or paid_per_base <= 0:
        return None
    return round(100 * (shelf_per_base - paid_per_base) / paid_per_base, 1)


# Two sightings of the same item at the same price minutes apart are one
# sighting, not corroboration. Generous enough to cover a confirm followed by a
# question about it, short enough that a genuine revisit next week still counts.
SAME_SIGHTING_WINDOW_SECONDS = 15 * 60


def _is_the_same_sighting(
    meta: dict, item_description: Optional[str], shelf_price: Optional[float]
) -> bool:
    """True when a retrieved observation IS the price being asked about.

    Identity by content rather than by id, because the caller that knows the id
    is not always the caller that needs the exclusion: the confirm card passes
    it, the agent — asking about the same price seconds later — cannot.
    """
    if shelf_price is None or not item_description:
        return False
    try:
        if abs(float(meta.get("shelf_price") or 0) - float(shelf_price)) > 0.001:
            return False
    except (TypeError, ValueError):
        return False

    seen = _as_datetime(meta.get("observed_on"))
    if seen is None:
        return False
    age = (datetime.now(seen.tzinfo) - seen).total_seconds()
    if not (0 <= age <= SAME_SIGHTING_WINDOW_SECONDS):
        return False

    return normalize_item_name(meta.get("description")) == normalize_item_name(item_description)


def _as_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def describe_product(config: Optional[dict], description: str, **context) -> Optional[str]:
    """One or two sentences on what a product IS. Returns None on any failure.

    Shared by the shelf-price and receipt-line paths so both describe a product
    the same way, and so a correction can rewrite a description that a size fix
    has just made false.

    `context` is anything that helps read an abbreviation — shop, brand, size.
    The shop matters most: "GV LF 2 GAL" searched alone came back as Lemon Fresh
    BLEACH, and as Great Value low-fat milk once Walmart was named.
    """
    if not config or not description:
        return None
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.prompts import ChatPromptTemplate

        provider = (config.get("llm_provider") or "").lower()
        model = init_chat_model(
            config.get("decode_model") or "gemini-3-flash-preview",
            model_provider="google_genai" if provider == "google" else "openai",
            api_key=config.get("api_key"),
        )
        known = ", ".join(f"{k}: {v}" for k, v in context.items() if v)
        prompt = ChatPromptTemplate.from_messages([(
            "human",
            "In one or two plain sentences, say what this grocery product IS: what "
            "kind of thing, and any variant that affects price (organic, low-fat, "
            "brand tier). Say NOTHING about price or value.\n"
            "Use the known facts below and do NOT contradict them — the size given "
            "there has been confirmed, and a description that disagrees with it "
            "will be quoted back to the user as if it were true.\n"
            "A BARE NUMBER IN THE NAME IS USUALLY A VARIANT, NOT A COUNT. Milk is "
            "labelled by fat: '3' is whole (3.25%), '2' is reduced fat, '1' is low "
            "fat — 'GV LF 2 GAL' is a ONE gallon jug of 2% milk, not two gallons. "
            "The same goes for grades and percentages elsewhere. Only call "
            "something a multi-pack if the name actually says so ('2PK', 'TWIN', "
            "'CASE'). If you cannot tell, describe the product and leave the "
            "quantity out rather than inventing one.\n\n"
            "Item as printed: {description}\nKnown: {known}",
        )])
        return _message_text(
            (prompt | model).invoke({"description": description, "known": known or "nothing else"}).content
        ) or None
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not describe %r: %s", description, e)
        return None
