import asyncio
import hashlib
import logging
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

from backend.dependencies import client_for as _client, get_supabase
from backend.services import background, config_service
from backend.services.categories import normalize_category

logger = logging.getLogger("moneyrag.services.transaction")

# Fields whose value feeds the pgvector document text (merchant, category, and
# the user's note) — a change to any of them requires re-embedding the vector.
_VECTOR_TEXT_FIELDS = ("merchant_name", "category", "note")

# Of those, the ones that appear ONLY in the parent transaction's document.
#
# A line item's vector is built from the item's own identity plus its
# enrichment, and its metadata carries the parent's category — so a merchant or
# category change must re-embed the children too. A note appears in neither,
# which makes a note-only edit the case where re-embedding line items is pure
# waste: identical strings sent back to the embedding API, one call per item.
_PARENT_ONLY_VECTOR_FIELDS = frozenset({"note"})


def _vector_fields_changed(
    changes: Dict[str, Any], old: Dict[str, Any]
) -> frozenset:
    """Which document-affecting fields this edit actually alters.

    Compares values rather than trusting presence: re-saving a form without
    touching the note must not spend an embedding call to rewrite the identical
    string.
    """
    return frozenset(
        f for f in _VECTOR_TEXT_FIELDS if f in changes and changes[f] != old.get(f)
    )


def _children_need_reembed(changed_fields: frozenset) -> bool:
    """True when line-item vectors are stale too, not just the parent's."""
    return bool(changed_fields - _PARENT_ONLY_VECTOR_FIELDS)


def _note_change(review: Dict[str, Any]) -> Dict[str, Any]:
    """The note portion of a verify, as fields to write — possibly none.

    Verifying a receipt writes the same dict on insert and on re-verify, and on
    re-verify it becomes an UPDATE. So an absent note must not appear in that
    dict at all: including it as null would erase a note the user added later
    from the transaction screen, and nothing on the review form suggests it
    touches anything but the receipt.

      absent/None -> {}                 leave whatever is stored
      ""          -> {"note": None}     explicit clear
      "  x  "     -> {"note": "x"}      set, trimmed
    """
    note = review.get("note")
    if note is None:
        return {}
    return {"note": note.strip() or None}

# Columns returned for the browser list. `enriched_info` is used only as a
# lightweight indicator that Deep Enrichment completed for this receipt.
_LIST_COLUMNS = (
    "id,trans_date,description,amount,category,merchant_name,location,"
    "subtotal,tax_total,tax_breakdown,discount_total,savings_total,"
    "source,source_csv_id,source_bill_file_id,"
    "note,enriched_info,created_at"
)


def _merchant_match_key(transaction: Dict[str, Any]) -> str:
    """Return a stable comparison key for bank and receipt merchant names."""
    value = str(transaction.get("merchant_name") or transaction.get("description") or "").lower()
    return re.sub(r"[^a-z]", "", value)


def _link_verified_bill_transaction(user: dict, bill_transaction_id: str) -> int:
    """Link a newly verified receipt to matching CSV transactions.

    CSV ingestion already creates these links when the CSV arrives last.  This
    handles the reverse order: a bank statement was imported first and its
    supporting receipt is verified later.  It deliberately creates links only
    (never removes a source transaction) and retains the same conservative
    merchant/date/amount rules used for CSV ingestion.
    """
    client = _client(user)
    bill_rows = (
        client.table("Transaction")
        .select("id,trans_date,amount,merchant_name,description,source")
        .eq("id", bill_transaction_id)
        .eq("user_id", user["id"])
        .eq("source", "bill")
        .limit(1)
        .execute().data or []
    )
    if not bill_rows:
        return 0
    bill = bill_rows[0]
    bill_merchant = _merchant_match_key(bill)
    if len(bill_merchant) < 4:
        return 0
    try:
        bill_date = date.fromisoformat(str(bill["trans_date"])[:10])
        bill_amount = float(bill["amount"])
    except (TypeError, ValueError):
        return 0

    candidates = (
        client.table("Transaction")
        .select("id,trans_date,amount,merchant_name,description,source")
        .eq("user_id", user["id"])
        .eq("source", "csv")
        .execute().data or []
    )
    links = []
    for candidate in candidates:
        candidate_merchant = _merchant_match_key(candidate)
        if (
            len(candidate_merchant) < 4
            or (bill_merchant not in candidate_merchant and candidate_merchant not in bill_merchant)
        ):
            continue
        try:
            date_gap = abs((bill_date - date.fromisoformat(str(candidate["trans_date"])[:10])).days)
            amount_gap = abs(bill_amount - float(candidate["amount"]))
        except (TypeError, ValueError):
            continue
        if date_gap > 1 or amount_gap > 0.10:
            continue
        left_id, right_id = sorted((str(bill["id"]), str(candidate["id"])))
        links.append({
            "user_id": user["id"],
            "transaction_id": left_id,
            "linked_transaction_id": right_id,
            "match_type": "csv_receipt",
            "confidence": 1,
        })
    if links:
        client.table("TransactionLink").upsert(
            links, on_conflict="user_id,transaction_id,linked_transaction_id"
        ).execute()
    return len(links)


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
        transactions = res.data or []
        # A file can be hidden without deleting it or its source records.
        hidden_csv_ids = {
            str(row["id"]) for row in (
                _client(user).table("CSVFile").select("id").eq("user_id", user["id"])
                .eq("is_hidden", True).execute().data or []
            )
        }
        hidden_bill_ids = {
            str(row["id"]) for row in (
                _client(user).table("BillFile").select("id").eq("user_id", user["id"])
                .eq("is_hidden", True).execute().data or []
            )
        }
        transactions = [
            transaction for transaction in transactions
            if str(transaction.get("source_csv_id") or "") not in hidden_csv_ids
            and str(transaction.get("source_bill_file_id") or "") not in hidden_bill_ids
        ]
        if not transactions:
            return transactions
        # Return link metadata so the UI can represent one real-world purchase
        # once while preserving all of its underlying CSV/receipt records.
        try:
            links = (
                _client(user).table("TransactionLink")
                .select("transaction_id,linked_transaction_id")
                .eq("user_id", user["id"]).execute().data or []
            )
            linked_by_id: Dict[str, List[str]] = {}
            for link in links:
                left, right = str(link["transaction_id"]), str(link["linked_transaction_id"])
                linked_by_id.setdefault(left, []).append(right)
                linked_by_id.setdefault(right, []).append(left)
            for transaction in transactions:
                transaction["linked_transaction_ids"] = linked_by_id.get(str(transaction["id"]), [])
        except Exception as e:
            logger.warning("Could not load transaction links: %s", e)
            for transaction in transactions:
                transaction["linked_transaction_ids"] = []
        return transactions

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
        tx["linked_transactions"] = _linked_transactions_sync(client, user, transaction_id)
        return tx

    return await asyncio.to_thread(_run)


def _linked_transactions_sync(client, user: dict, transaction_id: str) -> List[Dict[str, Any]]:
    """The other records of this same real-world purchase.

    A purchase can arrive twice — once off a bank statement and once as a
    photographed receipt — and the two are matched on merchant, date and amount
    rather than merged, because neither is safe to throw away: the statement is
    what the bank says, the receipt is what was actually bought.

    The list screen already collapses a linked pair to one row, silently. Here
    it is said out loud, with enough of the other record to be recognised —
    otherwise a user sees one transaction where they know there were two, and
    has no way to tell whether the other was absorbed or lost.
    """
    links = (
        client.table("TransactionLink")
        .select("transaction_id,linked_transaction_id,match_type")
        .eq("user_id", user["id"])
        .or_(
            f"transaction_id.eq.{transaction_id},"
            f"linked_transaction_id.eq.{transaction_id}"
        )
        .execute().data or []
    )
    # The link is undirected — this row may be on either side of it.
    other_ids = {
        str(link["linked_transaction_id"])
        if str(link["transaction_id"]) == str(transaction_id)
        else str(link["transaction_id"])
        for link in links
    }
    other_ids.discard(str(transaction_id))
    if not other_ids:
        return []

    match_types = {
        (
            str(link["linked_transaction_id"])
            if str(link["transaction_id"]) == str(transaction_id)
            else str(link["transaction_id"])
        ): link.get("match_type")
        for link in links
    }

    rows = (
        client.table("Transaction")
        .select("id,trans_date,amount,merchant_name,description,source,source_csv_id")
        .eq("user_id", user["id"])
        .in_("id", list(other_ids))
        .execute().data or []
    )
    if not rows:
        return []

    # Whether the other side has line items decides which of the two is worth
    # reading, and the UI says so rather than making the user go and find out.
    counts = (
        client.table("TransactionDetail")
        .select("transaction_id")
        .eq("user_id", user["id"])
        .in_("transaction_id", [str(r["id"]) for r in rows])
        .execute().data or []
    )
    detail_counts: Dict[str, int] = {}
    for row in counts:
        key = str(row["transaction_id"])
        detail_counts[key] = detail_counts.get(key, 0) + 1

    return [
        {
            "id": str(row["id"]),
            "trans_date": row.get("trans_date"),
            "amount": row.get("amount"),
            "merchant_name": row.get("merchant_name") or row.get("description"),
            "source": row.get("source"),
            "match_type": match_types.get(str(row["id"])),
            "detail_count": detail_counts.get(str(row["id"]), 0),
        }
        for row in rows
    ]


async def get_receipt_review(user: dict, file_id: str) -> Optional[Dict[str, Any]]:
    """Return a user's persisted OCR draft without exposing any other receipt."""

    def _run():
        res = (
            _client(user)
            .table("BillFile")
            .select("id,filename,raw_ocr_string")
            .eq("id", file_id)
            .eq("user_id", user["id"])
            .limit(1)
            .execute()
        )
        if not res.data or not res.data[0].get("raw_ocr_string"):
            return None
        import json
        try:
            extracted = json.loads(res.data[0]["raw_ocr_string"])
        except (TypeError, ValueError):
            return None
        return {"file_id": file_id, "filename": res.data[0]["filename"], "extracted": extracted}

    return await asyncio.to_thread(_run)


def merchant_key(merchant: Optional[str]) -> str:
    """A merchant name reduced to what OCR cannot vary.

    Every non-alphanumeric character goes, over the WHOLE name. This used to
    take `split()[0]` — the first word only — so the same shop read once as
    "STOP & SHOP" and once as "STOP&SHOP" produced 'stop' and 'stopshop', and a
    receipt uploaded twice was not recognised as the same receipt. Whitespace is
    exactly the sort of thing a vision model varies between runs, and it must
    not be load-bearing.
    """
    return "".join(c for c in (merchant or "").lower() if c.isalnum())


def receipt_content_hash(
    date_value: Any, receipt_time: str, total: float, merchant: Optional[str]
) -> str:
    """Identify a receipt by the things printed clearly enough to read twice.

    Date, time, total and merchant. All four came back identical across two OCR
    passes of the same receipt; the line-item descriptions did not, and used to
    be part of this. Abbreviations are where the vision model wanders —
    "SL LGHT FLUID32Z" one run, "SL LIGHT FLUID" the next — so a single
    character across fourteen lines was enough to defeat the match and write a
    second transaction for a purchase that happened once.

    Dropping them costs almost no discrimination: two different purchases at the
    same shop, in the same MINUTE, for the same total is not a case worth
    protecting against, and the time is what makes that true.
    """
    return hashlib.sha256(
        f"{date_value}{receipt_time}{total}{merchant_key(merchant)}".encode()
    ).hexdigest()


def _verify_receipt_row(user: dict, file_id: str, review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Commit a reviewed OCR draft and its line items in one blocking DB task."""
    import json
    client = _client(user)
    bill = (
        client.table("BillFile").select("id,filename").eq("id", file_id)
        .eq("user_id", user["id"]).limit(1).execute()
    )
    if not bill.data:
        return None

    merchant = review["merchant_name"].strip()
    date_value = review["date"]
    receipt_time = (review.get("time") or "").strip()
    # Rename only the display filename — the object-storage key remains stable.
    # This makes later review easy to find when the OCR merchant/date was fixed.
    import re
    original_filename = bill.data[0].get("filename") or "receipt.jpg"
    extension = os.path.splitext(original_filename)[1] or ".jpg"
    merchant_slug = re.sub(r"[^A-Za-z0-9]+", "_", merchant).strip("_") or "receipt"
    date_slug = str(date_value).replace("-", "") or "nodate"
    filename = f"{merchant_slug}_{date_slug}{extension}"

    # Preserve the user's corrections so opening the receipt again shows the
    # latest review data rather than the original OCR response.
    client.table("BillFile").update({
        "raw_ocr_string": json.dumps(review),
        "filename": filename,
    }).eq(
        "id", file_id
    ).eq("user_id", user["id"]).execute()

    items = review.get("line_items") or []
    details = []
    subtotal = 0.0
    tax_total = 0.0
    item_savings_total = 0.0
    for item in items:
        raw_quantity = item.get("item_quantity")
        quantity = 1.0 if raw_quantity is None else float(raw_quantity)
        # unit_price is the NET price actually paid per unit (item markdowns are
        # already applied), so the line totals already sum to the balance.
        unit_price = float(item.get("item_unit_price") or 0)
        item_savings = round(float(item.get("item_savings") or 0), 2)
        tax_rate = float(item.get("tax_rate") or 0)
        item_subtotal = round(quantity * unit_price, 2)
        tax_amount = round(item_subtotal * tax_rate / 100, 2)
        subtotal += item_subtotal
        tax_total += tax_amount
        item_savings_total += item_savings
        description = item.get("item_description", "")
        # Size is deliberately NOT parsed into columns. It is already in
        # item_description ("WW SPAG 16OZ"), so storing a parse added nothing but
        # a place for it to be wrong — measured at ~26% on real rows, including
        # "5L" on a bag of potatoes read as litres and a product code read as a
        # 30-count pack. Retrieval is semantic and comparison is the agent's job,
        # so it reads the size from the text and a parser fix heals history
        # instead of leaving frozen mistakes behind.
        #
        # What item_quantity is measured in, when the model could read it. A
        # bare "2.25" is ambiguous between 2.25 lb of loose bananas and 2.25
        # packages, and the two imply different prices per unit.
        quantity_unit = (item.get("item_quantity_unit") or "").strip().lower() or None
        details.append({
            "user_id": user["id"], "bill_file_id": file_id,
            "item_description": description,
            "item_quantity": quantity,
            "item_quantity_unit": quantity_unit,
            # How much is in ONE of them. Separate from item_quantity: a 5 lb bag
            # is quantity 1 "each", size 5 "lb", and squeezing both into one
            # column is what made a bag of potatoes read as five litres.
            "size_value": item.get("size_value"),
            "size_unit": (item.get("size_unit") or "").strip().lower() or None,
            "unit_quantity_subtotal": unit_price,
            "item_subtotal_price": item_subtotal, "item_savings": item_savings,
            "tax_amount": tax_amount,
            "tax_rate": tax_rate, "taxable": tax_rate > 0,
            "item_total_price": round(item_subtotal + tax_amount, 2),
        })

    # Order-level coupons (e.g. "$5 off $50") reduce the whole basket and are the
    # only savings that get subtracted; item markdowns are already netted out above.
    discount_total = round(float(review.get("discount_total") or 0), 2)
    # A discount can't exceed the basket; clamp so a malformed payload (coupon
    # larger than the items, with no explicit total) can't store a negative amount.
    computed_total = round(max(subtotal + tax_total - discount_total, 0.0), 2)
    # A receipt can still include fees/rounding the line items can't capture, so
    # trust the OCR/user total when supplied; fall back to the computed one.
    total = round(float(review.get("total_amount") or computed_total), 2)
    # Total the shopper saved: per-item markdowns plus any order coupon. Display
    # only — never re-derives the total.
    savings_total = round(item_savings_total + discount_total, 2)
    content_hash = receipt_content_hash(date_value, receipt_time, total, merchant)

    existing_for_bill = (
        client.table("Transaction").select("id").eq("user_id", user["id"])
        .eq("source_bill_file_id", file_id).limit(1).execute()
    )
    existing = (
        client.table("Transaction").select("id").eq("user_id", user["id"])
        .eq("content_hash", content_hash).limit(1).execute()
    )
    tax_breakdown = []
    for rate in sorted({d["tax_rate"] for d in details if d["tax_rate"] > 0}):
        amount = round(sum(d["tax_amount"] for d in details if d["tax_rate"] == rate), 2)
        tax_breakdown.append({"label": "Tax", "rate": rate, "amount": amount})
    tx = {
        "user_id": user["id"], "trans_date": date_value, "amount": total,
        "description": merchant, "merchant_name": merchant,
        "category": normalize_category(review.get("category")),
        "location": review.get("location") or None, "subtotal": round(subtotal, 2),
        "tax_total": round(tax_total, 2), "tax_breakdown": tax_breakdown or None,
        "discount_total": discount_total or None, "savings_total": savings_total or None,
        "content_hash": content_hash, "source": "bill", "source_bill_file_id": file_id,
    }
    tx.update(_note_change(review))
    if existing_for_bill.data:
        # Re-verification edits the transaction already linked to this receipt.
        tx_id = existing_for_bill.data[0]["id"]
        client.table("Transaction").update(tx).eq("id", tx_id).eq("user_id", user["id"]).execute()
        client.table("TransactionDetail").delete().eq("transaction_id", tx_id).eq("user_id", user["id"]).execute()
    elif existing.data:
        # A separate re-upload of an identical receipt stays deduplicated — and
        # now says so. Returning the existing row bare was indistinguishable
        # from having saved this upload: the caller got a transaction back and
        # reasonably concluded the second copy had been recorded, when in fact
        # nothing was written and the new photo is left with no transaction
        # pointing at it.
        return {"id": existing.data[0]["id"], "duplicate_of": existing.data[0]["id"]}
    else:
        inserted = client.table("Transaction").insert(tx).execute().data or []
        if not inserted:
            raise RuntimeError("Could not save the verified receipt")
        tx_id = inserted[0]["id"]
    if details:
        for detail in details:
            detail["transaction_id"] = tx_id
        client.table("TransactionDetail").insert(details).execute()
    return {"id": tx_id}


async def verify_receipt(user: dict, file_id: str, review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    result = await asyncio.to_thread(_verify_receipt_row, user, file_id, review)
    if not result:
        return None
    try:
        # A receipt may be verified after its bank CSV was already imported.
        # Reconcile in that direction as well as during CSV ingestion.
        await asyncio.to_thread(_link_verified_bill_transaction, user, result["id"])
    except Exception as e:
        # A link is supplementary: never make a successfully reviewed receipt
        # fail to save because reconciliation was temporarily unavailable.
        logger.warning("Could not link verified receipt %s: %s", file_id, e)
    tx = await get_transaction(user, result["id"])
    if not tx:
        return None
    # Carried to the client so it can say "already recorded" rather than
    # presenting someone else's earlier transaction as the thing just saved.
    tx["is_duplicate"] = bool(result.get("duplicate_of"))
    if tx["is_duplicate"]:
        logger.info(
            "Receipt %s matched existing transaction %s — nothing written",
            file_id, tx["id"],
        )
    try:
        config = await config_service.get_config(user)
        if config and config.get("deep_enrichment"):
            # The user has already verified the OCR fields. Enrichment is an
            # automatic follow-up and never requires a second confirmation.
            background.spawn(
                _deep_enrich_verified_receipt(user, tx["id"], config),
                name=f"enrich:{tx['id']}",
            )
        elif config:
            # This is the first point at which a reviewed receipt enters search.
            await asyncio.to_thread(_reembed_transaction, tx, tx["details"], user["id"], config)
    except Exception as e:
        logger.warning("Receipt %s was saved but vector indexing failed: %s", file_id, e)
    return tx


def _build_embeddings(config: dict):
    """Construct the user's embedding model from their AccountConfig.

    The key goes to the constructor, not os.environ — this process handles every
    user's requests, so a global would let the most recent caller's key be used
    to embed someone else's data.
    """
    provider = (config.get("llm_provider") or "").lower()
    api_key = config.get("api_key")
    model = config.get("embedding_model")
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model or "gemini-embedding-001", google_api_key=api_key
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model or "text-embedding-3-small", openai_api_key=api_key
    )


def _save_enrichment_sync(
    user: dict, transaction_id: str, merchant_info: str, item_info: List[str]
) -> None:
    """Persist enrichment text on the verified receipt and its line items."""
    client = _client(user)
    client.table("Transaction").update({"enriched_info": merchant_info}).eq(
        "id", transaction_id
    ).eq("user_id", user["id"]).execute()
    details = (
        client.table("TransactionDetail").select("id").eq("transaction_id", transaction_id)
        .eq("user_id", user["id"]).order("created_at").execute().data or []
    )
    for detail, info in zip(details, item_info):
        client.table("TransactionDetail").update({"enriched_info": info}).eq(
            "id", detail["id"]
        ).eq("user_id", user["id"]).execute()


async def _deep_enrich_verified_receipt(user: dict, transaction_id: str, config: dict) -> None:
    """Enrich a reviewed receipt asynchronously, then re-index the result."""
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.tools import DuckDuckGoSearchRun

        tx = await get_transaction(user, transaction_id)
        if not tx:
            return
        merchant = tx.get("merchant_name") or tx.get("description") or "Merchant"
        names = [merchant] + [
            detail.get("item_description") or "Item" for detail in tx.get("details", [])
        ]
        search_tool = DuckDuckGoSearchRun()
        # Search the merchant + each line item concurrently instead of one at a
        # time (a 10-item receipt was ~10 serial round-trips, taking up to a
        # minute). A small semaphore keeps us well under DuckDuckGo's rate limit.
        # Names repeat across line items, so dedup before searching.
        unique_names = list(dict.fromkeys(names))
        search_semaphore = asyncio.Semaphore(4)

        async def _search_name(name: str):
            # The MERCHANT goes in the query with the item. A receipt line is a
            # private abbreviation of one chain's own catalogue — "GV LF 2 GAL"
            # searched alone is close to meaningless, while "Walmart GV LF 2 GAL"
            # resolves GV to Great Value. Enrichment feeds the embedded text, so
            # a vague answer here weakens every later product match.
            query = (
                f"What type of business is '{name}'?"
                if name == merchant
                else f"What product is '{name}' sold at {merchant}?"
            )
            async with search_semaphore:
                try:
                    return name, await search_tool.ainvoke(query)
                except Exception:
                    return name, ""

        search_results = dict(
            await asyncio.gather(*(_search_name(name) for name in unique_names))
        )
        search_context = "\n".join(
            f"- {name}: {result[:200]}" for name, result in search_results.items() if result
        )
        provider = (config.get("llm_provider") or "").lower()
        provider_name = "google_genai" if provider == "google" else "openai"
        llm = init_chat_model(
            config.get("decode_model") or "gemini-3-flash-preview",
            model_provider=provider_name,
            api_key=config["api_key"],
        )
        prompt = ChatPromptTemplate.from_template("""
You are a financial data assistant. Give a concise, helpful one-sentence description
for the merchant followed by each receipt line item. Use the research context when useful.

Every item below was bought at {merchant}, so read the abbreviations as that
chain's own: "GV" on a Walmart receipt is Great Value, its store brand. Say what
the product IS and any variant that affects price (organic, low-fat, brand tier).
A number beside a unit is often a variant rather than a size — "LF 2" is low-fat
2% milk, not two of something.

Items: {names_json}
Research context:
{search_context}

Return ONLY a JSON array in the same order:
[
  {{"enriched_info": "one sentence"}}
]
""")
        enriched = await (prompt | llm | JsonOutputParser()).ainvoke({
            "names_json": __import__("json").dumps(names),
            "merchant": merchant,
            "search_context": search_context or "No additional research available.",
        })
        if not isinstance(enriched, list):
            raise ValueError("Enrichment response was not a list")
        descriptions = [
            str(entry.get("enriched_info") or "") if isinstance(entry, dict) else ""
            for entry in enriched
        ]
        await asyncio.to_thread(
            _save_enrichment_sync, user, transaction_id,
            descriptions[0] if descriptions else "", descriptions[1:],
        )
        updated = await get_transaction(user, transaction_id)
        if updated:
            await asyncio.to_thread(
                _reembed_transaction, updated, updated["details"], user["id"], config
            )
        logger.info("Deep enrichment complete for verified receipt %s", transaction_id)
    except Exception as e:
        # The verified transaction is already saved; enrichment is best-effort.
        logger.warning("Deep enrichment failed for receipt %s: %s", transaction_id, e, exc_info=True)


def _reembed_transaction(tx: Dict[str, Any], details: List[Dict[str, Any]], user_id: str, config: dict) -> None:
    """Re-embed one transaction's vector(s) so semantic search reflects the edit.

    Pass an empty `details` to re-embed only the parent document — correct when
    the change cannot appear in a line item's text or metadata.
    """
    from backend.vector_db_client import get_vector_client

    embeddings = _build_embeddings(config)
    get_vector_client().sync_single_transaction(tx, details, user_id, embeddings)


def _apply_update(user: dict, transaction_id: str, changes: Dict[str, Any]):
    """Blocking: verify ownership, update, and re-read row + details.

    Returns (updated_tx, details, changed_vector_fields) or None if the row does
    not exist / is not the user's. The caller uses the changed-field set to
    decide how much to re-embed, not merely whether to.
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
    changed_vector_fields = _vector_fields_changed(changes, old)
    return updated, details, changed_vector_fields


async def update_transaction(
    user: dict, transaction_id: str, changes: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Update editable fields; re-embed the vector if merchant/category changed.

    A re-embed failure (e.g. LLM quota) is logged but does not fail the edit —
    the DB is the source of truth and a later full sync repairs the vector.
    """
    if "category" in changes:
        changes["category"] = normalize_category(changes["category"])
    result = await asyncio.to_thread(_apply_update, user, transaction_id, changes)
    if result is None:
        return None
    updated, details, changed_vector_fields = result

    if changed_vector_fields:
        config = await config_service.get_config(user)
        if config:
            # Editing a note used to re-embed every line item on the receipt as
            # well — up to 29 here — each producing a byte-identical string,
            # synchronously, against the embedding quota. Children are only
            # re-sent when something they actually contain has changed.
            reembed_details = (
                details if _children_need_reembed(changed_vector_fields) else []
            )
            try:
                await asyncio.to_thread(
                    _reembed_transaction, updated, reembed_details, user["id"], config
                )
                logger.info(
                    "Re-embedded transaction %s (%d document(s); changed: %s)",
                    transaction_id,
                    len(reembed_details) + 1,
                    ", ".join(sorted(changed_vector_fields)),
                )
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
        unit = d.get("unit_quantity_subtotal")
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
            # Carried through rather than dropped: the rows are rewritten wholesale
            # on every edit, so omitting this here wiped the unit off every line
            # the moment a user touched an unrelated field.
            "item_quantity_unit": (d.get("item_quantity_unit") or "").strip().lower() or None,
            "size_value": d.get("size_value"),
            "size_unit": (d.get("size_unit") or "").strip().lower() or None,
            "unit_quantity_subtotal": unit,
            "item_subtotal_price": subtotal,
            "item_savings": d.get("item_savings"),
            "tax_amount": tax_amount,
            "tax_rate": rate,
            "taxable": taxable,
            "item_total_price": total,
            "enriched_info": d.get("enriched_info"),
        })
    return rows


def _header_totals_from_details(
    details: List[Dict[str, Any]],
    existing_breakdown,
    discount_total: float = 0.0,
    previous: Optional[Dict[str, Any]] = None,
):
    """Roll line items up into header subtotal / tax_total / amount / breakdown /
    savings_total, so the transaction stays consistent with its items. An
    order-level discount (unchanged by item edits) is subtracted from amount and
    folded into the savings shown to the user.

    `previous` is the transaction as it stands before the edit. When given, the
    part of its amount the line items never explained — a bag fee, a deposit,
    a rounding adjustment, anything printed on the receipt but not itemised —
    is carried across.

    Verification deliberately trusts the receipt's own printed total over the
    computed one, so that gap is normal and meaningful. This function did not
    know about it, and re-derived the amount from scratch: editing one item's
    spelling silently moved a verified transaction off the figure the receipt
    actually shows. The edit's own effect still applies in full; only the
    unexplained remainder survives it.
    """
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    discount_total = round(_f(discount_total), 2)
    subtotal = round(sum(_f(d.get("item_subtotal_price")) for d in details), 2)
    tax_total = round(sum(_f(d.get("tax_amount")) for d in details), 2)
    amount = round(subtotal + tax_total - discount_total, 2)

    if previous:
        # What the old items added up to, versus what the transaction actually
        # said. Zero for a transaction whose items already explained it, which
        # is the common case and leaves the behaviour exactly as it was.
        previously_computed = round(
            _f(previous.get("subtotal"))
            + _f(previous.get("tax_total"))
            - _f(previous.get("discount_total")),
            2,
        )
        remainder = round(_f(previous.get("amount")) - previously_computed, 2)
        if remainder:
            logger.info(
                "Carrying %.2f of unitemised total across a line-item edit", remainder
            )
            amount = round(amount + remainder, 2)

    savings_total = round(
        sum(_f(d.get("item_savings")) for d in details) + discount_total, 2
    )

    # Group tax by rate; preserve the receipt's rate labels where we can.
    label_by_rate: Dict[float, Any] = {}
    for e in (existing_breakdown or []):
        try:
            label_by_rate[float(e.get("rate"))] = e.get("label")
        except (TypeError, ValueError):
            pass
    by_rate: Dict[float, float] = {}
    for d in details:
        rate = d.get("tax_rate")
        tax = _f(d.get("tax_amount"))
        if rate and float(rate) > 0 and tax:
            r = float(rate)
            by_rate[r] = round(by_rate.get(r, 0.0) + tax, 2)
    breakdown = [
        {"label": label_by_rate.get(r, f"{r}% tax"), "rate": r, "amount": amt}
        for r, amt in sorted(by_rate.items())
    ] or None

    return subtotal, tax_total, amount, breakdown, savings_total


def _replace_details_rows(
    user: dict, transaction_id: str, details_input: List[Dict[str, Any]]
):
    """Blocking: verify ownership, swap line items, keep the header totals in
    sync, return (tx_row, new_details) or None if not found / not the user's."""
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

    # Keep header subtotal / tax_total / amount solid with the line items. The
    # order-level discount is a header property the item edits don't touch, so
    # carry it through the rollup rather than dropping it (which would inflate
    # amount back past the balance).
    if details:
        subtotal, tax_total, amount, breakdown, savings_total = _header_totals_from_details(
            details,
            tx_row.get("tax_breakdown"),
            tx_row.get("discount_total"),
            # So a fee or rounding the items never covered is not lost to an
            # unrelated edit — see the note on _header_totals_from_details.
            previous=tx_row,
        )
        header_update = {
            "subtotal": subtotal,
            "tax_total": tax_total,
            "amount": amount,
            "tax_breakdown": breakdown,
            "savings_total": savings_total or None,
        }
        upd = (
            client.table("Transaction")
            .update(header_update)
            .eq("id", transaction_id)
            .eq("user_id", user["id"])
            .execute()
        )
        tx_row = upd.data[0] if upd.data else {**tx_row, **header_update}

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
