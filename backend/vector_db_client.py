"""Vector storage and search, backed by embedding columns on the rows themselves.

Vectors live on `Transaction.embedding` and `TransactionDetail.embedding` rather
than in a separate store. That placement is the design:

  * deleting a row takes its vector with it, so the orphaned-vector class of bug
    is structurally impossible. The previous store accumulated 49 stale
    line-item vectors, which still matched searches and let the agent quote
    line items that had been deleted during a receipt re-review;
  * the tables' own RLS policies apply, instead of tenancy resting on a
    `cmetadata->>'user_id'` filter that every query had to remember on a
    connection that bypasses RLS;
  * re-embedding is an UPDATE on a known row rather than an upsert keyed by a
    naming convention.

Two different texts get embedded, for two different jobs:

  Transaction        "{merchant} ({category}) — {enrichment} — Note: {note}"
                     Chat retrieval: "how much did I spend on fast food?" needs
                     merchant and category in the vector.

  TransactionDetail  "{item_description} {enrichment}"
                     Product matching: "have I bought this before?". Merchant is
                     deliberately absent — embedding "Line item from Walmart:"
                     made every piece of produce land on top of every other
                     (STRAWBERRIES scored 0.937 against a BANANAS probe).
                     Enrichment carries the abbreviation decoding a receipt line
                     cannot ("GV LF 2 GAL" -> "a two-gallon container of Great
                     Value low-fat milk"), which is what makes matching work at
                     all — see price_service.MIN_SEMANTIC_SCORE.

Ranking lives in SQL (`match_transactions`, `match_purchase_history`), so the
floors are applied once, next to the data.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_core.embeddings import Embeddings

from backend.config import get_settings
from backend.services.price_service import embedding_text, to_vector_literal

logger = logging.getLogger("moneyrag.vector_db_client")


def _to_psycopg_url(database_url: str) -> str:
    """Normalize a Postgres URL to the psycopg (v3) SQLAlchemy driver."""
    if database_url.startswith("postgresql+"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


_engine: Optional[Engine] = None


def _get_engine() -> Engine:
    """Lazily build a shared SQLAlchemy engine from DATABASE_URL."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            _to_psycopg_url(settings.DATABASE_URL),
            pool_pre_ping=True,
            # pgvector lives in `extensions` on Supabase; the app's tables are in
            # `public`. Both must resolve for `<=>` and ::vector casts to work.
            connect_args={"options": "-csearch_path=public,extensions"},
        )
    return _engine


def transaction_document(row: Dict[str, Any]) -> str:
    """The text embedded for a transaction — merchant, category, context, note.

    Kept as one function because it is used at write time AND to render
    `page_content` at read time. Reconstructing it separately in the search path
    would let the two drift, so a result could describe a transaction in words
    that were never embedded.
    """
    def clean(value: Any) -> str:
        # pandas turns absent values into NaN, which str() would render as the
        # literal "nan" and embed as if it were content.
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    merchant = clean(row.get("merchant_name")) or clean(row.get("description"))
    category = clean(row.get("category")) or "Uncategorized"
    parts = [f"{merchant} ({category})"]

    enrichment = clean(row.get("enriched_info"))
    if enrichment:
        parts.append(enrichment)

    # Prefixed with "Note:" so the model can tell the user's own words from the
    # generated description. Handled separately rather than in a loop — telling
    # the two fields apart by identity broke when both held the same string.
    note = clean(row.get("note"))
    if note:
        parts.append(f"Note: {note}")

    return " — ".join(parts)


def detail_document(row: Dict[str, Any]) -> str:
    """The text embedded for a line item — product identity only, no merchant."""
    enrichment = row.get("enriched_info")
    if enrichment is not None and isinstance(enrichment, float) and pd.isna(enrichment):
        enrichment = None
    return embedding_text(row.get("item_description"), None, None, enrichment)


def observation_document(row: Dict[str, Any]) -> str:
    """The text embedded for a shelf price — must match what prices.py wrote.

    The tag's own words play the part enrichment plays on the receipt side: a
    bare "Broccoli" is close to unmatchable, and the surrounding text is what
    makes it findable.
    """
    return embedding_text(
        row.get("item_description"),
        row.get("brand_name"),
        None,
        row.get("item_qualitative_description"),
    )


# What a search may look in. Three separate corpora because they answer
# different questions and mixing them silently is how a shelf price ends up
# quoted as something the user bought.
SEARCH_SCOPES = ("all", "transactions", "line_items", "price_observations")


class VectorDBClient:
    """Vector operations over the embedding columns on Transaction/TransactionDetail."""

    def __init__(self):
        self.settings = get_settings()

    # ── writing ─────────────────────────────────────────────────────────────

    def sync_transactions(
        self,
        df: pd.DataFrame,
        details_df: pd.DataFrame,
        user_id: str,
        embeddings_model: Embeddings,
        progress_callback=None,
    ) -> Optional[int]:
        """Embed transactions and their line items, writing onto their rows.

        Returns the number of vectors written. Passing an empty `details_df`
        embeds only the parents — correct when the change cannot affect a line
        item's own text.
        """
        jobs: List[tuple] = []

        for _, row in df.iterrows():
            document = transaction_document(row.to_dict())
            if document.strip():
                jobs.append(("Transaction", str(row["id"]), document))

        if not details_df.empty:
            for _, row in details_df.iterrows():
                document = detail_document(row.to_dict())
                if document.strip():
                    jobs.append(("TransactionDetail", str(row["id"]), document))

        if not jobs:
            return 0

        model_name = getattr(embeddings_model, "model", None) or "unknown"
        total = len(jobs)
        if progress_callback:
            progress_callback("Embedding & saving", total, 0)

        written = 0
        engine = _get_engine()
        BATCH = 50
        for start in range(0, total, BATCH):
            chunk = jobs[start : start + BATCH]
            try:
                vectors = embeddings_model.embed_documents([d for _, _, d in chunk])
            except Exception as e:  # noqa: BLE001
                # The rows are already saved; a vector can be rebuilt later. Never
                # let an embedding-quota failure lose ingested data.
                logger.warning("Embedding batch failed (%d docs): %s", len(chunk), e)
                continue

            with engine.begin() as conn:
                for (table, row_id, _), vector in zip(chunk, vectors):
                    conn.execute(
                        # CAST(... AS ...) rather than `:vec::extensions.vector`:
                        # SQLAlchemy's :param syntax collides with Postgres's ::
                        # cast operator, and silently leaves :vec unbound.
                        text(
                            f'UPDATE public."{table}" '
                            "SET embedding = CAST(:vec AS extensions.vector), "
                            "    embedding_model = :model "
                            "WHERE id = :id AND user_id = :user_id"
                        ),
                        {
                            "vec": to_vector_literal(vector),
                            "model": model_name,
                            "id": row_id,
                            "user_id": user_id,
                        },
                    )
                    written += 1
            if progress_callback:
                progress_callback("Embedding & saving", total, min(start + BATCH, total))

        logger.info("Wrote %d/%d vectors for user_id=%s", written, total, user_id)
        return written

    def sync_single_transaction(
        self,
        transaction: Dict[str, Any],
        details: List[Dict[str, Any]],
        user_id: str,
        embeddings_model: Embeddings,
    ) -> Optional[int]:
        """Re-embed one transaction, and its line items when they are supplied.

        Pass an empty `details` to embed only the parent — correct when the edit
        cannot appear in a line item's own text (a note, for instance).
        """
        return self.sync_transactions(
            pd.DataFrame([transaction]),
            pd.DataFrame(details) if details else pd.DataFrame(),
            user_id,
            embeddings_model,
        )

    # ── searching ───────────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        embeddings_model: Optional[Embeddings] = None,
        scope: str = "all",
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Search one or all of the three corpora, best matches first.

        `scope` picks what to look in — see SEARCH_SCOPES. It matters because the
        three answer different questions and the vectors are built from different
        text: a transaction embeds merchant and category, a line item embeds
        product identity with merchant deliberately absent, and a price
        observation embeds a shelf tag nobody bought anything from. Searching
        everything for "how much did I spend on groceries" pulls in shelf prices
        that are not spending at all.

        `min_score` is the cosine floor. It defaults to 0.0 — top-k regardless of
        relevance — which is right for open-ended chat retrieval, where "nothing
        found" is a worse answer than a weak one. Price comparison passes
        price_service.MIN_SEMANTIC_SCORE instead, because there a weak match is
        not a weak answer but a WRONG one: quoting what you paid for cilantro as
        the going rate for shampoo.

        Returns dicts with 'page_content' and 'metadata', the shape callers have
        always received. Tenancy is enforced by the WHERE clause here AND by RLS
        on the underlying tables; the previous store had no policies at all.
        """
        if embeddings_model is None:
            return []
        if scope not in SEARCH_SCOPES:
            raise ValueError(
                f"Unknown search scope {scope!r}. Expected one of: {', '.join(SEARCH_SCOPES)}"
            )
        vector = to_vector_literal(embeddings_model.embed_query(query))
        if vector is None:
            return []
        model_name = getattr(embeddings_model, "model", None)
        wants = {s: scope in ("all", s) for s in SEARCH_SCOPES if s != "all"}

        results: List[Dict] = []
        engine = _get_engine()
        with engine.connect() as conn:
            for row in [] if not wants["transactions"] else conn.execute(
                text(
                    "SELECT id, trans_date, merchant_name, description, category, "
                    "amount, note, enriched_info, score "
                    "FROM public.match_transactions(:vec, :model, :k, CAST(:floor AS real)) "
                    "WHERE id IN (SELECT id FROM public.\"Transaction\" WHERE user_id = :uid)"
                ),
                {"vec": vector, "model": model_name, "k": top_k, "uid": user_id, "floor": min_score},
            ).mappings():
                results.append({
                    "page_content": transaction_document(dict(row)),
                    "metadata": {
                        "id": str(row["id"]),
                        "user_id": user_id,
                        "vector_type": "transaction",
                        "amount": float(row["amount"]) if row["amount"] is not None else None,
                        "category": row["category"],
                        "merchant_name": row["merchant_name"],
                        "transaction_date": str(row["trans_date"]),
                        "score": float(row["score"]),
                    },
                })

            for row in [] if not wants["line_items"] else conn.execute(
                text(
                    "SELECT m.*, d.user_id FROM public.match_purchase_history("
                    "  :vec, :model, :k, CAST(:floor AS real)) m "
                    "JOIN public.\"TransactionDetail\" d ON d.id = m.id "
                    "WHERE d.user_id = :uid"
                ),
                {"vec": vector, "model": model_name, "k": top_k, "uid": user_id, "floor": min_score},
            ).mappings():
                results.append({
                    "page_content": detail_document(dict(row)),
                    "metadata": {
                        "id": str(row["transaction_id"]),
                        "detail_id": str(row["id"]),
                        "user_id": user_id,
                        "vector_type": "line_item",
                        "amount": float(row["item_total_price"]) if row["item_total_price"] is not None else None,
                        "merchant_name": row["merchant_name"],
                        # Where it was bought. A price from another city is not
                        # the going rate here, and without this the comparison
                        # could not tell.
                        "location": row.get("location"),
                        # Which receipt this line came from, so "show me proof"
                        # fetches THAT photo instead of searching for one.
                        "bill_file_id": (
                            str(row["bill_file_id"]) if row.get("bill_file_id") else None
                        ),
                        "transaction_date": str(row["trans_date"]),
                        "quantity": float(row["item_quantity"]) if row.get("item_quantity") is not None else None,
                        "quantity_unit": row.get("item_quantity_unit"),
                        # The size of ONE unit, confirmed rather than parsed.
                        "size_value": float(row["size_value"]) if row.get("size_value") is not None else None,
                        "size_unit": row.get("size_unit"),
                        "unit_price": (
                            float(row["unit_quantity_subtotal"])
                            if row.get("unit_quantity_subtotal") is not None else None
                        ),
                        # Carried so a price comparison can tell an ordinary price
                        # from one that was on offer. Treating a marked-down
                        # purchase as "what you usually pay" makes every normal
                        # shelf price look like a rip-off.
                        "item_savings": (
                            float(row["item_savings"]) if row.get("item_savings") is not None else None
                        ),
                        "discount_total": (
                            float(row["discount_total"]) if row.get("discount_total") is not None else None
                        ),
                        "item_description": row.get("item_description"),
                        "score": float(row["score"]),
                    },
                })

            for row in [] if not wants["price_observations"] else conn.execute(
                text(
                    "SELECT m.* FROM public.match_price_observations("
                    "  :vec, :model, :k, CAST(:floor AS real)) m "
                    "JOIN public.\"PriceObservation\" p ON p.id = m.id "
                    "WHERE p.user_id = :uid"
                ),
                {"vec": vector, "model": model_name, "k": top_k, "uid": user_id, "floor": min_score},
            ).mappings():
                results.append({
                    "page_content": observation_document(dict(row)),
                    "metadata": {
                        "id": str(row["id"]),
                        "user_id": user_id,
                        "vector_type": "price_observation",
                        # The shelf price, NOT money spent. Named apart from
                        # "amount" on purpose: a caller that sums `amount` across
                        # results must not pick this up and call it spending.
                        "shelf_price": (
                            float(row["item_subtotal_price"])
                            if row["item_subtotal_price"] is not None else None
                        ),
                        "merchant_name": row["merchant_name"],
                        "location": row["location"],
                        "observed_on": str(row["created_at"]),
                        "size_value": float(row["size_value"]) if row["size_value"] is not None else None,
                        "size_unit": row["size_unit"],
                        "unit_price": (
                            float(row["unit_quantity_subtotal"])
                            if row["unit_quantity_subtotal"] is not None else None
                        ),
                        # The tag's own words — offers, sale end dates, damage.
                        # Carried through because a price without them is
                        # routinely misread as what the item normally costs.
                        "tag_says": row["item_qualitative_description"],
                        "note": row["note"],
                        "score": float(row["score"]),
                    },
                })

        results.sort(key=lambda r: r["metadata"]["score"], reverse=True)
        return results[:top_k]

    # ── deletion ────────────────────────────────────────────────────────────
    #
    # Both are no-ops now, kept so callers need no change. Vectors are columns
    # on the rows, so deleting a transaction, a line item or a file removes them
    # via the existing ON DELETE CASCADE. The explicit cleanup these replaced was
    # the thing that could be forgotten — and was: re-verifying a receipt deleted
    # its line items without deleting their vectors, leaving 49 orphans behind.

    def delete_file_vectors(self, file_id: str, file_type: str) -> None:
        logger.debug("delete_file_vectors(%s) is a no-op — CASCADE handles it", file_id)

    def delete_transaction_vectors(
        self, transaction_id: str, detail_ids: Optional[List[str]] = None
    ) -> None:
        logger.debug(
            "delete_transaction_vectors(%s) is a no-op — CASCADE handles it", transaction_id
        )


def get_vector_client() -> VectorDBClient:
    return VectorDBClient()
