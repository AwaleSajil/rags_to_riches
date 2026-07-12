import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from backend.config import get_settings

logger = logging.getLogger("moneyrag.vector_db_client")

COLLECTION_NAME = "transactions"

# langchain_postgres default table names (created automatically on first use).
_EMBEDDING_TABLE = "langchain_pg_embedding"
_COLLECTION_TABLE = "langchain_pg_collection"


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
            # Ensure the pgvector type (installed in `extensions` on Supabase) is
            # resolvable, while langchain's own tables land in `public`.
            connect_args={"options": "-csearch_path=public,extensions"},
        )
    return _engine


class VectorDBClient:
    """Vector store operations backed by Supabase Postgres + pgvector."""

    def __init__(self):
        self.settings = get_settings()
        self.collection_name = COLLECTION_NAME

    def _store(self, embeddings_model: Embeddings) -> PGVector:
        # The vector extension is pre-installed on Supabase (schema `extensions`),
        # so we skip CREATE EXTENSION here. The embedding column is unbounded
        # (`vector`), which lets different users store different embedding
        # dimensions — search filters by user_id, so per-user dims stay uniform.
        return PGVector(
            embeddings=embeddings_model,
            collection_name=self.collection_name,
            connection=_get_engine(),
            use_jsonb=True,
            create_extension=False,
        )

    def sync_transactions(
        self,
        df: pd.DataFrame,
        details_df: pd.DataFrame,
        user_id: str,
        embeddings_model: Embeddings,
        progress_callback=None,
    ) -> Optional[PGVector]:
        """
        Embed and ingest transactions and line items into the vector database.
        progress_callback(stage_detail, total, done) — optional callable for progress.
        """
        if df.empty:
            print("No transactions found in database for this user. Skipping vector sync.")
            return None

        total_items = len(df) + (len(details_df) if not details_df.empty else 0)
        built = 0

        texts: List[str] = []
        metadatas: List[dict] = []
        vector_ids: List[str] = []

        # 1. Build parent transaction payloads
        for _, row in df.iterrows():
            merchant = row.get('merchant_name', '') or row.get('description', '')
            category = row.get('category', 'Uncategorized')
            enriched = row.get('enriched_info', '')
            base_text = f"{merchant} ({category})"
            texts.append(f"{base_text} — {enriched}" if enriched else base_text)

            meta_cols = ['id', 'amount', 'category', 'trans_date']
            if 'merchant_name' in row: meta_cols.append('merchant_name')
            if 'location' in row: meta_cols.append('location')
            if 'source_csv_id' in row: meta_cols.append('source_csv_id')

            meta = {k: row[k] for k in meta_cols if k in row and pd.notna(row[k])}
            if 'source_bill_file_id' in row and pd.notna(row['source_bill_file_id']):
                meta['bill_file_id'] = row['source_bill_file_id']

            meta['user_id'] = user_id
            meta['transaction_date'] = str(meta.pop('trans_date'))
            meta['vector_type'] = 'transaction'
            meta['page_content'] = texts[-1]

            metadatas.append(meta)
            vector_ids.append(str(row['id']))
            built += 1
            if progress_callback and built % 50 == 0:
                progress_callback("Building payloads", total_items, built)

        # 2. Build line item payloads
        if not details_df.empty:
            for _, d_row in details_df.iterrows():
                parent_row = df[df['id'] == d_row['transaction_id']].iloc[0]
                merchant = parent_row.get('merchant_name', parent_row.get('description', ''))
                texts.append(f"Line item from {merchant}: {d_row['item_description']} — {d_row.get('enriched_info', '')}")

                meta = {
                    'id': str(parent_row['id']),
                    'detail_id': str(d_row['id']),
                    'amount': float(d_row['item_total_price'] if pd.notna(d_row.get('item_total_price')) else 0),
                    'category': parent_row.get('category', 'Uncategorized'),
                    'user_id': user_id,
                    'transaction_date': str(parent_row['trans_date']),
                    'vector_type': 'line_item',
                    'merchant_name': str(merchant),
                }
                if 'source_csv_id' in parent_row and pd.notna(parent_row['source_csv_id']):
                    meta['source_csv_id'] = parent_row['source_csv_id']
                if 'source_bill_file_id' in parent_row and pd.notna(parent_row['source_bill_file_id']):
                    meta['bill_file_id'] = parent_row['source_bill_file_id']

                meta['page_content'] = texts[-1]
                metadatas.append(meta)
                vector_ids.append(str(d_row['id']))
                built += 1
                if progress_callback and built % 50 == 0:
                    progress_callback("Building payloads", total_items, built)

        if progress_callback:
            progress_callback("Building payloads", total_items, total_items)

        if not texts:
            return None

        # 3. Embed + upsert into pgvector in batches (add_texts embeds internally).
        #    Passing ids makes re-syncs upsert the same rows rather than duplicate.
        store = self._store(embeddings_model)
        total = len(texts)
        print(f"   🧠 Embedding {total} documents into pgvector...")
        BATCH = 50
        if progress_callback:
            progress_callback("Embedding & saving", total, 0)
        for i in range(0, total, BATCH):
            end = min(i + BATCH, total)
            store.add_texts(
                texts=texts[i:end],
                metadatas=metadatas[i:end],
                ids=vector_ids[i:end],
            )
            if progress_callback:
                progress_callback("Embedding & saving", total, end)

        return store

    def semantic_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        embeddings_model: Optional[Embeddings] = None,
    ) -> List[Dict]:
        """Search the vector store, returning dicts with 'page_content' and 'metadata'."""
        store = self._store(embeddings_model)
        results = store.similarity_search(
            query,
            k=top_k,
            filter={"user_id": {"$eq": user_id}},
        )
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]

    def delete_file_vectors(self, file_id: str, file_type: str) -> None:
        """Delete all vectors that originated from a specific file (by metadata)."""
        meta_key = "source_csv_id" if file_type == 'csv' else "bill_file_id"
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    DELETE FROM {_EMBEDDING_TABLE} e
                    USING {_COLLECTION_TABLE} c
                    WHERE e.collection_id = c.uuid
                      AND c.name = :cname
                      AND e.cmetadata ->> :meta_key = :file_id
                """),
                {"cname": self.collection_name, "meta_key": meta_key, "file_id": str(file_id)},
            )

    def delete_transaction_vectors(
        self, transaction_id: str, detail_ids: Optional[List[str]] = None
    ) -> None:
        """Delete a single transaction's vectors (parent + its line items).

        Both the parent vector and every line-item vector carry
        cmetadata->>'id' = <transaction_id>, so that filter alone removes the
        whole family. We also match line-item vectors by cmetadata->>'detail_id'
        as a belt-and-suspenders guard when detail ids are known.
        """
        txid = str(transaction_id)
        detail_ids = [str(d) for d in (detail_ids or [])]
        # Only add the detail_id branch when we have ids — an empty array would
        # otherwise trip Postgres type inference on `= ANY(:detail_ids)`.
        detail_clause = (
            " OR e.cmetadata ->> 'detail_id' = ANY(:detail_ids)" if detail_ids else ""
        )
        params: Dict[str, Any] = {"cname": self.collection_name, "txid": txid}
        if detail_ids:
            params["detail_ids"] = detail_ids
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    DELETE FROM {_EMBEDDING_TABLE} e
                    USING {_COLLECTION_TABLE} c
                    WHERE e.collection_id = c.uuid
                      AND c.name = :cname
                      AND ( e.cmetadata ->> 'id' = :txid{detail_clause} )
                """),
                params,
            )

    def sync_single_transaction(
        self,
        transaction: Dict[str, Any],
        details: List[Dict[str, Any]],
        user_id: str,
        embeddings_model: Embeddings,
    ) -> Optional[PGVector]:
        """Re-embed one transaction (and its line items), upserting by id.

        Reuses sync_transactions so the document text / metadata stay identical
        to the bulk ingestion path. Passing the same ids upserts in place.
        """
        df = pd.DataFrame([transaction])
        details_df = pd.DataFrame(details) if details else pd.DataFrame()
        return self.sync_transactions(df, details_df, user_id, embeddings_model)


def get_vector_client() -> VectorDBClient:
    return VectorDBClient()
