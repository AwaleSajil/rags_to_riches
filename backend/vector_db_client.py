import os
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain_core.embeddings import Embeddings
from backend.config import get_settings

class VectorDBClient:
    """Abstract interface for Vector Database Operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.collection_name = "transactions"
        
        # Setup Qdrant
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(
            url=self.settings.QDRANT_URL,
            api_key=self.settings.QDRANT_API_KEY
        )

    def sync_transactions(self, df: pd.DataFrame, details_df: pd.DataFrame, user_id: str, embeddings_model: Embeddings, progress_callback=None) -> None:
        """
        Embed and ingest transactions and line items into the vector database.
        Returns the initialized Langchain VectorStore (for Qdrant) or None (for Actian, which is managed directly).
        progress_callback(stage_detail, total, done) — optional callable for progress updates.
        """
        if df.empty:
            print("No transactions found in database for this user. Skipping vector sync.")
            return None
        sample_embedding = embeddings_model.embed_query("test")
        embedding_dim = len(sample_embedding)
        
        total_items = len(df) + (len(details_df) if not details_df.empty else 0)
        built = 0
        
        texts = []
        metadatas = []
        vector_ids = []
        
        # 1. Build parent transaction payloads
        for _, row in df.iterrows():
            merchant = row.get('merchant_name', '') or row.get('description', '')
            category = row.get('category', 'Uncategorized')
            enriched = row.get('enriched_info', '')
            base_text = f"{merchant} ({category})"
            texts.append(f"{base_text} — {enriched}" if enriched else base_text)

            meta_cols = ['id', 'amount', 'category', 'trans_date']
            if 'merchant_name' in row: meta_cols.append('merchant_name')
            if 'source_csv_id' in row: meta_cols.append('source_csv_id')
                
            meta = {k: row[k] for k in meta_cols if k in row and pd.notna(row[k])}
            if 'source_bill_file_id' in row and pd.notna(row['source_bill_file_id']):
                meta['bill_file_id'] = row['source_bill_file_id']
                
            meta['user_id'] = user_id
            meta['transaction_date'] = str(meta.pop('trans_date'))
            meta['vector_type'] = 'transaction'
            # For Actian compatibility, ensure we save the raw page_content in the payload
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
                    'merchant_name': str(merchant)
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

        # Generate embeddings in batches for progress tracking
        total_texts = len(texts)
        print(f"   🧠 Embedding {total_texts} documents into Qdrant...")
        EMBED_BATCH = 50
        vectors = []
        for i in range(0, total_texts, EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            batch_vectors = embeddings_model.embed_documents(batch)
            vectors.extend(batch_vectors)
            if progress_callback:
                progress_callback("Embedding", total_texts, min(i + EMBED_BATCH, total_texts))

        return self._sync_qdrant(texts, metadatas, vector_ids, embedding_dim, embeddings_model, progress_callback)



    def _sync_qdrant(self, texts, metadatas, vector_ids, dim, embeddings_model, progress_callback=None):
        from qdrant_client.http import models as qdrant_models
        from langchain_qdrant import QdrantVectorStore
        
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE),
            )
            
        self.qdrant_client.create_payload_index(self.collection_name, "metadata.user_id", qdrant_models.PayloadSchemaType.KEYWORD)
        self.qdrant_client.create_payload_index(self.collection_name, "metadata.source_csv_id", qdrant_models.PayloadSchemaType.KEYWORD)
        self.qdrant_client.create_payload_index(self.collection_name, "metadata.bill_file_id", qdrant_models.PayloadSchemaType.KEYWORD)
        
        vs = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name, embedding=embeddings_model)
        
        # Batch add_texts for progress tracking
        total = len(texts)
        UPSERT_BATCH = 50
        if progress_callback:
            progress_callback("Uploading to vector DB", total, 0)
        for i in range(0, total, UPSERT_BATCH):
            end = min(i + UPSERT_BATCH, total)
            vs.add_texts(
                texts=texts[i:end],
                metadatas=metadatas[i:end],
                ids=vector_ids[i:end],
            )
            if progress_callback:
                progress_callback("Uploading to vector DB", total, end)
        
        return vs

    def semantic_search(self, query: str, user_id: str, top_k: int = 5, embeddings_model: Optional[Embeddings] = None) -> List[Dict]:
        """Search the vector database, returning a list of dicts with 'page_content' and 'metadata'."""
        from qdrant_client.http import models
        from langchain_qdrant import QdrantVectorStore
        
        q_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
        )
        
        vs = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name, embedding=embeddings_model)
        results = vs.similarity_search(query, k=top_k, filter=q_filter)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]

    def delete_file_vectors(self, file_id: str, file_type: str) -> None:
        """Deletes all vectors originating from a specific file."""
        filter_key = "metadata.source_csv_id" if file_type == 'csv' else "metadata.bill_file_id"
        from qdrant_client.http import models
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[models.FieldCondition(key=filter_key, match=models.MatchValue(value=file_id))]
            )
        )

def get_vector_client() -> VectorDBClient:
    return VectorDBClient()
