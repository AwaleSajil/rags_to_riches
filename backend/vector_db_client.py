import os
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain_core.embeddings import Embeddings
from backend.config import get_settings

class VectorDBClient:
    """Abstract interface for Vector Database Operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.stack = self.settings.VECTOR_DB_STACK.lower()
        self.collection_name = "transactions"
        
        if self.stack == "actian":
            try:
                from cortex import CortexClient
                self.actian_client = CortexClient(
                    address=self.settings.ACTIAN_ADDRESS,
                    api_key=self.settings.ACTIAN_API_KEY
                )
            except ImportError:
                raise RuntimeError("Actian Cortex SDK ('actiancortex') is not installed.")
        else:
            # Setup Qdrant
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY
            )

    def is_actian(self) -> bool:
        return self.stack == "actian"

    def sync_transactions(self, df: pd.DataFrame, details_df: pd.DataFrame, user_id: str, embeddings_model: Embeddings) -> None:
        """
        Embed and ingest transactions and line items into the vector database.
        Returns the initialized Langchain VectorStore (for Qdrant) or None (for Actian, which is managed directly).
        """
        if df.empty:
            raise ValueError("No transactions found in database for this user. Please upload files first.")
            
        sample_embedding = embeddings_model.embed_query("test")
        embedding_dim = len(sample_embedding)
        
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

        # Generate actual embeddings
        print(f"   🧠 Embedding {len(texts)} documents into {self.stack}...")
        vectors = embeddings_model.embed_documents(texts)

        if self.is_actian():
            self._sync_actian(vectors, metadatas, vector_ids, embedding_dim)
            return None
        else:
            return self._sync_qdrant(texts, metadatas, vector_ids, embedding_dim, embeddings_model)

    def _sync_actian(self, vectors: List[List[float]], metadatas: List[Dict], vector_ids: List[str], dim: int):
        from cortex import DistanceMetric
        import uuid
        
        with self.actian_client as client:
            client.get_or_create_collection(
                name=self.collection_name,
                dimension=dim,
                distance_metric=DistanceMetric.COSINE
            )
            
            # Actian expects integer IDs, but our DB uses UUID strings.
            # We map strings to integer IDs deterministically via hashing for Actian.
            int_ids = [int(uuid.UUID(vid).int >> 64) for vid in vector_ids]
            
            client.batch_upsert(
                collection_name=self.collection_name,
                ids=int_ids,
                vectors=vectors,
                payloads=metadatas
            )

    def _sync_qdrant(self, texts, metadatas, vector_ids, dim, embeddings_model):
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
        vs.add_texts(texts=texts, metadatas=metadatas, ids=vector_ids)
        return vs

    def semantic_search(self, query: str, user_id: str, top_k: int = 5, embeddings_model: Optional[Embeddings] = None) -> List[Dict]:
        """Search the vector database, returning a list of dicts with 'page_content' and 'metadata'."""
        if self.is_actian():
            if not embeddings_model:
                raise ValueError("Actian search requires passing the embeddings_model to generated query vectors.")
            
            query_vector = embeddings_model.embed_query(query)
            from cortex import Filter, Field
            
            f = Filter().must(Field("user_id").eq(user_id))
            
            with self.actian_client as client:
                if not client.has_collection(self.collection_name):
                    return []
                results = client.search(
                    collection_name=self.collection_name,
                    query=query_vector,
                    top_k=top_k,
                    filter=f,
                    with_payload=True
                )
            
            return [{"page_content": r.payload.get("page_content", ""), "metadata": r.payload} for r in results]
            
        else:
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
        if self.is_actian():
            from cortex import Filter, Field
            filter_key = "source_csv_id" if file_type == 'csv' else "bill_file_id"
            f = Filter().must(Field(filter_key).eq(file_id))
            
            with self.actian_client as client:
                if not client.has_collection(self.collection_name): return
                
                # Actian batch_delete requires IDs, so we query to find matching IDs first
                records = client.query(self.collection_name, filter=f, limit=10000)
                ids_to_delete = [r.id for r in records]
                
                if ids_to_delete:
                    client.batch_delete(self.collection_name, ids=ids_to_delete)
        else:
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
