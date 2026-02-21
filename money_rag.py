import os
import sys
import uuid
import asyncio
import pandas as pd
import sqlite3
import shutil
import tempfile
from typing import List, Optional
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.runtime import get_runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient  
from qdrant_client.http import models as qdrant_models

# Import specific embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from supabase import create_client, ClientOptions

from dotenv import load_dotenv
load_dotenv()

class MoneyRAG:
    def __init__(self, llm_provider: str, model_name: str, embedding_model_name: str, api_key: str, user_id: str, access_token: str = None):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.user_id = user_id
        
        # Initialize Supabase Client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        # Security: Inject the logged-in user's JWT so RLS policies pass!
        if access_token:
            opts = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
            self.supabase = create_client(url, key, options=opts)
        else:
            self.supabase = create_client(url, key)
        
        # Set API Keys
        if self.llm_provider == "google":
            os.environ["GOOGLE_API_KEY"] = api_key
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
            provider_name = "google_genai"
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
            provider_name = "openai"

        # Initialize LLM
        self.llm = init_chat_model(
            self.model_name,
            model_provider=provider_name,
        )

        # Temporary paths for this session
        self.temp_dir = tempfile.mkdtemp()
        os.environ["DATA_DIR"] = self.temp_dir # Harmonize with mcp_server.py 
        self.db_path = os.path.join(self.temp_dir, "money_rag.db")
        self.qdrant_path = os.path.join(self.temp_dir, "qdrant_db")
        
        self.db: Optional[SQLDatabase] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.agent = None
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.search_tool = DuckDuckGoSearchRun()
        self.merchant_cache = {}  # Session-based cache for merchant enrichment
        self.memory = InMemorySaver()  # Session-based cache for chat memory

    async def setup_session(self, uploaded_files: List[dict]):
        """Ingests CSVs and Bills, then sets up DBs."""
        # uploaded_files format: [{"path": "/temp/file.csv", "file_id": "uuid"}, ...]
        for file_info in uploaded_files:
            ext = file_info["path"].lower().split('.')[-1]
            if ext in ['png', 'jpg', 'jpeg']:
                await self._ingest_bill(file_info["path"], file_info.get("file_id"))
            else:
                await self._ingest_csv(file_info["path"], file_info.get("file_id"))
        
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        self.vector_store = self._sync_to_qdrant()

    async def _ingest_csv(self, file_path, csv_id=None):
        import hashlib
        import json
        
        df = pd.read_csv(file_path)
        headers = df.columns.tolist()
        sample_data = df.head(10).to_json()

        prompt = ChatPromptTemplate.from_template("""
        Act as a financial data parser. Analyze this CSV data:
        Filename: {filename}
        Headers: {headers}
        Sample Data: {sample}

        TASK:
        1. Map the CSV columns to standard fields: date, description, amount, and category.
        2. Determine the 'sign_convention' for spending.
        
        RULES:
        - If the filename suggests 'Discover' credit card, spending are usually POSITIVE.
        - If the filename suggests 'Chase' credit card, spending are usually NEGATIVE.
                                                
        - Analyze the 'sign_convention' for spending (outflows):
            - Look at the sample data for known merchants or spending patterns.
            - If spending (like a restaurant or store) is NEGATIVE (e.g., -25.00), the convention is 'spending_is_negative'.
            - If spending is POSITIVE (e.g., 25.00), the convention is 'spending_is_positive'.

        OUTPUT FORMAT (JSON ONLY):
        {{
        "date_col": "column_name",
        "desc_col": "column_name",
        "amount_col": "column_name",
        "category_col": "column_name or null",
        "sign_convention": "spending_is_negative" | "spending_is_positive"
        }}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        mapping = await chain.ainvoke({"headers": headers, "sample": sample_data, "filename": os.path.basename(file_path)})

        standard_df = pd.DataFrame()
        standard_df['trans_date'] = pd.to_datetime(df[mapping['date_col']]).dt.strftime('%Y-%m-%d')
        # Assign user_id AFTER trans_date establishes the DataFrame length, or else it defaults to NaN!
        standard_df['user_id'] = self.user_id
        standard_df['description'] = df[mapping['desc_col']]
        if csv_id:
            standard_df['source_csv_id'] = csv_id
        
        raw_amounts = pd.to_numeric(df[mapping['amount_col']])
        standard_df['amount'] = raw_amounts * -1 if mapping['sign_convention'] == "spending_is_negative" else raw_amounts
        standard_df['category'] = df[mapping.get('category_col')] if mapping.get('category_col') else 'Uncategorized'

        # --- Async Enrichment Step ---
        print(f"   ✨ Enriching descriptions for {os.path.basename(file_path)}...")
        unique_descriptions = standard_df['description'].unique().tolist()
        sem = asyncio.Semaphore(5)

        extract_prompt = ChatPromptTemplate.from_template("""
You are a financial data assistant. Given a raw bank transaction description and a web search snippet about the merchant, extract structured information.

Transaction description: {description}
Web search result: {search_result}

Return ONLY valid JSON with exactly these two fields:
{{
  "merchant_name": "<clean 1-4 word business name, e.g. 'Chipotle', 'Amazon', 'Spotify'>",
  "enriched_info": "<one sentence describing what type of business this is>"
}}
""")
        extract_chain = extract_prompt | self.llm | JsonOutputParser()

        async def get_merchant_info(description):
            if description in self.merchant_cache:
                return self.merchant_cache[description]
            async with sem:
                try:
                    await asyncio.sleep(0.05)
                    print(f"      🔍 Web searching: {description}...")
                    search_result = await self.search_tool.ainvoke(
                        f"What type of business / store is '{description}'?"
                    )
                    # Extract structured merchant name + description in one LLM call
                    structured = await extract_chain.ainvoke({
                        "description": description,
                        "search_result": search_result[:500],
                    })
                    result = {
                        "merchant_name": structured.get("merchant_name", description),
                        "enriched_info": structured.get("enriched_info", search_result[:200]),
                    }
                    self.merchant_cache[description] = result
                    return result
                except Exception as e:
                    print(f"      ⚠️ Enrichment failed for {description}: {e}")
                    return {"merchant_name": description, "enriched_info": ""}

        tasks = [get_merchant_info(desc) for desc in unique_descriptions]
        enrichment_results = await asyncio.gather(*tasks)

        desc_map = dict(zip(unique_descriptions, enrichment_results))
        standard_df['enriched_info'] = standard_df['description'].map(
            lambda d: desc_map.get(d, {}).get("enriched_info", "")
        )
        standard_df['merchant_name'] = standard_df['description'].map(
            lambda d: desc_map.get(d, {}).get("merchant_name", d)
        )

        print(f"   ✅ Enriched {len(unique_descriptions)} unique merchants.")


        # Save to Supabase transactions table
        records = json.loads(standard_df.to_json(orient='records'))

        # Calculate content_hash and source for deduplication
        def generate_hash(row):
            date_str = str(row['trans_date']).strip()
            amount_str = str(round(float(row['amount']), 2))
            merch = str(row.get('merchant_name', row['description'])).lower().strip().split()[0]
            merch = ''.join(c for c in merch if c.isalnum())
            hash_input = f"{date_str}{amount_str}{merch}"
            return hashlib.sha256(hash_input.encode()).hexdigest()

        for r in records:
            r['content_hash'] = generate_hash(r)
            r['source'] = 'csv'

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                # Upsert based on content_hash
                self.supabase.table("Transaction").upsert(batch, on_conflict="content_hash").execute()
            except Exception as e:
                # Fallback if DB migration hasn't been run yet (no content_hash / merchant_name)
                for r in batch:
                    r.pop('merchant_name', None)
                    r.pop('content_hash', None)
                    r.pop('source', None)
                try:
                    self.supabase.table("Transaction").insert(batch).execute()
                except Exception as ex:
                    print(f"Failed fallback insert: {ex}")

    async def _ingest_bill(self, file_path, file_id=None):
        import base64
        import hashlib
        import json
        from langchain_core.messages import HumanMessage
        
        print(f"   📸 Processing bill image: {os.path.basename(file_path)}...")
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        ext = file_path.lower().split('.')[-1]
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

        schema = {
            "date": "YYYY-MM-DD",
            "total_amount": 123.45,
            "merchant_name": "Merchant",
            "category": "Dining",
            "line_items": [{"item_description": "Item", "item_quantity": 1, "item_unit_price": 10.0, "tax_amount": 0.0, "item_total_price": 10.0}]
        }
        
        prompt = f"Extract structured data from this receipt/bill. Return strictly valid JSON exactly matching this schema: {json.dumps(schema)}"
        message = HumanMessage(
            content=[
                 {"type": "text", "text": prompt},
                 {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded_string}"}}
            ]
        )
        
        extract_chain = self.llm | JsonOutputParser()
        try:
            extracted = await extract_chain.ainvoke([message])
        except Exception as e:
            print(f"   ❌ Vision extraction failed: {e}")
            return
            
        print(f"   ✅ Extracted {extracted.get('merchant_name')} for {extracted.get('total_amount')}")
        
        # Save the raw OCR JSON back to the BillFile record so we don't lose the raw extraction
        if file_id:
            try:
                self.supabase.table("BillFile").update({"raw_ocr_string": json.dumps(extracted)}).eq("id", file_id).execute()
            except Exception as e:
                print(f"   ⚠️ Failed to save raw_ocr_string to BillFile: {e}")
        
        # Calculate content_hash
        date_str = str(extracted.get('date', '')).strip()
        amount_str = str(round(float(extracted.get('total_amount', 0)), 2))
        merch = str(extracted.get('merchant_name', '')).lower().strip().split()[0] if extracted.get('merchant_name') else ""
        merch = ''.join(c for c in merch if c.isalnum())
        hash_input = f"{date_str}{amount_str}{merch}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Build transaction record
        tx_record = {
            "user_id": self.user_id,
            "trans_date": date_str,
            # In this app, spending is traditionally positive, so let's keep it strictly positive for bills
            "amount": abs(float(extracted.get('total_amount', 0))), 
            "description": extracted.get('merchant_name', 'Unknown'),
            "merchant_name": extracted.get('merchant_name', 'Unknown'),
            "category": extracted.get('category', 'Uncategorized'),
            "content_hash": content_hash,
            "source": 'bill'
        }
        if file_id:
            tx_record["source_bill_file_id"] = file_id
            
        # Optional: Enrich line items
        sem = asyncio.Semaphore(5)
        async def enrich_item(item):
            name = item.get('item_description', '')
            async with sem:
                try:
                    await asyncio.sleep(0.05)
                    res = await self.search_tool.ainvoke(f"What type of product is '{name}'?")
                    item['enriched_info'] = res[:200]
                except Exception:
                    item['enriched_info'] = ""
            return item
            
        line_items = extracted.get('line_items', [])
        print(f"   ✨ Enriching {len(line_items)} line items...")
        tasks = [enrich_item(i) for i in line_items]
        line_items = await asyncio.gather(*tasks)

        # Upsert Transaction
        try:
            upsert_res = self.supabase.table("Transaction").upsert([tx_record], on_conflict="content_hash").execute()
        except Exception as e:
            print(f"   ⚠️ Upsert failed (migration not run?), falling back to insert: {e}")
            tx_record.pop('merchant_name', None)
            tx_record.pop('content_hash', None)
            tx_record.pop('source', None)
            upsert_res = self.supabase.table("Transaction").insert([tx_record]).execute()
            
        # Get the transaction ID to link details (upsert doesn't always return data on conflict update, so explicit select is safest)
        try:
            fetch_res = self.supabase.table("Transaction").select("id").eq("content_hash", content_hash).eq("user_id", self.user_id).execute()
            if fetch_res.data:
                tx_id = fetch_res.data[0]['id']
            else:
                tx_id = None
        except Exception as e:
            print(f"   ⚠️ Failed to fetch transaction ID: {e}")
            tx_id = None
        
        if tx_id and line_items:
            details = []
            for item in line_items:
                details.append({
                    "transaction_id": tx_id,
                    "user_id": self.user_id,
                    "bill_file_id": file_id,
                    "item_description": item.get('item_description', ''),
                    "item_quantity": item.get('item_quantity', 1),
                    "item_unit_price": item.get('item_unit_price', item.get('item_total_price', 0)),
                    "tax_amount": item.get('tax_amount', 0),
                    "item_total_price": item.get('item_total_price', 0),
                    "enriched_info": item.get('enriched_info', '')
                })
            try:
                self.supabase.table("TransactionDetail").insert(details).execute()
                print(f"   ✅ Saved {len(details)} line items.")
            except Exception as e:
                print(f"   ⚠️ Failed to insert details (table might not exist): {e}")

    def _sync_to_qdrant(self):
        # client = QdrantClient(path=self.qdrant_path)
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = "transactions"
        
        # Fetch only THIS USER'S transactions from Supabase to sync into VectorDB
        res = self.supabase.table("Transaction").select("*").eq("user_id", self.user_id).execute()
        df = pd.DataFrame(res.data)
        
        # Check for empty dataframe
        if df.empty:
            raise ValueError("No transactions found in database for this user. Please upload files first.")
        
        # Dynamically detect embedding dimension
        sample_embedding = self.embeddings.embed_query("test")
        embedding_dim = len(sample_embedding)

        # Safely create the collection only if it doesn't already exist to preserve multi-tenant pool
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=qdrant_models.VectorParams(size=embedding_dim, distance=qdrant_models.Distance.COSINE),
            )
        
        # Security: Create a strict Payload Index on the user_id field so we can filter by it securely!
        client.create_payload_index(
            collection_name=collection,
            field_name="metadata.user_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        
        vs = QdrantVectorStore(client=client, collection_name=collection, embedding=self.embeddings)
        
        # Try to fetch TransactionDetail
        try:
            detail_res = self.supabase.table("TransactionDetail").select("*").execute()
            details_df = pd.DataFrame(detail_res.data)
        except Exception:
            details_df = pd.DataFrame()
            
        if not details_df.empty:
            # Filter line items to only those belonging to this user's transactions
            details_df = details_df[details_df['transaction_id'].isin(df['id'])]
            
        texts = []
        metadatas = []
        vector_ids = []
        
        # 1. Build vectors for parent transactions
        for _, row in df.iterrows():
            merchant = row.get('merchant_name', '') or row.get('description', '')
            category = row.get('category', 'Uncategorized')
            enriched = row.get('enriched_info', '')
            base_text = f"{merchant} ({category})"
            if enriched:
                texts.append(f"{base_text} — {enriched}")
            else:
                texts.append(base_text)
                
            meta_cols = ['id', 'amount', 'category', 'trans_date']
            if 'merchant_name' in row:
                meta_cols.append('merchant_name')
            if 'source_csv_id' in row:
                meta_cols.append('source_csv_id')
                
            meta = {k: row[k] for k in meta_cols if k in row and pd.notna(row[k])}
            meta['user_id'] = self.user_id
            meta['transaction_date'] = str(meta.pop('trans_date'))
            meta['vector_type'] = 'transaction'
            metadatas.append(meta)
            vector_ids.append(str(row['id']))

        # 2. Build explicit separate vectors for every line item
        if not details_df.empty:
            for _, d_row in details_df.iterrows():
                parent_row = df[df['id'] == d_row['transaction_id']].iloc[0]
                merchant = parent_row.get('merchant_name', parent_row.get('description', ''))
                texts.append(f"Line item from {merchant}: {d_row['item_description']} — {d_row.get('enriched_info', '')}")
                
                # Keep parent ID in standard 'id' field so the SQL agent resolves it nicely
                meta = {
                    'id': str(parent_row['id']),
                    'detail_id': str(d_row['id']),
                    'amount': float(d_row['item_total_price'] if pd.notna(d_row.get('item_total_price')) else 0),
                    'category': parent_row.get('category', 'Uncategorized'),
                    'user_id': self.user_id,
                    'transaction_date': str(parent_row['trans_date']),
                    'vector_type': 'line_item',
                    'merchant_name': str(merchant)
                }
                if 'source_csv_id' in parent_row and pd.notna(parent_row['source_csv_id']):
                    meta['source_csv_id'] = parent_row['source_csv_id']
                    
                metadatas.append(meta)
                # Use the detail_id for the Qdrant point ID to avoid UUID collisions with parents
                vector_ids.append(str(d_row['id']))
                
        vs.add_texts(texts=texts, metadatas=metadatas, ids=vector_ids)
        return vs

    async def delete_file(self, file_id: str, file_type: str = 'csv'):
        """Force delete a file and all its transactions from Postgres and Qdrant."""
        try:
            # 1. Delete from Postgres (Transactions cascade automatically if foreign keyed... but we'll manually ensure they wipe just in case)
            if file_type == 'csv':
                self.supabase.table("Transaction").delete().eq("source_csv_id", file_id).execute()
                self.supabase.table("CSVFile").delete().eq("id", file_id).execute()
            else:
                self.supabase.table("TransactionDetail").delete().eq("bill_file_id", file_id).execute()
                self.supabase.table("BillFile").delete().eq("id", file_id).execute()
            
            # 2. Delete from Qdrant via payload filter
            client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
            filter_key = "metadata.source_csv_id" if file_type == 'csv' else "metadata.bill_file_id"
            
            client.delete(
                collection_name="transactions",
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=filter_key,
                            match=qdrant_models.MatchValue(value=file_id)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Error purging file data: {e}")

    async def chat(self, query: str):
        """Async generator that yields status events + final response."""
        server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        
        mcp_client = MultiServerMCPClient(
            {
                "money_rag": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [server_path],
                    "env": {**os.environ.copy(), "CURRENT_USER_ID": self.user_id},
                }
            }
        )

        try:
            mcp_tools = await mcp_client.get_tools()

            system_prompt = (
                "You are a financial analyst. Use the provided tools to query the database "
                "and perform semantic searches. Spending is POSITIVE (>0). "
                "Always explain your findings clearly."
                "IMPORTANT: Whenever possible and relevant (e.g. when discussing trends, comparing categories, or showing breakdowns), "
                "you MUST proactively use the 'generate_interactive_chart' tool to generate visual plots (bar, pie, or line charts) to accompany your analysis. "
                "WARNING: You MUST use the actual tool call to generate the chart. DO NOT simply output a json block with chart parameters as your final text answer."
            )
            
            agent = create_agent(
                model=self.llm,
                tools=mcp_tools,
                system_prompt=system_prompt,
                checkpointer=self.memory,
            )

            config = {"configurable": {"thread_id": "session_1"}}
            
            chart_path = os.path.join(self.temp_dir, "latest_chart.json")
            if os.path.exists(chart_path):
                os.remove(chart_path)

            # Stream events so we can yield live tool-call updates
            final_content = ""
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": query}]},
                config,
                version="v2",
            ):
                kind = event.get("event")

                if kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    tool_input = event.get("data", {}).get("input", {})
                    # Summarise long inputs
                    if isinstance(tool_input, dict):
                        snippet = ", ".join(f"{k}={str(v)[:60]}" for k, v in tool_input.items())
                    else:
                        snippet = str(tool_input)[:120]
                    yield {"type": "tool_start", "name": tool_name, "input": snippet}

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    output = event.get("data", {}).get("output", "")
                    snippet = str(output)[:200].replace("\n", " ")
                    yield {"type": "tool_end", "name": tool_name, "snippet": snippet}

                elif kind == "on_chain_end":
                    # Capture the final AI message from the root chain
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "messages" in output:
                        last_msg = output["messages"][-1]
                        content = last_msg.content if hasattr(last_msg, "content") else ""
                        if isinstance(content, list):
                            final_content = "\n".join(
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        elif content:
                            final_content = content

            # Build final response (with optional chart)
            if os.path.exists(chart_path):
                with open(chart_path, "r") as f:
                    chart_json = f.read()
                yield {"type": "final", "content": f"{final_content}\n\n===CHART===\n{chart_json}\n===ENDCHART==="}
            else:
                yield {"type": "final", "content": final_content}
            
        finally:
            try:
                await mcp_client.close()
            except Exception as close_e:
                print(f"Warning on closing MCP Client: {close_e}")

    async def cleanup(self):
        """Delete temporary session files and close MCP client."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temp directory: {e}")