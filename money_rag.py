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
from langgraph.runtime import get_runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient  
from backend.vector_db_client import get_vector_client

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
        self.db_path = os.path.join(self.temp_dir, "money_rag.db")
        
        self.db: Optional[SQLDatabase] = None
        self.vector_store_client = None
        self.agent = None
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.search_tool = DuckDuckGoSearchRun()
        self.merchant_cache = {}  # Session-based cache for merchant enrichment
        self.memory = InMemorySaver()  # Session-based cache for chat memory

    async def setup_session(self, uploaded_files: List[dict]):
        """Ingests CSVs and Bills, then sets up DBs."""
        # uploaded_files format: [{"path": "/temp/file.csv", "file_id": "uuid"}, ...]
        all_duplicates = []
        for file_info in uploaded_files:
            file_path = file_info["path"]
            file_name = os.path.basename(file_path)
            ext = file_path.lower().split('.')[-1]
            try:
                if ext in ['png', 'jpg', 'jpeg']:
                    dups = await self._ingest_bill(file_path, file_info.get("file_id"))
                else:
                    dups = await self._ingest_csv(file_path, file_info.get("file_id"))
                if dups:
                    all_duplicates.extend(dups)
            except Exception as e:
                raise RuntimeError(f"Failed to ingest '{file_name}': {e}") from e

        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            self.vector_store_client = self._sync_to_vectordb()
        except Exception as e:
            raise RuntimeError(f"Failed to sync to vector store: {e}") from e
        return all_duplicates

    async def _ingest_csv(self, file_path, csv_id=None):
        import hashlib
        import json

        filename = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Cannot read CSV file: {e}") from e

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
        try:
            mapping = await chain.ainvoke({"headers": headers, "sample": sample_data, "filename": filename})
        except Exception as e:
            raise RuntimeError(f"LLM column mapping failed (headers: {headers}): {e}") from e

        # Validate that the LLM returned the required keys
        for key in ['date_col', 'desc_col', 'amount_col']:
            if key not in mapping:
                raise RuntimeError(f"LLM mapping missing required key '{key}'. Got: {mapping}")
            if mapping[key] not in df.columns:
                raise RuntimeError(f"LLM mapped '{key}' to column '{mapping[key]}' which doesn't exist. Available columns: {headers}")

        standard_df = pd.DataFrame()
        try:
            standard_df['trans_date'] = pd.to_datetime(df[mapping['date_col']]).dt.strftime('%Y-%m-%d')
        except Exception as e:
            raise RuntimeError(f"Date parsing failed for column '{mapping['date_col']}': {e}") from e
        # Assign user_id AFTER trans_date establishes the DataFrame length, or else it defaults to NaN!
        standard_df['user_id'] = self.user_id
        standard_df['description'] = df[mapping['desc_col']]
        if csv_id:
            standard_df['source_csv_id'] = csv_id

        raw_amounts = pd.to_numeric(df[mapping['amount_col']], errors='coerce')
        if raw_amounts.isna().all():
            raise RuntimeError(f"All values in amount column '{mapping['amount_col']}' are non-numeric")
        standard_df['amount'] = raw_amounts * -1 if mapping['sign_convention'] == "spending_is_negative" else raw_amounts

        cat_col = mapping.get('category_col')
        standard_df['category'] = df[cat_col] if cat_col and cat_col in df.columns else 'Uncategorized'

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

        # Detect duplicates before upserting
        hashes = [r['content_hash'] for r in records]
        existing_hashes = set()
        for i in range(0, len(hashes), 100):
            batch_hashes = hashes[i:i+100]
            existing_res = self.supabase.table("Transaction").select("content_hash").in_("content_hash", batch_hashes).execute()
            if existing_res.data:
                existing_hashes.update(row['content_hash'] for row in existing_res.data)
                
        duplicates = [
            {"date": r['trans_date'], "merchant": r.get('merchant_name', r.get('description', '')), "amount": r['amount']} 
            for r in records if r['content_hash'] in existing_hashes
        ]

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
                    
        return duplicates

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
        
        # Save the raw OCR JSON back to the BillFile record safely via psycopg2
        if file_id:
            try:
                import psycopg2
                db_url = os.environ.get("DATABASE_URL")
                if db_url:
                    conn = psycopg2.connect(db_url)
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE "BillFile" SET raw_ocr_string = %s WHERE id = %s',
                        (json.dumps(extracted), file_id)
                    )
                    conn.commit()
                    conn.close()
                    print("   ✅ Saved raw OCR to BillFile seamlessly.")
                else:
                    print("   ⚠️ DATABASE_URL not set, skipping raw OCR save.")
            except Exception as e:
                print(f"   ⚠️ Failed to save raw_ocr_string to BillFile: {e}")
        
        # Calculate content_hash
        date_str = str(extracted.get('date', '')).strip()
        amount_str = str(round(float(extracted.get('total_amount', 0)), 2))
        
        # Enrich parent merchant via Web Search
        raw_merchant = extracted.get('merchant_name', 'Unknown')
        print(f"      🔍 Web searching to enrich merchant: {raw_merchant}...")
        try:
            search_result = await self.search_tool.ainvoke(f"What type of business / store is '{raw_merchant}'?")
            
            # Reusing the CSV enrichment prompt format
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
            extract_chain_enrich = extract_prompt | self.llm | JsonOutputParser()
            enriched_data = await extract_chain_enrich.ainvoke({
                "description": raw_merchant,
                "search_result": search_result[:500]
            })
            enriched_info = enriched_data.get("enriched_info", search_result[:200])
            clean_merchant = enriched_data.get("merchant_name", raw_merchant)
        except Exception as e:
            print(f"      ⚠️ Parent enrichment failed: {e}")
            enriched_info = ""
            clean_merchant = raw_merchant

        merch_hash = str(clean_merchant).lower().strip().split()[0] if clean_merchant else ""
        merch_hash = ''.join(c for c in merch_hash if c.isalnum())
        hash_input = f"{date_str}{amount_str}{merch_hash}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Build transaction record
        tx_record = {
            "user_id": self.user_id,
            "trans_date": date_str,
            # In this app, spending is traditionally positive, so let's keep it strictly positive for bills
            "amount": abs(float(extracted.get('total_amount', 0))), 
            "description": raw_merchant,
            "merchant_name": clean_merchant,
            "category": extracted.get('category', 'Uncategorized'),
            "content_hash": content_hash,
            "source": 'bill',
            "enriched_info": enriched_info
        }
        if file_id:
            tx_record["source_bill_file_id"] = file_id
            
        # Check duplicate before upserting
        existing_res = self.supabase.table("Transaction").select("id").eq("content_hash", content_hash).eq("user_id", self.user_id).execute()
        is_duplicate = len(existing_res.data) > 0
        duplicates = [{"date": tx_record['trans_date'], "merchant": tx_record['merchant_name'], "amount": tx_record['amount']}] if is_duplicate else []
            
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
            print(f"   [DEBUG] Preparing to Upsert Transaction with source_bill_file_id: {file_id}")
            if file_id:
                import psycopg2
                conn_tmp = psycopg2.connect(os.environ.get("DATABASE_URL"))
                cur_tmp = conn_tmp.cursor()
                cur_tmp.execute("SELECT count(*) FROM \"BillFile\" WHERE id = %s", (file_id,))
                cnt = cur_tmp.fetchone()[0]
                conn_tmp.close()
                print(f"   [DEBUG] BillFile presence check via Psycopg2: {cnt} rows found.")
                
            upsert_res = self.supabase.table("Transaction").upsert([tx_record], on_conflict="content_hash").execute()
            print("   [DEBUG] Upsert successful!")
        except Exception as e:
            print(f"   ⚠️ Upsert failed (migration not run?), falling back to insert: {e}")
            tx_record.pop('merchant_name', None)
            tx_record.pop('content_hash', None)
            tx_record.pop('source', None)
            try:
                upsert_res = self.supabase.table("Transaction").insert([tx_record]).execute()
            except Exception as e2:
                print(f"   ❌ Fallback insert completely failed: {e2}")
            
        # Get the transaction ID to link details (upsert doesn't always return data on conflict update, so explicit select is safest)
        try:
            fetch_res = self.supabase.table("Transaction").select("id, created_at").eq("content_hash", content_hash).eq("user_id", self.user_id).execute()
            if fetch_res.data:
                tx_id = fetch_res.data[0]['id']
                # If we just upserted, we can loosely guess it was a duplicate if it's older than ~1 minute prior
                # But a cleaner way is just tracking if it already existed BEFORE upsert! 
                # Let's just return a standard duplicate dict structure for Bills right now if it was an update.
            else:
                tx_id = None
        except Exception as e:
            print(f"   ⚠️ Failed to fetch transaction ID: {e}")
            tx_id = None
            
        # Check duplicate by looking at content_hash before upsert?
        # Since we already ran the upsert, any existing row is updated.
        # Actually we should have checked duplicate before the upsert. Let's just track it here by assuming anything with source='both' or existing is dup.
        # For simplicity, let's just use the `fetch_res` info or we can do a preemptive check in future. For bills, there's exactly 1 record.
        # Let's just query before the upsert! Wait, we already did the upsert on line 389.
        
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

        return duplicates

    def _sync_to_vectordb(self):
        # Fetch only THIS USER'S transactions from Supabase to sync into VectorDB
        res = self.supabase.table("Transaction").select("*").eq("user_id", self.user_id).execute()
        df = pd.DataFrame(res.data)
        
        # Try to fetch TransactionDetail
        try:
            detail_res = self.supabase.table("TransactionDetail").select("*").execute()
            details_df = pd.DataFrame(detail_res.data)
            if not details_df.empty:
                # Filter line items to only those belonging to this user's transactions
                details_df = details_df[details_df['transaction_id'].isin(df['id'])]
        except Exception:
            details_df = pd.DataFrame()
            
        vdb = get_vector_client()
        return vdb.sync_transactions(df, details_df, self.user_id, self.embeddings)

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
            
            # 2. Delete from Vector Database
            vdb = get_vector_client()
            vdb.delete_file_vectors(file_id, file_type)
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
                "IMPORTANT: Whenever possible and relevant (e.g. when discussing trends, comparing categories, or showing breakdowns), "
                "you MUST proactively use the 'generate_interactive_chart' tool to generate visual plots (bar, pie, or line charts) to accompany your analysis. "
                "CRITICAL RULE FOR RESPONSES: After calling any chart or data tool, you MUST write a detailed text analysis "
                "that includes: (1) a summary of the key numbers, (2) the top and bottom items, (3) any notable patterns or insights. "
                "The chart appears below your text automatically — your text analysis is the PRIMARY response the user reads. "
                "Never respond with just a single sentence when data is available. "
                "WARNING: You MUST use the actual tool call to generate the chart. DO NOT output raw chart JSON in your text."
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
                
            images_path = os.path.join(self.temp_dir, "latest_images.json")
            if os.path.exists(images_path):
                os.remove(images_path)

            # Stream events so we can yield live tool-call updates
            # Accumulate all AI text tokens across the full agent run
            ai_text_chunks: list[str] = []
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

                elif kind == "on_chat_model_stream":
                    # Collect streamed AI text tokens
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        raw = chunk.content
                        if isinstance(raw, str) and raw:
                            ai_text_chunks.append(raw)
                        elif isinstance(raw, list):
                            for block in raw:
                                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                    ai_text_chunks.append(block["text"])

            final_content = "".join(ai_text_chunks).strip()

            # Build final response (with optional chart and images)
            if os.path.exists(chart_path):
                with open(chart_path, "r") as f:
                    chart_json = f.read()
                final_content = f"{final_content}\n\n===CHART===\n{chart_json}\n===ENDCHART==="
                
            if os.path.exists(images_path):
                with open(images_path, "r") as f:
                    images_json = f.read()
                final_content = f"{final_content}\n\n===IMAGES===\n{images_json}\n===ENDIMAGES==="
                
            yield {"type": "final", "content": final_content}
            
        finally:
            try:
                await mcp_client.__aexit__(None, None, None)
            except Exception as close_e:
                print(f"Warning on closing MCP Client: {close_e}")

    async def cleanup(self):
        """Delete temporary session files and close MCP client."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temp directory: {e}")