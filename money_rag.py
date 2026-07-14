import os
import sys
import uuid
import asyncio
import pandas as pd
import sqlite3
import shutil
import tempfile
import re
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
    def __init__(self, llm_provider: str, model_name: str, embedding_model_name: str, api_key: str, user_id: str, access_token: str = None, deep_enrichment: bool = False):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.user_id = user_id
        self.access_token = access_token
        self.deep_enrichment = deep_enrichment
        # Initialize Supabase Client (always needed for auth, storage, and data)
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
        
        self.db: Optional[SQLDatabase] = None
        self.vector_store_client = None
        self.agent = None
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.search_tool = DuckDuckGoSearchRun()
        self.merchant_cache = {}  # Session-based cache for merchant enrichment
        self.memory = InMemorySaver()  # Session-based cache for chat memory

    # --- Database abstraction helpers (Supabase vs Databricks) ---

    def _db_select(self, table: str, columns: str = "*", filters: dict = None) -> List[dict]:
        """SELECT rows from a table. Returns list of dicts."""
        q = self.supabase.table(table).select(columns)
        if filters:
            for k, v in filters.items():
                q = q.eq(k, v)
        res = q.execute()
        return res.data or []

    def _db_select_in(self, table: str, columns: str, field: str, values_list: list) -> List[dict]:
        """SELECT rows WHERE field IN (...)."""
        if not values_list:
            return []
        res = self.supabase.table(table).select(columns).in_(field, values_list).execute()
        return res.data or []

    def _db_upsert(self, table: str, records: List[dict], conflict_key: str = None):
        """Upsert records into a table."""
        if conflict_key:
            self.supabase.table(table).upsert(records, on_conflict=conflict_key).execute()
        else:
            self.supabase.table(table).insert(records).execute()

    def _db_insert(self, table: str, records: List[dict]):
        """Insert records into a table."""
        self.supabase.table(table).insert(records).execute()

    def _db_delete(self, table: str, filters: dict):
        """Delete rows matching filters."""
        q = self.supabase.table(table).delete()
        for k, v in filters.items():
            q = q.eq(k, v)
        q.execute()

    def _db_update(self, table: str, data: dict, filters: dict):
        """Update rows matching filters."""
        q = self.supabase.table(table).update(data)
        for k, v in filters.items():
            q = q.eq(k, v)
        q.execute()

    def _link_new_csv_transactions(self, csv_id: str):
        """Link, but never merge/delete, high-confidence matches from another
        CSV or a receipt. This intentionally skips rows from the CSV currently
        being ingested so identical purchases inside one statement remain distinct.
        """
        if not csv_id:
            return
        columns = "id,trans_date,amount,merchant_name,description,enriched_info,source,source_csv_id"
        rows = self._db_select("Transaction", columns, {"user_id": self.user_id})
        new_rows = [row for row in rows if row.get("source_csv_id") == csv_id]

        def merchant_key(row):
            value = str(row.get("merchant_name") or row.get("description") or "").lower()
            return re.sub(r"[^a-z]", "", value)

        def matches(left, right):
            left_merchant, right_merchant = merchant_key(left), merchant_key(right)
            if len(left_merchant) < 4 or len(right_merchant) < 4:
                return False
            # Receipt OCR often drops store numbers/suffixes, so allow one
            # normalized merchant value to contain the other.
            if left_merchant not in right_merchant and right_merchant not in left_merchant:
                return False
            try:
                date_gap = abs((pd.to_datetime(left["trans_date"]).date() - pd.to_datetime(right["trans_date"]).date()).days)
                amount_gap = abs(float(left["amount"]) - float(right["amount"]))
            except (TypeError, ValueError):
                return False
            # Bank posting dates may differ by one day; allow small receipt/bank
            # rounding differences but avoid merging independently made purchases.
            return date_gap <= 1 and amount_gap <= 0.10

        links = []
        for row in new_rows:
            for candidate in rows:
                if candidate["id"] == row["id"] or candidate.get("source_csv_id") == csv_id:
                    continue
                if candidate.get("source") not in ("csv", "bill") or not matches(row, candidate):
                    continue
                left_id, right_id = sorted((str(row["id"]), str(candidate["id"])))
                links.append({
                    "user_id": self.user_id,
                    "transaction_id": left_id,
                    "linked_transaction_id": right_id,
                    "match_type": "csv_receipt" if candidate.get("source") == "bill" else "csv_csv",
                    "confidence": 1,
                })
                # CSV enrichment contains a useful merchant description. When
                # it is linked to a receipt that has not been deep-enriched,
                # carry that description to the receipt shown in the UI.
                if (
                    candidate.get("source") == "bill"
                    and not candidate.get("enriched_info")
                    and row.get("enriched_info")
                ):
                    self._db_update(
                        "Transaction", {"enriched_info": row["enriched_info"]}, {"id": candidate["id"]}
                    )
        if links:
            self._db_upsert(
                "TransactionLink", links,
                conflict_key="user_id,transaction_id,linked_transaction_id",
            )

    async def setup_session(self, uploaded_files: List[dict]):
        """Ingest CSVs and extract receipt drafts for the user to review."""
        # uploaded_files format: [{"path": "/temp/file.csv", "file_id": "uuid"}, ...]
        all_duplicates = []
        receipt_review_file_ids = []
        has_committed_transactions = False
        committed_csv_ids = []
        for file_info in uploaded_files:
            file_path = file_info["path"]
            file_name = os.path.basename(file_path)
            ext = file_path.lower().split('.')[-1]
            try:
                if ext in ['png', 'jpg', 'jpeg']:
                    dups = await self._ingest_bill(file_path, file_info.get("file_id"))
                    if file_info.get("file_id"):
                        receipt_review_file_ids.append(file_info["file_id"])
                else:
                    dups = await self._ingest_csv(file_path, file_info.get("file_id"))
                    has_committed_transactions = True
                    if file_info.get("file_id"):
                        committed_csv_ids.append(str(file_info["file_id"]))
                if dups:
                    all_duplicates.extend(dups)
            except Exception as e:
                raise RuntimeError(f"Failed to ingest '{file_name}': {e}") from e

        # Receipt data is deliberately not indexed until the user reviews it.
        # A normal CSV upload indexes only the rows from that file. An empty
        # file list is used by the settings flow to rebuild all vectors after
        # an embedding-model change.
        if committed_csv_ids:
            try:
                self._emit_progress("embedding", 0, 0, "Indexing newly uploaded transactions...")
                self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
                self.vector_store_client = self._sync_csv_files_to_vectordb(committed_csv_ids)
                self._emit_progress("embedding", 1, 1, "New transaction vectors complete")
            except Exception as e:
                raise RuntimeError(f"Failed to sync to vector store: {e}") from e
        elif has_committed_transactions or not uploaded_files:
            try:
                self._emit_progress("embedding", 0, 0, "Rebuilding vector database...")
                self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
                self.vector_store_client = self._sync_to_vectordb()
                self._emit_progress("embedding", 1, 1, "Vector rebuild complete")
            except Exception as e:
                raise RuntimeError(f"Failed to sync to vector store: {e}") from e
        return all_duplicates, receipt_review_file_ids

    def _emit_progress(self, stage: str, total: int, done: int, detail: str = ""):
        """Emit a progress update as a JSON line to stderr for the parent process to parse."""
        import json as _json
        import sys as _sys
        progress = {"progress": {"stage": stage, "total": total, "done": done}}
        if detail:
            progress["progress"]["detail"] = detail
        _sys.stderr.write(_json.dumps(progress) + "\n")
        _sys.stderr.flush()

    async def _ingest_csv(self, file_path, csv_id=None):
        import hashlib
        import json

        filename = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Cannot read CSV file: {e}") from e

        total_rows = len(df)
        self._emit_progress("parsing", total_rows, 0, f"Read {total_rows} rows from {filename}")

        headers = df.columns.tolist()
        sample_data = df.head(10).to_json()

        prompt = ChatPromptTemplate.from_template("""
        Act as a financial data parser. Analyze this CSV data:
        Filename: {filename}
        Headers: {headers}
        Sample Data: {sample}

        TASK:
        1. Map the CSV columns to standard fields: date, description, amount, category, and location.
           - 'location' is optional: map it only if a column clearly holds a place (city, state, or address). Otherwise return null.
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
        "location_col": "column_name or null",
        "sign_convention": "spending_is_negative" | "spending_is_positive"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            mapping = await chain.ainvoke({"headers": headers, "sample": sample_data, "filename": filename})
        except Exception as e:
            raise RuntimeError(f"LLM column mapping failed (headers: {headers}): {e}") from e

        self._emit_progress("parsing", total_rows, total_rows, "Column mapping complete")

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

        loc_col = mapping.get('location_col')
        standard_df['location'] = df[loc_col] if loc_col and loc_col in df.columns else None

        # --- Async Enrichment Step (batched for speed) ---
        unique_descriptions = standard_df['description'].unique().tolist()
        total_unique = len(unique_descriptions)
        self._emit_progress("enriching", total_unique, 0, f"{total_unique} unique merchants to enrich")
        print(f"   ✨ Enriching {total_unique} unique descriptions for {filename}...")

        sem = asyncio.Semaphore(15)  # Higher concurrency for faster enrichment
        enriched_count = 0

        # Batch enrichment prompt — process up to 10 descriptions at once
        batch_extract_prompt = ChatPromptTemplate.from_template("""
You are a financial data assistant. For each transaction description below, extract the clean merchant name and a one-sentence description of the business.

Transaction descriptions:
{descriptions_json}

Return ONLY a valid JSON array with one object per description, in the same order:
[
  {{"description": "<original description>", "merchant_name": "<clean 1-4 word name>", "enriched_info": "<one sentence about the business>"}},
  ...
]
""")
        batch_extract_chain = batch_extract_prompt | self.llm | JsonOutputParser()

        async def enrich_batch(descriptions_batch):
            nonlocal enriched_count
            # Check cache first, only process uncached
            results = {}
            uncached = []
            for desc in descriptions_batch:
                if desc in self.merchant_cache:
                    results[desc] = self.merchant_cache[desc]
                else:
                    uncached.append(desc)

            if uncached:
                async with sem:
                    try:
                        # Deep enrichment: do web searches first for extra context
                        search_context = ""
                        if self.deep_enrichment:
                            search_results = {}
                            for desc in uncached:
                                try:
                                    sr = await self.search_tool.ainvoke(f"What type of business is '{desc}'?")
                                    search_results[desc] = sr[:200]
                                except Exception:
                                    search_results[desc] = ""
                            if any(search_results.values()):
                                search_context = "\n\nWeb search context:\n" + "\n".join(
                                    f"- {d}: {s}" for d, s in search_results.items() if s
                                )

                        descriptions_json = json.dumps(uncached)
                        prompt_input = {"descriptions_json": descriptions_json + search_context}
                        structured_list = await batch_extract_chain.ainvoke(prompt_input)
                        if isinstance(structured_list, list):
                            for item in structured_list:
                                desc = item.get("description", "")
                                result = {
                                    "merchant_name": item.get("merchant_name", desc),
                                    "enriched_info": item.get("enriched_info", ""),
                                }
                                # Match to uncached by best effort
                                matched_desc = desc
                                for uc in uncached:
                                    if uc.lower().strip() == desc.lower().strip() or uc in desc or desc in uc:
                                        matched_desc = uc
                                        break
                                results[matched_desc] = result
                                self.merchant_cache[matched_desc] = result
                        # Fill any that weren't matched
                        for desc in uncached:
                            if desc not in results:
                                results[desc] = {"merchant_name": desc, "enriched_info": ""}
                                self.merchant_cache[desc] = results[desc]
                    except Exception as e:
                        print(f"      ⚠️ Batch enrichment failed: {e}")
                        for desc in uncached:
                            results[desc] = {"merchant_name": desc, "enriched_info": ""}

            enriched_count += len(descriptions_batch)
            self._emit_progress("enriching", total_unique, enriched_count)
            return results

        # Process in batches of 10
        BATCH_SIZE = 10
        desc_map = {}
        batch_tasks = []
        for i in range(0, total_unique, BATCH_SIZE):
            batch = unique_descriptions[i:i + BATCH_SIZE]
            batch_tasks.append(enrich_batch(batch))

        batch_results = await asyncio.gather(*batch_tasks)
        for batch_result in batch_results:
            desc_map.update(batch_result)

        standard_df['enriched_info'] = standard_df['description'].map(
            lambda d: desc_map.get(d, {}).get("enriched_info", "")
        )
        standard_df['merchant_name'] = standard_df['description'].map(
            lambda d: desc_map.get(d, {}).get("merchant_name", d)
        )

        print(f"   ✅ Enriched {total_unique} unique merchants.")

        # Save to Supabase transactions table
        self._emit_progress("saving", total_rows, 0, "Saving transactions to database")
        records = json.loads(standard_df.to_json(orient='records'))

        # Calculate content_hash and source for deduplication.
        def base_sig(row):
            date_str = str(row['trans_date']).strip()
            amount_str = str(round(float(row['amount']), 2))
            words = str(row.get('merchant_name', row['description'])).lower().strip().split()
            merch = ''.join(c for c in (words[0] if words else "") if c.isalnum())
            # The source file id makes every CSV row durable. A later CSV export
            # can be linked to it, but must never overwrite or remove it.
            return f"{csv_id or filename}|{date_str}|{amount_str}|{merch}"

        # An occurrence counter (in file order) distinguishes genuinely-distinct rows
        # that share date+amount+merchant (e.g. two $5 coffees the same day), while
        # staying stable when the SAME file is re-uploaded so real re-uploads still dedup.
        occ = {}
        for r in records:
            sig = base_sig(r)
            n = occ.get(sig, 0)
            occ[sig] = n + 1
            r['content_hash'] = hashlib.sha256(f"{sig}#{n}".encode()).hexdigest()
            r['source'] = 'csv'

        # Detect duplicates before upserting
        hashes = [r['content_hash'] for r in records]
        existing_hashes = set()
        for i in range(0, len(hashes), 100):
            batch_hashes = hashes[i:i+100]
            existing_rows = self._db_select_in("Transaction", "content_hash", "content_hash", batch_hashes)
            existing_hashes.update(row['content_hash'] for row in existing_rows)

        duplicates = [
            {"date": r['trans_date'], "merchant": r.get('merchant_name', r.get('description', '')), "amount": r['amount']}
            for r in records if r['content_hash'] in existing_hashes
        ]

        batch_size = 100
        saved_rows = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                self._db_upsert("Transaction", batch, conflict_key="user_id,content_hash")
            except Exception as e:
                # Fallback if DB migration hasn't been run yet (no content_hash / merchant_name)
                for r in batch:
                    r.pop('merchant_name', None)
                    r.pop('content_hash', None)
                    r.pop('source', None)
                try:
                    self._db_insert("Transaction", batch)
                except Exception as ex:
                    print(f"Failed fallback insert: {ex}")
            saved_rows += len(batch)
            self._emit_progress("saving", total_rows, saved_rows)

        try:
            self._link_new_csv_transactions(csv_id)
        except Exception as e:
            # Linking is supplementary; never reject a valid CSV upload if the
            # migration has not yet been applied or matching fails.
            print(f"   ⚠️ Transaction linking skipped: {e}")

        return duplicates

    async def _ingest_bill(self, file_path, file_id=None):
        import base64
        import json
        from langchain_core.messages import HumanMessage
        
        filename = os.path.basename(file_path)
        self._emit_progress("parsing", 1, 0, f"Reading image: {filename}")
        
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        ext = file_path.lower().split('.')[-1]
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

        schema = {
            "date": "YYYY-MM-DD",
            "time": "HH:MM in 24-hour local receipt time, or null if not visible",
            "merchant_name": "Merchant",
            "category": "One of: Groceries, Dining, Shopping, Transportation, Utilities, Healthcare, Entertainment, Travel, Personal Care, or Uncategorized",
            "location": "Store address or city/state if visible on the receipt, else null",
            "subtotal": 100.00,
            "tax_lines": [{"label": "Sales Tax", "rate": 8.25, "amount": 8.25}],
            "total_amount": 108.25,
            "discounts": [{"label": "Coupon", "amount": 5.00}],
            "line_items": [{"item_description": "Item", "item_quantity": 1, "item_unit_price": 10.0, "item_savings": 0.0, "tax_rate": 8.25}]
        }

        # Vision extraction — single LLM call to extract all structured data
        self._emit_progress("parsing", 1, 0, "Extracting data from receipt...")
        prompt = (
            "Extract structured data from this receipt/bill. Return strictly valid JSON matching this schema: "
            f"{json.dumps(schema)}.\n"
            "- 'total_amount' is the final amount actually charged (the balance/total paid).\n"
            "- 'subtotal' is the pre-tax sum of what was paid, AFTER any item markdowns.\n"
            "- Extract the receipt's purchase time when shown, normalized to 24-hour HH:MM. "
            "Use null when the time is not visible.\n"
            "- Choose category only from the category list in the schema; use Uncategorized if unsure.\n"
            "- 'tax_lines': one entry per DISTINCT tax rate shown. Receipts may have several "
            "(e.g. 0% for groceries and 8.25% for general goods). Use [] if no tax is shown.\n"
            "- Different item TYPES can carry different tax rates (e.g. groceries are often "
            "exempt at 0% while general goods are taxed, and this varies by locale). Set each "
            "line item's 'tax_rate' to the percentage applied to THAT item (0 if not taxed).\n"
            "DISCOUNTS AND SAVINGS — read carefully, they are commonly misread:\n"
            "- 'item_unit_price' MUST be the price ACTUALLY PAID per unit. When an item shows a "
            "markdown (a 'SAVINGS x.xx-' line and/or a 'PRICE YOU PAY' amount below its regular "
            "price), use the reduced/'PRICE YOU PAY' figure, NOT the higher struck/regular price.\n"
            "- 'item_savings': when a line shows a markdown — a 'SAVINGS x.xx-' amount printed under "
            "it and/or a struck regular price above its 'PRICE YOU PAY' — set item_savings to that "
            "printed SAVINGS figure as a POSITIVE number (e.g. 'SAVINGS 1.95-' -> 1.95). Copy the "
            "number shown on the receipt; do not compute it. Use 0 only when no savings/markdown is "
            "printed for that line. It is informational only — the price is already net, so never "
            "subtract item_savings from any total.\n"
            "- 'discounts': ONLY order-level coupons that reduce the whole basket (e.g. '$5 off $50', "
            "a store coupon applied at the end). These ARE subtracted from the total. Use [] if none. "
            "Do NOT put per-item markdowns here.\n"
            "- IGNORE the summary recap block ('SAVINGS SUMMARY', 'Card Savings', 'Your Total "
            "Savings', 'CARD SAVINGS', loyalty points). It only restates savings already applied to "
            "the item prices — it is NOT a line item, tax, or discount and must never be subtracted."
        )
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
            
        self._emit_progress("parsing", 1, 1, f"Extracted: {extracted.get('merchant_name')}")
        
        # Save the raw OCR JSON back to the BillFile record
        if file_id:
            try:
                import re
                extension = os.path.splitext(filename)[1] or ".jpg"
                merchant_slug = re.sub(
                    r"[^A-Za-z0-9]+", "_", str(extracted.get("merchant_name") or "receipt")
                ).strip("_") or "receipt"
                date_slug = str(extracted.get("date") or "nodate").replace("-", "")
                time_slug = re.sub(r"[^0-9]", "", str(extracted.get("time") or ""))
                friendly_name = f"{merchant_slug}_{date_slug}{f'_{time_slug}' if time_slug else ''}{extension}"
                # Only the display name changes; s3_key remains stable for previews.
                self._db_update("BillFile", {
                    "raw_ocr_string": json.dumps(extracted),
                    "filename": friendly_name,
                }, {"id": file_id})
            except Exception as e:
                print(f"   ⚠️ Failed to save OCR receipt data: {e}")

        # Do not create a transaction from OCR automatically. The mobile app
        # presents this draft to the user, who may correct it before verifying.
        self._emit_progress("review", 1, 1, "Receipt extracted — waiting for review")
        return []

    def _sync_to_vectordb(self):
        # Fetch only THIS USER'S transactions to sync into VectorDB
        rows = self._db_select("Transaction", "*", {"user_id": self.user_id})
        df = pd.DataFrame(rows)

        # Try to fetch TransactionDetail
        try:
            detail_rows = self._db_select("TransactionDetail", "*", {"user_id": self.user_id})
            details_df = pd.DataFrame(detail_rows)
            if not details_df.empty and not df.empty:
                details_df = details_df[details_df['transaction_id'].isin(df['id'])]
        except Exception:
            details_df = pd.DataFrame()

        def _progress(detail, total, done):
            self._emit_progress("embedding", total, done, detail)

        vdb = get_vector_client()
        return vdb.sync_transactions(df, details_df, self.user_id, self.embeddings, progress_callback=_progress)

    def _sync_csv_files_to_vectordb(self, csv_ids: List[str]):
        """Incrementally embed only transactions belonging to newly uploaded CSVs."""
        rows = self._db_select("Transaction", "*", {"user_id": self.user_id})
        csv_id_set = {str(csv_id) for csv_id in csv_ids}
        new_rows = [row for row in rows if str(row.get("source_csv_id") or "") in csv_id_set]
        df = pd.DataFrame(new_rows)
        if df.empty:
            return None

        def _progress(detail, total, done):
            self._emit_progress("embedding", total, done, detail)

        # CSV transactions do not have receipt line items, so there is no need
        # to fetch or re-embed the user's entire TransactionDetail table.
        return get_vector_client().sync_transactions(
            df, pd.DataFrame(), self.user_id, self.embeddings, progress_callback=_progress
        )

    async def delete_file(self, file_id: str, file_type: str = 'csv'):
        """Force delete a file and all its transactions from the database and vector store."""
        try:
            if file_type == 'csv':
                self._db_delete("Transaction", {"source_csv_id": file_id})
                self._db_delete("CSVFile", {"id": file_id})
            else:
                self._db_delete("TransactionDetail", {"bill_file_id": file_id})
                self._db_delete("BillFile", {"id": file_id})

            # Delete from Vector Database
            vdb = get_vector_client()
            vdb.delete_file_vectors(file_id, file_type)
        except Exception as e:
            print(f"Error purging file data: {e}")

    async def chat(self, query: str, history: Optional[list] = None):
        """Async generator that yields status events + final response.

        `history` is a list of prior {role, content} turns (from the DB) that give
        the agent conversation context. Passed explicitly so memory survives
        restarts rather than living only in the in-process checkpointer.
        """
        server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        
        mcp_client = MultiServerMCPClient(
            {
                "money_rag": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [server_path],
                    "env": {
                        **os.environ.copy(),
                        "CURRENT_USER_ID": self.user_id,
                        "CURRENT_EMBEDDING_PROVIDER": self.llm_provider,
                        "CURRENT_EMBEDDING_MODEL": self.embedding_model_name,
                        # So MCP tools can sign private-bucket storage URLs as this user.
                        "CURRENT_ACCESS_TOKEN": self.access_token or "",
                    },
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
                "WARNING: You MUST use the actual tool call to generate the chart. DO NOT output raw chart JSON in your text.\n"
                "MANUAL TRANSACTIONS: When the user describes a transaction they made (e.g., 'I gave X $100', "
                "'I spent $50 at Target', 'paid rent $1200'), you MUST use the 'propose_transaction' tool. "
                "Extract the amount, description, date (default today), category, and merchant name from the user's message. "
                "Do NOT insert transactions directly — always let the user confirm via the UI card. "
                "CRITICAL: You MUST include the ===CONFIRM_TX=== marker output from the tool in your response EXACTLY as returned. "
                "Do not remove or modify the marker content."
            )
            
            agent = create_agent(
                model=self.llm,
                tools=mcp_tools,
                system_prompt=system_prompt,
                checkpointer=self.memory,
            )

            import uuid
            # Fresh thread per call — conversation memory comes from the passed
            # `history`, so we don't double-accumulate in the checkpointer.
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            input_messages = list(history or []) + [{"role": "user", "content": query}]
            
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
                {"messages": input_messages},
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
