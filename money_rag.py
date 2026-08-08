import os
import sys
import uuid
import asyncio
import logging
import pandas as pd
import sqlite3
import shutil
import tempfile
import re
from datetime import timedelta
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
from backend.services import stream_gate as markers

# Import specific embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from supabase import create_client, ClientOptions

from dotenv import load_dotenv
load_dotenv()

# Ingestion used to print() its progress and swallow its failures into stdout,
# where they are invisible to any log aggregator and carry no level, timestamp
# or user context. Progress for the UI still goes to stderr as JSON — see
# _emit_progress — which is a separate channel with a parser on the other end.
logger = logging.getLogger("moneyrag.money_rag")


def _chunk_calls_tool(chunk) -> bool:
    """True when this streamed chunk is part of the model choosing a tool.

    Providers signal it differently — LangChain normalises most into
    `tool_call_chunks`, but Gemini and older OpenAI paths put it in
    additional_kwargs — so check both rather than trusting one shape.
    """
    if getattr(chunk, "tool_call_chunks", None):
        return True
    if getattr(chunk, "tool_calls", None):
        return True
    extra = getattr(chunk, "additional_kwargs", None) or {}
    return bool(extra.get("function_call") or extra.get("tool_calls"))


def _chunk_text(chunk) -> str:
    """The human-readable text in a streamed chunk, if any.

    `content` is a plain string for most providers and a list of typed blocks
    for others; only the text blocks are the answer.
    """
    raw = getattr(chunk, "content", None)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [
            block["text"]
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ]
        return "".join(parts)
    return ""



# Bank posting dates slip by a day, and a receipt total can round against the
# statement by a penny or two. Wider than this starts matching purchases that
# genuinely happened separately.
MAX_LINK_DAY_GAP = 1
MAX_LINK_AMOUNT_GAP = 0.10
# Shorter names identify nothing: "bp", "atm" and "h&m" would match half a
# statement between them.
MIN_MERCHANT_KEY = 4


def _merchant_key(row: dict) -> str:
    value = str(row.get("merchant_name") or row.get("description") or "").lower()
    return re.sub(r"[^a-z]", "", value)


def pair_same_purchase(rows: List[dict], csv_id: str) -> List[tuple]:
    """Pair rows from CSV `csv_id` with earlier rows describing the same purchase.

    Returns (new_row, existing_row) pairs, at most one per row on either side.

    That one-to-one rule is the whole point. The obvious version — emit every
    pair that matches — chains: three £4.50 coffees on consecutive days all
    match each other inside the ±1 day posting window, and because the
    transactions list collapses a linked component into a single row, three
    purchases displayed as one and the month under-reported. Here each row may
    claim at most one partner, and the closest pairings are settled first.

    Pure and database-free so the pairing can actually be tested; the caller
    turns the result into TransactionLink rows.
    """
    prepared = []
    for row in rows:
        key = _merchant_key(row)
        if len(key) < MIN_MERCHANT_KEY:
            continue
        try:
            when = pd.to_datetime(row["trans_date"]).date()
            amount = float(row["amount"])
        except (TypeError, ValueError):
            continue
        prepared.append({"row": row, "key": key, "date": when, "amount": amount})

    # Bucketed by date, the tightest filter available, so each new row consults
    # three buckets instead of the entire history. The merchant key and the
    # parsed date are computed once per row rather than once per comparison —
    # they used to sit inside the inner loop, which meant a regex and two date
    # parses per pair, millions of them on a long history.
    by_date: dict = {}
    for info in prepared:
        if info["row"].get("source_csv_id") == csv_id:
            continue
        if info["row"].get("source") not in ("csv", "bill"):
            continue
        by_date.setdefault(info["date"], []).append(info)

    # Score every allowable pairing, then settle them best-first. Scored
    # globally rather than row by row: taking the first match in file order
    # would let an early row claim a partner that fits a later one better.
    scored = []
    for info in prepared:
        if info["row"].get("source_csv_id") != csv_id:
            continue
        for offset in range(-MAX_LINK_DAY_GAP, MAX_LINK_DAY_GAP + 1):
            for other in by_date.get(info["date"] + timedelta(days=offset), []):
                if str(other["row"]["id"]) == str(info["row"]["id"]):
                    continue
                # Receipt OCR often drops store numbers and suffixes, so one
                # normalised name containing the other still counts.
                if info["key"] not in other["key"] and other["key"] not in info["key"]:
                    continue
                amount_gap = abs(info["amount"] - other["amount"])
                if amount_gap > MAX_LINK_AMOUNT_GAP:
                    continue
                scored.append((
                    (
                        abs(offset),                                  # same day first
                        round(amount_gap, 2),                         # then closest amount
                        0 if info["key"] == other["key"] else 1,      # then exact name
                        str(info["row"]["id"]),                       # deterministic tiebreak
                        str(other["row"]["id"]),
                    ),
                    info["row"],
                    other["row"],
                ))
    scored.sort(key=lambda item: item[0])

    claimed = set()
    pairs = []
    for _, new_row, other in scored:
        new_id, other_id = str(new_row["id"]), str(other["id"])
        if new_id in claimed or other_id in claimed:
            continue
        claimed.add(new_id)
        claimed.add(other_id)
        pairs.append((new_row, other))
    return pairs


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
        
        # Keys are passed straight to each client rather than exported to
        # os.environ: one process serves every logged-in user, so a global would
        # mean whoever constructed a MoneyRAG last decides which key everyone's
        # requests are billed to.
        self.api_key = api_key
        if self.llm_provider == "google":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model_name, google_api_key=api_key
            )
            provider_name = "google_genai"
        else:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model_name, openai_api_key=api_key
            )
            provider_name = "openai"

        # Initialize LLM
        self.llm = init_chat_model(
            self.model_name,
            model_provider=provider_name,
            api_key=api_key,
        )

        # Temporary paths for this session. temp_dir is handed to the MCP server
        # through its own env at launch (see chat()) — never via os.environ here,
        # which another user's instance would immediately overwrite.
        self.temp_dir = tempfile.mkdtemp()
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
        """Record the pairings found by `pair_same_purchase` as TransactionLinks.

        Links, never merges or deletes: a bank statement line and a receipt are
        both durable records of one purchase and neither is safe to discard.
        """
        if not csv_id:
            return
        columns = "id,trans_date,amount,merchant_name,description,enriched_info,source,source_csv_id"
        rows = self._db_select("Transaction", columns, {"user_id": self.user_id})

        links = []
        for new_row, other in pair_same_purchase(rows, csv_id):
            left_id, right_id = sorted((str(new_row["id"]), str(other["id"])))
            links.append({
                "user_id": self.user_id,
                "transaction_id": left_id,
                "linked_transaction_id": right_id,
                "match_type": "csv_receipt" if other.get("source") == "bill" else "csv_csv",
                "confidence": 1,
            })
            # CSV enrichment carries a useful merchant description. When it is
            # linked to a receipt that has not been deep-enriched, hand it over
            # so the row the UI actually shows is the described one.
            if (
                other.get("source") == "bill"
                and not other.get("enriched_info")
                and new_row.get("enriched_info")
            ):
                self._db_update(
                    "Transaction",
                    {"enriched_info": new_row["enriched_info"]},
                    {"id": other["id"]},
                )
        if links:
            self._db_upsert(
                "TransactionLink", links,
                conflict_key="user_id,transaction_id,linked_transaction_id",
            )

    async def setup_session(self, uploaded_files: List[dict]):
        """Ingest CSVs and extract photo drafts for the user to review.

        Returns (duplicates, review_items) where review_items is
        [{"file_id": ..., "kind": "receipt"|"price_tag"|"unknown"}] — the caller
        routes each photo to the matching review screen, or to the "which is
        this?" prompt when the model could not tell.
        """
        # uploaded_files format: [{"path": "/temp/file.csv", "file_id": "uuid"}, ...]
        all_duplicates = []
        review_items = []
        has_committed_transactions = False
        committed_csv_ids = []
        for file_info in uploaded_files:
            file_path = file_info["path"]
            file_name = os.path.basename(file_path)
            ext = file_path.lower().split('.')[-1]
            try:
                if ext in ['png', 'jpg', 'jpeg']:
                    # First value is the draft, not duplicates — a photo cannot
                    # duplicate anything until it is reviewed, and dedup happens
                    # at verify time against content_hash.
                    _, kind = await self._ingest_bill(file_path, file_info.get("file_id"))
                    dups = []
                    if file_info.get("file_id"):
                        review_items.append({
                            "file_id": str(file_info["file_id"]),
                            "kind": kind,
                        })
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
        return all_duplicates, review_items

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
        logger.info("Enriching %d unique descriptions for %s", total_unique, filename)

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
                        logger.warning("Batch enrichment failed, falling back to raw descriptions: %s", e)
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

        logger.info("Enriched %d unique merchants", total_unique)

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
                # Fallback if DB migration hasn't been run yet (no content_hash /
                # merchant_name). Loud, because rows saved this way carry no
                # content_hash and can therefore never be de-duplicated: the same
                # statement uploaded twice silently doubles the user's spending.
                logger.error(
                    "Transaction upsert failed (%s) — retrying without "
                    "content_hash/merchant_name/source. These %d row(s) will NOT "
                    "de-duplicate on re-upload; check that migrations are applied.",
                    e, len(batch),
                )
                for r in batch:
                    r.pop('merchant_name', None)
                    r.pop('content_hash', None)
                    r.pop('source', None)
                try:
                    self._db_insert("Transaction", batch)
                except Exception as ex:
                    logger.error("Fallback insert also failed, %d row(s) lost: %s", len(batch), ex)
            saved_rows += len(batch)
            self._emit_progress("saving", total_rows, saved_rows)

        try:
            self._link_new_csv_transactions(csv_id)
        except Exception as e:
            # Linking is supplementary; never reject a valid CSV upload if the
            # migration has not yet been applied or matching fails.
            logger.warning("Transaction linking skipped: %s", e)

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
            # item_quantity_unit is what item_quantity counts. A bare "2.25" is
            # ambiguous between 2.25 lb of loose bananas and 2.25 packages, and
            # the two mean different prices per unit — which is the whole basis
            # of comparing a shelf price against what was paid before.
            "line_items": [{"item_description": "Item", "item_quantity": 1, "item_quantity_unit": "each|lb|oz|fl oz|kg|g|ml|l|gal|ct", "size_value": 5, "size_unit": "lb|gal|fl oz|oz|...", "item_unit_price": 10.0, "item_savings": 0.0, "tax_rate": 8.25}]
        }

        # A shelf/price tag is a completely different document: one product, the
        # price it is being offered at, and no purchase. It must never become a
        # transaction — see PriceObservation in migration 015.
        # A LIST, because a shelf photo is usually a shelf: a dairy case shot
        # holds a tag per product. Returning one tag meant silently discarding
        # the rest, and the user could not tell which one had been kept.
        price_tag_schema = {
            "item_description": "Product name as printed on the tag",
            "brand_name": "Brand if distinguishable, else null",
            # Deliberately the same shape as a receipt line, because the two are
            # compared against each other. "$4.29 / 12 OZ" is quantity 12, unit
            # 'oz', unit_quantity_subtotal 0.3575, item_subtotal_price 4.29.
            "size_value": 12,
            "size_unit": "oz|fl oz|lb|g|kg|ml|l|gal|qt|pt|ct|each — the unit the size is printed in",
            "unit_quantity_subtotal": "The per-unit price PRINTED on the tag ('$0.36/OZ'), as a number. null if the tag prints none — do NOT compute it.",
            "unit_price_unit": "The unit that PRINTED per-unit price is quoted in ('$0.36/OZ' -> 'oz'). Often NOT the package unit: a one-gallon milk jug is commonly tagged PER QUART. null if the tag does not say.",
            "item_subtotal_price": "The shelf price the shopper pays for the package, e.g. 4.29",
            "merchant_name": "Store name if visible anywhere in the photo, else null",
            "item_qualitative_description": (
                "Everything the photo shows about this price that is not a number, "
                "in the tag's own words. Offers ('2 for $5', 'BOGO', 'with card'), "
                "sale end wording exactly as printed ('Sale ends 8/15'), any use-by "
                "or best-before on the product, shelf labels like 'CLEARANCE' or "
                "'MANAGER'S SPECIAL', and visible condition such as a dented or "
                "opened package. Quote rather than interpret; do not convert dates "
                "and do not judge whether the price is good. Null if the tag shows "
                "only a name and a price."
            ),
        }

        # Vision extraction — ONE call that both classifies and extracts. A
        # separate classification pass would double latency and cost on the
        # critical path, and the user is standing in a shop waiting.
        self._emit_progress("parsing", 1, 0, "Reading photo...")
        prompt = (
            "You are given ONE photo. First decide what it is, then extract accordingly.\n\n"
            "KIND — choose exactly one:\n"
            "- 'receipt': a proof of purchase. Multiple line items, a total paid, usually tax, "
            "a date/time, often a payment method.\n"
            "- 'price_tag': a shelf label, price sticker, or product tag showing what ONE item "
            "costs. Nothing has been bought. Often shows a per-unit price and a size.\n"
            "- 'unknown': anything else, OR a photo you cannot confidently place in either "
            "category (e.g. a receipt lying on a shelf, a blurred or partial image).\n"
            "Do NOT guess between receipt and price_tag. 'unknown' is the correct, expected "
            "answer when the photo is ambiguous — the user will be asked.\n\n"
            "Return strictly valid JSON of the form: "
            '{"kind": "receipt|price_tag|unknown", "receipt": <receipt object or null>, '
            '"price_tags": <ARRAY of price tag objects, or null>}\n'
            "Populate only the field matching the chosen kind; set the other to null. "
            "For 'unknown', set both to null.\n\n"
            f"PRICE TAG schema — 'price_tags' is an ARRAY of objects shaped like: {json.dumps(price_tag_schema)}\n"
            "- Return EVERY price tag you can read, ordered as they appear left-to-right, "
            "top-to-bottom. A photo of a shelf normally shows several, and returning only "
            "one silently loses the others.\n"
            "- Only include a tag whose product name AND price you can actually read. Skip "
            "any that is cut off, blurred, or angled past legibility rather than guessing: "
            "a misread price is stored as fact and compared against for months, while a "
            "skipped tag costs one more photo.\n"
            "- Do not invent a tag for a product that is merely visible on the shelf. A tag "
            "is a printed label showing a price.\n"
            "- 'item_subtotal_price' is what the shopper pays right now for the package "
            "(the large/highlighted figure). If a member/loyalty price and a regular price are "
            "both shown, use the member price and record the regular one in "
            "'item_qualitative_description'.\n"
            "- 'unit_quantity_subtotal': copy the per-unit price the tag PRINTS ('$0.36/OZ') as "
            "a number. Do NOT compute it — the store already did the pack-size arithmetic for "
            "the exact package on the shelf, and its figure is authoritative. Null if none is "
            "printed.\n"
            "- 'unit_price_unit': what that printed figure is PER, copied from the tag. Stores "
            "very often quote a unit different from the package — a one-gallon milk jug tagged "
            "'UNIT PRICE PER QUART 0.87' is 'qt', not 'gal'. Getting this wrong labels $0.87 as "
            "the price of a whole gallon that costs $3.49. Null if the tag does not say.\n"
            "- 'size_value' / 'size_unit': the package size AS PRINTED, split into a number and "
            "its unit ('12 OZ' -> 12 and 'oz'). Do NOT convert: a tag reading 'Gallon' is 1 "
            "'gal', never 128 'oz'. Converting is done later, exactly, by code — your job is "
            "to record what the label says and to decide WHICH unit it is.\n"
            "- That decision is yours alone and code cannot make it: 'oz' is a WEIGHT and "
            "'fl oz' is a VOLUME, and the two never compare. You can see what the product is, "
            "so use 'fl oz' for anything poured — milk, juice, soda, oil, sauce — and plain "
            "'oz' only for something weighed, like cheese or nuts. Getting this wrong files a "
            "gallon of milk as a weight, and it can then never be compared to another gallon.\n"
            "- 'item_qualitative_description': quote what the tag says, do not interpret it. Copy sale end "
            "wording as printed ('Sale ends 8/15', 'Prices good thru Sunday') rather than "
            "converting to a date — resolving 'Sunday' without a year is guesswork, and a wrong "
            "end date silently turns a limited offer into what the item normally costs. Include "
            "any use-by / best-before printed on the PRODUCT: a markdown is often cheap "
            "precisely because the item expires imminently, and that has to be sayable rather "
            "than read as a bargain. Also include clearance or manager's-special labelling and "
            "visible package damage.\n\n"
            f"RECEIPT schema: {json.dumps(schema)}.\n"
            "- 'total_amount' is the final amount actually charged (the balance/total paid).\n"
            "- 'subtotal' is the pre-tax sum of what was paid, AFTER any item markdowns.\n"
            "- Extract the receipt's purchase time when shown, normalized to 24-hour HH:MM. "
            "Use null when the time is not visible.\n"
            "- Choose category only from the category list in the schema; use Uncategorized if unsure.\n"
            "- 'size_value' / 'size_unit' describe how much is in ONE unit, and are NOT the same "
            "as item_quantity: a 5 lb bag of potatoes is item_quantity 1 'each' with size_value 5 "
            "and size_unit 'lb'. Receipts abbreviate this badly ('+RED POTA 5L US#' is five POUNDS, "
            "not five litres), so read the size from the product name and use null when it is "
            "genuinely unclear rather than guessing.\n"
            "- A NUMBER SITTING NEXT TO A UNIT IS OFTEN A VARIANT, NOT A SIZE. 'GV LF 2 GAL' is "
            "Great Value LOW FAT 2% milk in a ONE gallon jug — size_value 1, size_unit 'gal' — not "
            "two gallons; the same receipt lists 'GV FF GAL' with no number at all. '1% MILK', "
            "'2 PCT', '100 CALORIE' and grades like 'AA' behave the same way. You know what the "
            "product IS, which is the only way to tell: ask whether the number describes the "
            "CONTENTS or the CONTAINER, and use null if you cannot tell. Reading a fat percentage "
            "as a pack size halves the unit price and makes an ordinary shelf price look like a "
            "rip-off.\n"
            "- 'item_quantity_unit' is what 'item_quantity' counts. Weighed items show a "
            "fractional quantity with a per-weight price ('2.25 lb @ $0.50/lb') — use 'lb' "
            "(or kg/oz/g as printed). Packaged items are counted, so use 'each'. This is not "
            "the package size: a single 30-count bag of cilantro is quantity 1 'each', not 30 "
            "'ct'. Use null when the receipt genuinely does not say.\n"
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
            classified = await extract_chain.ainvoke([message])
        except Exception as e:
            logger.error("Vision extraction failed for %s: %s", filename, e)
            return {}, "unknown"

        kind, extracted = self._split_classified_photo(classified)

        # The draft stored for review is the branch that matched, so the review
        # screens keep receiving the same flat shape they always have.
        if kind == "price_tag":
            tags = extracted.get("tags") or []
            first = (tags[0].get("item_description") if tags else None) or "item"
            label = f"{first} +{len(tags) - 1} more" if len(tags) > 1 else first
        else:
            label = extracted.get("merchant_name") or "photo"
        self._emit_progress("parsing", 1, 1, f"Read {kind.replace('_', ' ')}: {label}")

        # Save the extracted draft and what kind of photo this is.
        if file_id:
            try:
                self._db_update("BillFile", {
                    "raw_ocr_string": json.dumps(extracted),
                    "kind": kind,
                    # Only the display name changes; s3_key remains stable for previews.
                    "filename": self._photo_filename(filename, kind, extracted),
                }, {"id": file_id})
            except Exception as e:
                logger.error("Failed to save extracted photo data for file_id=%s: %s", file_id, e)

        # Nothing is committed from OCR automatically — for either kind. The app
        # shows the draft and the user corrects it before anything is written.
        #
        # The draft is RETURNED as well as stored, because `file_id` is optional:
        # a chat capture has no row yet — a shelf price gets one only if the user
        # confirms it — so the caller needs the draft handed back directly.
        self._emit_progress("review", 1, 1, f"{kind.replace('_', ' ').title()} read — waiting for review")
        return extracted, kind

    @staticmethod
    def _split_classified_photo(classified: dict) -> tuple[str, dict]:
        """Pull (kind, draft) out of the vision response.

        Tolerates a model that ignores the envelope and returns a bare receipt
        object, which is what the previous single-purpose prompt always produced
        — treating that as a receipt keeps older behaviour working.
        """
        if not isinstance(classified, dict):
            return "unknown", {}

        kind = str(classified.get("kind") or "").strip().lower()
        if kind == "price_tag":
            # Accepts the array, and a lone object from a model that ignored the
            # instruction — one tag read is still worth keeping.
            payload = classified.get("price_tags")
            if payload is None:
                payload = classified.get("price_tag")
            tags = MoneyRAG._as_tag_list(payload)
            return ("price_tag", {"tags": tags}) if tags else ("unknown", {})

        if kind in ("receipt", "unknown"):
            payload = classified.get(kind) if kind != "unknown" else None
            if isinstance(payload, dict) and payload:
                return kind, payload
            # A declared kind with an empty body gives the review screen nothing
            # to show, so it is treated as undecided and the user is asked —
            # the same rule as an ambiguous photo, rather than opening a blank
            # form and inviting a confirmed-but-empty record.
            return "unknown", {}

        # No envelope: infer from the keys present rather than discarding it.
        if "line_items" in classified or "total_amount" in classified:
            return "receipt", classified
        if "price_tags" in classified:
            tags = MoneyRAG._as_tag_list(classified.get("price_tags"))
            return ("price_tag", {"tags": tags}) if tags else ("unknown", {})
        if "item_subtotal_price" in classified or "unit_quantity_subtotal" in classified:
            return "price_tag", {"tags": [classified]}
        return "unknown", classified

    @staticmethod
    def _as_tag_list(payload) -> list:
        """Normalise a price-tag payload to a list of non-empty tag objects."""
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []
        return [tag for tag in payload if isinstance(tag, dict) and tag]

    @staticmethod
    def _photo_filename(original: str, kind: str, extracted: dict) -> str:
        """A recognisable display name, so a stored photo is findable later."""
        import re

        extension = os.path.splitext(original)[1] or ".jpg"

        def slug(value, fallback):
            cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")
            return cleaned or fallback

        if kind == "price_tag":
            # Named after the first tag; a shelf photo holding several is still
            # one file, and listing them all would blow past any sane filename.
            tags = extracted.get("tags") or [extracted]
            first = tags[0] if isinstance(tags[0], dict) else {}
            item = slug(first.get("item_description"), "item")
            if len(tags) > 1:
                item = f"{item}_and_{len(tags) - 1}_more"
            store = slug(first.get("merchant_name"), "")
            # A tag carries no date of its own; when it was seen is the date.
            date_slug = pd.Timestamp.now().strftime("%Y%m%d")
            parts = [p for p in ("pricetag", item, store, date_slug) if p]
            return "_".join(parts)[:120] + extension

        merchant = slug(extracted.get("merchant_name"), "receipt")
        date_slug = str(extracted.get("date") or "nodate").replace("-", "")
        time_slug = re.sub(r"[^0-9]", "", str(extracted.get("time") or ""))
        return f"{merchant}_{date_slug}{f'_{time_slug}' if time_slug else ''}{extension}"

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
            # Raised, not swallowed. Reporting success on a failed delete is
            # exactly how a "Discarded" card ended up sitting next to a file that
            # was still there — the caller has to be able to say so.
            logger.error("Error purging file data for %s (%s): %s", file_id, file_type, e)
            raise

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
                        # This user's own key, so the MCP server never depends on
                        # a process-wide GOOGLE_API_KEY/OPENAI_API_KEY.
                        "CURRENT_LLM_API_KEY": self.api_key or "",
                        # So MCP tools can sign private-bucket storage URLs as this user.
                        "CURRENT_ACCESS_TOKEN": self.access_token or "",
                        # Chart/image handoff directory, unique to this instance —
                        # otherwise concurrent users overwrite each other's output.
                        "DATA_DIR": self.temp_dir,
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
                "CRITICAL RULE FOR RESPONSES (spending analysis and charts ONLY — NOT price checks, see below): After calling any chart or data tool, you MUST write a detailed text analysis "
                "that includes: (1) a summary of the key numbers, (2) the top and bottom items, (3) any notable patterns or insights. "
                "The chart appears below your text automatically — your text analysis is the PRIMARY response the user reads. "
                "Never respond with just a single sentence when data is available. "
                "WARNING: You MUST use the actual tool call to generate the chart. DO NOT output raw chart JSON in your text.\n"
                "MANUAL TRANSACTIONS: When the user describes a transaction they made (e.g., 'I gave X $100', "
                "'I spent $50 at Target', 'paid rent $1200'), you MUST use the 'propose_transaction' tool. "
                "Extract the amount, description, date (default today), category, and merchant name from the user's message. "
                "Do NOT insert transactions directly — always let the user confirm via the UI card. "
                f"CRITICAL: You MUST include the {markers.CONFIRM_TX} marker output from the tool in your response EXACTLY as returned. "
                "Do not remove or modify the marker content.\n"
                "FINDING PRODUCTS: receipt line items are printed abbreviated past recognition "
                "('GV LF 2 GAL' is Great Value low-fat milk, 'BB GRND TRKY1LB' is ground turkey), "
                "so a SQL LIKE on a product name finds nothing. Use 'semantic_search' for anything "
                "about a product, and pass its 'scope' argument deliberately: 'transactions' for "
                "spending themes, 'line_items' for products actually bought, 'price_observations' "
                "for shelf prices the user photographed.\n"
                "SHELF PRICES ARE NOT SPENDING: a price observation is a photo of a price tag. "
                "Nothing was bought and no money left the user's account, so NEVER add one to a "
                "spending total, an average spend, or a category breakdown. They live in the "
                '"PriceObservation" table and are labelled [SHELF PRICE] in search results.\n'
                "IS THIS A GOOD PRICE: lead with ONE SHORT SENTENCE of verdict, then list the 2-3 most comparable prices you actually have, one per line, in the shortest form that is still clear: date - price per unit - shop. Showing several is the point: one number reads like a rule, three show the range the user is really choosing within and whether this shelf price is an outlier or ordinary. Add a caveat line only if it would change their decision.\n""Keep it to that. No headings, no 'Price Analysis' / 'Insights' sections, no restating the tag back to them, no paragraph of reasoning around each line — someone standing in a shop needs an answer, not a report. The detailed-analysis rule above does NOT apply here. List ONLY prices for the same product: a near-miss with a similar name is worse than a short list.\n""Use the 'check_price' tool. When the user says the sighting is ALREADY RECORDED, pass record=false — the confirm card saved it before asking you, and saving again stores one sighting twice. Otherwise it records the sighting, "
                "researches the product, RANKS past purchases by similarity without filtering them, "
                "converts both sides to the same unit and weights recent evidence more heavily. "
                "Do NOT hand-roll this with SQL — the per-unit conversion and the recency "
                "weighting are the parts that are easy to get wrong.\n"
                "NEVER report a per-unit price as a total, and never drop its unit: '$1.00/l' and 'paid $4.99' are different facts. If a unit looks wrong for the product ('5L' on a bag of potatoes is five POUNDS, not litres), say the size is uncertain rather than reasoning from it.\n""FIXING A WRONG VALUE: when the user says something stored is wrong — a misread size, a unit "
                "that makes no sense for the product, a mangled name, the wrong shop — use 'propose_correction'. "
                "It only PROPOSES: the user sees a card and confirms, and nothing is written until they do. "
                f"You MUST include the {markers.CONFIRM_FIX} marker in your reply EXACTLY as returned. "
                "It cannot delete anything and it cannot touch money on a receipt — amounts, totals and tax "
                "are corrected on the receipt review screen because they have to stay consistent with each "
                "other. A shelf price in PriceObservation IS correctable. Never invent a row id: take it from "
                "a tool result.\n""SHOWING PROOF: when the user asks to see a receipt behind a price you quoted, pass the 'receipt=<id>' value check_price gave you for THAT line to get_bill_images as bill_file_ids. Never re-find a receipt by searching item text — receipt lines are abbreviated past recognition, so the search matches a different purchase and you attach someone else's receipt as evidence. If a line has no receipt id, say the receipt is not available rather than showing a substitute.\n""PRICES ARE LOCAL: check WHERE each past purchase happened before quoting it. A price from another city or state is not the going rate where the user is shopping now — they may have moved — so say so plainly rather than calling the current price a rip-off, and prefer a nearer purchase even if it scores slightly lower.\n""ALWAYS compare PER UNIT. A $3.49 gallon versus a $3.38 two-gallon jug is $3.49/gal versus "
                "$1.69/gal, not 'about the same'; the tool normalises both sides for you and marks "
                "anything it could not normalise. Compare against BOTH what the user PAID and prices "
                "they have SEEN before, and never describe a sighting as something they paid.\n"
                "check_price RANKS candidates rather than filtering them: rows marked '~' are the "
                "closest things found, not confirmed matches. Read each description and decide "
                "whether it is really the same product before using it - low-fat vs whole milk, "
                "organic vs conventional, and a 2-gallon vs a 1-gallon jug all rank high on "
                "wording while costing different amounts.\n"
                "check_price returns EVIDENCE, never a verdict: the judgement is yours. ALWAYS "
                "read what the tag said before judging — a price well under the usual one is very "
                "often a clearance on something expiring within days, a multi-buy needing several, "
                "or a loyalty-card rate, and you must say what the catch is. Never present a "
                "purchase marked ON OFFER as what the user normally pays. When there is no "
                "purchase history for the item, say so plainly instead of guessing. When the size "
                "is unknown, say you are comparing package to package rather than per unit."
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
            # LangGraph defaults to 25 steps, and a comparison question spends
            # several before it starts: the schema lookup, a rejected query that
            # forgot the user_id filter, a retry for an unquoted table name. A
            # question across two cities then runs out mid-answer and fails
            # outright. 40 leaves room without letting a genuinely stuck agent
            # spin indefinitely.
            config = {
                "configurable": {"thread_id": uuid.uuid4().hex},
                "recursion_limit": 40,
            }
            input_messages = list(history or []) + [{"role": "user", "content": query}]
            
            chart_path = os.path.join(self.temp_dir, "latest_chart.json")
            if os.path.exists(chart_path):
                os.remove(chart_path)
                
            images_path = os.path.join(self.temp_dir, "latest_images.json")
            if os.path.exists(images_path):
                os.remove(images_path)

            # Stream events so we can yield live tool-call updates.
            #
            # Text is accumulated PER MODEL TURN, not across the whole run. A
            # ReAct agent calls the model repeatedly, and the turns that decide
            # which tool to use often think out loud first ("Let me look that
            # up..."). That is working-out, not an answer: it used to be
            # concatenated into the reply, and streaming would put it on screen.
            # A turn that calls a tool has its text discarded.
            ai_text_chunks: list[str] = []
            turn_text: list[str] = []
            turn_calls_tool = False
            # astream_events surfaces NESTED events too, and several tools here
            # call an LLM of their own (merchant enrichment, price comparison).
            # Their tokens are indistinguishable from the agent's own at this
            # level, so without this the user would watch a tool's internal
            # monologue stream into the answer.
            tool_depth = 0

            async for event in agent.astream_events(
                {"messages": input_messages},
                config,
                version="v2",
            ):
                kind = event.get("event")

                if kind == "on_chat_model_start":
                    if tool_depth == 0:
                        turn_text = []
                        turn_calls_tool = False
                    continue

                if kind == "on_chat_model_end":
                    if tool_depth == 0 and not turn_calls_tool:
                        ai_text_chunks.extend(turn_text)
                        turn_text = []
                    continue

                if kind == "on_tool_start":
                    tool_depth += 1
                    tool_name = event.get("name", "tool")
                    tool_input = event.get("data", {}).get("input", {})
                    # Summarise long inputs
                    if isinstance(tool_input, dict):
                        snippet = ", ".join(f"{k}={str(v)[:60]}" for k, v in tool_input.items())
                    else:
                        snippet = str(tool_input)[:120]
                    yield {"type": "tool_start", "name": tool_name, "input": snippet}

                elif kind == "on_tool_end":
                    tool_depth = max(0, tool_depth - 1)
                    tool_name = event.get("name", "tool")
                    output = event.get("data", {}).get("output", "")
                    snippet = str(output)[:200].replace("\n", " ")
                    yield {"type": "tool_end", "name": tool_name, "snippet": snippet}

                elif kind == "on_chat_model_stream":
                    # A model running inside a tool is not answering the user.
                    if tool_depth > 0:
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue

                    # A turn that emits tool-call chunks is choosing a tool, so
                    # whatever prose came with it is working-out. Stop emitting
                    # from this turn and drop what it produced.
                    if _chunk_calls_tool(chunk):
                        turn_calls_tool = True
                        turn_text = []
                        continue
                    if turn_calls_tool:
                        continue

                    text = _chunk_text(chunk)
                    if text:
                        turn_text.append(text)
                        # Raw. The caller gates it — knowledge of the ===MARKER===
                        # blocks lives in one place, in the router that strips
                        # them from the final content.
                        yield {"type": "token", "text": text}

            # A run cut short (no on_chat_model_end for the last turn) still has
            # its answer in turn_text; losing it would blank the reply.
            if turn_text and not turn_calls_tool:
                ai_text_chunks.extend(turn_text)

            final_content = "".join(ai_text_chunks).strip()

            # Build final response (with optional chart and images)
            if os.path.exists(chart_path):
                with open(chart_path, "r") as f:
                    chart_json = f.read()
                final_content = f"{final_content}\n\n{markers.wrap(markers.CHART, chart_json)}"

            if os.path.exists(images_path):
                with open(images_path, "r") as f:
                    images_json = f.read()
                final_content = f"{final_content}\n\n{markers.wrap(markers.IMAGES, images_json)}"
                
            yield {"type": "final", "content": final_content}
            
        finally:
            try:
                await mcp_client.__aexit__(None, None, None)
            except Exception as close_e:
                logger.warning("Error closing MCP client: %s", close_e)

    async def cleanup(self):
        """Delete temporary session files and close MCP client."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning("Failed to remove temp directory %s: %s", self.temp_dir, e)
