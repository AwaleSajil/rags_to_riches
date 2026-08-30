import json
import os
import re
from typing import Optional

import pandas as pd
import plotly.express as px
from fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from textwrap import dedent

from backend.sql_guard import SqlGuardError, readonly_cursor, validate_select
from backend.services import stream_gate as markers

# Load environment variables (API keys, etc.)
load_dotenv()

# Per-chat scratch directory for chart/image handoff to the UI, injected by
# money_rag when it launches this server. It must be per-instance: a shared
# default would let one user's chart be picked up by another user's response.
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize the MCP Server
mcp = FastMCP("Money RAG Financial Analyst")

from supabase import create_client, Client

def get_db_connection():
    """Returns a database connection (psycopg2 for Postgres)."""
    import psycopg2
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL must be defined to construct raw SQL connections.")
    return psycopg2.connect(db_url)

def get_current_user_id() -> str:
    user_id = os.environ.get("CURRENT_USER_ID")
    if not user_id:
        raise ValueError("CURRENT_USER_ID not injected into MCP environment!")
    return user_id

# The names as they are actually spelled in the database. sql_guard's allowlist
# is lowercased for comparison, which loses the casing needed to tell the agent
# what to write. test_schema_doc holds the two in step.
TABLE_NAMES = (
    "Transaction",
    "TransactionDeduped",
    "TransactionDetail",
    "TransactionLink",
    "BillFile",
    "CSVFile",
    "PriceObservation",
)


def _quote_table(name: str) -> str:
    """Quote a table name appropriately for Postgres."""
    return f'"{name}"'  # Postgres uses double-quoted identifiers

def _run_guarded_query(query: str):
    """Validate LLM-written SQL, then run it read-only. Returns (rows, columns).

    Every tool that executes model-authored SQL goes through here. The
    connection uses DATABASE_URL, which is the superuser role and bypasses RLS,
    so validation is the only thing standing between a prompt injection and the
    whole database — see backend/sql_guard.py. Raises SqlGuardError when the
    query is rejected; callers surface that text to the model so it can retry.
    """
    user_id = get_current_user_id()
    validate_select(query, user_id)

    conn = get_db_connection()
    try:
        cursor = readonly_cursor(conn)
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        return results, column_names
    finally:
        conn.close()

# The single description of the schema, shared by the schema resource and the
# query_database tool. It used to be written out twice, and the two copies drifted
# apart from the database and from each other: both still named a
# TransactionDetail column that migration 023 renamed, so every agent query that
# touched a unit price failed. One string, injected in both places.
SCHEMA_DOC = dedent(f"""
    Schema for the authenticated user's data. All money is in the account currency.

    CRITICAL RULE:
    You MUST add `WHERE user_id = '{{current_user_id}}'` to EVERY SINGLE query you write.
    Never query data without filtering by user_id!

    Table names are case-sensitive and MUST be double-quoted: {_quote_table("Transaction")}.

    Never SELECT the `embedding` columns. They are 3072-dimension vectors, they
    are useless as text, and one row of them will swamp your context. Use the
    semantic_search tool when you need similarity.

    TABLE: {_quote_table("Transaction")}  — one purchase.
      - id (UUID), user_id (UUID)
      - trans_date (DATE)
      - description (TEXT) — merchant as written on the statement/receipt
      - merchant_name (TEXT)
      - amount (NUMERIC) — the final amount CHARGED. Positive = spending,
        negative = payment/refund. Coupons are already subtracted.
      - category (TEXT)
      - location (TEXT, nullable) — city/state/address when known
      - subtotal (NUMERIC, nullable) — pre-tax, after item markdowns
      - tax_total (NUMERIC, nullable)
      - tax_breakdown (JSONB, nullable) — [{{label, rate, amount}}] per distinct rate
      - discount_total (NUMERIC, nullable) — order-level coupons. ALREADY
        subtracted from `amount`; never subtract it again.
      - savings_total (NUMERIC, nullable) — item markdowns + coupons combined.
        DISPLAY ONLY. It is what the shopper saved, not money spent or owed —
        never subtract it from anything.
      - note (TEXT, nullable) — free text the USER wrote about this purchase
      - enriched_info (TEXT, nullable) — researched description of the merchant
      - source (TEXT) — 'bill' (photographed receipt) or 'csv' (bank statement)
      - source_csv_id (UUID, nullable), source_bill_file_id (UUID, nullable)
      - content_hash (TEXT) — deduplication key, not meaningful to report
      - embedding (VECTOR), embedding_model (TEXT) — never select
      - created_at (TIMESTAMPTZ)

    VIEW: {_quote_table("TransactionDeduped")}  — the same columns as
    {_quote_table("Transaction")} (minus the embedding) with linked duplicates
    already removed: exactly one row per real-world purchase.
    USE THIS, NOT {_quote_table("Transaction")}, for every sum, average, count
    or category breakdown of spending. Querying the table directly counts a
    photographed receipt and its bank-statement line as two separate purchases.
    Reach for {_quote_table("Transaction")} only when you specifically want a row
    this view drops — e.g. showing BOTH records of one purchase.

    TABLE: {_quote_table("TransactionDetail")}  — one line on a receipt.
    Only receipts have these; a bank-statement transaction has no line items.
      - id (UUID), transaction_id (UUID), user_id (UUID)
      - bill_file_id (UUID, nullable) — the photo this line was read from
      - item_description (TEXT) — as PRINTED, so heavily abbreviated
        ("GV LF 2 GAL" = Great Value low-fat milk, 2 gallon). Do not assume a
        readable product name; prefer semantic_search over LIKE for products.
      - item_quantity (NUMERIC) — how many purchase units were bought
      - item_quantity_unit (TEXT, nullable) — what item_quantity counts:
        'each','lb','oz','kg','g','ml','l','ct'. NULL on rows imported before
        this column existed, so treat a NULL as unknown rather than 'each'.
      - size_value (NUMERIC, nullable) / size_unit (TEXT, nullable) — how much is
        in ONE purchase unit: a 5 lb bag is item_quantity 1 'each' with
        size_value 5, size_unit 'lb'. NOT the same as item_quantity. NULL means
        unknown, and the size then has to be read out of item_description, which
        is a guess ("+RED POTA 5L US#" is five POUNDS, not litres).
      - unit_quantity_subtotal (NUMERIC) — pre-tax price per unit ACTUALLY PAID,
        net of any markdown
      - item_subtotal_price (NUMERIC) — pre-tax line total
        = item_quantity x unit_quantity_subtotal
      - item_savings (NUMERIC) — how much this line was marked down. DISPLAY
        ONLY: the prices above are already net, so never subtract it.
      - tax_rate (NUMERIC, nullable) — % applied to THIS item, 0 = exempt
      - tax_amount (NUMERIC), taxable (BOOLEAN, nullable)
      - item_total_price (NUMERIC) — post-tax line total
      - enriched_info (TEXT, nullable) — researched description of the product
      - embedding (VECTOR), embedding_model (TEXT) — never select
      - created_at (TIMESTAMPTZ)

    TABLE: {_quote_table("PriceObservation")}  — a shelf price the user
    photographed. NOT a purchase: nothing was bought and no money was spent, so
    it must NEVER be added to any spending total. Use it to answer "what does X
    cost / have I seen this price before", and compare it against
    TransactionDetail to answer "is this a good price".
      - id (UUID), user_id (UUID)
      - bill_file_id (UUID, nullable) — the photo of the tag
      - merchant_name (TEXT, nullable), location (TEXT, nullable)
      - item_description (TEXT), brand_name (TEXT, nullable)
      - size_value (NUMERIC, nullable) / size_unit (TEXT, nullable) — the package
        the tag prices, e.g. 12 'oz' for "$4.29 / 12 OZ". Nothing is bought from
        a shelf, so there is no count column here.
      - unit_quantity_subtotal (NUMERIC, nullable) — the per-unit price the tag
        PRINTED where it printed one; the store's own figure
      - item_subtotal_price (NUMERIC, nullable) — shelf price for the package
      - item_qualitative_description (TEXT, nullable) — everything the tag showed
        that is not a number, in its own words: "2 for $5 with card",
        "CLEARANCE", "Sale ends 8/15", "best before 08/05", visible damage.
        Deliberately unparsed — READ IT before judging a price, because a cheap
        price is often cheap for a reason stated here.
      - note (TEXT, nullable) — what the USER said about the item in chat
      - enriched_info (TEXT, nullable)
      - embedding (VECTOR), embedding_model (TEXT) — never select
      - created_at (TIMESTAMPTZ) — when the price was SEEN

    TABLE: {_quote_table("BillFile")}  — an uploaded photo.
      - id (UUID), user_id (UUID), filename (TEXT), s3_key (TEXT)
      - kind (TEXT) — 'receipt' | 'price_tag' | 'unknown'
      - raw_ocr_string (TEXT) — the extracted draft as JSON
      - is_hidden (BOOLEAN) — when true the user excluded it; respect this
      - upload_date (TIMESTAMPTZ)

    TABLE: {_quote_table("CSVFile")}  — an uploaded bank statement.
      - id (UUID), user_id (UUID), filename (TEXT), s3_key (TEXT)
      - is_hidden (BOOLEAN), upload_date (TIMESTAMPTZ)

    TABLE: {_quote_table("TransactionLink")}  — reconciliation between a
    photographed receipt and the bank-statement row for the same purchase.
    Both rows exist, so counting both DOUBLE COUNTS that spending.
      - id (UUID), user_id (UUID)
      - transaction_id (UUID), linked_transaction_id (UUID)
      - match_type (TEXT), confidence (NUMERIC), created_at (TIMESTAMPTZ)
    """)


def get_schema_info() -> str:
    """Get database schema information."""
    return SCHEMA_DOC


@mcp.resource("schema://database/tables")
def get_database_schema() -> str:
    """Complete schema information for the money_rag database."""
    return get_schema_info()

def query_database(query: str) -> str:
    try:
        results, column_names = _run_guarded_query(query)

        if not results:
            return "No results found"

        # Format results nicely
        formatted_results = []
        formatted_results.append(f"Columns: {', '.join(column_names)}")
        for row in results:
            formatted_results.append(str(row))

        return "\n".join(formatted_results)
    except SqlGuardError as e:
        return f"Query rejected: {e}"
    except Exception as e:
        return _explain_db_error(e)


# Postgres folds an unquoted identifier to lowercase, so `FROM Transaction`
# looks for a table named `transaction` and reports it missing. The schema
# already says to quote table names, and the agent still forgets — it costs two
# steps of a 25-step budget on almost every run, because the raw Postgres text
# never says what to do about it. This does.
_MISSING_RELATION = re.compile(r'relation "([a-z_]+)" does not exist', re.IGNORECASE)


def _explain_db_error(exc: Exception) -> str:
    message = str(exc)
    match = _MISSING_RELATION.search(message)
    if match:
        lowercased = match.group(1)
        correct = next(
            (t for t in TABLE_NAMES if t.lower() == lowercased.lower()), None
        )
        if correct:
            return (
                f'Database Error: relation "{lowercased}" does not exist.\n'
                f'The table is named {_quote_table(correct)} — Postgres lowercases '
                f'unquoted names, so you must write it with double quotes. '
                f'Re-run the SAME query with {_quote_table(correct)} instead of '
                f'{correct}.'
            )
    return f"Database Error: {message}"


# Registered by hand rather than with @mcp.tool() so the schema can be injected:
# a docstring cannot be an f-string, and writing the tables out a second time is
# what let this tool's copy fall behind the database in the first place.
query_database.__doc__ = dedent("""
    Execute a read-only SQL SELECT against the user's financial data.

    Args:
        query: The SQL SELECT query to execute

    Returns:
        Query results, or an error message

    Rules:
    - SELECT only. Anything that writes is rejected before it reaches the database.
    - Every query MUST filter by user_id.
    - Quote table names: FROM "Transaction", not FROM Transaction.
    - Prefer semantic_search over LIKE when looking for a PRODUCT: receipt line
      items are abbreviated past recognition ("GV LF 2 GAL", "BB GRND TRKY1LB").
    - Counting spending: sum "Transaction".amount. Do NOT sum line items to get a
      total (they are pre-tax and exclude fees), and do NOT subtract
      savings_total or discount_total — amount is already final.
    - "PriceObservation" is not spending. Never include it in a spending total.

    Example queries:
    - Walmart spending: SELECT SUM(amount) FROM "Transaction" WHERE user_id = '...' AND merchant_name ILIKE '%walmart%' AND amount > 0;
    - Recent purchases: SELECT trans_date, merchant_name, amount, category FROM "Transaction" WHERE user_id = '...' ORDER BY trans_date DESC LIMIT 5;
    - Spending by category: SELECT category, SUM(amount) FROM "Transaction" WHERE user_id = '...' AND amount > 0 GROUP BY category;
    - What a product cost before: SELECT t.trans_date, t.merchant_name, d.item_description, d.item_quantity, d.item_quantity_unit, d.unit_quantity_subtotal FROM "TransactionDetail" d JOIN "Transaction" t ON t.id = d.transaction_id WHERE d.user_id = '...' AND d.item_description ILIKE '%milk%' ORDER BY t.trans_date DESC;
    """) + SCHEMA_DOC
query_database = mcp.tool()(query_database)


@mcp.tool()
def semantic_search(query: str, top_k: int = 5, scope: str = "all") -> str:
    """
    Search the user's financial data by meaning rather than by exact text.

    Use this instead of SQL LIKE whenever you are looking for a PRODUCT or a
    THEME. Receipt lines are printed abbreviated past recognition — "GV LF 2 GAL"
    is Great Value low-fat milk, "BB GRND TRKY1LB" is ground turkey — so a LIKE
    on "milk" finds neither.

    Args:
        query: What to look for, in natural words.
        top_k: Number of results to return (default 5).
        scope: Which corpus to search. Pick deliberately — the three hold
            different things and are embedded from different text:

            'transactions'       Whole purchases. Embeds merchant + category +
                                 the user's note. Use for "how much did I spend
                                 on fast food", "subscriptions", "that trip".
            'line_items'         Individual products BOUGHT, from receipts.
                                 Embeds product identity only, no merchant. Use
                                 for "have I bought this before", "what do I
                                 usually pay for X".
            'price_observations' Shelf prices the user PHOTOGRAPHED. Nothing was
                                 bought and NO MONEY WAS SPENT — never add these
                                 to a spending total. Use for "what did that cost
                                 in the shop", "have I seen this price before".
            'all'                (default) All three, ranked together.

            To answer "is this a good price?", search 'line_items' for what the
            user actually paid and compare. Do not mix scopes in one total.

    Results are labelled by kind. A price observation line carries what the tag
    itself said ("2 for $5 with card", "CLEARANCE", "Sale ends 8/15", a
    best-before date) — read it before judging the price, because a cheap price
    is very often cheap for a reason printed right there.
    """
    try:
        user_id = get_current_user_id()
        
        from backend.vector_db_client import get_vector_client
        vdb = get_vector_client()

        # Use the SAME embedding model the transactions were synced with, injected
        # by money_rag when it launches this MCP server. Falls back to Gemini.
        # The key is passed to the constructor rather than read from a process-wide
        # env var, so it stays tied to the user this server was launched for.
        provider = os.environ.get("CURRENT_EMBEDDING_PROVIDER", "google")
        model = os.environ.get("CURRENT_EMBEDDING_MODEL", "gemini-embedding-001")
        api_key = os.environ.get("CURRENT_LLM_API_KEY") or None
        if provider == "google":
            embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        else:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)

        results = vdb.semantic_search(
            query, user_id=user_id, top_k=top_k, embeddings_model=embeddings, scope=scope
        )

        if not results:
            return f"No matches found in {scope}."

        # Formatted per kind rather than through one template. The old single
        # line printed "Amount" for everything, which would have labelled a shelf
        # price as money spent — the one confusion this whole table exists to
        # prevent.
        output = []
        for doc in results:
            meta = doc["metadata"]
            kind = meta.get("vector_type")
            money = lambda v: "?" if v is None else f"${v:.2f}"

            if kind == "price_observation":
                parts = [
                    f"[SHELF PRICE — not a purchase] Seen {str(meta.get('observed_on'))[:10]}",
                    doc["page_content"],
                    f"Price: {money(meta.get('shelf_price'))}",
                ]
                if meta.get("unit_price") is not None:
                    parts.append(f"{money(meta['unit_price'])}/{meta.get('quantity_unit') or 'unit'}")
                if meta.get("merchant_name"):
                    parts.append(f"at {meta['merchant_name']}")
                if meta.get("tag_says"):
                    parts.append(f"Tag says: {meta['tag_says']}")
                if meta.get("note"):
                    parts.append(f"User note: {meta['note']}")
            elif kind == "line_item":
                parts = [
                    f"[BOUGHT] {str(meta.get('transaction_date'))[:10]}",
                    doc["page_content"],
                    f"Paid: {money(meta.get('amount'))}",
                ]
                if meta.get("unit_price") is not None:
                    parts.append(f"{money(meta['unit_price'])}/{meta.get('quantity_unit') or 'unit'}")
                if meta.get("merchant_name"):
                    parts.append(f"at {meta['merchant_name']}")
            else:
                parts = [
                    f"[PURCHASE] {str(meta.get('transaction_date'))[:10]}",
                    doc["page_content"],
                    f"Total: {money(meta.get('amount'))}",
                ]
            output.append(" | ".join(parts))

        return "\n".join(output)
        
    except Exception as e:
        import traceback
        return f"Error performing search: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def generate_interactive_chart(sql_query: str, chart_type: str, x_col: str, y_col: str, title: str, color_col: Optional[str] = None) -> str:
    """
    Generate an interactive Plotly chart using SQL data.
    IMPORTANT: The table name MUST be "Transaction" exactly with quotes.

    Args:
        sql_query: The SQL SELECT query to retrieve the data for the chart from the "Transaction" table.
            - Must use 'user_id' filter.
        chart_type: The type of chart: 'bar', 'line', 'pie', 'scatter'
        x_col: The name of the column to use for the X axis (or labels for pie charts)
        y_col: The name of the column to use for the Y axis (or values for pie charts)
        title: The title of the chart
        color_col: (Optional) Column to use for color grouping

    Returns:
        A natural language summary confirming chart generation.
    """
    try:
        results, columns = _run_guarded_query(sql_query)
        df = pd.DataFrame(results, columns=columns)
        if df.empty:
            return json.dumps({"error": "No data found for this query."})
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title, color=color_col)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title, color=color_col)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color_col)
        else:
            return json.dumps({"error": f"Unsupported chart type: {chart_type}"})

        # Mobile-friendly color palette (high contrast, accessible)
        mobile_colors = [
            "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4",
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#64748b",
        ]
        fig.update_layout(colorway=mobile_colors)

        # Mobile-optimized layout defaults
        fig.update_layout(
            font=dict(size=12),
            margin=dict(l=4, r=4, t=40, b=4, autoexpand=True),
        )

        # Bar chart: limit visible bars and use horizontal if many categories
        if chart_type == "bar" and len(df) > 10:
            fig.update_layout(xaxis_tickangle=-45)

        # Pie chart: compact label positioning
        if chart_type == "pie":
            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
                hole=0.3,
                pull=0.02,
            )

        # Write the huge JSON to a temp file instead of returning it directly to LLM context
        chart_path = os.path.join(DATA_DIR, "latest_chart.json")
        with open(chart_path, "w") as f:
            f.write(fig.to_json())
            
        # Summarise the data so the LLM can write a useful text analysis
        summary_parts = [f"Chart generated ({chart_type}): '{title}'"]
        summary_parts.append(f"Data: {len(df)} rows, columns={list(df.columns)}")
        if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            summary_parts.append(f"Total {y_col}: {df[y_col].sum():.2f}")
            summary_parts.append(f"Top: {df.loc[df[y_col].idxmax(), x_col]} ({df[y_col].max():.2f})")
            summary_parts.append(f"Bottom: {df.loc[df[y_col].idxmin(), x_col]} ({df[y_col].min():.2f})")
        return (
            " | ".join(summary_parts)
            + "\n\nThe chart will be displayed in the UI automatically. "
            "You MUST now write a detailed text summary of the data and key insights for the user. "
            "Do NOT include the raw chart JSON in your response."
        )

    except SqlGuardError as e:
        return json.dumps({"error": f"Query rejected: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to generate chart: {e}"})
@mcp.tool()
def get_bill_images(sql_query: Optional[str] = None, bill_file_ids: Optional[list] = None) -> str:
    """
    Show the user photos of specific receipts.

    PREFER `bill_file_ids`. Every purchase check_price returns carries
    `receipt=<id>`; pass those ids straight here and you get exactly the
    receipts behind the prices you quoted.

    Do NOT re-find a receipt by searching item text. Receipt lines are
    abbreviated past recognition, so the search does not match what you quoted —
    it matches something else and attaches it as proof. Asked to evidence
    "$1.69/gal at Walmart", a search for "milk" returned a COSTCO receipt,
    because "GV LF 2 GAL" contains no such word and "3 WHOLE MILK" does.

    `sql_query` remains for cases with no id in hand — a whole transaction's
    receipt, say. It must SELECT b.s3_key, join "BillFile", and filter by
    user_id.

    Args:
        bill_file_ids: BillFile ids, e.g. the `receipt=` values from check_price.
        sql_query: Fallback SELECT returning b.s3_key.

    Returns:
        JSON with signed image URLs.
    """
    try:
        if bill_file_ids:
            user_id = get_current_user_id()
            conn = get_db_connection()
            try:
                cursor = readonly_cursor(conn)
                cursor.execute(
                    'SELECT s3_key FROM public."BillFile" WHERE user_id = %s AND id = ANY(%s)',
                    (user_id, [str(i) for i in bill_file_ids]),
                )
                results = cursor.fetchall()
            finally:
                conn.close()
        elif sql_query:
            results, _ = _run_guarded_query(sql_query)
        else:
            return json.dumps({"error": "Pass bill_file_ids (preferred) or a sql_query."})

        if not results:
            return json.dumps({"error": "No bills found for this query."})

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        access_token = os.environ.get("CURRENT_ACCESS_TOKEN")

        # The bucket is private, so a public URL won't load. Sign a temporary URL
        # instead — authenticated as the user so it passes storage RLS.
        if access_token:
            from supabase import ClientOptions
            opts = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
            supabase = create_client(url, key, options=opts)
        else:
            supabase = create_client(url, key)

        urls = []
        # One receipt can hold the same product on several lines — Costco listed
        # "3 WHOLE MILK" twice — and each line carries the same photo, so the
        # unfiltered list showed the identical receipt more than once.
        seen_keys = set()
        for row in results:
            if row[0] and row[0] not in seen_keys:
                seen_keys.add(row[0])
                signed = supabase.storage.from_("money-rag-files").create_signed_url(row[0], 3600)
                signed_url = (
                    signed.get("signedURL")
                    or signed.get("signedUrl")
                    or signed.get("signed_url")
                )
                if signed_url and signed_url.startswith("/"):
                    signed_url = url.rstrip("/") + signed_url
                if signed_url:
                    urls.append(signed_url)

        if not urls:
            return json.dumps({"error": "No image keys found in result set."})

        # Write image URLs to a temp file that the main UI can pick up and render alongside the chat
        chart_path = os.path.join(DATA_DIR, "latest_images.json")
        with open(chart_path, "w") as f:
            json.dump(urls, f)
            
        return "Images retrieved successfully! I have sent the image URLs to the user's UI. You can tell the user the receipt is attached."

    except SqlGuardError as e:
        return json.dumps({"error": f"Query rejected: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve images: {e}"})

@mcp.tool()
def propose_transaction(
    description: str,
    amount: float,
    date: Optional[str] = None,
    category: Optional[str] = None,
    merchant_name: Optional[str] = None,
) -> str:
    """
    Propose a new manually-entered transaction for user confirmation.
    Use this when the user tells you about a transaction in natural language,
    such as "I gave Simran 100 dollars" or "I spent $50 on groceries today".

    IMPORTANT: This tool does NOT insert anything into the database.
    It returns a structured proposal that will be shown to the user for confirmation.
    The user must explicitly confirm before the transaction is saved.

    Args:
        description: What the transaction was for (e.g., "Gave Simran", "Lunch at subway")
        amount: The transaction amount as a positive number (spending is positive)
        date: The date in YYYY-MM-DD format. If not specified, defaults to today.
        category: Category like "Transfer", "Food", "Shopping", etc. Defaults to "Uncategorized".
        merchant_name: Clean merchant/recipient name. Defaults to description.

    Returns:
        JSON string with the proposed transaction details for UI confirmation.
    """
    import json
    from datetime import date as date_type

    # Default date to today if not provided
    if not date:
        date = date_type.today().isoformat()

    # Validate date format
    try:
        parsed = date_type.fromisoformat(date)
        date = parsed.isoformat()
    except ValueError:
        return json.dumps({"error": f"Invalid date format: {date}. Use YYYY-MM-DD."})

    # Validate amount
    if amount <= 0:
        return json.dumps({"error": "Amount must be positive."})

    proposal = {
        "description": description,
        "amount": round(amount, 2),
        "trans_date": date,
        "category": category or "Uncategorized",
        "merchant_name": merchant_name or description,
    }

    result = json.dumps(proposal)

    return (
        f"{markers.wrap(markers.CONFIRM_TX, result)}\n\n"
        "I've prepared this transaction for your review. "
        "Please check the details in the confirmation card and tap Confirm to save it."
    )


def correction_service_columns() -> str:
    """The allowlist, read from the service so the two can never disagree."""
    from backend.services import correction_service

    return correction_service.describe_correctable()


def _user_client():
    """A Supabase client carrying the user's own token, so RLS applies.

    The raw DATABASE_URL connection used elsewhere in this file is the superuser
    role and bypasses RLS. Anything that WRITES goes through here instead.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    token = os.environ.get("CURRENT_ACCESS_TOKEN")
    if not (url and key and token):
        raise ValueError("Supabase credentials were not injected into the MCP environment")
    client = create_client(url, key)
    client.postgrest.auth(token)
    return client


def _account_config():
    """The user's AccountConfig, read under their own token."""
    rows = (
        _user_client().table("AccountConfig")
        .select("*").eq("user_id", get_current_user_id()).limit(1).execute().data or []
    )
    return rows[0] if rows else None


@mcp.tool()
def check_price(
    item_description: str,
    shelf_price: Optional[float] = None,
    quantity: Optional[float] = None,
    quantity_unit: Optional[str] = None,
    brand_name: Optional[str] = None,
    merchant_name: Optional[str] = None,
    tag_says: Optional[str] = None,
    note: Optional[str] = None,
    record: bool = True,
) -> str:
    """Check a shelf price against what the user has actually paid before.

    Use this whenever the user asks whether a price is good, mentions seeing a
    price in a shop, or asks what something usually costs them.

    It does three things: records the sighting so it is there next time,
    researches what the product is (which is what makes matching work on
    abbreviated receipt lines), and retrieves comparable past purchases.

    IMPORTANT: this returns EVIDENCE, not a verdict. Deciding whether the price
    is good is YOUR job, and it depends on more than the numbers:

    - Read 'tag_says' before judging. A price well below the usual one is very
      often a clearance on something expiring within days, a multi-buy that
      requires taking several, or a loyalty-card rate. Say what the catch is.
    - Purchases marked 'was_on_offer' are excluded from the baseline on purpose.
      Do not present a marked-down price as what the user normally pays.
    - ALWAYS compare PER UNIT, never package against package. A $3.49 gallon and
      a $3.38 two-gallon jug are not "about the same" — they are $3.49/gal and
      $1.69/gal. Every figure this returns is already normalised to a common
      unit; rows saying "not comparable per unit" could NOT be, so do not
      compare their headline prices instead.
    - NEVER quote a per-unit price as if it were the total, and never drop the
      unit from it. "$1.00/l" and "paid $4.99 total" are two different facts
      about the same purchase.
    - "not comparable per unit" means the two sides are in different unit
      FAMILIES and code will not guess between them. That is the one case where
      converting is YOUR job, because it needs knowing what the product is: 128
      'oz' of milk is 128 FLUID ounces, which is a gallon; "13.2G" on a $34.46
      step-on bin is gallons, not grams. Say what you think the real unit is and
      that you inferred it — never present it as recorded fact.
    - A unit that looks wrong for the product probably is. Receipt text is
      abbreviated and gets misread: "5L" on a bag of potatoes is five POUNDS,
      not five litres. If a unit makes no sense for the item, say the size is
      uncertain rather than reasoning from it.
    - CHECK WHERE each purchase happened. Prices are local. A comparison across
      cities or states is not a comparison of the same market — the user may
      simply have moved. If the merchant or city differs from where they are
      shopping now, SAY SO instead of presenting the old price as the going
      rate, and prefer a nearer purchase even if it scores slightly lower.
    - Compare against BOTH sources: what the user PAID (line items) and prices
      they have SEEN before (observations). A sighting is not a purchase — never
      say they paid it — but it is real evidence of what the item costs.
    - Show the user SEVERAL comparable prices when you have them, not just the
      best or the closest. One number reads like a rule; two or three show the
      range they are really choosing within, and whether the shelf price is an
      outlier or ordinary. Two or three is the useful number — a longer list
      stops being readable in a shop.
    - Results are RANKED, not filtered. Rows marked '~' are the closest things
      found, not confirmed matches — read the description and decide. Watch for a
      different variant: low-fat vs whole milk, organic vs conventional, and a
      2-gallon vs a 1-gallon jug are different products at different prices, and
      they rank high because they are near-identical words.
    - When 'purchases' is empty, say plainly that there is no history for this
      item. Do NOT guess whether the price is good.
    - When 'size' is null, the sizes could not be compared. Compare package to
      package and SAY that is what you are doing.

    Args:
        item_description: The product, as printed on the tag.
        shelf_price: What the shopper pays for the package. Omit to look up
            history without recording a new sighting.
        quantity: The package size as a number ("12 OZ" -> 12).
        quantity_unit: Its unit ("12 OZ" -> "oz"). Do not convert units.
        brand_name: Brand, if distinguishable.
        merchant_name: The shop, if known.
        tag_says: Everything on the tag that is not a number, in its own words:
            "2 for $5 with card", "CLEARANCE", "Sale ends 8/15", "best before
            08/05", visible damage. Quote it; do not interpret it.
        note: What the USER said about the item, if anything.
        record: Whether to save this sighting (default true). Pass FALSE when
            the user is only asking what something usually costs, and whenever
            they say the sighting is already recorded — the confirm card saves
            the price before asking you about it, and saving again would store
            one sighting twice and make it look like corroborating evidence.
    """
    try:
        user_id = get_current_user_id()
        config = _account_config()

        from backend.services import price_service

        saved = None
        if record and shelf_price is not None:
            saved = price_service.record_observation(
                _user_client(), config,
                {
                    "item_description": item_description,
                    "brand_name": brand_name,
                    # The names migration 034 gave these. They were still
                    # spelled item_quantity/item_quantity_unit here, which
                    # record_observation does not read, so every price the agent
                    # recorded was stored with no size AND embedded without one —
                    # the exact 12oz-jar-equals-64oz-jar collapse the confirm
                    # card path was fixed for.
                    "size_value": quantity,
                    "size_unit": quantity_unit,
                    "item_subtotal_price": shelf_price,
                    "merchant_name": merchant_name,
                    "item_qualitative_description": tag_says,
                    "note": note,
                },
                user_id=user_id,
            )
            # Enriched straight away rather than in the background: the vector
            # written at insert has only the tag's words in it, and the receipt
            # side is matched on a researched product description. Until the two
            # carry the same kind of text the measured floor does not transfer.
            saved = price_service.enrich_observation(_user_client(), config, saved)

        # Queried with the tag's own short phrasing, NOT the enrichment stored on
        # the row. The match floor was calibrated in that direction; enriching
        # both sides drops a true BANANAS match from 0.772 to 0.748, below it.
        evidence = price_service.compare_price(
            config, user_id, item_description,
            shelf_price=shelf_price, quantity=quantity, quantity_unit=quantity_unit,
            brand_name=brand_name,
            exclude_observation_id=(saved or {}).get("id"),
        )

        lines = [f"ITEM: {item_description}" + (f" ({evidence['size']})" if evidence.get("size") else "")]
        if shelf_price is not None:
            lines.append(f"SHELF PRICE: ${shelf_price:.2f}" + (" — recorded" if saved else ""))
        if tag_says:
            lines.append(f"TAG SAYS: {tag_says}")

        if evidence["purchases"]:
            lines.append(
                "\nWHAT YOU PAID BEFORE (ranked by similarity; '~' means a LOOSE "
                "match — judge for yourself whether it is the same product):"
            )
            for p in evidence["purchases"]:
                bits = [f"  {'  ' if p['confident'] else '~ '}{p['date']}", p["description"]]
                # Package price AND per-unit price, always together and always
                # with the unit attached. Given only "$1.00/l" for a $4.99 bag of
                # potatoes, the model reported "you paid $1.00 for a bag" — the
                # per-unit figure quoted as the total, with the unit dropped.
                if p.get("amount") is not None:
                    bits.append(f"paid ${p['amount']:.2f} total")
                if p.get("unit_price_display"):
                    bits.append(f"= {p['unit_price_display']}")
                elif p.get("paid_per_unit") is not None:
                    bits.append(
                        f"= ${p['paid_per_unit']:.2f} per "
                        f"{p.get('quantity_unit') or 'UNKNOWN UNIT'}"
                    )
                if p.get("merchant"):
                    bits.append(f"at {p['merchant']}")
                if p.get("location"):
                    bits.append(p["location"])
                if p["vs_shelf_percent"] is not None:
                    bits.append(f"this shelf price is {p['vs_shelf_percent']:+.1f}% vs that")
                else:
                    bits.append("not comparable per unit (different unit family)")
                if p["was_on_offer"]:
                    bits.append("[ON OFFER — excluded from the baseline]")
                bits.append(f"similarity {p['score']}")
                if p.get("bill_file_id"):
                    # Quote this id verbatim to get_bill_images. It is the ONLY
                    # correct way to show proof of this line.
                    bits.append(f"receipt={p['bill_file_id']}")
                lines.append(" | ".join(bits))
                for caveat in p["caveats"]:
                    lines.append(f"      note: {caveat}")

        if evidence["prior_observations"]:
            lines.append("\nPRICES YOU HAVE SEEN BEFORE (not purchases):")
            for o in evidence["prior_observations"]:
                bits = [f"  {'  ' if o['confident'] else '~ '}{o['seen']}", o["description"]]
                if o.get("price") is not None:
                    bits.append(f"${o['price']:.2f}")
                if o.get("unit_price_display"):
                    bits.append(o["unit_price_display"])
                if o.get("merchant"):
                    bits.append(f"at {o['merchant']}")
                if o.get("location"):
                    bits.append(o["location"])
                if o.get("vs_shelf_percent") is not None:
                    bits.append(f"this shelf price is {o['vs_shelf_percent']:+.1f}% vs that")
                lines.append(" | ".join(bits))
                if o.get("tag_says"):
                    lines.append(f"      tag said: {o['tag_says']}")

        closest = evidence.get("closest_comparable")
        if closest and not evidence.get("baseline"):
            lines.append(
                f"\nCLOSEST LIKE-FOR-LIKE (per unit, "
                f"{'something you BOUGHT' if closest['kind'] == 'paid' else 'a price you SAW but did not buy'}): "
                f"{evidence.get('shelf_unit_price')} vs "
                f"{closest['their_unit_price']} ({closest['description']}, "
                f"{closest['merchant']}"
                + (f", {closest['location']}" if closest.get("location") else "")
                + f", {closest['date']}) — "
                f"{closest['percent']:+.1f}%."
                + ("" if closest["confident"] else
                   " NOTE: a loose match, not confirmed. Check it is the same product "
                   "before quoting this figure.")
            )

        baseline = evidence.get("baseline")
        if baseline:
            lines.append(
                f"\nTYPICALLY PAID: ${baseline['typical']:.2f} per unit "
                f"(range ${baseline['low']:.2f}-${baseline['high']:.2f}, "
                f"{baseline['count']} purchases, recency-weighted confidence "
                f"{baseline['confidence']})"
            )
        comparison = evidence.get("comparison")
        if comparison:
            lines.append(
                f"THIS PRICE IS {comparison['percent']:+.1f}% vs what you typically pay "
                f"(${comparison['shelf_per_unit']:.2f} vs ${comparison['typical_paid_per_unit']:.2f} per unit)"
            )

        if evidence["cautions"]:
            lines.append("\nCAUTIONS — reflect these in your answer:")
            lines.extend(f"  - {c}" for c in evidence["cautions"])

        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"Error checking price: {e}\n{traceback.format_exc()}"


def propose_correction(table: str, row_id: str, changes: dict, reason: str = "") -> str:
    """PLACEHOLDER — docstring assembled below so the allowlist cannot drift."""
    from backend.services import correction_service

    try:
        cleaned = correction_service.validate(table, changes)
    except ValueError as e:
        # Handed back to the model so it can retry within the rules rather than
        # telling the user something was fixed when nothing was.
        return f"Correction rejected: {e}"

    # Read the row first: it proves the id exists and is the user's before a card
    # is shown, and it gives the card something to show the change FROM. A fix
    # presented without the old value asks the user to confirm a change they
    # cannot see.
    try:
        rows = (
            _user_client().table(table)
            .select(",".join(["id", *cleaned]))
            .eq("id", str(row_id)).eq("user_id", get_current_user_id())
            .limit(1).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        return f"Could not read that row: {e}"
    if not rows:
        return (
            f"No {table} row with id {row_id} belongs to this user. "
            "Get the id from a tool result rather than constructing one."
        )
    current = {c: rows[0].get(c) for c in cleaned}

    unchanged = [c for c, v in cleaned.items() if str(current.get(c)) == str(v)]
    if len(unchanged) == len(cleaned):
        return "That row already holds those values — nothing to fix."

    proposal = json.dumps({
        "table": table,
        "row_id": str(row_id),
        "changes": cleaned,
        "current": current,
        "labels": {c: correction_service.CORRECTABLE[table][c] for c in cleaned},
        "reason": reason or "",
    })
    return (
        f"{markers.wrap(markers.CONFIRM_FIX, proposal)}\n\n"
        "I've prepared that fix. Check the card and tap Confirm to apply it."
    )


propose_correction.__doc__ = dedent("""
    Propose a correction to ONE row, for the user to confirm.

    Use this when the user says a stored value is wrong — a misread size, a unit
    that makes no sense for the product, a mangled item name, the wrong shop.

    This does NOT change anything. It returns a proposal the user sees as a card
    and confirms; only then is anything written, on their own request under their
    own permissions. You cannot write to the database and this is not a way to.

    There is NO delete. Rows are never removed this way — if something should not
    exist, say so and let the user remove it themselves.

    Only these tables and columns can be corrected, and only rows the user owns:

    """) + correction_service_columns() + dedent("""

    Everything else is refused, including anything about MONEY on a receipt —
    amount, subtotal, tax, line totals. Those have to stay consistent with each
    other, so they are corrected on the receipt review screen, which recomputes
    the whole receipt. A shelf price in PriceObservation IS correctable: nothing
    is derived from it and it never reaches a spending total.

    Get `row_id` from what a tool already gave you — check_price returns the
    receipt behind each line, query_database can select an id. Never guess one.

    Args:
        table: "Transaction", "TransactionDetail" or "PriceObservation".
        row_id: The row's id.
        changes: {column: new value}, only the columns actually changing.
        reason: One short line on why, shown on the card.

    Returns:
        A confirmation marker to include in your reply EXACTLY as returned.
    """)
propose_correction = mcp.tool()(propose_correction)


if __name__ == "__main__":
    # Runs the server over stdio
    mcp.run(transport="stdio")
