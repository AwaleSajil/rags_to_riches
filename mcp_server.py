import json
import os
from typing import Optional

import pandas as pd
import plotly.express as px
from fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from textwrap import dedent

from backend.sql_guard import SqlGuardError, readonly_cursor, validate_select

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

def get_schema_info() -> str:
    """Get database schema information."""
    tbl = _quote_table("Transaction")
    dtbl = _quote_table("TransactionDetail")
    return dedent(f"""
    Here is the database schema for the authenticated user's data.

    CRITICAL RULE:
    You MUST add `WHERE user_id = '{{current_user_id}}'` to EVERY SINGLE query you write.
    Never query data without filtering by user_id!

    TABLE: {tbl}
    Columns:
      - id (UUID/STRING)
      - user_id (UUID/STRING)
      - trans_date (DATE)
      - description (TEXT)
      - amount (DECIMAL/DOUBLE)
      - category (VARCHAR/STRING)
      - merchant_name (TEXT)
      - location (TEXT, nullable — store city/state/address when known)
      - subtotal (DECIMAL, nullable — pre-tax total, receipts)
      - tax_total (DECIMAL, nullable — total tax)
      - tax_breakdown (JSONB, nullable — [{{label, rate, amount}}] per tax rate)

    TABLE: {dtbl}
    Columns:
      - id (UUID/STRING)
      - transaction_id (UUID/STRING)
      - user_id (UUID/STRING)
      - item_description (TEXT)
      - item_quantity (DECIMAL/DOUBLE)
      - item_unit_subtotal_price (DECIMAL/DOUBLE — pre-tax unit price)
      - item_subtotal_price (DECIMAL/DOUBLE — pre-tax line total = qty x unit)
      - tax_amount (DECIMAL/DOUBLE — tax for this item = subtotal x rate)
      - taxable (BOOLEAN, nullable — was this item taxed)
      - tax_rate (DECIMAL, nullable — % rate applied to this item, 0 = exempt)
      - item_total_price (DECIMAL/DOUBLE — post-tax line total = subtotal + tax_amount)
      - enriched_info (TEXT)
    """)


@mcp.resource("schema://database/tables")
def get_database_schema() -> str:
    """Complete schema information for the money_rag database."""
    return get_schema_info()

@mcp.tool()
def query_database(query: str) -> str:
    """
    Execute a raw SQL query against the database.
    IMPORTANT STRICT SCHEMA:
    Table: Transaction
    - id (UUID/STRING)
    - user_id (UUID/STRING)
    - trans_date (DATE)
    - description (TEXT)
    - merchant_name (TEXT)
    - amount (NUMERIC/DOUBLE)
    - category (TEXT)
    - location (TEXT, nullable)
    - subtotal (NUMERIC, nullable — pre-tax)
    - tax_total (NUMERIC, nullable)
    - tax_breakdown (JSONB, nullable — [{label, rate, amount}])

    Table: TransactionDetail
    - id (UUID/STRING)
    - transaction_id (UUID/STRING)
    - user_id (UUID/STRING)
    - item_description (TEXT)
    - item_quantity (NUMERIC/DOUBLE)
    - item_unit_subtotal_price (NUMERIC/DOUBLE — pre-tax unit price)
    - item_subtotal_price (NUMERIC/DOUBLE — pre-tax line total)
    - tax_amount (NUMERIC/DOUBLE — computed tax for this item)
    - taxable (BOOLEAN, nullable)
    - tax_rate (NUMERIC, nullable — % rate for this item, 0 = exempt)
    - item_total_price (NUMERIC/DOUBLE — post-tax line total = subtotal + tax_amount)
    - enriched_info (TEXT)

    Args:
        query: The SQL SELECT query to execute

    Returns:
        Query results or error message

    Important Notes:
    - Only SELECT queries are allowed (read-only)
    - Use 'description' column for text search
    - 'amount' column: positive values = spending, negative values = payments/refunds

    Example queries:
    - Find Walmart spending: SELECT SUM(amount) FROM Transaction WHERE description LIKE '%Walmart%' AND amount > 0;
    - List recent transactions: SELECT trans_date, description, amount, category FROM Transaction ORDER BY trans_date DESC LIMIT 5;
    - Spending by category: SELECT category, SUM(amount) FROM Transaction WHERE amount > 0 GROUP BY category;
    """
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
        return f"Database Error: {str(e)}"

@mcp.tool()
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Search for personal financial transactions semantically.
    
    Use this to find spending when specific merchant names are unknown or ambiguous.
    Examples: "how much did I spend on fast food?", "subscriptions", "travel expenses".
    
    Args:
        query: The description or category of spending to look for.
        top_k: Number of results to return (default 5).
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

        results = vdb.semantic_search(query, user_id=user_id, top_k=top_k, embeddings_model=embeddings)
        
        if not results:
            return "No matching transactions found."
            
        output = []
        for doc in results:
            meta = doc['metadata']
            amount = meta.get('amount', 'N/A')
            date = meta.get('transaction_date', 'N/A')
            output.append(f"Date: {date} | Match: {doc['page_content']} | Amount: {amount}")
            
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
def get_bill_images(sql_query: str) -> str:
    """
    Get image URLs for specific uploaded bills and receipts so they can be displayed to the user in the UI.
    IMPORTANT: You must provide a valid SQL SELECT query that targets the "Transaction" or "TransactionDetail" table,
    and it MUST join or link to the "BillFile" table to retrieve the 's3_key' column.
    
    Example queries:
    - Get receipt for transaction:
      SELECT b.s3_key FROM "Transaction" t JOIN "BillFile" b ON t.source_bill_file_id = b.id WHERE t.user_id = '{user_id}' AND t.description LIKE '%McDonalds%'
    - Get receipt for a specific line item:
      SELECT b.s3_key FROM "TransactionDetail" d JOIN "BillFile" b ON d.bill_file_id = b.id WHERE d.user_id = '{user_id}' AND d.item_description LIKE '%Fries%'
      
    Args:
        sql_query: The SQL SELECT query to retrieve the 's3_key' (string).

    Returns:
        JSON string containing the public image URLs.
    """
    try:
        results, _ = _run_guarded_query(sql_query)

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
        for row in results:
            if row[0]:
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
        f"===CONFIRM_TX==={result}===ENDCONFIRM_TX===\n\n"
        "I've prepared this transaction for your review. "
        "Please check the details in the confirmation card and tap Confirm to save it."
    )


if __name__ == "__main__":
    # Runs the server over stdio
    mcp.run(transport="stdio")
