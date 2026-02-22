import pandas as pd
import plotly.express as px
from fastmcp import FastMCP
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from typing import Optional

import shutil

from textwrap import dedent

# Load environment variables (API keys, etc.)
load_dotenv()

# Define paths to your data
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_data"))

# Initialize the MCP Server
mcp = FastMCP("Money RAG Financial Analyst")

import psycopg2
from supabase import create_client, Client

def get_db_connection():
    """Returns a psycopg2 connection to Supabase Postgres."""
    # Supabase provides postgres connection strings, but typically doesn't default in plain OS vars unless you build it
    # Supabase gives a postgres:// connection string in the dashboard under Database Settings.
    # Alternatively we can build it manually or just use the Supabase python client.
    # To support raw LLM SQL, we use psycopg2 instead of Supabase client.
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL must be defined to construct raw SQL connections.")
    return psycopg2.connect(db_url)

def get_current_user_id() -> str:
    user_id = os.environ.get("CURRENT_USER_ID")
    if not user_id:
        raise ValueError("CURRENT_USER_ID not injected into MCP environment!")
    return user_id

def get_schema_info() -> str:
    """Get database schema information for Postgres tables."""
    return dedent("""
    Here is the PostgreSQL database schema for the authenticated user's data.
    
    CRITICAL RULE:
    You MUST add `WHERE user_id = '{current_user_id}'` to EVERY SINGLE query you write.
    Never query data without filtering by user_id!
    
    TABLE: "Transaction"
    Columns:
      - id (UUID)
      - user_id (UUID)
      - trans_date (DATE)
      - description (TEXT)
      - amount (DECIMAL)
      - category (VARCHAR)
      - merchant_name (TEXT)

    TABLE: "TransactionDetail"
    Columns:
      - id (UUID)
      - transaction_id (UUID)
      - user_id (UUID)
      - item_description (TEXT)
      - item_quantity (DECIMAL)
      - item_unit_price (DECIMAL)
      - item_total_price (DECIMAL)
      - tax_amount (DECIMAL)
      - enriched_info (TEXT)
    """)


@mcp.resource("schema://database/tables")
def get_database_schema() -> str:
    """Complete schema information for the money_rag database."""
    return get_schema_info()

@mcp.tool()
def query_database(query: str) -> str:
    """
    Execute a raw SQL query against the Postgres database.
    The main table is named "Transaction" (you MUST INCLUDE QUOTES in your SQL!).
    IMPORTANT STRICT SCHEMA:
    Table: "Transaction"
    - id (UUID)
    - user_id (UUID text)
    - trans_date (DATE)
    - description (TEXT)
    - merchant_name (TEXT)
    - amount (NUMERIC)
    - category (TEXT)
    
    Table: "TransactionDetail"
    - id (UUID)
    - transaction_id (UUID)
    - user_id (UUID text)
    - item_description (TEXT)
    - item_quantity (NUMERIC)
    - item_total_price (NUMERIC)
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
    - Find Walmart spending: SELECT SUM(amount) FROM "Transaction" WHERE description LIKE '%Walmart%' AND amount > 0;
    - List recent transactions: SELECT trans_date, description, amount, category FROM "Transaction" ORDER BY trans_date DESC LIMIT 5;
    - Spending by category: SELECT category, SUM(amount) FROM "Transaction" WHERE amount > 0 GROUP BY category;
    """
    # Security: Only allow SELECT queries
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
        return "Error: Only SELECT queries are allowed"

    # Forbidden operations
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"]
    if any(f" {word} " in f" {query_upper} " for word in forbidden):
        return f"Error: Query contains forbidden operation. Only SELECT queries allowed."

    user_id = get_current_user_id()
    if user_id not in query:
        return f"Error: You forgot to include the security filter (WHERE user_id = '{user_id}') in your query! Try again."

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Get column names to make result more readable
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        
        conn.close()

        if not results:
            return "No results found"
            
        # Format results nicely
        formatted_results = []
        formatted_results.append(f"Columns: {', '.join(column_names)}")
        for row in results:
            formatted_results.append(str(row))

        return "\n".join(formatted_results)
    except psycopg2.Error as e:
        return f"Database Error: {str(e)}"

def get_vector_store():
    """Initialize connection to the Qdrant vector store"""
    # Initialize Embedding Model using Google AI Studio
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Connect to Qdrant Cloud
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    return QdrantVectorStore(
        client=client,
        collection_name="transactions",
        embedding=embeddings,
    )

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
        vector_store = get_vector_store()
        
        # Apply strict multi-tenant filtering based on the payload we injected in money_rag.py
        from qdrant_client.http import models
        filter = models.Filter(
            must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
        )

        results = vector_store.similarity_search(query, k=top_k, filter=filter)
        
        if not results:
            return "No matching transactions found."
            
        output = []
        for doc in results:
            amount = doc.metadata.get('amount', 'N/A')
            date = doc.metadata.get('transaction_date', 'N/A')
            output.append(f"Date: {date} | Match: {doc.page_content} | Amount: {amount}")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error performing search: {str(e)}"


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
        user_id = get_current_user_id()
        if user_id not in sql_query:
            return f'{{"error": "You forgot the WHERE user_id = \\"{user_id}\\" security clause!"}}'
            
        conn = get_db_connection()
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty:
            return '{"error": "No data found for this query."}'
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title, color=color_col)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title, color=color_col)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color_col)
        else:
            return f'{{"error": "Unsupported chart type: {chart_type}"}}'

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
        
    except Exception as e:
        return f'{{"error": "Failed to generate chart: {str(e)}"}}'
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
        user_id = get_current_user_id()
        if user_id not in sql_query:
            return f'{{"error": "You forgot the WHERE user_id = \\"{user_id}\\" security clause!"}}'
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return '{"error": "No bills found for this query."}'
            
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        supabase = create_client(url, key)
        
        urls = []
        for row in results:
            if row[0]:
                public_url = supabase.storage.from_("money-rag-files").get_public_url(row[0])
                urls.append(public_url)
                
        if not urls:
            return '{"error": "No image keys found in result set."}'
            
        import json
        
        # Write image URLs to a temp file that the main UI can pick up and render alongside the chat
        chart_path = os.path.join(DATA_DIR, "latest_images.json")
        with open(chart_path, "w") as f:
            json.dump(urls, f)
            
        return "Images retrieved successfully! I have sent the image URLs to the user's UI. You can tell the user the receipt is attached."
        
    except Exception as e:
        return f'{{"error": "Failed to retrieve images: {str(e)}"}}'

if __name__ == "__main__":
    # Runs the server over stdio
    mcp.run(transport="stdio")
