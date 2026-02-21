from fastmcp import FastMCP
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
import os

import shutil

# Load environment variables (API keys, etc.)
load_dotenv()

# Define paths to your data
# For Hugging Face Spaces (Ephemeral):
# We use a temporary directory that gets wiped on restart.
# If DATA_DIR is set (e.g., by your deployment config), use it.
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_data"))
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_db")
DB_PATH = os.path.join(DATA_DIR, "money_rag.db")

# Initialize the MCP Server
mcp = FastMCP("Money RAG Financial Analyst")

import sqlite3

def get_schema_info() -> str:
    """Get database schema information."""
    if not os.path.exists(DB_PATH):
        return "Database file does not exist yet. Please upload data."

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_info = []
        for (table_name,) in tables:
            schema_info.append(f"\nTable: {table_name}")

            # Get column info for each table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            schema_info.append("Columns:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                schema_info.append(f"  - {col_name} ({col_type})")

        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error reading schema: {e}"


@mcp.resource("schema://database/tables")
def get_database_schema() -> str:
    """Complete schema information for the money_rag database."""
    return get_schema_info()


@mcp.tool()
def query_database(query: str) -> str:
    """Execute a SELECT query on the money_rag SQLite database.

    Args:
        query: The SQL SELECT query to execute

    Returns:
        Query results or error message

    Important Notes:
    - Only SELECT queries are allowed (read-only)
    - Use 'description' column for text search 
    - 'amount' column: positive values = spending, negative values = payments/refunds

    Example queries:
    - Find Walmart spending: SELECT SUM(amount) FROM transactions WHERE description LIKE '%Walmart%' AND amount > 0;
    - List recent transactions: SELECT transaction_date, description, amount, category FROM transactions ORDER BY transaction_date DESC LIMIT 5;
    - Spending by category: SELECT category, SUM(amount) FROM transactions WHERE amount > 0 GROUP BY category;
    """
    if not os.path.exists(DB_PATH):
        return "Database file does not exist yet. Please upload data."

    # Security: Only allow SELECT queries
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT") and not query_upper.startswith("PRAGMA"):
        return "Error: Only SELECT and PRAGMA queries are allowed"

    # Forbidden operations
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "DETACH"]
    # Check for forbidden words as standalone words to avoid false positives (e.g. "update_date" column)
    # Simple check: space-surrounded or end-of-string
    if any(f" {word} " in f" {query_upper} " for word in forbidden):
        return f"Error: Query contains forbidden operation. Only SELECT queries allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Get column names to make result more readable
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        conn.close()

        if not results:
            return "No results found"
            
        # Format results nicely
        formatted_results = []
        formatted_results.append(f"Columns: {', '.join(column_names)}")
        for row in results:
            formatted_results.append(str(row))

        return "\n".join(formatted_results)
    except sqlite3.Error as e:
        return f"Error: {str(e)}"

def get_vector_store():
    """Initialize connection to the Qdrant vector store"""
    # Initialize Embedding Model
    # Uses environment variables/default gcloud auth
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

    # Connect to Qdrant (Persistent Disk Mode at specific path)
    # We ensure the directory exists so Qdrant can write to it.
    os.makedirs(QDRANT_PATH, exist_ok=True)
    
    client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists (it might be empty in a new ephemeral session)
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if "transactions" not in collection_names:
        # In a real app, you would probably trigger ingestion here or handle the empty state
        pass

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
        vector_store = get_vector_store()
        
        # Safety check: if no data has been ingested yet
        if not os.path.exists(QDRANT_PATH) or not os.listdir(QDRANT_PATH):
             return "No matching transactions found (Database is empty. Please upload data first)."

        results = vector_store.similarity_search(query, k=top_k)
        
        if not results:
            return "No matching transactions found."
            
        output = []
        for doc in results:
            # Format the output clearly for the LLM/User
            amount = doc.metadata.get('amount', 'N/A')
            date = doc.metadata.get('transaction_date', 'N/A')
            output.append(f"Date: {date} | Match: {doc.page_content} | Amount: {amount}")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error performing search: {str(e)}"

# A helper to clear data (useful for session reset)
@mcp.tool()
def clear_database() -> str:
    """Clear all stored transaction data to reset the session."""
    try:
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
        return "Database cleared successfully."
    except Exception as e:
        return f"Error clearing database: {e}"

if __name__ == "__main__":
    # Runs the server over stdio
    mcp.run()
