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

    async def setup_session(self, csv_files: List[dict]):
        """Ingests CSVs and sets up DBs."""
        # csv_files format: [{"path": "/temp/file.csv", "csv_id": "uuid"}, ...]
        for file_info in csv_files:
            await self._ingest_csv(file_info["path"], file_info.get("csv_id"))
        
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        self.vector_store = self._sync_to_qdrant()

    async def _ingest_csv(self, file_path, csv_id=None):
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
        unique_descriptions = standard_df['description'].unique()
        sem = asyncio.Semaphore(5)

        async def get_merchant_info(description):
            if description in self.merchant_cache:
                return self.merchant_cache[description]
            
            async with sem:
                try:
                    await asyncio.sleep(0.05) # Jitter
                    print(f"      🔍 Web searching: {description}...")
                    result = await self.search_tool.ainvoke(f"What type of business / store is '{description}'?")
                    self.merchant_cache[description] = result
                    return result
                except Exception as e:
                    print(f"      ⚠️ Search failed for {description}: {e}")
                    return "Unknown"

        tasks = [get_merchant_info(desc) for desc in unique_descriptions]
        enrichment_results = await asyncio.gather(*tasks)
        
        desc_map = dict(zip(unique_descriptions, enrichment_results))
        standard_df['enriched_info'] = standard_df['description'].map(desc_map).fillna("")

        # Save to Supabase transactions table instead of local SQLite
        # Use simplejson roundtrip to guarantee all Pandas NaNs, NaTs, and weird floats become strict JSON nulls
        import json
        records = json.loads(standard_df.to_json(orient='records'))
        
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            # If insertion fails, it raises an exception so Streamlit surfaces the error
            self.supabase.table("Transaction").insert(batch).execute()

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
        
        # Use description + category + enrichment for vectorization
        texts = []
        for _, row in df.iterrows():
            enriched = row.get('enriched_info', '')
            base_text = f"{row['description']} ({row['category']})"
            if enriched and enriched != "Unknown" and enriched != "":
                texts.append(f"{base_text} - {enriched}")
            else:
                texts.append(base_text)
        
        # Inject critical user_id payload to Qdrant so we can filter on it during retrieval
        metadatas = df[['id', 'amount', 'category', 'trans_date']].copy()
        if 'source_csv_id' in df.columns:
            metadatas['source_csv_id'] = df['source_csv_id']
        metadatas = metadatas.to_dict('records')
        
        vector_ids = []
        for m in metadatas: 
            vector_ids.append(str(m['id'])) # Keep original Postgres UUID as Vector ID to prevent duplication
            m['user_id'] = self.user_id # Secure payload identifier
            m['transaction_date'] = str(m['trans_date']) # Rename for agent consistency
            del m['trans_date']
            
        vs.add_texts(texts=texts, metadatas=metadatas, ids=vector_ids)
        return vs

    async def delete_file(self, csv_id: str):
        """Force delete a file and all its transactions from Postgres and Qdrant."""
        try:
            # 1. Delete from Postgres (Transactions cascade automatically if foreign keyed... but we'll manually ensure they wipe just in case)
            self.supabase.table("Transaction").delete().eq("source_csv_id", csv_id).execute()
            self.supabase.table("CSVFile").delete().eq("id", csv_id).execute()
            
            # 2. Delete from Qdrant via payload filter
            client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
            client.delete(
                collection_name="transactions",
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.source_csv_id",
                            match=qdrant_models.MatchValue(value=csv_id)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Error purging file data: {e}")

    async def chat(self, query: str):
        # 1. Initialize MCP client dynamically to guarantee fresh bindings
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
            # 2. Extract tools from the safely established subprocess
            mcp_tools = await mcp_client.get_tools()

            # 3. Create the LangGraph agent for this turn, preserving historical memory cache
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
            
            # Clear out any previous chart so we don't carry over stale plots
            chart_path = os.path.join(self.temp_dir, "latest_chart.json")
            if os.path.exists(chart_path):
                os.remove(chart_path)
            
            # 4. Invoke the agent against the LLM, triggering our nested Tools locally
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config,
            )
            
            # Extract content - handle both string and list formats
            content = result["messages"][-1].content
            
            # If content is a list (Gemini format), extract text from blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                final_text = "\n".join(text_parts)
            else:
                final_text = content
                
            # Check for generated chart
            if os.path.exists(chart_path):
                with open(chart_path, "r") as f:
                    chart_json = f.read()
                return f"{final_text}\n\n===CHART===\n{chart_json}\n===ENDCHART==="
                
            return final_text
            
        finally:
            # 5. Destroy the subprocess safely so we don't leak FastMCP zombies across Streamlit reruns
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