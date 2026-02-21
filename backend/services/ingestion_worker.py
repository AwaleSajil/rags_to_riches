"""
Standalone ingestion worker — runs in a subprocess so primp/duckduckgo-search
can't hold the GIL and freeze the main FastAPI process.

Usage: python -m backend.services.ingestion_worker <json_args>
"""
import asyncio
import json
import sys
import os

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from money_rag import MoneyRAG


async def run(config: dict, user_id: str, access_token: str, uploaded_files_info: list):
    rag = MoneyRAG(
        llm_provider=config["llm_provider"],
        model_name=config.get("decode_model", "gemini-3-flash-preview"),
        embedding_model_name=config.get("embedding_model", "gemini-embedding-001"),
        api_key=config["api_key"],
        user_id=user_id,
        access_token=access_token,
    )
    await rag.setup_session(uploaded_files_info)


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    asyncio.run(run(
        config=args["config"],
        user_id=args["user_id"],
        access_token=args["access_token"],
        uploaded_files_info=args["uploaded_files_info"],
    ))
