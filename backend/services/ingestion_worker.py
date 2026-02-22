"""
Standalone ingestion worker — runs in a subprocess so primp/duckduckgo-search
can't hold the GIL and freeze the main FastAPI process.

Usage: python -m backend.services.ingestion_worker <json_args>
"""
import asyncio
import json
import logging
import sys
import os
import time

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("moneyrag.ingestion_worker")

from money_rag import MoneyRAG


async def run(config: dict, user_id: str, access_token: str, uploaded_files_info: list):
    logger.info(
        "Ingestion worker started — user_id=%s, %d files, provider=%s, model=%s",
        user_id, len(uploaded_files_info), config["llm_provider"],
        config.get("decode_model", "gemini-3-flash-preview"),
    )
    logger.debug("Files to ingest: %s", uploaded_files_info)

    logger.debug("Creating MoneyRAG instance")
    start = time.perf_counter()
    rag = MoneyRAG(
        llm_provider=config["llm_provider"],
        model_name=config.get("decode_model", "gemini-3-flash-preview"),
        embedding_model_name=config.get("embedding_model", "gemini-embedding-001"),
        api_key=config["api_key"],
        user_id=user_id,
        access_token=access_token,
    )
    logger.debug("MoneyRAG instance created in %.1fms", (time.perf_counter() - start) * 1000)

    logger.debug("Calling rag.setup_session with %d files", len(uploaded_files_info))
    session_start = time.perf_counter()
    duplicates = await rag.setup_session(uploaded_files_info)
    session_ms = (time.perf_counter() - session_start) * 1000
    logger.info(
        "Ingestion worker complete — user_id=%s, %d files processed in %.1fms, %d duplicates",
        user_id, len(uploaded_files_info), session_ms, len(duplicates or []),
    )

    # Output result as JSON on stdout for the parent process to parse
    print(json.dumps({"duplicates": duplicates or []}))


if __name__ == "__main__":
    logger.debug("Ingestion worker subprocess starting — PID=%d", os.getpid())
    logger.debug("sys.argv[1] length: %d chars", len(sys.argv[1]))
    args = json.loads(sys.argv[1])
    logger.debug("Parsed args — user_id=%s, %d files", args["user_id"], len(args["uploaded_files_info"]))
    asyncio.run(run(
        config=args["config"],
        user_id=args["user_id"],
        access_token=args["access_token"],
        uploaded_files_info=args["uploaded_files_info"],
    ))
    logger.info("Ingestion worker subprocess exiting — PID=%d", os.getpid())
