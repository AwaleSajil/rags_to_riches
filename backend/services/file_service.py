import asyncio
import json
import logging
import os
import sys
import time
from typing import List
from backend.dependencies import get_supabase
from backend.db_client import get_db_client
from backend.services.rag_manager import rag_manager
from backend.services import config_service

logger = logging.getLogger("moneyrag.services.file_service")

# Per-user ingestion status: user_id -> {"status": "processing"|"complete"|"failed", "error": str|None}
ingestion_status: dict[str, dict] = {}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── List files ──────────────────────────────────────────────────────────────

def _list_files_sync(access_token: str, user_id: str) -> List[dict]:
    logger.debug("Querying CSVFile and BillFile tables for user_id=%s", user_id)
    with get_db_client(access_token) as db:
        res_csv, res_bill = db.list_files(user_id)

    csv_count = len(res_csv)
    bill_count = len(res_bill)
    logger.debug("Found %d CSV files, %d bill files for user_id=%s", csv_count, bill_count, user_id)

    files = []
    for d in res_csv:
        d["type"] = "csv"
        files.append(d)
    for d in res_bill:
        d["type"] = "bill"
        files.append(d)
    return files


async def list_files(user: dict) -> List[dict]:
    logger.debug("list_files called for user_id=%s", user["id"])
    result = await asyncio.to_thread(_list_files_sync, user["access_token"], user["id"])
    logger.debug("list_files returning %d files for user_id=%s", len(result), user["id"])
    return result


# ── Upload + ingest ─────────────────────────────────────────────────────────

def _upload_to_storage_sync(user: dict, saved_files: List[dict]) -> tuple[list, list]:
    """Uploads files to Supabase storage + creates DB records."""
    logger.debug("_upload_to_storage_sync — %d files for user_id=%s", len(saved_files), user["id"])
    client = get_supabase(user["access_token"])
    uploaded_files_info = []
    file_ids = []

    with get_db_client(user["access_token"]) as db:
        for file_info in saved_files:
            local_path = file_info["local_path"]
            filename = file_info["filename"]
            is_image = filename.lower().endswith((".png", ".jpg", ".jpeg"))
            folder = "bills" if is_image else "csvs"
            s3_key = f"{user['id']}/{folder}/{filename}"

            content_type = "text/csv"
            if filename.lower().endswith(".png"):
                content_type = "image/png"
            elif filename.lower().endswith((".jpg", ".jpeg")):
                content_type = "image/jpeg"

            logger.debug(
                "Uploading '%s' to storage — s3_key=%s, content_type=%s",
                filename, s3_key, content_type,
            )
            start = time.perf_counter()
            client.storage.from_("money-rag-files").upload(
                file=local_path,
                path=s3_key,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            upload_ms = (time.perf_counter() - start) * 1000
            logger.debug("Storage upload complete for '%s' in %.1fms", filename, upload_ms)

            table = "BillFile" if is_image else "CSVFile"
            logger.debug("Inserting DB record into %s for '%s'", table, filename)
            
            file_id = db.insert_file_record(table, user["id"], filename, s3_key)

            file_ids.append(file_id)
            uploaded_files_info.append({"path": local_path, "file_id": file_id})
            logger.debug("DB record created — file_id=%s for '%s'", file_id, filename)

    logger.debug(
        "All %d files uploaded — file_ids=%s",
        len(file_ids), file_ids,
    )
    return uploaded_files_info, file_ids


async def upload_and_ingest(user: dict, saved_files: List[dict]) -> List[str]:
    """
    Uploads to storage + creates DB records (in thread), then kicks off
    RAG ingestion in a subprocess. Returns file_ids immediately.
    """
    logger.debug("upload_and_ingest called — %d files for user_id=%s", len(saved_files), user["id"])

    logger.debug("Fetching config for ingestion check")
    config = await config_service.get_config(user)
    if not config:
        logger.warning("No config found — cannot ingest for user_id=%s", user["id"])
        raise ValueError("Account config required before uploading files")

    logger.debug("Starting storage upload in thread for user_id=%s", user["id"])
    uploaded_files_info, file_ids = await asyncio.to_thread(
        _upload_to_storage_sync, user, saved_files
    )
    logger.debug("Storage upload thread complete — %d files uploaded", len(file_ids))

    # Spawn ingestion in a separate PROCESS so primp/duckduckgo can't hold the GIL
    if uploaded_files_info:
        ingestion_status[user["id"]] = {"status": "processing", "error": None}
        logger.info(
            "Spawning ingestion subprocess for user_id=%s — %d files",
            user["id"], len(uploaded_files_info),
        )
        asyncio.create_task(_run_ingestion_subprocess(user, config, uploaded_files_info))
    else:
        logger.debug("No files to ingest for user_id=%s", user["id"])

    return file_ids


async def _run_ingestion_subprocess(user: dict, config: dict, uploaded_files_info: List[dict]):
    """Spawns ingestion_worker.py as a subprocess — fully isolated from the main process."""
    user_id = user["id"]
    logger.debug(
        "Preparing ingestion subprocess for user_id=%s — %d files",
        user_id, len(uploaded_files_info),
    )

    worker_args = json.dumps({
        "config": config,
        "user_id": user_id,
        "access_token": user.get("access_token"),
        "uploaded_files_info": uploaded_files_info,
    })

    worker_script = os.path.join(os.path.dirname(__file__), "ingestion_worker.py")
    logger.debug("Worker script: %s", worker_script)
    logger.debug("Worker args length: %d chars", len(worker_args))

    try:
        logger.debug("Launching subprocess: %s %s", sys.executable, worker_script)
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, worker_script, worker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )
        logger.debug("Subprocess launched — PID=%d, waiting for completion", proc.pid)
        stdout, stderr = await proc.communicate()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if stdout:
            logger.debug("Subprocess stdout:\n%s", stdout.decode())
        if stderr:
            logger.debug("Subprocess stderr:\n%s", stderr.decode())

        if proc.returncode == 0:
            duplicates = []
            if stdout:
                for line in reversed(stdout.decode().strip().split("\n")):
                    try:
                        result = json.loads(line)
                        duplicates = result.get("duplicates", [])
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
            ingestion_status[user_id] = {"status": "complete", "error": None, "duplicates": duplicates}
            logger.info(
                "Background ingestion complete for user_id=%s — PID=%d, %.1fms, %d duplicates",
                user_id, proc.pid, elapsed_ms, len(duplicates),
            )
            await rag_manager.invalidate(user_id)
        else:
            # Extract meaningful error: last 5 lines for context
            stderr_lines = stderr.decode().strip().split("\n") if stderr else []
            error_msg = "\n".join(stderr_lines[-5:]) if stderr_lines else "Unknown error"
            ingestion_status[user_id] = {"status": "failed", "error": error_msg}
            logger.error(
                "Background ingestion FAILED for user_id=%s — returncode=%d, error=%s",
                user_id, proc.returncode, error_msg,
            )
    except Exception as e:
        ingestion_status[user_id] = {"status": "failed", "error": str(e)}
        logger.error(
            "Background ingestion subprocess exception for user_id=%s: %s",
            user_id, e, exc_info=True,
        )


# ── Delete file ─────────────────────────────────────────────────────────────

def _delete_file_sync(access_token: str, file_id: str, file_type: str) -> tuple[str, str]:
    """Sync: look up file record + delete from storage. Returns (filename, file_type)."""
    logger.debug("_delete_file_sync — file_id=%s, type=%s", file_id, file_type)
    client = get_supabase(access_token)
    table = "CSVFile" if file_type == "csv" else "BillFile"
    logger.debug("Looking up file in %s table", table)
    
    with get_db_client(access_token) as db:
        record = db.get_file_record(table, file_id)
        if not record:
            logger.warning("File not found — file_id=%s in table %s", file_id, table)
            raise ValueError("File not found")

        s3_key = record["s3_key"]
        filename = record["filename"]
        logger.debug("Found file '%s' — s3_key=%s, deleting from storage", filename, s3_key)

        try:
            client.storage.from_("money-rag-files").remove([s3_key])
            logger.debug("Storage delete succeeded for s3_key=%s", s3_key)
        except Exception as e:
            logger.warning("Storage delete failed for s3_key=%s: %s", s3_key, e)

    return filename, file_type


async def delete_file(user: dict, file_id: str, file_type: str):
    logger.debug(
        "delete_file called — file_id=%s, type=%s, user_id=%s",
        file_id, file_type, user["id"],
    )

    filename, _ = await asyncio.to_thread(
        _delete_file_sync, user["access_token"], file_id, file_type
    )
    logger.debug("File '%s' removed from storage — now cleaning up RAG data", filename)

    config = await config_service.get_config(user)
    if config:
        logger.debug("Using RAG instance to delete file data for file_id=%s", file_id)
        rag = await rag_manager.get_or_create(user, config)
        await rag.delete_file(file_id, file_type)
        logger.debug("RAG delete_file complete for file_id=%s", file_id)
    else:
        logger.debug("No config — using fallback DB delete for file_id=%s", file_id)
        await asyncio.to_thread(_delete_fallback_sync, user["access_token"], file_id, file_type)
        logger.debug("Fallback delete complete for file_id=%s", file_id)

    logger.info("File '%s' (id=%s) fully deleted for user_id=%s", filename, file_id, user["id"])
    return filename


def _delete_fallback_sync(access_token: str, file_id: str, file_type: str):
    logger.debug("_delete_fallback_sync — file_id=%s, type=%s", file_id, file_type)
    table = "CSVFile" if file_type == "csv" else "BillFile"
    with get_db_client(access_token) as db:
        logger.debug("Deleting %s record for file_id=%s", table, file_id)
        db.delete_file_record(table, file_id)
        logger.debug("Fallback DB delete complete for file_id=%s", file_id)
