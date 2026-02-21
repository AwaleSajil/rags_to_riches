import asyncio
import json
import os
import sys
from typing import List
from backend.dependencies import get_supabase
from backend.services.rag_manager import rag_manager
from backend.services import config_service

# Per-user ingestion status: user_id -> {"status": "processing"|"complete"|"failed", "error": str|None}
ingestion_status: dict[str, dict] = {}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── List files ──────────────────────────────────────────────────────────────

def _list_files_sync(access_token: str, user_id: str) -> List[dict]:
    client = get_supabase(access_token)
    res_csv = client.table("CSVFile").select("*").eq("user_id", user_id).execute()
    res_bill = client.table("BillFile").select("*").eq("user_id", user_id).execute()

    files = []
    for d in (res_csv.data or []):
        d["type"] = "csv"
        files.append(d)
    for d in (res_bill.data or []):
        d["type"] = "bill"
        files.append(d)
    return files


async def list_files(user: dict) -> List[dict]:
    return await asyncio.to_thread(_list_files_sync, user["access_token"], user["id"])


# ── Upload + ingest ─────────────────────────────────────────────────────────

def _upload_to_storage_sync(user: dict, saved_files: List[dict]) -> tuple[list, list]:
    """Uploads files to Supabase storage + creates DB records."""
    client = get_supabase(user["access_token"])
    uploaded_files_info = []
    file_ids = []

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

        client.storage.from_("money-rag-files").upload(
            file=local_path,
            path=s3_key,
            file_options={"content-type": content_type, "upsert": "true"},
        )

        table = "BillFile" if is_image else "CSVFile"
        file_record = client.table(table).insert({
            "user_id": user["id"],
            "filename": filename,
            "s3_key": s3_key,
        }).execute()

        file_id = file_record.data[0]["id"]
        file_ids.append(file_id)
        uploaded_files_info.append({"path": local_path, "file_id": file_id})

    return uploaded_files_info, file_ids


async def upload_and_ingest(user: dict, saved_files: List[dict]) -> List[str]:
    """
    Uploads to storage + creates DB records (in thread), then kicks off
    RAG ingestion in a subprocess. Returns file_ids immediately.
    """
    config = await config_service.get_config(user)
    if not config:
        raise ValueError("Account config required before uploading files")

    uploaded_files_info, file_ids = await asyncio.to_thread(
        _upload_to_storage_sync, user, saved_files
    )

    # Spawn ingestion in a separate PROCESS so primp/duckduckgo can't hold the GIL
    if uploaded_files_info:
        ingestion_status[user["id"]] = {"status": "processing", "error": None}
        asyncio.create_task(_run_ingestion_subprocess(user, config, uploaded_files_info))

    return file_ids


async def _run_ingestion_subprocess(user: dict, config: dict, uploaded_files_info: List[dict]):
    """Spawns ingestion_worker.py as a subprocess — fully isolated from the main process."""
    user_id = user["id"]
    worker_args = json.dumps({
        "config": config,
        "user_id": user_id,
        "access_token": user.get("access_token"),
        "uploaded_files_info": uploaded_files_info,
    })

    worker_script = os.path.join(os.path.dirname(__file__), "ingestion_worker.py")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, worker_script, worker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )
        stdout, stderr = await proc.communicate()

        if stdout:
            print(stdout.decode(), end="")
        if stderr:
            print(stderr.decode(), end="")

        if proc.returncode == 0:
            ingestion_status[user_id] = {"status": "complete", "error": None}
            print(f"✅ Background ingestion complete for user {user_id}")
            await rag_manager.invalidate(user_id)
        else:
            error_msg = stderr.decode().strip().split("\n")[-1] if stderr else "Unknown error"
            ingestion_status[user_id] = {"status": "failed", "error": error_msg}
            print(f"❌ Background ingestion failed for user {user_id}: {error_msg}")
    except Exception as e:
        ingestion_status[user_id] = {"status": "failed", "error": str(e)}
        print(f"❌ Background ingestion failed for user {user_id}: {e}")


# ── Delete file ─────────────────────────────────────────────────────────────

def _delete_file_sync(access_token: str, file_id: str, file_type: str) -> tuple[str, str]:
    """Sync: look up file record + delete from storage. Returns (filename, file_type)."""
    client = get_supabase(access_token)
    table = "CSVFile" if file_type == "csv" else "BillFile"
    record = client.table(table).select("*").eq("id", file_id).execute()
    if not record.data:
        raise ValueError("File not found")

    try:
        client.storage.from_("money-rag-files").remove([record.data[0]["s3_key"]])
    except Exception as e:
        print(f"Warning: storage delete failed: {e}")

    return record.data[0]["filename"], file_type


async def delete_file(user: dict, file_id: str, file_type: str):
    filename, _ = await asyncio.to_thread(
        _delete_file_sync, user["access_token"], file_id, file_type
    )

    config = await config_service.get_config(user)
    if config:
        rag = await rag_manager.get_or_create(user, config)
        await rag.delete_file(file_id, file_type)
    else:
        await asyncio.to_thread(_delete_fallback_sync, user["access_token"], file_id, file_type)

    return filename


def _delete_fallback_sync(access_token: str, file_id: str, file_type: str):
    client = get_supabase(access_token)
    table = "CSVFile" if file_type == "csv" else "BillFile"
    if file_type == "csv":
        client.table("Transaction").delete().eq("source_csv_id", file_id).execute()
    client.table(table).delete().eq("id", file_id).execute()
