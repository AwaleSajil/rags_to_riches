import logging
import os
import shutil
import tempfile
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from backend.dependencies import get_current_user
from backend.services import file_service

logger = logging.getLogger("moneyrag.routers.files")

router = APIRouter()

# Anything not recognised as an image is handed to the CSV parser, so the
# allowlist is what keeps a .pdf or an iPhone .heic from being silently parsed
# as a spreadsheet. Matches the picker's filter in frontend/app/(tabs)/ingest.tsx.
ALLOWED_EXTENSIONS = frozenset({".csv", ".png", ".jpg", ".jpeg"})

# Receipts and bank exports are small; this is generous for both and keeps a
# single request from filling the disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _safe_filename(raw: str | None) -> str:
    """Reduce a client-supplied filename to a bare, allowlisted basename.

    The name reaches both `os.path.join(temp_dir, ...)` and the storage key, so
    a value like `../../etc/passwd` would otherwise escape the temp directory
    and land somewhere it shouldn't. Strips any directory component (both
    separators, since the client may be on Windows) plus null bytes, then
    requires a known extension.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].replace("\x00", "").strip()
    # Leading dots would make the file hidden, or resolve to "." / ".." outright.
    name = name.lstrip(".")
    if not name:
        raise ValueError("A file was uploaded without a usable filename")

    extension = os.path.splitext(name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"'{name}' has an unsupported type. Upload a CSV or a PNG/JPG image."
        )
    return name


async def _save_within_limit(upload: UploadFile, destination: str) -> int:
    """Stream one upload to disk, aborting if it exceeds MAX_UPLOAD_BYTES.

    Written in chunks rather than a single `.read()` so a large file is never
    held in memory in full.
    """
    total = 0
    with open(destination, "wb") as fh:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"'{upload.filename}' is larger than the "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit"
                )
            fh.write(chunk)
    if total == 0:
        raise ValueError(f"'{upload.filename}' is empty")
    return total


@router.get("")
async def list_files(user: dict = Depends(get_current_user)):
    logger.debug("Listing files for user_id=%s", user["id"])
    try:
        files = await file_service.list_files(user)
        logger.debug("Found %d files for user_id=%s", len(files), user["id"])
        return {"files": files}
    except Exception as e:
        logger.error("Failed to list files for user_id=%s: %s", user["id"], e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load files: {e}")


@router.post("/upload")
async def upload_files(
    files: Annotated[List[UploadFile], File(description="CSV or image files to upload")],
    user: dict = Depends(get_current_user),
):
    logger.debug(
        "Upload request from user_id=%s — %d file(s): %s",
        user["id"], len(files), [f.filename for f in files],
    )
    if not files:
        logger.warning("Empty file upload from user_id=%s", user["id"])
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    logger.debug("Created temp dir: %s", temp_dir)
    saved_files = []

    try:
        for f in files:
            filename = _safe_filename(f.filename)
            local_path = os.path.join(temp_dir, filename)
            written = await _save_within_limit(f, local_path)
            logger.debug("Saved file '%s' (%d bytes) to %s", filename, written, local_path)
            saved_files.append({"local_path": local_path, "filename": filename})

        logger.debug("All files saved to temp — calling upload_and_ingest")
        file_ids = await file_service.upload_and_ingest(user, saved_files)
        logger.info(
            "Upload complete for user_id=%s — file_ids=%s",
            user["id"], file_ids,
        )
        return {
            "message": f"Uploaded {len(file_ids)} file(s). Ingestion is processing in the background.",
            "file_ids": file_ids,
        }
    except ValueError as e:
        # Rejected before ingestion started, so nothing else is reading temp_dir.
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Upload validation error for user_id=%s: %s", user["id"], e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # upload_and_ingest only spawns its background task on success, so a
        # raise here likewise means no one is left holding these paths.
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Upload failed for user_id=%s: %s", user["id"], e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/ingestion-status")
async def get_ingestion_status(user: dict = Depends(get_current_user)):
    """Poll this to check if background ingestion is done."""
    status = file_service.ingestion_status.get(user["id"])
    logger.debug("Ingestion status for user_id=%s: %s", user["id"], status)
    if not status:
        return {"status": "idle"}
    return status


@router.patch("/{file_id}/visibility")
async def set_file_visibility(
    file_id: str,
    type: str = Query(..., description="File type: csv or bill"),
    hidden: bool = Query(..., description="Whether to hide this file's transactions"),
    user: dict = Depends(get_current_user),
):
    try:
        is_hidden = await file_service.set_file_visibility(user, file_id, type, hidden)
        return {"message": "File visibility updated", "is_hidden": is_hidden}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    type: str = Query(..., description="File type: csv or bill"),
    user: dict = Depends(get_current_user),
):
    logger.debug(
        "Delete request — file_id=%s, type=%s, user_id=%s",
        file_id, type, user["id"],
    )
    try:
        filename = await file_service.delete_file(user, file_id, type)
        logger.info("Deleted file '%s' (id=%s) for user_id=%s", filename, file_id, user["id"])
        return {"message": f"Deleted {filename}"}
    except ValueError as e:
        logger.warning("File not found — file_id=%s: %s", file_id, e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Delete failed for file_id=%s: %s", file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
