import logging
import os
import shutil
import tempfile
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from backend.dependencies import get_current_user
from backend.services import file_service
from backend.services.upload_utils import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    safe_filename as _safe_filename,
    save_within_limit as _save_within_limit,
)

logger = logging.getLogger("moneyrag.routers.files")

router = APIRouter()

# ALLOWED_EXTENSIONS / MAX_UPLOAD_BYTES / the two helpers now live in
# services/upload_utils so the single-photo /captures route enforces exactly the
# same rules. Re-exported under their original names because the tests and the
# rest of this module already refer to them that way.
__all__ = ["router", "ALLOWED_EXTENSIONS", "MAX_UPLOAD_BYTES"]


@router.get("")
async def list_files(user: dict = Depends(get_current_user)):
    logger.debug("Listing files for user_id=%s", user["id"])
    files = await file_service.list_files(user)
    logger.debug("Found %d files for user_id=%s", len(files), user["id"])
    return {"files": files}


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
        file_ids, already_imported = await file_service.upload_and_ingest(user, saved_files)
        logger.info(
            "Upload complete for user_id=%s — file_ids=%s, skipped=%d",
            user["id"], file_ids, len(already_imported),
        )
        if file_ids:
            message = f"Uploaded {len(file_ids)} file(s). Ingestion is processing in the background."
        else:
            # Everything was a re-upload. Saying "uploaded 0 files" reads as a
            # failure; this is the system working.
            message = "Nothing new to import — you already have these."
        return {
            "message": message,
            "file_ids": file_ids,
            "already_imported": already_imported,
        }
    except ValueError as e:
        # Rejected before ingestion started, so nothing else is reading temp_dir.
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Upload validation error for user_id=%s: %s", user["id"], e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Real work, not boilerplate: upload_and_ingest only spawns its
        # background task on success, so a raise here likewise means no one is
        # left holding these paths. Re-raised so the global handler logs and
        # formats it like every other failure.
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@router.get("/ingestion-status")
async def get_ingestion_status(user: dict = Depends(get_current_user)):
    """Poll this to check if background ingestion is done."""
    status = file_service.ingestion_status.get(user["id"])
    logger.debug("Ingestion status for user_id=%s: %s", user["id"], status)
    if not status:
        return {"status": "idle"}
    return status


# Registered AFTER every literal path above. FastAPI matches routes in
# declaration order, so a "/{file_id}" placed earlier would swallow
# GET /files/ingestion-status as a file called "ingestion-status" and silently
# break upload polling.
@router.get("/{file_id}")
async def get_file(file_id: str, user: dict = Depends(get_current_user)):
    """One of the user's files, in the same shape as the list.

    404 rather than 403 for someone else's id: the underlying list is already
    scoped to this user, so a missing file and a forbidden one are
    indistinguishable here — which is the safer of the two to report.
    """
    file = await file_service.get_file(user, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file


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


@router.patch("/{file_id}/rotation")
async def set_file_rotation(
    file_id: str,
    degrees: int = Query(..., description="Quarter turns clockwise: 0, 90, 180 or 270"),
    user: dict = Depends(get_current_user),
):
    """Persist a photo's viewing orientation. Does not touch the stored image."""
    try:
        rotation = await file_service.set_file_rotation(user, file_id, degrees)
        return {"message": "Rotation saved", "rotation": rotation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
