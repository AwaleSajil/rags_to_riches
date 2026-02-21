import os
import tempfile
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from backend.dependencies import get_current_user
from backend.services import file_service

router = APIRouter()


@router.get("/")
async def list_files(user: dict = Depends(get_current_user)):
    try:
        files = await file_service.list_files(user)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load files: {e}")


@router.post("/upload")
async def upload_files(
    files: Annotated[List[UploadFile], File(description="CSV or image files to upload")],
    user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    saved_files = []

    try:
        for f in files:
            local_path = os.path.join(temp_dir, f.filename)
            content = await f.read()
            with open(local_path, "wb") as fh:
                fh.write(content)
            saved_files.append({"local_path": local_path, "filename": f.filename})

        file_ids = await file_service.upload_and_ingest(user, saved_files)
        return {
            "message": f"Uploaded {len(file_ids)} file(s). Ingestion is processing in the background.",
            "file_ids": file_ids,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/ingestion-status")
async def get_ingestion_status(user: dict = Depends(get_current_user)):
    """Poll this to check if background ingestion is done."""
    status = file_service.ingestion_status.get(user["id"])
    if not status:
        return {"status": "idle"}
    return status


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    type: str = Query(..., description="File type: csv or bill"),
    user: dict = Depends(get_current_user),
):
    try:
        filename = await file_service.delete_file(user, file_id, type)
        return {"message": f"Deleted {filename}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
