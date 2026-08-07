import logging
import os
import shutil
import tempfile
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.dependencies import get_current_user
from backend.services import capture_service
from backend.services.upload_utils import IMAGE_EXTENSIONS, safe_filename, save_within_limit

logger = logging.getLogger("moneyrag.routers.captures")

router = APIRouter()


@router.post("")
async def create_capture(
    file: Annotated[UploadFile, File(description="A single photo: receipt or price tag")],
    location: Annotated[Optional[str], Form()] = None,
    user: dict = Depends(get_current_user),
):
    """Upload one photo, classify it, and return the extracted draft.

    Runs inline rather than handing off to the ingestion subprocess: the caller
    is a chat message, not a batch import, so it waits for the answer instead of
    polling. Nothing is committed — the draft is confirmed by the user first.

    `location` is an already-resolved place name ("Main St, Norwalk"), optional
    and only used to say where a price tag was seen. The device does the GPS fix
    and the reverse geocode, so no coordinate reaches the server. Location
    capture is opt-in client-side and every downstream step works without it.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Only images here; a CSV arriving on this route is a client bug.
        filename = safe_filename(file.filename, allowed=IMAGE_EXTENSIONS)
        local_path = os.path.join(temp_dir, filename)
        written = await save_within_limit(file, local_path)
        logger.info(
            "Capture received from user_id=%s: %s (%d bytes), located=%s",
            user["id"], filename, written, location is not None,
        )

        result = await capture_service.capture_photo(
            user, local_path, filename, location=location,
        )
        logger.info(
            "Capture classified for user_id=%s file_id=%s kind=%s",
            user["id"], result["file_id"], result["kind"],
        )
        return result
    except ValueError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Capture failed for user_id=%s: %s", user["id"], e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process that photo: {e}")
    # No cleanup on the success path: reading the photo now happens AFTER this
    # response, and the background task still needs these bytes. It removes the
    # directory itself when it finishes.


@router.get("/{file_id}")
async def get_capture(
    file_id: str,
    user: dict = Depends(get_current_user),
):
    """Re-open a photo that was read earlier.

    Reached when a price tag (or an unclassified photo) arrives from the batch
    upload path, which returns only a file id — the app needs the draft back to
    render the card that confirms it.
    """
    try:
        return await capture_service.get_capture(user, file_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Reading capture %s failed: %s", file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load that photo: {e}")


@router.delete("/{file_id}")
async def discard_capture(
    file_id: str,
    user: dict = Depends(get_current_user),
):
    """Throw away a captured photo.

    An unconfirmed capture was never stored, so this just drops what is held in
    memory and its temp file. Once confirmed it is an ordinary BillFile and goes
    through the usual delete, which takes its observations with it.
    """
    if capture_service.forget_pending(file_id, user["id"]):
        return {"message": "Discarded"}
    try:
        from backend.services import file_service

        filename = await file_service.delete_file(user, file_id, "bill")
        return {"message": f"Deleted {filename}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Discarding capture %s failed: %s", file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not discard that photo: {e}")


@router.post("/{file_id}/kind")
async def set_capture_kind(
    file_id: str,
    kind: Annotated[str, Form(description="receipt or price_tag")],
    user: dict = Depends(get_current_user),
):
    """Record the user's answer for a photo the model could not classify.

    Reached from the "Receipt or price tag?" prompt. Deliberately a user
    decision rather than a lower confidence threshold: a price tag filed as a
    receipt invents spending that never happened.
    """
    try:
        return await capture_service.set_kind(user, file_id, kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Setting capture kind failed for %s: %s", file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not update that photo: {e}")
