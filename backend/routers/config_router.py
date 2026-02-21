from fastapi import APIRouter, Depends, HTTPException
from backend.dependencies import get_current_user
from backend.schemas.config_schema import ConfigUpdate
from backend.services import config_service
from backend.services.rag_manager import rag_manager

router = APIRouter()


@router.get("/")
async def get_config(user: dict = Depends(get_current_user)):
    config = await config_service.get_config(user)
    return config


@router.put("/")
async def update_config(body: ConfigUpdate, user: dict = Depends(get_current_user)):
    try:
        record = await config_service.upsert_config(user, body.model_dump())
        # Invalidate cached RAG instance so it picks up new config
        await rag_manager.invalidate(user["id"])
        return record
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")
