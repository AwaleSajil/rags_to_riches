import logging

from fastapi import APIRouter, Depends, HTTPException
from backend.dependencies import get_current_user
from backend.schemas.config_schema import ConfigUpdate
from backend.services import config_service
from backend.services.rag_manager import rag_manager

logger = logging.getLogger("moneyrag.routers.config")

router = APIRouter()


@router.get("")
async def get_config(user: dict = Depends(get_current_user)):
    logger.debug("GET config for user_id=%s", user["id"])
    config = await config_service.get_config(user)
    logger.debug(
        "Config result for user_id=%s: %s",
        user["id"],
        "found" if config else "not found",
    )
    return config


@router.put("")
async def update_config(body: ConfigUpdate, user: dict = Depends(get_current_user)):
    logger.debug(
        "PUT config for user_id=%s — provider=%s, model=%s",
        user["id"], body.llm_provider, body.decode_model,
    )
    try:
        # Check old config to see if embedding model changed
        old_config = await config_service.get_config(user)
        old_embed = old_config.get("embedding_model") if old_config else None
        
        record = await config_service.upsert_config(user, body.model_dump())
        logger.debug("Config saved for user_id=%s — invalidating RAG cache", user["id"])
        
        # Invalidate cached RAG instance so it picks up new config
        await rag_manager.invalidate(user["id"])
        
        new_embed = record.get("embedding_model")
        if old_embed and old_embed != new_embed:
            logger.info("Embedding model changed from %s to %s for user_id=%s — triggering vector sync", old_embed, new_embed, user["id"])
            from backend.services.file_service import _run_ingestion_subprocess, ingestion_status
            import asyncio
            ingestion_status[user["id"]] = {"status": "processing", "error": None}
            # Pass empty file list; worker will just sync existing DB transactions to new Qdrant collection
            asyncio.create_task(_run_ingestion_subprocess(user, record, []))
            
        logger.info("Config updated and RAG invalidated for user_id=%s", user["id"])
        return record
    except Exception as e:
        logger.error("Failed to save config for user_id=%s: %s", user["id"], e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")
