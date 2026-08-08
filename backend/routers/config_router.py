import logging

from fastapi import APIRouter, Depends, HTTPException
from backend.crypto import mask_secret
from backend.dependencies import get_current_user
from backend.schemas.config_schema import ConfigResponse, ConfigUpdate
from backend.services import config_service
from backend.services.rag_manager import rag_manager

logger = logging.getLogger("moneyrag.routers.config")

router = APIRouter()


def _public_config(config: dict) -> ConfigResponse:
    """Strip the API key before anything goes back over the wire.

    Every route in this module goes through here. The config dict carries a
    PLAINTEXT key (db_client decrypts on read, and upsert returns what it was
    given), so returning it directly — which is what used to happen on both GET
    and PUT — handed the user's credentials back to the client on every save.
    """
    key = config.get("api_key") or ""
    return ConfigResponse(
        id=config.get("id"),
        user_id=config["user_id"],
        llm_provider=config.get("llm_provider") or "",
        decode_model=config.get("decode_model") or "",
        embedding_model=config.get("embedding_model") or "",
        deep_enrichment=bool(config.get("deep_enrichment", False)),
        api_key_set=bool(key),
        api_key_hint=mask_secret(key),
    )


@router.get("", response_model=ConfigResponse | None)
async def get_config(user: dict = Depends(get_current_user)):
    logger.debug("GET config for user_id=%s", user["id"])
    config = await config_service.get_config(user)
    logger.debug(
        "Config result for user_id=%s: %s",
        user["id"],
        "found" if config else "not found",
    )
    return _public_config(config) if config else None


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

        payload = body.model_dump()
        # A blank key means "leave it alone", not "erase it". The client no
        # longer receives the key, so it cannot send one back when the user is
        # only switching model — without this, saving a model change would wipe
        # the stored credentials.
        if not (payload.get("api_key") or "").strip():
            payload["api_key"] = (old_config or {}).get("api_key") or ""
        if not payload["api_key"]:
            raise HTTPException(status_code=400, detail="An API key is required.")

        record = await config_service.upsert_config(user, payload)
        logger.debug("Config saved for user_id=%s — invalidating RAG cache", user["id"])
        
        # Invalidate cached RAG instance so it picks up new config
        await rag_manager.invalidate(user["id"])
        
        new_embed = record.get("embedding_model")
        if old_embed and old_embed != new_embed:
            logger.info("Embedding model changed from %s to %s for user_id=%s — triggering vector sync", old_embed, new_embed, user["id"])
            from backend.services.file_service import _run_ingestion_subprocess, ingestion_status
            from backend.services import background
            ingestion_status[user["id"]] = {"status": "processing", "error": None}
            # Pass empty file list; worker will re-embed existing DB transactions into pgvector
            background.spawn(
                _run_ingestion_subprocess(user, record, []),
                name=f"reembed:{user['id']}",
            )
            
        logger.info("Config updated and RAG invalidated for user_id=%s", user["id"])
        # `record` holds the plaintext key the subprocess above needs — strip it
        # before responding.
        return _public_config(record)
    except HTTPException:
        # A deliberate 4xx (e.g. the missing-key check above). Without this the
        # catch-all below would relabel it a 500 and hide the real reason.
        raise
