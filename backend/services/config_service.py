import asyncio
import logging

from backend.db_client import get_db_client

logger = logging.getLogger("moneyrag.services.config")


def _get_config_sync(access_token: str, user_id: str) -> dict | None:
    logger.debug("Querying AccountConfig for user_id=%s", user_id)
    with get_db_client(access_token) as db:
        res = db.get_account_config(user_id)
        if res:
            logger.debug(
                "AccountConfig found for user_id=%s — provider=%s, model=%s",
                user_id, res.get("llm_provider"), res.get("decode_model"),
            )
            return res
        logger.debug("No AccountConfig found for user_id=%s", user_id)
        return None


def _upsert_config_sync(access_token: str, user_id: str, data: dict) -> dict:
    logger.debug(
        "Upserting AccountConfig for user_id=%s — provider=%s, model=%s, embedding=%s",
        user_id, data["llm_provider"], data["decode_model"], data["embedding_model"],
    )
    with get_db_client(access_token) as db:
        record = db.upsert_account_config(user_id, data)
        logger.debug("AccountConfig upsert complete for user_id=%s", user_id)
        return record


async def get_config(user: dict) -> dict | None:
    logger.debug("get_config called for user_id=%s", user["id"])
    result = await asyncio.to_thread(_get_config_sync, user["access_token"], user["id"])
    logger.debug("get_config returning %s for user_id=%s", "config" if result else "None", user["id"])
    return result


async def upsert_config(user: dict, data: dict) -> dict:
    logger.debug("upsert_config called for user_id=%s", user["id"])
    result = await asyncio.to_thread(_upsert_config_sync, user["access_token"], user["id"], data)
    logger.debug("upsert_config complete for user_id=%s", user["id"])
    return result
