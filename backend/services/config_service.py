import asyncio

from backend.dependencies import get_supabase


def _get_config_sync(access_token: str, user_id: str) -> dict | None:
    client = get_supabase(access_token)
    res = client.table("AccountConfig").select("*").eq("user_id", user_id).execute()
    if res.data:
        return res.data[0]
    return None


def _upsert_config_sync(access_token: str, user_id: str, data: dict) -> dict:
    client = get_supabase(access_token)
    record = {
        "user_id": user_id,
        "llm_provider": data["llm_provider"],
        "api_key": data["api_key"],
        "decode_model": data["decode_model"],
        "embedding_model": data["embedding_model"],
    }
    existing = client.table("AccountConfig").select("id").eq("user_id", user_id).execute()
    if existing.data:
        client.table("AccountConfig").update(record).eq("id", existing.data[0]["id"]).execute()
    else:
        client.table("AccountConfig").insert(record).execute()
    return record


async def get_config(user: dict) -> dict | None:
    return await asyncio.to_thread(_get_config_sync, user["access_token"], user["id"])


async def upsert_config(user: dict, data: dict) -> dict:
    return await asyncio.to_thread(_upsert_config_sync, user["access_token"], user["id"], data)
