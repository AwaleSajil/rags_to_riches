import asyncio

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client, ClientOptions
from backend.config import get_settings, Settings

bearer_scheme = HTTPBearer()


def get_supabase(access_token: str | None = None):
    settings = get_settings()
    if access_token:
        opts = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY, options=opts)
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def _validate_token_sync(token: str, supabase_url: str, supabase_key: str) -> dict:
    """Sync Supabase auth call — runs in thread pool to avoid blocking the event loop."""
    client = create_client(supabase_url, supabase_key)
    res = client.auth.get_user(token)
    if not res or not res.user:
        raise ValueError("Invalid token")
    return {"id": res.user.id, "email": res.user.email}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> dict:
    token = credentials.credentials
    try:
        user_info = await asyncio.to_thread(
            _validate_token_sync, token, settings.SUPABASE_URL, settings.SUPABASE_KEY
        )
        return {**user_info, "access_token": token}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {e}")
