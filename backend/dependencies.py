import asyncio
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client, ClientOptions
from backend.config import get_settings, Settings

logger = logging.getLogger("moneyrag.dependencies")

bearer_scheme = HTTPBearer(auto_error=False)


def get_supabase(access_token: str | None = None):
    settings = get_settings()
    if access_token:
        logger.debug("Creating Supabase client WITH access_token (token=%s...)", access_token[:20])
        opts = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY, options=opts)
    logger.debug("Creating Supabase client WITHOUT access_token (service-level)")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def _validate_token_sync(token: str, supabase_url: str, supabase_key: str) -> dict:
    """Sync Supabase auth call — runs in thread pool to avoid blocking the event loop."""
    logger.debug("Validating token via Supabase auth.get_user (token=%s...)", token[:20])
    client = create_client(supabase_url, supabase_key)
    res = client.auth.get_user(token)
    if not res or not res.user:
        logger.warning("Token validation failed — no user returned")
        raise ValueError("Invalid token")
    logger.debug("Token validated — user_id=%s, email=%s", res.user.id, res.user.email)
    return {"id": res.user.id, "email": res.user.email}


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> dict | None:
    """Resolve the bearer token if one was sent, else None.

    Only for endpoints that legitimately serve anonymous callers — currently
    just login/register, which accept either a token or email+password. Every
    other route wants `get_current_user`, which rejects anonymous requests
    instead of handing back a None that the route would dereference.
    """
    if not credentials:
        return None

    token = credentials.credentials
    logger.debug("get_optional_user called — token=%s...", token[:20])
    try:
        user_info = await asyncio.to_thread(
            _validate_token_sync, token, settings.SUPABASE_URL, settings.SUPABASE_KEY
        )
        logger.debug("Authenticated user: id=%s, email=%s", user_info["id"], user_info["email"])
        return {**user_info, "access_token": token}
    except ValueError as e:
        logger.warning("Auth failed (ValueError): %s", e)
        raise HTTPException(status_code=401, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Auth failed (unexpected): %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail=f"Token validation failed: {e}")


async def get_current_user(
    user: dict | None = Depends(get_optional_user),
) -> dict:
    """The authenticated user, or a 401. Never returns None.

    Routes annotate `user: dict` and immediately index `user["id"]`, so an
    anonymous request used to raise TypeError and surface as a 500 — this
    turns that into the 401 the client is equipped to handle and retry.
    """
    if user is None:
        logger.debug("Rejecting unauthenticated request — no bearer credentials")
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
