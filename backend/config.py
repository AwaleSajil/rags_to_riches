import base64
import binascii
import json
import logging
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

logger = logging.getLogger("moneyrag.config")

# Local Expo dev servers. Used only when ALLOWED_ORIGINS is unset, so a
# development machine works out of the box without opening production up.
DEFAULT_DEV_ORIGINS = (
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:19006",
)


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str
    # Fernet key for secrets stored in the database (backend/crypto.py).
    # Declared here so it can live in .env alongside everything else — the app
    # refuses to start when it is blank, so the default is not a fallback.
    APP_ENCRYPTION_KEY: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    logger.debug("Loading settings from .env")
    settings = Settings()
    logger.debug(
        "Settings loaded — SUPABASE_URL=%s, DATABASE_URL=%s",
        settings.SUPABASE_URL,
        settings.DATABASE_URL[:30] + "..." if len(settings.DATABASE_URL) > 30 else settings.DATABASE_URL,
    )
    return settings


def allowed_origins() -> list[str]:
    """Browser origins permitted to call the API.

    The mobile app does not use CORS at all and the packaged web build is served
    same-origin by this process, so in production this list is usually empty or
    a single marketing domain — not the wildcard it used to be.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw == "*":
        # Still reachable, but it has to be asked for, and it is said out loud.
        logger.warning("ALLOWED_ORIGINS=* — every website may call this API")
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if origins:
        return origins
    logger.info("ALLOWED_ORIGINS unset — allowing local dev origins only")
    return list(DEFAULT_DEV_ORIGINS)


def _jwt_role(token: str) -> str | None:
    """The `role` claim of a Supabase JWT, without verifying the signature.

    Only used to tell an anon key from a service_role key. We are inspecting our
    own configuration, not trusting a caller, so verification is beside the
    point — the question is which key the operator pasted in.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload)).get("role")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None


def verify_public_key_is_not_privileged() -> None:
    """Refuse to start if SUPABASE_KEY is a key that bypasses RLS.

    `/api/v1/public-config` serves this value to any unauthenticated caller,
    because the frontend needs it to talk to Supabase. That is correct for the
    anon/publishable key and catastrophic for a service_role key, which ignores
    every row-level security policy in the database.

    The two are interchangeable-looking strings from the same dashboard page and
    the service key makes local experiments "just work", so this is an easy
    mistake to make once and never notice. Failing at boot is the only point
    where noticing is guaranteed.
    """
    key = get_settings().SUPABASE_KEY or ""
    role = _jwt_role(key)
    if key.startswith("sb_secret_") or role == "service_role":
        raise RuntimeError(
            "SUPABASE_KEY looks like a Supabase SERVICE key "
            f"({'sb_secret_ prefix' if key.startswith('sb_secret_') else 'role=service_role'}). "
            "It is served publicly by /api/v1/public-config and bypasses row-level "
            "security. Use the anon / publishable key instead."
        )
    logger.info("SUPABASE_KEY role=%s — safe to expose publicly", role or "publishable")
