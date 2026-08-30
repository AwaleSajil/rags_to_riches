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
    # Supabase SERVICE key. Bypasses RLS, so it is kept strictly apart from
    # SUPABASE_KEY above, which is handed to every anonymous caller by
    # /public-config. Used on exactly one path — deleting a user's auth record,
    # which the anon key cannot do and which both app stores require — via
    # dependencies.admin_client(). Optional so an existing deploy still boots;
    # account deletion returns a clear error until it is set.
    SUPABASE_SERVICE_KEY: str = ""
    # Shown on the public account-deletion page, which Play requires to be
    # reachable without installing the app. Left obviously unset rather than
    # defaulted to a plausible-looking address: a deletion request that silently
    # goes nowhere is worse than a page that admits it is unconfigured.
    SUPPORT_EMAIL: str = ""
    # Who publishes the app — the party a privacy policy has to name as the one
    # responsible for the data. Blank renders as an obvious placeholder rather
    # than a guess, because naming the wrong entity in a published policy is
    # worse than admitting it is unfilled.
    PUBLISHER_NAME: str = ""

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


def verify_service_key_is_privileged() -> None:
    """Warn at startup if the service key is missing or is not actually one.

    A warning rather than a refusal, deliberately: this key exists for account
    deletion, and a deployment that predates it should still boot. But both
    stores require deletion to work, so it must not be possible to ship without
    noticing — a silently disabled delete button is the version of this failure
    that gets an app pulled rather than rejected.

    The wrong-key case is the one worth catching loudly. An anon key pasted here
    fails only at the moment a user asks to delete their account, deep inside a
    Supabase permission error, which is the worst possible time to find out.
    """
    key = (get_settings().SUPABASE_SERVICE_KEY or "").strip()
    if not key:
        logger.warning(
            "SUPABASE_SERVICE_KEY is not set — account deletion will be "
            "unavailable. Both app stores require a working in-app delete, so "
            "set it before submitting a build."
        )
        return
    role = _jwt_role(key)
    if not (key.startswith("sb_secret_") or role == "service_role"):
        # Named specifically, because the dashboard shows the publishable key
        # prominently and hides the secret one behind a Reveal (or a Create),
        # so reaching for the wrong one is the easy mistake rather than an
        # exotic one. "Not a service key" sends someone back to a page where
        # everything looks like what they already copied.
        looks_like = (
            "the PUBLISHABLE key (sb_publishable_…), which is public"
            if key.startswith("sb_publishable_")
            else "a personal access token (sbp_…), which is account-level"
            if key.startswith("sbp_")
            else f"the ANON key (role={role}), which is public"
            if role == "anon"
            else f"an unrecognised key (role={role or 'not a JWT'})"
        )
        logger.error(
            "SUPABASE_SERVICE_KEY is %s — not a service key. Account deletion "
            "will fail at the moment a user asks for it. Use the secret key "
            "(sb_secret_…) or the legacy service_role key from Project "
            "Settings -> API Keys.",
            looks_like,
        )
        return
    logger.info("SUPABASE_SERVICE_KEY present — account deletion is available")
