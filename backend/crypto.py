"""Encryption for secrets held in the database.

LLM API keys used to sit in ``AccountConfig.api_key`` as plaintext. Anyone who
could read the database — a dump, a backup, a mis-scoped RLS policy, a support
query — could read every user's API credentials and spend their money. They are
now encrypted with Fernet (AES-128-CBC + HMAC-SHA256) under a key that lives
only in the environment, so the database alone is no longer enough.

The key is NOT optional. Making it optional would mean a deployment that forgot
to set it silently falls back to storing plaintext, which is exactly the failure
this module exists to prevent — so a missing key raises, loudly, at startup.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("moneyrag.crypto")

ENV_VAR = "APP_ENCRYPTION_KEY"

# Tags values this module produced. A stored value without it predates
# encryption and is legacy plaintext — see decrypt_secret.
_PREFIX = "enc:v1:"

_MISSING_KEY_HELP = (
    f"{ENV_VAR} is not set, so API keys cannot be encrypted at rest.\n"
    "Generate one with:\n"
    '  python -c "from cryptography.fernet import Fernet; '
    'print(Fernet.generate_key().decode())"\n'
    f"then add {ENV_VAR}=<that value> to .env (locally) or set it as a "
    "repository/Space secret (in deployment).\n"
    "Keep it safe: losing it makes every stored API key unreadable, and users "
    "will have to re-enter theirs."
)


class EncryptionKeyError(RuntimeError):
    """Raised when the encryption key is missing or malformed."""


def _key_material() -> str:
    """The key, from the environment or from .env.

    Both, and in that order. pydantic-settings reads .env into a Settings object
    and NOT into os.environ, so reading os.environ alone meant a key sitting in
    .env — where every other secret in this project lives — was invisible, and
    the app refused to start while looking correctly configured.

    os.environ still wins, so tests and scripts can override without a file.
    """
    raw = os.environ.get(ENV_VAR, "").strip()
    if raw:
        return raw
    try:
        from backend.config import get_settings

        return (get_settings().APP_ENCRYPTION_KEY or "").strip()
    except Exception:
        # Settings itself failing is a different, louder problem; don't mask it
        # behind a confusing message about encryption keys.
        return ""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    raw = _key_material()
    if not raw:
        raise EncryptionKeyError(_MISSING_KEY_HELP)
    try:
        return Fernet(raw.encode())
    except Exception as exc:
        raise EncryptionKeyError(
            f"{ENV_VAR} is set but is not a valid Fernet key "
            f"(expected 32 url-safe base64-encoded bytes). {_MISSING_KEY_HELP}"
        ) from exc


def verify_encryption_key() -> None:
    """Fail fast at startup rather than at the first config save.

    Called from the app's lifespan. Without it, a deployment missing the key
    boots happily and only breaks when someone tries to save their API key —
    at which point the failure looks like an unrelated 500.
    """
    _fernet()
    logger.info("Encryption key loaded — API keys will be encrypted at rest")


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret for storage. Blank input is stored as blank."""
    if not plaintext:
        return ""
    if plaintext.startswith(_PREFIX):
        # Already encrypted. Encrypting twice would still decrypt correctly the
        # first time and hand back ciphertext, which is the kind of bug that
        # only shows up as a mysterious auth failure days later.
        return plaintext
    token = _fernet().encrypt(plaintext.encode())
    return _PREFIX + token.decode()


def decrypt_secret(stored: str) -> str:
    """Decrypt a stored secret, tolerating rows written before encryption.

    Rows created before this module existed hold plaintext with no prefix.
    Those are returned as-is so existing users keep working; they are re-written
    encrypted the next time the user saves their config.
    """
    if not stored:
        return ""
    if not stored.startswith(_PREFIX):
        logger.warning(
            "Found a legacy plaintext secret in the database. It will be "
            "encrypted the next time this user saves their config; run "
            "scripts/encrypt_existing_api_keys.py to convert them all now."
        )
        return stored
    try:
        return _fernet().decrypt(stored[len(_PREFIX):].encode()).decode()
    except InvalidToken as exc:
        # Almost always a changed or wrong APP_ENCRYPTION_KEY. Say that plainly
        # rather than surfacing a bare InvalidToken from three layers down.
        raise EncryptionKeyError(
            "Could not decrypt a stored API key. This usually means "
            f"{ENV_VAR} has changed since the value was written. Restore the "
            "previous key, or have affected users re-enter their API key."
        ) from exc


def mask_secret(plaintext: str) -> str:
    """A display form safe to send to a client: last 4 characters only.

    Takes the DECRYPTED key — callers hold plaintext by the time they render a
    response, and decrypting again here would log a spurious legacy-plaintext
    warning on every request. Enough for someone to recognise which key is
    configured without the value ever leaving the server again.
    """
    if not plaintext:
        return ""
    if len(plaintext) <= 4:
        return "••••"
    return "•" * 8 + plaintext[-4:]
