"""API keys must be unreadable in the database and unreachable from the client.

Two separate promises, and both used to be broken: keys sat in AccountConfig as
plaintext, and every /config response — GET and PUT alike — handed the key back
to whoever asked. A leak of either one lets someone else spend the user's money,
so each has its own test here.
"""

import pytest

from backend import crypto


# --- at rest -----------------------------------------------------------------

def test_round_trip():
    assert crypto.decrypt_secret(crypto.encrypt_secret("AIzaSecret123")) == "AIzaSecret123"


def test_ciphertext_does_not_contain_the_key():
    """The point of the exercise: a database dump reveals nothing."""
    stored = crypto.encrypt_secret("AIzaSecret123")
    assert "AIzaSecret123" not in stored
    assert stored.startswith("enc:v1:")


def test_encryption_is_not_deterministic():
    """Same key twice must not produce the same ciphertext, or the storage
    itself tells you which two users share a key."""
    assert crypto.encrypt_secret("same-key") != crypto.encrypt_secret("same-key")


def test_encrypting_twice_is_a_no_op():
    """Guards the upsert path: double-encrypting would decrypt to ciphertext,
    which fails later as a confusing auth error rather than here."""
    once = crypto.encrypt_secret("AIzaSecret123")
    assert crypto.encrypt_secret(once) == once


def test_blank_stays_blank():
    assert crypto.encrypt_secret("") == ""
    assert crypto.decrypt_secret("") == ""


def test_legacy_plaintext_is_still_readable():
    """Rows written before encryption existed have no prefix. They must keep
    working, or every existing user is locked out on deploy."""
    assert crypto.decrypt_secret("AIzaLegacyPlaintext") == "AIzaLegacyPlaintext"


def test_wrong_key_is_reported_as_a_key_problem(monkeypatch):
    """An InvalidToken three layers down is not an actionable error message."""
    stored = crypto.encrypt_secret("AIzaSecret123")
    crypto._fernet.cache_clear()
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "kUYSi6ebhfEMiryZmiFYbeCLOd8wqfCbZGOAyRaug5A=")
    try:
        with pytest.raises(crypto.EncryptionKeyError, match="APP_ENCRYPTION_KEY"):
            crypto.decrypt_secret(stored)
    finally:
        crypto._fernet.cache_clear()


def test_missing_key_refuses_rather_than_storing_plaintext(monkeypatch):
    """The whole failure mode this module exists to prevent.

    Both sources have to be blanked. The key can come from os.environ OR from
    .env via Settings, and a developer machine with a real .env would otherwise
    make this pass for the wrong reason.
    """
    import backend.config

    crypto._fernet.cache_clear()
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "")
    monkeypatch.setattr(
        backend.config, "get_settings", lambda: type("S", (), {"APP_ENCRYPTION_KEY": ""})()
    )
    try:
        with pytest.raises(crypto.EncryptionKeyError):
            crypto.encrypt_secret("AIzaSecret123")
    finally:
        crypto._fernet.cache_clear()


def test_key_is_read_from_dotenv_when_not_in_the_environment(monkeypatch):
    """Regression: crypto read only os.environ, but pydantic-settings loads .env
    into Settings and NOT into os.environ — so a key sitting in .env, where
    every other secret in this project lives, was invisible and the API refused
    to start while looking correctly configured."""
    import backend.config

    crypto._fernet.cache_clear()
    monkeypatch.delenv("APP_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(
        backend.config,
        "get_settings",
        lambda: type(
            "S", (), {"APP_ENCRYPTION_KEY": "kUYSi6ebhfEMiryZmiFYbeCLOd8wqfCbZGOAyRaug5A="}
        )(),
    )
    try:
        assert crypto.decrypt_secret(crypto.encrypt_secret("from-dotenv")) == "from-dotenv"
    finally:
        crypto._fernet.cache_clear()


# --- in transit --------------------------------------------------------------

def test_mask_shows_only_the_last_four():
    masked = crypto.mask_secret("AIzaSyD-abcdefgh9fK2")
    assert masked.endswith("9fK2")
    assert "AIzaSyD" not in masked


def test_mask_hides_a_short_key_entirely():
    """Last-4 of a 4-character key is the whole key."""
    assert crypto.mask_secret("abcd") == "••••"


def test_mask_of_nothing_is_nothing():
    assert crypto.mask_secret("") == ""


# --- the /config boundary ----------------------------------------------------
#
# db_client decrypts on read, so everything past it holds a PLAINTEXT key. These
# routes are the only place it can escape to a client, and both used to let it.

SECRET = "AIzaSyD-supersecret-value-9fK2"

STORED_CONFIG = {
    "id": "cfg-1",
    "user_id": "11111111-2222-3333-4444-555555555555",
    "llm_provider": "Google",
    "api_key": SECRET,
    "decode_model": "gemini-3-flash-preview",
    "embedding_model": "gemini-embedding-001",
    "deep_enrichment": False,
}


@pytest.fixture
def stubbed_config(monkeypatch):
    """Stands in for the database, and records what the route tried to save."""
    from backend.services import config_service

    saved = {}

    async def fake_get(user):
        return dict(STORED_CONFIG)

    async def fake_upsert(user, data):
        saved.update(data)
        return {**STORED_CONFIG, **data}

    monkeypatch.setattr(config_service, "get_config", fake_get)
    monkeypatch.setattr(config_service, "upsert_config", fake_upsert)
    return saved


def test_get_config_never_returns_the_key(client, stubbed_config):
    response = client.get("/api/v1/config")
    # Asserted so a 500 cannot pass this test by returning a body with no key.
    assert response.status_code == 200
    body = response.json()
    assert "api_key" not in body
    assert SECRET not in str(body)
    # But it still says a key is configured, and which one.
    assert body["api_key_set"] is True
    assert body["api_key_hint"].endswith("9fK2")


def test_put_config_never_returns_the_key(client, stubbed_config):
    """The save response used to echo the stored key straight back."""
    response = client.put(
        "/api/v1/config",
        json={
            "llm_provider": "Google",
            "decode_model": "gemini-3-flash-preview",
            "embedding_model": "gemini-embedding-001",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "api_key" not in body
    assert SECRET not in str(body)


def test_saving_without_a_key_keeps_the_stored_one(client, stubbed_config):
    """The client can no longer echo the key back, so an omitted one has to mean
    'leave it alone' — otherwise changing the model wipes the credentials."""
    response = client.put(
        "/api/v1/config",
        json={
            "llm_provider": "Google",
            "decode_model": "gemini-3-pro",
            "embedding_model": "gemini-embedding-001",
        },
    )
    assert response.status_code == 200
    assert stubbed_config["api_key"] == SECRET
    assert stubbed_config["decode_model"] == "gemini-3-pro"


def test_a_supplied_key_replaces_the_stored_one(client, stubbed_config):
    response = client.put(
        "/api/v1/config",
        json={
            "llm_provider": "Google",
            "api_key": "AIzaBrandNewKey",
            "decode_model": "gemini-3-flash-preview",
            "embedding_model": "gemini-embedding-001",
        },
    )
    assert response.status_code == 200
    assert stubbed_config["api_key"] == "AIzaBrandNewKey"
