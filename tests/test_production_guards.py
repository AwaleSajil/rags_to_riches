"""The bounds that stop one box serving many users from falling over.

Every one of these covers a failure that does not show up in development, where
there is one user, one question at a time, and a process that gets restarted
every few minutes by the reloader. They only bite on a long-lived server, which
is the worst place to discover them.
"""

import asyncio
import base64
import json

import pytest

from backend.services import rag_manager as rm


class FakeRAG:
    """Stands in for a MoneyRAG without building LLM clients or temp dirs."""

    def __init__(self):
        self.cleaned = False

    async def cleanup(self):
        self.cleaned = True


@pytest.fixture
def manager(monkeypatch):
    """A fresh manager whose instances are cheap fakes."""
    made: list[FakeRAG] = []

    def _fake_ctor(**kwargs):
        instance = FakeRAG()
        made.append(instance)
        return instance

    monkeypatch.setattr(rm, "MoneyRAG", _fake_ctor)
    mgr = rm.RAGManager()
    mgr.made = made  # type: ignore[attr-defined]
    return mgr


CONFIG = {"llm_provider": "google", "api_key": "k"}


def user(uid: str) -> dict:
    return {"id": uid, "access_token": "t"}


# --- the leak that ends the process ------------------------------------------

@pytest.mark.asyncio
async def test_an_idle_instance_is_swept(manager, monkeypatch):
    await manager.get_or_create(user("a"), CONFIG)
    assert len(manager._instances) == 1

    now = [1000.0]
    monkeypatch.setattr(rm.time, "monotonic", lambda: now[0])
    # get_or_create stamped last_used with the real clock, so re-stamp it.
    manager._last_used["a"] = now[0]
    now[0] += rm.IDLE_TTL_SECONDS + 1

    assert await manager.sweep() == 1
    assert manager._instances == {}
    assert manager.made[0].cleaned, "cleanup() must run so the temp dir goes too"


@pytest.mark.asyncio
async def test_a_recently_used_instance_survives_the_sweep(manager):
    await manager.get_or_create(user("a"), CONFIG)
    assert await manager.sweep() == 0
    assert "a" in manager._instances


@pytest.mark.asyncio
async def test_an_instance_mid_generation_is_never_swept(manager, monkeypatch):
    """Its temp dir holds the chart the running answer is about to read."""
    await manager.get_or_create(user("a"), CONFIG)
    now = [1000.0]
    monkeypatch.setattr(rm.time, "monotonic", lambda: now[0])
    manager._last_used["a"] = now[0]
    now[0] += rm.IDLE_TTL_SECONDS + 1

    async with manager.chatting("a"):
        assert await manager.sweep() == 0
        assert "a" in manager._instances


@pytest.mark.asyncio
async def test_the_ceiling_evicts_least_recently_used(manager, monkeypatch):
    monkeypatch.setattr(rm, "MAX_INSTANCES", 3)
    for uid in ("a", "b", "c"):
        await manager.get_or_create(user(uid), CONFIG)
    # Touch 'a' so 'b' becomes the oldest.
    await manager.get_or_create(user("a"), CONFIG)
    await manager.get_or_create(user("d"), CONFIG)

    assert len(manager._instances) == 3
    assert "b" not in manager._instances
    assert {"a", "c", "d"} == set(manager._instances)


# --- the subprocess-per-chat bound -------------------------------------------

@pytest.mark.asyncio
async def test_a_second_concurrent_chat_is_refused(manager):
    async with manager.chatting("a"):
        assert manager.active_chats("a") == 1
        with pytest.raises(rm.ChatBusy):
            async with manager.chatting("a"):
                pass


@pytest.mark.asyncio
async def test_the_slot_is_released_even_when_the_chat_raises(manager):
    with pytest.raises(RuntimeError):
        async with manager.chatting("a"):
            raise RuntimeError("provider exploded")
    assert manager.active_chats("a") == 0
    # And a retry is allowed straight away rather than being locked out.
    async with manager.chatting("a"):
        pass


@pytest.mark.asyncio
async def test_different_users_do_not_block_each_other(manager):
    async with manager.chatting("a"):
        async with manager.chatting("b"):
            assert manager.active_chats("a") == 1
            assert manager.active_chats("b") == 1


# --- the key that must never be published ------------------------------------

def _fake_jwt(role: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.mark.parametrize(
    "key",
    [_fake_jwt("service_role"), "sb_secret_abc123"],
    ids=["legacy-service-role-jwt", "new-style-secret-key"],
)
def test_startup_refuses_a_privileged_supabase_key(key, monkeypatch):
    """/api/v1/public-config serves this value to anyone. A service key there
    hands the whole database, RLS bypassed, to any unauthenticated caller."""
    from backend import config as cfg

    monkeypatch.setattr(cfg, "get_settings", lambda: type("S", (), {"SUPABASE_KEY": key})())
    with pytest.raises(RuntimeError, match="SERVICE key"):
        cfg.verify_public_key_is_not_privileged()


@pytest.mark.parametrize(
    "key",
    [_fake_jwt("anon"), "sb_publishable_abc123", "test-anon-key"],
    ids=["legacy-anon-jwt", "new-style-publishable", "opaque"],
)
def test_startup_accepts_a_public_supabase_key(key, monkeypatch):
    from backend import config as cfg

    monkeypatch.setattr(cfg, "get_settings", lambda: type("S", (), {"SUPABASE_KEY": key})())
    cfg.verify_public_key_is_not_privileged()  # must not raise


# --- CORS --------------------------------------------------------------------

def test_origins_default_to_local_dev_not_wildcard(monkeypatch):
    from backend import config as cfg

    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    origins = cfg.allowed_origins()
    assert "*" not in origins
    assert all(o.startswith("http://localhost") or o.startswith("http://127.0.0.1") for o in origins)


def test_origins_come_from_the_environment(monkeypatch):
    from backend import config as cfg

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com, https://www.example.com")
    assert cfg.allowed_origins() == ["https://app.example.com", "https://www.example.com"]


def test_wildcard_still_possible_but_must_be_explicit(monkeypatch):
    from backend import config as cfg

    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    assert cfg.allowed_origins() == ["*"]


# --- credential hygiene -------------------------------------------------------

def test_authorization_header_is_never_logged():
    from backend.main import _safe_headers

    safe = _safe_headers({
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.secret.sig",
        "cookie": "session=abc",
        "content-type": "application/json",
    })
    assert "secret" not in json.dumps(safe)
    assert safe["authorization"] == "<redacted>"
    assert safe["cookie"] == "<redacted>"
    # Everything else still readable, or the log stops being useful.
    assert safe["content-type"] == "application/json"


# --- readiness ----------------------------------------------------------------

def test_readiness_reports_503_when_the_database_is_unreachable(client, monkeypatch):
    import backend.vector_db_client as vdb

    def _boom():
        raise OSError("connection refused")

    monkeypatch.setattr(vdb, "_get_engine", _boom)
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_liveness_stays_up_even_when_the_database_is_down(client, monkeypatch):
    """Liveness and readiness are different questions — a DB blip must not get
    a healthy container killed and restarted."""
    import backend.vector_db_client as vdb

    monkeypatch.setattr(vdb, "_get_engine", lambda: (_ for _ in ()).throw(OSError("down")))
    assert client.get("/api/v1/health").status_code == 200


# --- the one place an unexpected failure becomes a response --------------------
#
# Nineteen routes each built their own 500 and interpolated the exception into
# `detail`, so a database error handed the caller table names, column names and
# on a bad day a connection string. One handler now, and the text stays in the
# log where whoever can fix it is looking.

def test_an_unexpected_failure_is_a_generic_500(client, monkeypatch):
    from backend.services import file_service

    async def _boom(user):
        raise RuntimeError("relation \"Transaction\" does not exist at 10.1.2.3:5432")

    monkeypatch.setattr(file_service, "list_files", _boom)
    response = client.get("/api/v1/files")

    assert response.status_code == 500
    body = response.text
    # The parts that must never travel to a client.
    assert "relation" not in body
    assert "10.1.2.3" not in body
    assert "RuntimeError" not in body
    assert response.json()["detail"] == "Something went wrong. Please try again."


def test_a_deliberate_http_error_is_still_passed_through(client, monkeypatch):
    """The global handler must not swallow the 404s routes raise on purpose."""
    from backend.services import file_service

    async def _none(user, file_id):
        return None

    monkeypatch.setattr(file_service, "get_file", _none)
    response = client.get("/api/v1/files/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_a_value_error_still_maps_to_its_own_status(client, monkeypatch):
    """Routes map ValueError to 400/404 themselves; that is logic, not boilerplate."""
    from backend.services import file_service

    async def _missing(user, file_id, type):
        raise ValueError("File not found")

    monkeypatch.setattr(file_service, "delete_file", _missing)
    response = client.delete("/api/v1/files/abc?type=bill")
    assert response.status_code == 404


# --- the privileged key must never leave the server -------------------------
#
# SUPABASE_SERVICE_KEY bypasses every RLS policy in the database. It was added
# for account deletion, which both stores require, and it now sits in the same
# Settings object as the anon key that /public-config hands to anonymous
# callers. The two must never be confused, so the boundary is tested rather
# than trusted.

def test_public_config_never_serves_the_service_key(anon_client, monkeypatch):
    from backend.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "sb_secret_do_not_leak_me")
    try:
        body = anon_client.get("/api/v1/public-config").json()
        assert "sb_secret_do_not_leak_me" not in str(body)
        # And it still serves the key the frontend legitimately needs.
        assert body["supabase_anon_key"]
    finally:
        get_settings.cache_clear()


def test_the_service_key_is_not_reachable_through_the_ordinary_client_helpers():
    """Only admin_client() may hold it, so `grep admin_client` finds every
    place RLS is skipped. get_supabase/client_for must stay on the anon key."""
    from backend.config import get_settings
    from backend.dependencies import client_for, get_supabase

    service = "sb_secret_do_not_leak_me"
    anon = get_settings().SUPABASE_KEY
    assert anon != service
    for client in (get_supabase(), get_supabase("tok"), client_for({"access_token": "tok"})):
        assert service not in str(getattr(client, "supabase_key", ""))


# --- telling the operator WHICH wrong key they pasted -------------------------
#
# The dashboard shows the publishable key prominently and hides the secret one
# behind a Reveal or a Create, so reaching for the wrong one is the ordinary
# mistake. This actually happened during setup. "Not a service key" would send
# someone back to a page where everything looks like what they already copied,
# so the check names the specific key instead.

@pytest.mark.parametrize("key,expected", [
    ("sb_publishable_" + "a" * 30, "publishable"),
    ("sbp_" + "a" * 40, "personal access token"),
])
def test_a_wrong_key_is_named_not_just_rejected(monkeypatch, caplog, key, expected):
    import logging

    from backend.config import get_settings, verify_service_key_is_privileged

    get_settings.cache_clear()
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", key)
    try:
        with caplog.at_level(logging.ERROR, logger="moneyrag.config"):
            verify_service_key_is_privileged()
        assert expected in caplog.text.lower()
        # And it says where to get the right one.
        assert "sb_secret_" in caplog.text or "service_role" in caplog.text
    finally:
        get_settings.cache_clear()


def test_a_real_secret_key_is_accepted(monkeypatch, caplog):
    import logging

    from backend.config import get_settings, verify_service_key_is_privileged

    get_settings.cache_clear()
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "sb_secret_" + "a" * 30)
    try:
        with caplog.at_level(logging.INFO, logger="moneyrag.config"):
            verify_service_key_is_privileged()
        assert "account deletion is available" in caplog.text.lower()
        assert "ERROR" not in caplog.text
    finally:
        get_settings.cache_clear()


def test_a_missing_key_warns_rather_than_erroring(monkeypatch, caplog):
    """An existing deploy must still boot; it just cannot delete accounts."""
    import logging

    from backend.config import get_settings, verify_service_key_is_privileged

    get_settings.cache_clear()
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "")
    try:
        with caplog.at_level(logging.WARNING, logger="moneyrag.config"):
            verify_service_key_is_privileged()
        assert "not set" in caplog.text.lower()
        assert "before submitting" in caplog.text.lower()
    finally:
        get_settings.cache_clear()
