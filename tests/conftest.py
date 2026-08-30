"""Shared test setup.

Settings are required at import time, so dummy values go into the environment
before anything under `backend` is imported. Nothing here touches a real
Supabase project or database — these tests are pure logic plus in-process HTTP.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SUPABASE_URL", "https://test-project.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-anon-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:test@localhost:5432/postgres"
)
# A throwaway Fernet key. backend.crypto refuses to start without one, so tests
# need their own — never reuse this anywhere real.
os.environ.setdefault(
    "APP_ENCRYPTION_KEY", "0EJ7QNCLNs3v6nPRJXcUCTQtR1z8sSPlB2h_dl1S7XM="
)

import pytest

TEST_USER_ID = "11111111-2222-3333-4444-555555555555"
OTHER_USER_ID = "99999999-8888-7777-6666-555555555555"


@pytest.fixture
def user_id() -> str:
    return TEST_USER_ID


@pytest.fixture
def client():
    """TestClient with auth stubbed out, for exercising route logic."""
    from fastapi.testclient import TestClient

    from backend.dependencies import get_current_user
    from backend.main import app

    app.dependency_overrides[get_current_user] = lambda: {
        "id": TEST_USER_ID,
        "email": "test@example.com",
        "access_token": "test-token",
    }
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client():
    """TestClient with no auth override, so real 401 handling is exercised."""
    from fastapi.testclient import TestClient

    from backend.main import app

    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
