"""Unauthenticated requests must be rejected with 401.

`get_current_user` used to return None when no bearer token was sent, and every
route then indexed `user["id"]` — so anonymous calls raised TypeError and came
back as a 500 the client had no way to act on.
"""

import pytest

PROTECTED_ROUTES = [
    ("GET", "/api/v1/files"),
    ("POST", "/api/v1/files/upload"),
    ("GET", "/api/v1/files/ingestion-status"),
    ("DELETE", "/api/v1/files/abc?type=csv"),
    ("PATCH", "/api/v1/files/abc/visibility?type=csv&hidden=true"),
    ("GET", "/api/v1/config"),
    ("PUT", "/api/v1/config"),
    ("POST", "/api/v1/chat"),
    ("GET", "/api/v1/conversations"),
    ("POST", "/api/v1/conversations"),
    ("GET", "/api/v1/conversations/abc/messages"),
    ("DELETE", "/api/v1/conversations/abc"),
    ("GET", "/api/v1/transactions"),
    ("POST", "/api/v1/transactions"),
    ("GET", "/api/v1/transactions/abc"),
    ("PATCH", "/api/v1/transactions/abc"),
    ("PUT", "/api/v1/transactions/abc/details"),
    ("DELETE", "/api/v1/transactions/abc"),
    ("GET", "/api/v1/transactions/receipt-review/abc"),
    ("POST", "/api/v1/auth/logout"),
]


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_require_auth(anon_client, method, path):
    response = anon_client.request(method, path, json={})
    assert response.status_code == 401, (
        f"{method} {path} returned {response.status_code}, expected 401"
    )


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_reject_a_garbage_token(anon_client, method, path):
    """A malformed token must not fall through to a 500 either."""
    response = anon_client.request(
        method, path, json={}, headers={"Authorization": "Bearer not-a-real-jwt"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/public-config"])
def test_public_routes_stay_open(anon_client, path):
    assert anon_client.get(path).status_code == 200


def test_401_carries_the_www_authenticate_header(anon_client):
    """Tells the client this is an auth problem, not a permissions one."""
    response = anon_client.get("/api/v1/transactions")
    assert response.headers.get("www-authenticate") == "Bearer"


def test_login_is_reachable_without_a_token(anon_client):
    """login/register accept credentials in the body, so the strict dependency
    must not be applied to them."""
    response = anon_client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "wrong"},
    )
    # It fails (no such user / no network), but not with the dependency's
    # "Not authenticated" — that would mean the guard blocked it too early.
    assert response.json().get("detail") != "Not authenticated"
