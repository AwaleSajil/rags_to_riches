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
    ("POST", "/api/v1/captures"),
    ("POST", "/api/v1/captures/abc/kind"),
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


# --- one account per email --------------------------------------------------
#
# The User table mirrors auth.users but, until migration 019, asserted nothing
# about email: both writers conflict on `id`, so two auth rows carrying the same
# address would have landed as two rows. The index is the backstop; these cover
# the normalisation that keeps callers from ever reaching it.


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sam@Example.com", "sam@example.com"),
        ("  sam@example.com  ", "sam@example.com"),
        ("SAM@EXAMPLE.COM", "sam@example.com"),
        ("sam@example.com", "sam@example.com"),
    ],
)
def test_email_is_folded_to_one_form(raw, expected):
    """Matches the lower(email) unique index, so a case variant is the same
    account rather than a second row the index has to refuse."""
    from backend.routers.auth import normalize_email

    assert normalize_email(raw) == expected


def test_email_normalisation_survives_none():
    from backend.routers.auth import normalize_email

    assert normalize_email(None) is None


def test_gmail_local_conventions_are_left_alone():
    """Dots and +suffixes are Gmail's convention, not email's. Collapsing them
    would merge addresses that are different people at other providers."""
    from backend.routers.auth import normalize_email

    assert normalize_email("a.b+tag@example.com") == "a.b+tag@example.com"


@pytest.mark.parametrize(
    "message",
    [
        "User already registered",
        "duplicate key value violates unique constraint \"user_email_lower_key\"",
        "AuthApiError: user_already_exists",
        "23505",
    ],
)
def test_duplicate_signup_is_recognised(message):
    """Both layers can refuse the second account and they word it differently;
    each must still reach the user as one clear 409."""
    from backend.routers.auth import _is_duplicate_email

    assert _is_duplicate_email(Exception(message)) is True


@pytest.mark.parametrize("message", ["network timeout", "invalid password", ""])
def test_unrelated_failures_are_not_called_duplicates(message):
    """A 409 telling someone the address is taken when the real fault was a
    timeout sends them off to recover an account that does not exist."""
    from backend.routers.auth import _is_duplicate_email

    assert _is_duplicate_email(Exception(message)) is False
