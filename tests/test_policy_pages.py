"""The policy pages both stores require, and what has to be true of them.

Three pages, all public and all served from the API so their URLs exist wherever
the backend does:

    /privacy           required by Apple and Play; the listing will not submit
                       without a resolving URL
    /terms             Apple expects an EULA; Play expects it in the listing
    /account-deletion  Play requires this specifically to work WITHOUT the app

The interesting tests here are not "does it return 200". They are: does the page
still describe what the code actually does, and can it be published with a
placeholder still in it.
"""

import re

import pytest

PAGES = ("/privacy", "/terms", "/account-deletion")


def prose(response) -> str:
    """A page's text with markup and line wrapping flattened.

    Assertions here are about what the page SAYS, and HTML wraps a sentence
    across lines and splits it with <strong> whenever the author felt like it.
    Matching the raw source would make these tests fail on reflowing a
    paragraph, which trains people to weaken them.
    """
    text = re.sub(r"<[^>]+>", " ", response.text)
    return re.sub(r"\s+", " ", text).lower()


@pytest.mark.parametrize("path", PAGES)
def test_pages_are_public(anon_client, path):
    """Behind a login they would satisfy nobody — Play's whole point is that
    someone who already uninstalled can still read them."""
    response = anon_client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.parametrize("path", PAGES)
def test_no_unsubstituted_placeholder_survives_to_the_page(anon_client, path):
    """A published policy showing '{{PUBLISHER_NAME}}' is worse than no policy:
    it is evidence nobody read it."""
    assert "{{" not in anon_client.get(path).text


@pytest.mark.parametrize("path", PAGES)
def test_an_unset_value_is_visibly_unset(anon_client, path, monkeypatch):
    """Not silently blank, and never a plausible-looking guess. Naming the wrong
    company in a privacy policy is a worse failure than an obvious gap."""
    from backend.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PUBLISHER_NAME", "")
    monkeypatch.setenv("SUPPORT_EMAIL", "")
    try:
        assert "not configured" in prose(anon_client.get(path))
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("path", PAGES)
def test_configured_values_appear(anon_client, path, monkeypatch):
    from backend.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PUBLISHER_NAME", "Example Ltd")
    monkeypatch.setenv("SUPPORT_EMAIL", "privacy@example.com")
    try:
        body = prose(anon_client.get(path))
        assert "privacy@example.com" in body
        assert "not configured" not in body
    finally:
        get_settings.cache_clear()


# --- the policy has to keep matching the code -------------------------------
#
# A privacy policy is a claim about behaviour, and behaviour drifts. These pin
# the disclosures that would become false if someone changed the code without
# reopening the page — which is the ordinary way a policy stops being true.

def test_the_ai_provider_disclosure_is_present(anon_client):
    """The single most consequential fact: receipt images and transaction text
    leave the system and go to a third party."""
    body = prose(anon_client.get("/privacy"))
    assert "ai provider" in body
    for claim in ("receipt", "image", "transaction description", "chat message"):
        assert claim in body
    # And that deleting the R2R account does not reach it.
    assert "does not remove anything from your ai provider" in body


def test_the_web_search_disclosure_matches_that_it_is_opt_in(anon_client):
    """DuckDuckGo receives merchant and item names, but only when Deep
    Enrichment is on — and it defaults to off. If that default ever flips, this
    page becomes untrue."""
    from backend.schemas.config_schema import ConfigUpdate

    assert ConfigUpdate.model_fields["deep_enrichment"].default is False
    body = prose(anon_client.get("/privacy"))
    assert "duckduckgo" in body
    assert "off by default" in body


def test_the_no_tracking_claim_matches_the_dependency_list():
    """The policy says there are no analytics, ad or crash SDKs. That is only
    true until someone adds one, and adding one is a one-line change."""
    import json
    from pathlib import Path

    package = json.loads(
        (Path(__file__).resolve().parent.parent / "frontend" / "package.json").read_text()
    )
    installed = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    banned = re.compile(
        r"sentry|analytics|amplitude|mixpanel|firebase|segment|facebook|"
        r"appsflyer|adjust|posthog|bugsnag|branch|onesignal",
        re.IGNORECASE,
    )
    offenders = [name for name in installed if banned.search(name)]
    assert not offenders, (
        f"{offenders} would make the privacy policy's 'no analytics, advertising, "
        "tracking or crash-reporting SDKs' claim false. Update /privacy and the "
        "store data-safety declarations before adding it."
    )


def test_the_location_claim_matches_what_is_sent(anon_client):
    """The policy promises no coordinates ever leave the device. The capture
    route takes a resolved place NAME and has no latitude/longitude parameter —
    if that changes, this claim has to change with it."""
    from backend.routers.captures import create_capture

    params = create_capture.__annotations__
    assert "location" in params
    assert not {"latitude", "longitude", "lat", "lon"} & set(params)
    assert "no gps coordinates" in prose(anon_client.get("/privacy"))


def test_the_deletion_claim_matches_the_route_that_exists(client, monkeypatch):
    """The policy tells people Settings -> Delete account works. It has to."""
    from backend.services import account_service

    async def _fake(user):
        return {"deleted": True, "objects_removed": 0}

    monkeypatch.setattr(account_service, "delete_account", _fake)
    assert client.delete("/api/v1/auth/account").status_code == 200


def test_the_pages_link_to_each_other(anon_client):
    """A reviewer lands on one and needs to reach the others."""
    assert "/terms" in anon_client.get("/privacy").text
    assert "/privacy" in anon_client.get("/terms").text
    assert "/account-deletion" in anon_client.get("/privacy").text


def test_the_governing_law_placeholder_is_loud(anon_client):
    """Deliberately not defaulted to a jurisdiction — that is a real legal
    choice. It must be impossible to publish without noticing."""
    assert "set this before publishing" in prose(anon_client.get("/terms"))
