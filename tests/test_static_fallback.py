"""The SPA fallback must not serve files from outside the web build.

`request.url.path` reaches the middleware percent-DECODED, because uvicorn
unquotes the request target before it builds the ASGI scope. So "%2e%2e" is
".." by the time the fallback sees it, and the fallback joined that straight
onto the static root and served whatever `is_file()` accepted.

In the container the build sits at /app/static, which puts /proc/self/environ
two levels up — and on HF Spaces that holds DATABASE_URL, SUPABASE_KEY and
APP_ENCRYPTION_KEY, the key every stored user API key is encrypted under. No
authentication was involved.

Only the plain "/../x" spelling is normalised away by HTTP clients before it is
sent, which is why this went unnoticed: the obvious thing to try is the one
thing that does not work.
"""

from pathlib import Path

import pytest

from backend.main import static_file_within


@pytest.fixture
def build(tmp_path: Path) -> Path:
    """A web build with a secret sitting next to it, as in the container."""
    root = tmp_path / "static"
    (root / "_expo").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html>")
    (root / "_expo" / "app.js").write_text("console.log(1)")
    (tmp_path / ".env").write_text("APP_ENCRYPTION_KEY=hunter2")
    return root


@pytest.mark.parametrize("url_path", [
    "/index.html",
    "/_expo/app.js",
    # A leading slash is stripped, not treated as the filesystem root.
    "index.html",
])
def test_files_inside_the_build_are_served(build, url_path):
    resolved = static_file_within(build, url_path)
    assert resolved is not None
    assert resolved.read_text()


@pytest.mark.parametrize("url_path", [
    "/../.env",                       # what %2e%2e/.env decodes to
    "/../../etc/passwd",
    "/_expo/../../.env",              # escaping from inside a real subdirectory
    "/./../.env",
    "/....//.env",
    "/../../proc/self/environ",       # the one that leaks the whole environment
])
def test_paths_that_escape_the_build_are_refused(build, url_path):
    assert static_file_within(build, url_path) is None


def test_an_absolute_looking_path_cannot_reach_the_filesystem_root(build, tmp_path):
    # "/etc/passwd" lstrips to "etc/passwd" and lands under the build, where it
    # does not exist. It must never resolve to the real /etc/passwd.
    assert static_file_within(build, "/etc/passwd") is None


def test_a_missing_file_inside_the_build_is_not_an_error(build):
    # The caller falls back to index.html for this — a route the SPA handles
    # client-side is a normal request, not an attack.
    assert static_file_within(build, "/transactions/123") is None


def test_a_directory_is_not_served_as_a_file(build):
    assert static_file_within(build, "/_expo") is None


def test_a_symlink_out_of_the_build_is_refused(build, tmp_path):
    """Containment is checked after resolution, so a link cannot smuggle a path.

    Expo does not create symlinks in `dist`, but the check is only worth having
    if it holds for the case a prefix comparison on the unresolved path misses.
    """
    (build / "escape.env").symlink_to(tmp_path / ".env")
    assert static_file_within(build, "/escape.env") is None
