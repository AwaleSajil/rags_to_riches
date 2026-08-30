import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import (
    allowed_origins,
    verify_public_key_is_not_privileged,
    verify_service_key_is_privileged,
)
from backend.crypto import verify_encryption_key
from backend.routers import corrections, auth, captures, config_router, files, chat, prices, transactions, conversations
from backend.services.rag_manager import rag_manager, run_janitor

# ---------------------------------------------------------------------------
# Monkey-patch google-genai bug: HttpResponse.json crashes when response_stream
# is an aiohttp.ClientResponse (not subscriptable). This triggers when langchain
# error-handling calls hasattr(resp, "json") on a streaming response object.
# See: google/genai/_api_client.py HttpResponse.json property
# ---------------------------------------------------------------------------
try:
    from google.genai._api_client import HttpResponse as _GenaiHttpResponse

    @property  # type: ignore[misc]
    def _safe_json(self):  # type: ignore[no-untyped-def]
        rs = self.response_stream
        if rs is None:
            return ""
        if isinstance(rs, list):
            if not rs or not rs[0]:
                return ""
            return self._load_json_from_response(rs[0])
        # rs is a raw ClientResponse or httpx.Response — can't subscript
        return ""

    _GenaiHttpResponse.json = _safe_json  # type: ignore[assignment]
except Exception as _patch_error:
    # Said out loud. If google-genai moves this class the patch silently stops
    # applying, and the crash it exists to prevent comes back looking like a
    # brand-new bug in error handling.
    logging.getLogger("moneyrag.main").warning(
        "Could not apply google-genai HttpResponse.json patch (%s) — "
        "streaming error handling may crash on that provider", _patch_error,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("moneyrag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MoneyRAG API starting up")
    # Refuse to start without it. A deployment that silently fell back to
    # plaintext would look perfectly healthy while storing every user's API key
    # in the clear — which is the failure this check exists to make impossible.
    verify_encryption_key()
    # And refuse to start if the key the frontend is handed can bypass RLS.
    verify_public_key_is_not_privileged()
    # Says out loud whether account deletion — which both stores require — can
    # actually run on this deployment.
    verify_service_key_is_privileged()
    logger.debug("Registered routers: auth, config, files, chat")

    # Releases idle RAG instances and abandoned photo captures. Without it both
    # grow until the container is restarted for memory.
    janitor = asyncio.create_task(run_janitor(), name="janitor")

    yield

    logger.info("MoneyRAG API shutting down — cleaning up RAG instances")
    janitor.cancel()
    try:
        await janitor
    except asyncio.CancelledError:
        pass
    await rag_manager.cleanup_all()
    logger.info("Shutdown complete")


app = FastAPI(title="MoneyRAG API", version="1.0.0", lifespan=lifespan, redirect_slashes=False)


# Headers that carry credentials. Logged as a placeholder rather than dropped,
# so "was a token even sent?" is still answerable while debugging.
_REDACTED_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "x-api-key"})


def _safe_headers(headers) -> dict:
    return {
        k: ("<redacted>" if k.lower() in _REDACTED_HEADERS else v)
        for k, v in headers.items()
    }


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """The one place an unexpected failure becomes a response.

    Every route used to wrap itself in `except Exception` and build its own 500,
    which meant nineteen slightly different messages, and any route that forgot
    the wrapper behaved differently from its neighbours.

    The exception text is deliberately NOT returned. It was being interpolated
    into `detail`, so a database error handed the caller table names, column
    names and sometimes a connection string. It goes to the log, which is where
    whoever can fix it is looking.
    """
    from starlette.responses import JSONResponse

    logger.error(
        "Unhandled error on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again."},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Every header used to be logged verbatim. At DEBUG level in production that
    # writes each user's JWT to the log, where it is a working credential for
    # anyone who can read logs.
    logger.debug(
        ">>> %s %s | headers: %s",
        request.method,
        request.url.path,
        _safe_headers(request.headers),
    )
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug(
        "<<< %s %s | status=%d | %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    # Fix file upload schemas: Swagger UI needs "format: binary" not "contentMediaType"
    for comp in (schema.get("components", {}).get("schemas", {}) or {}).values():
        for prop in (comp.get("properties", {}) or {}).values():
            items = prop.get("items", {})
            if items.get("contentMediaType"):
                items.pop("contentMediaType")
                items["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# allow_credentials is deliberately False: authentication is a Bearer token in
# the Authorization header, never a cookie. With credentials enabled, Starlette
# echoes the caller's own origin back for allow_origins=["*"], which made every
# website on the internet an allowed credentialed origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(config_router.router, prefix="/api/v1/config", tags=["config"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(captures.router, prefix="/api/v1/captures", tags=["captures"])
app.include_router(prices.router, prefix="/api/v1/price-observations", tags=["prices"])
app.include_router(corrections.router, prefix="/api/v1/corrections", tags=["corrections"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["transactions"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])


@app.get("/api/v1/health")
async def health():
    """Liveness only — is this process up and serving?

    Deliberately does not touch the database, so an orchestrator does not
    restart a perfectly good container during a brief Supabase blip.
    """
    logger.debug("Health check hit")
    return {"status": "ok"}


@app.get("/api/v1/ready")
async def ready():
    """Readiness — can this process actually do its job?

    A container that cannot reach the database answered the old health check
    happily and kept receiving traffic, turning an outage into a stream of 500s
    instead of a failed deploy. This is the check a load balancer should use.
    """
    from starlette.responses import JSONResponse
    from backend.vector_db_client import _get_engine
    from sqlalchemy import text

    try:
        engine = await asyncio.to_thread(_get_engine)

        def _ping():
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        await asyncio.to_thread(_ping)
    except Exception as e:  # noqa: BLE001
        logger.error("Readiness check failed: %s", e, exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unavailable", "database": "unreachable"})
    return {"status": "ready", "database": "ok"}


@app.get("/api/v1/public-config")
async def public_config():
    """Return public (non-secret) config for the frontend."""
    from backend.config import get_settings
    s = get_settings()
    return {
        "supabase_url": s.SUPABASE_URL,
        "supabase_anon_key": s.SUPABASE_KEY,
    }


# The legal and policy pages, served from the API so their URLs exist wherever
# the backend does. Both stores require a reachable privacy policy URL, and Play
# additionally requires the deletion page to be readable WITHOUT installing the
# app — which rules out putting any of this behind the login.
#
# Kept as HTML files rather than strings in Python so they can be edited (and
# reviewed by someone who is not a programmer) without touching code.
_PAGES = {
    "account-deletion": "account_deletion.html",
    "privacy": "privacy.html",
    "terms": "terms.html",
}


def _render_page(slug: str) -> str:
    """One policy page, with the deployment's own details filled in.

    Placeholders are substituted rather than hard-coded so the same file serves
    every deployment. An UNSET placeholder renders as a visible admission that
    it is unset — never as a plausible-looking address or company name, because
    a policy naming the wrong party is worse than one that is obviously
    incomplete, and the second kind gets fixed.
    """
    from backend.config import get_settings

    html = (Path(__file__).resolve().parent / "pages" / _PAGES[slug]).read_text(
        encoding="utf-8"
    )
    settings = get_settings()
    values = {
        "SUPPORT_EMAIL": (settings.SUPPORT_EMAIL or "").strip(),
        "PUBLISHER_NAME": (settings.PUBLISHER_NAME or "").strip(),
    }
    for name, value in values.items():
        if value:
            html = html.replace("{{%s}}" % name, value)
        else:
            logger.warning("%s is not set — %s page renders a placeholder", name, slug)
            html = html.replace(
                "{{%s}}" % name,
                f'<mark style="background:#fee;color:#900">[{name} not configured]</mark>',
            )
    return html


@app.get("/account-deletion", include_in_schema=False)
async def account_deletion_page():
    """How to delete your account, readable without installing the app.

    Play's deletion policy requires a publicly reachable URL for this, separate
    from the in-app flow — the point being that someone who has already
    uninstalled, or who never had a working sign-in, can still ask. Apple only
    requires the in-app path, but one page serves both.
    """
    from starlette.responses import HTMLResponse

    return HTMLResponse(_render_page("account-deletion"))


@app.get("/privacy", include_in_schema=False)
async def privacy_page():
    """The privacy policy. Both stores require this URL to exist and resolve."""
    from starlette.responses import HTMLResponse

    return HTMLResponse(_render_page("privacy"))


@app.get("/terms", include_in_schema=False)
async def terms_page():
    """Terms of service. Apple expects an EULA; Play expects a link in-listing."""
    from starlette.responses import HTMLResponse

    return HTMLResponse(_render_page("terms"))


def static_file_within(root: Path, url_path: str) -> Path | None:
    """The file `url_path` names inside `root`, or None if it escapes it.

    `request.url.path` arrives percent-DECODED — uvicorn unquotes the target
    before it builds the ASGI scope — so a request for "/%2e%2e/%2e%2e/.env"
    reaches the SPA fallback as "/../../.env". Joining that onto the static root
    and calling is_file() served anything this process could read:
    /proc/self/environ alone carries DATABASE_URL, SUPABASE_KEY and
    APP_ENCRYPTION_KEY, which is every stored user's LLM credentials. Only the
    plain "/../x" spelling is normalised away by clients; the encoded ones
    arrive intact.

    The two StaticFiles mounts below are safe because Starlette does this check
    itself. The fallback is hand-rolled and has to do it too.

    Module level, and taking its root as an argument, so it is reachable from a
    test — the version that lived inside `if _static_dir.is_dir()` could only be
    exercised by a deployment that had already shipped.
    """
    root = root.resolve()
    candidate = (root / url_path.lstrip("/")).resolve()
    if not candidate.is_relative_to(root):
        logger.warning("Refusing static path outside the build: %s", url_path)
        return None
    return candidate if candidate.is_file() else None


# --- Serve Expo web build as static files (for Docker / HF Spaces) ---
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    from starlette.responses import FileResponse

    logger.info("Serving static frontend from %s", _static_dir)

    # The build root with symlinks resolved, so a candidate path can be tested
    # against it. Computed once — it never changes, and resolve() hits the disk.
    _static_root = _static_dir.resolve()

    # Mount known static asset directories so they're served directly
    app.mount("/_expo", StaticFiles(directory=str(_static_dir / "_expo")), name="expo-assets")
    if (_static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="static-assets")

    # SPA fallback middleware: if the app returns 404 and the request is NOT
    # for an API route, serve index.html (or the exact static file).
    # Using middleware avoids the catch-all route / mount problems that shadow
    # API routes registered via include_router.
    @app.middleware("http")
    async def spa_fallback(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path

        # Only intercept 404s for non-API GET requests
        if (
            response.status_code == 404
            and request.method == "GET"
            and not path.startswith("/api/")
        ):
            # Try exact static file first
            file_path = static_file_within(_static_root, path)
            if file_path is not None:
                return FileResponse(file_path)
            # SPA fallback
            return FileResponse(_static_root / "index.html")

        return response
