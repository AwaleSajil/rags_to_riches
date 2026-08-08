import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import allowed_origins, verify_public_key_is_not_privileged
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


# --- Serve Expo web build as static files (for Docker / HF Spaces) ---
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    from starlette.responses import FileResponse

    logger.info("Serving static frontend from %s", _static_dir)

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
            file_path = _static_dir / path.lstrip("/")
            if file_path.is_file():
                return FileResponse(file_path)
            # SPA fallback
            return FileResponse(_static_dir / "index.html")

        return response
