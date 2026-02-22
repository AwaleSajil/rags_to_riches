import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import auth, config_router, files, chat, transactions
from backend.services.rag_manager import rag_manager

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
except Exception:
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("moneyrag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MoneyRAG API starting up")
    logger.debug("Registered routers: auth, config, files, chat")
    yield
    logger.info("MoneyRAG API shutting down — cleaning up RAG instances")
    await rag_manager.cleanup_all()
    logger.info("Shutdown complete")


app = FastAPI(title="MoneyRAG API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(
        ">>> %s %s | headers: %s",
        request.method,
        request.url.path,
        dict(request.headers),
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(config_router.router, prefix="/api/v1/config", tags=["config"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["transactions"])


@app.get("/api/v1/health")
async def health():
    logger.debug("Health check hit")
    return {"status": "ok"}


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

    # Mount static assets (JS/CSS/images) at /_expo so they don't conflict with API
    app.mount("/_expo", StaticFiles(directory=str(_static_dir / "_expo")), name="expo-assets")

    # Catch-all: serve index.html for any non-API route (SPA client-side routing)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Try to serve the exact static file first
        file_path = _static_dir / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html (SPA fallback)
        return FileResponse(_static_dir / "index.html")
