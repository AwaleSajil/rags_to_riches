from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth, config_router, files, chat
from backend.services.rag_manager import rag_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await rag_manager.cleanup_all()


app = FastAPI(title="MoneyRAG API", version="1.0.0", lifespan=lifespan)


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


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
