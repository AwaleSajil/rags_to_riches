import logging
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

logger = logging.getLogger("moneyrag.config")


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    DATABASE_URL: str
    POSTGRESSQL_STACK: str = "supabase"
    DATABRICKS_SERVER_HOSTNAME: str | None = None
    DATABRICKS_HTTP_PATH: str | None = None
    DATABRICKS_TOKEN: str | None = None
    
    VECTOR_DB_STACK: str = "qdrant"
    ACTIAN_ADDRESS: str | None = None
    ACTIAN_API_KEY: str | None = None

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    logger.debug("Loading settings from .env")
    settings = Settings()
    logger.debug(
        "Settings loaded — SUPABASE_URL=%s, QDRANT_URL=%s, DATABASE_URL=%s",
        settings.SUPABASE_URL,
        settings.QDRANT_URL,
        settings.DATABASE_URL[:30] + "..." if len(settings.DATABASE_URL) > 30 else settings.DATABASE_URL,
    )
    return settings
