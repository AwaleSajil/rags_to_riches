from typing import Optional
from pydantic import BaseModel


class ConfigUpdate(BaseModel):
    llm_provider: str
    api_key: str
    decode_model: str
    embedding_model: str


class ConfigResponse(BaseModel):
    id: Optional[str] = None
    user_id: str
    llm_provider: str
    api_key: str
    decode_model: str
    embedding_model: str
