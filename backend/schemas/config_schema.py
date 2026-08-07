from typing import Optional
from pydantic import BaseModel


class ConfigUpdate(BaseModel):
    llm_provider: str
    # Optional so the client can save model/provider changes without holding a
    # copy of the key. Omitted or blank means "keep the key already stored" —
    # the server never sends the key back, so the client cannot echo it.
    api_key: Optional[str] = None
    decode_model: str
    embedding_model: str
    deep_enrichment: bool = False


class ConfigResponse(BaseModel):
    """What the client is allowed to see.

    Deliberately has no ``api_key``. The key goes in from the client and is used
    server-side; it never travels back out. ``api_key_hint`` carries just enough
    (last 4 characters) for someone to recognise which key is configured.
    """

    id: Optional[str] = None
    user_id: str
    llm_provider: str
    decode_model: str
    embedding_model: str
    deep_enrichment: bool = False
    api_key_set: bool = False
    api_key_hint: str = ""
