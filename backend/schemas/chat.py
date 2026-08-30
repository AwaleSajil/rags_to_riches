from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    # Photos this turn is about, as BillFile ids. Stored with the message so the
    # picture comes back when the conversation is reloaded — the local file URI
    # the app showed at the time does not survive a restart, and a signed URL
    # would have expired by then.
    bill_file_ids: Optional[List[str]] = None
