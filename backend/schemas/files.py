from typing import List, Optional
from pydantic import BaseModel


class FileItem(BaseModel):
    id: str
    filename: str
    s3_key: str
    upload_date: str
    type: str
    # Photos only: 'receipt' | 'price_tag' | 'unknown'. The client needs this to
    # decide where tapping a file leads — opening the receipt form on a price
    # tag is how a shelf price becomes a transaction that never happened.
    kind: Optional[str] = None
    is_hidden: Optional[bool] = None


class FileListResponse(BaseModel):
    files: List[FileItem]


class UploadResponse(BaseModel):
    message: str
    file_ids: List[str]
