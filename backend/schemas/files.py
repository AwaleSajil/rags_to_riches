from typing import List, Optional
from pydantic import BaseModel


class FileItem(BaseModel):
    id: str
    filename: str
    s3_key: str
    upload_date: str
    type: str


class FileListResponse(BaseModel):
    files: List[FileItem]


class UploadResponse(BaseModel):
    message: str
    file_ids: List[str]
