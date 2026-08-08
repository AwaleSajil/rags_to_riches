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
    # Receipts only (None on CSVs and price tags): whether this photo has been
    # reviewed into a transaction. Until it has, its spending is in no total,
    # and nothing in the file list used to distinguish the two.
    is_verified: Optional[bool] = None
    # Quarter turns clockwise needed to view this photo upright. Stored rather
    # than baked into the image so the original bytes stay untouched — see
    # migration 038. None/0 on CSVs and on rows written before that migration.
    rotation: Optional[int] = 0
    # Receipts only: this purchase is also recorded from a bank statement. The
    # two are linked, not merged, and counted once — see LinkedTransaction.
    is_linked: Optional[bool] = None


class FileListResponse(BaseModel):
    files: List[FileItem]


class AlreadyImported(BaseModel):
    """A CSV skipped because this user has these exact bytes already."""

    filename: str
    existing_filename: Optional[str] = None
    uploaded_at: Optional[str] = None


class UploadResponse(BaseModel):
    message: str
    file_ids: List[str]
    # Empty on every ordinary upload.
    already_imported: List[AlreadyImported] = []
