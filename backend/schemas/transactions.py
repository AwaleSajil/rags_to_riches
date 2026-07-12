from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionCreate(BaseModel):
    description: str
    amount: float = Field(gt=0)
    trans_date: date
    category: str = "Uncategorized"
    merchant_name: Optional[str] = None


class TransactionUpdate(BaseModel):
    """Editable fields for a transaction. Only supplied fields are updated."""

    trans_date: Optional[date] = None
    description: Optional[str] = None
    merchant_name: Optional[str] = None
    amount: Optional[float] = Field(default=None, gt=0)
    category: Optional[str] = None
    location: Optional[str] = None
    subtotal: Optional[float] = Field(default=None, ge=0)
    tax_total: Optional[float] = Field(default=None, ge=0)
    tax_breakdown: Optional[List[Dict[str, Any]]] = None


class TransactionResponse(BaseModel):
    id: str
    description: str
    amount: float
    trans_date: str
    category: str
    merchant_name: Optional[str]


class TransactionListItem(BaseModel):
    """A row in the transactions browser list. Numeric fields are coerced to float."""

    id: str
    trans_date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    merchant_name: Optional[str] = None
    location: Optional[str] = None
    subtotal: Optional[float] = None
    tax_total: Optional[float] = None
    tax_breakdown: Optional[Any] = None
    source: Optional[str] = None
    created_at: Optional[str] = None


class TransactionDetailItem(BaseModel):
    """A single line item belonging to a transaction (from a parsed receipt).

    Prices are split: *_subtotal_price are pre-tax, item_total_price is post-tax
    (= item_subtotal_price + tax_amount).
    """

    id: str
    item_description: Optional[str] = None
    item_quantity: Optional[float] = None
    item_unit_subtotal_price: Optional[float] = None
    item_subtotal_price: Optional[float] = None
    tax_amount: Optional[float] = None
    taxable: Optional[bool] = None
    tax_rate: Optional[float] = None
    item_total_price: Optional[float] = None
    enriched_info: Optional[str] = None


class TransactionWithDetails(TransactionListItem):
    """Full transaction for the detail screen: header fields + ordered line items."""

    enriched_info: Optional[str] = None
    content_hash: Optional[str] = None
    source_csv_id: Optional[str] = None
    source_bill_file_id: Optional[str] = None
    details: List[TransactionDetailItem] = Field(default_factory=list)
