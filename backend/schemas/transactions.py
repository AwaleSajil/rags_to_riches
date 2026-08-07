from datetime import date
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionCreate(BaseModel):
    description: str = Field(min_length=1)
    amount: float = Field(gt=0)
    trans_date: date
    category: str = "Uncategorized"
    merchant_name: Optional[str] = None

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, value: str) -> str:
        # min_length alone still admits "   ", which would store an unlabelled
        # transaction and leave the dedup hash with nothing to key on.
        stripped = value.strip()
        if not stripped:
            raise ValueError("description must not be blank")
        return stripped


class TransactionUpdate(BaseModel):
    """Editable fields for a transaction. Only supplied fields are updated."""

    trans_date: Optional[date] = None
    description: Optional[str] = None
    merchant_name: Optional[str] = None
    amount: Optional[float] = Field(default=None, gt=0)
    category: Optional[str] = None
    location: Optional[str] = None
    note: Optional[str] = None
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
    note: Optional[str] = None
    subtotal: Optional[float] = None
    tax_total: Optional[float] = None
    tax_breakdown: Optional[Any] = None
    discount_total: Optional[float] = None
    savings_total: Optional[float] = None
    source: Optional[str] = None
    enriched_info: Optional[str] = None
    linked_transaction_ids: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class TransactionDetailItem(BaseModel):
    """A single line item belonging to a transaction (from a parsed receipt).

    Prices are split: *_subtotal_price are pre-tax, item_total_price is post-tax
    (= item_subtotal_price + tax_amount).
    """

    id: str
    item_description: Optional[str] = None
    item_quantity: Optional[float] = None
    unit_quantity_subtotal: Optional[float] = None
    item_subtotal_price: Optional[float] = None
    item_savings: Optional[float] = None
    tax_amount: Optional[float] = None
    taxable: Optional[bool] = None
    tax_rate: Optional[float] = None
    item_total_price: Optional[float] = None
    enriched_info: Optional[str] = None


class TransactionDetailInput(BaseModel):
    """One line item in a bulk 'replace details' request. Derived fields
    (item_subtotal_price, tax_amount, item_total_price, taxable) are computed
    server-side from quantity/unit/tax_rate when omitted."""

    item_description: Optional[str] = None
    item_quantity: Optional[float] = None
    unit_quantity_subtotal: Optional[float] = None
    item_subtotal_price: Optional[float] = None
    item_savings: Optional[float] = None
    tax_amount: Optional[float] = None
    taxable: Optional[bool] = None
    tax_rate: Optional[float] = None
    item_total_price: Optional[float] = None
    enriched_info: Optional[str] = None


class TransactionDetailsReplace(BaseModel):
    """Replace the full set of line items for a transaction."""

    details: List[TransactionDetailInput] = Field(default_factory=list)


class ReceiptReviewLineItem(BaseModel):
    item_description: str = ""
    item_quantity: float = Field(default=1, ge=0)
    # What item_quantity counts: 'lb' for weighed produce, 'each' for packaged
    # goods. Distinct from package size — a 30-count bag is quantity 1 'each'.
    # Null when the receipt does not say; inferring it would feed a guess
    # straight into per-unit price comparisons.
    item_quantity_unit: Optional[str] = None
    # Net price actually paid per unit (after any item-level markdown).
    item_unit_price: float = Field(default=0, ge=0)
    # How much this line was marked down (regular price minus what was paid);
    # informational only, never subtracted from a total.
    item_savings: float = Field(default=0, ge=0)
    tax_rate: float = Field(default=0, ge=0)


class ReceiptReviewInput(BaseModel):
    """User-corrected OCR data. A receipt becomes a transaction only on verify."""

    date: date
    time: Optional[str] = None
    merchant_name: str = Field(min_length=1)
    category: str = "Uncategorized"
    location: Optional[str] = None
    total_amount: Optional[float] = Field(default=None, gt=0)
    # Order-level coupons subtracted from the whole basket (not item markdowns).
    discount_total: Optional[float] = Field(default=None, ge=0)
    line_items: List[ReceiptReviewLineItem] = Field(default_factory=list)
    # Free text, same field the transaction editor writes. Worth capturing here
    # because why a purchase happened is clearest while the receipt is in your
    # hand, not weeks later in the browser.
    #
    # None means "leave whatever is there alone" — re-verifying a receipt must
    # not erase a note added afterwards from the transaction screen. An empty
    # string is an explicit clear.
    note: Optional[str] = None

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        normalized = value.strip()
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", normalized):
            raise ValueError("time must use 24-hour HH:MM format")
        return normalized


class TransactionWithDetails(TransactionListItem):
    """Full transaction for the detail screen: header fields + ordered line items."""

    enriched_info: Optional[str] = None
    content_hash: Optional[str] = None
    source_csv_id: Optional[str] = None
    source_bill_file_id: Optional[str] = None
    details: List[TransactionDetailItem] = Field(default_factory=list)
    # True when verifying a receipt matched a transaction that already existed,
    # so this is the earlier record rather than one just created. Defaults false
    # on every other route that returns this shape.
    is_duplicate: bool = False
