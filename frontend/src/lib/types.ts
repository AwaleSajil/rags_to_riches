export interface User {
  id: string;
  email: string;
}

export interface AccountConfig {
  id?: string;
  user_id: string;
  llm_provider: string;
  decode_model: string;
  embedding_model: string;
  deep_enrichment?: boolean;
  // Deliberately no `api_key`. The key goes to the server and never comes back
  // — these two are all the client gets, and all it needs to show which key is
  // configured.
  api_key_set: boolean;
  /** Last 4 characters, e.g. "••••••••9fK2". Empty when no key is stored. */
  api_key_hint: string;
}

/** What the vision model decided a captured photo is. */
export type CaptureKind = "receipt" | "price_tag" | "unknown";

export interface FileItem {
  id: string;
  filename: string;
  s3_key: string;
  upload_date: string;
  type: "csv" | "bill";
  is_hidden?: boolean;
  /**
   * Photos only. Decides where tapping this file leads: only a receipt belongs
   * in the receipt review form. Absent on CSVs, and on bill rows written before
   * migration 014.
   */
  kind?: CaptureKind;
  /**
   * Receipts only (absent on CSVs and price tags): whether this photo has been
   * reviewed into a transaction. Until it has, none of its spending is in any
   * total — a distinction the file list otherwise cannot show.
   */
  is_verified?: boolean;
  /**
   * Quarter turns clockwise needed to view this photo upright, remembered
   * across sessions. The stored image is untouched — see migration 038.
   */
  rotation?: number;
  /**
   * Receipts only: the same purchase is also on a bank statement you imported.
   * Said here because the transactions list collapses the pair into one row,
   * so from Files a linked receipt otherwise looks like a duplicate.
   */
  is_linked?: boolean;
}

export interface PendingTransaction {
  description: string;
  amount: number;
  trans_date: string;
  category: string;
  merchant_name: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  charts?: string[];
  images?: string[];
  toolTraces?: ToolEvent[];
  pendingTransactions?: PendingTransaction[];
  /** Fixes the agent proposed. Nothing is written until the user confirms one. */
  pendingCorrections?: import("../services/correctionService").PendingCorrection[];
  isError?: boolean;
  /**
   * A photo the user just took, held as a local file URI. Kept separate from
   * `images` (which are signed remote URLs the agent attached) because these
   * are not fetched, are not persisted with the conversation, and disappear on
   * reload — the record of the capture is the BillFile row, not the bubble.
   */
  localImages?: string[];
  /** A captured photo awaiting the user's decision — see CaptureResult. */
  capture?: import("../services/captureService").CaptureResult;
}

export interface ToolEvent {
  type: "tool_start" | "tool_end";
  name: string;
  input?: string;
  snippet?: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  charts?: string[] | null;
  images?: string[] | null;
  pending_transactions?: PendingTransaction[] | null;
  /** Photos this turn was about. Resolved to fresh signed URLs server-side and
   *  merged into `images`, so the client needs nothing extra to render them. */
  bill_file_ids?: string[] | null;
  created_at: string;
}

export interface TaxBreakdownEntry {
  label?: string;
  rate?: number;
  amount?: number;
}

export interface TransactionListItem {
  id: string;
  trans_date: string | null;
  description: string | null;
  amount: number | null;
  category: string | null;
  merchant_name: string | null;
  location: string | null;
  note?: string | null; // free-text note the user attached; also indexed for search
  subtotal: number | null;
  tax_total: number | null;
  tax_breakdown: TaxBreakdownEntry[] | null;
  discount_total?: number | null; // order-level coupons subtracted from the basket
  savings_total?: number | null; // total saved (item markdowns + coupons), display only
  source: string | null;
  source_csv_id?: string | null;
  source_bill_file_id?: string | null;
  enriched_info?: string | null;
  linked_transaction_ids?: string[];
  created_at: string | null;
}

export interface TransactionDetailItem {
  id: string;
  item_description: string | null;
  item_quantity: number | null;
  item_quantity_unit: string | null; // what item_quantity counts: each|lb|oz|ml|ct…
  unit_quantity_subtotal: number | null; // pre-tax unit price actually paid (net of markdown)
  item_subtotal_price: number | null; // pre-tax line total (qty x unit)
  item_savings?: number | null; // how much this line was marked down, display only
  tax_amount: number | null; // tax for this item
  taxable: boolean | null;
  tax_rate: number | null;
  item_total_price: number | null; // post-tax line total (subtotal + tax)
  enriched_info: string | null;
}

/** Another record of the same real-world purchase — see LinkedTransaction. */
export interface LinkedTransaction {
  id: string;
  trans_date: string | null;
  amount: number | null;
  merchant_name: string | null;
  source: string | null;
  /** csv_receipt — a statement line and its receipt. csv_csv — two statements. */
  match_type: string | null;
  /** How many line items the other record has; 0 for a bank statement line. */
  detail_count: number;
}

export interface TransactionWithDetails extends TransactionListItem {
  enriched_info: string | null;
  content_hash?: string | null;
  source_csv_id?: string | null;
  source_bill_file_id?: string | null;
  details: TransactionDetailItem[];
  /**
   * Other records of the same purchase. The transactions list collapses a
   * linked pair to one row, so without this the user sees one entry where they
   * know there were two and cannot tell whether the other was absorbed or lost.
   */
  linked_transactions?: LinkedTransaction[];
  /**
   * Set when verifying a receipt matched a transaction that already existed.
   * The returned record is that earlier one — this upload wrote nothing.
   */
  is_duplicate?: boolean;
}

export interface ReceiptReviewLineItem {
  item_description?: string;
  item_quantity?: number;
  item_unit_price?: number; // net price actually paid per unit
  item_savings?: number; // markdown on this line, display only
  tax_rate?: number;
}

export interface ReceiptDiscount {
  label?: string;
  amount?: number;
}

export interface ReceiptReviewDraft {
  file_id: string;
  filename: string;
  extracted: {
    date?: string;
    time?: string | null;
    merchant_name?: string;
    category?: string;
    location?: string | null;
    total_amount?: number | string | null;
    discounts?: ReceiptDiscount[]; // order-level coupons, as raw OCR returns them
    /**
     * The single coupon figure this form submits. Present instead of
     * `discounts` when the receipt has already been verified once — verifying
     * writes the review back over the OCR draft, so a re-review reads its own
     * previous output rather than the original scan.
     */
    discount_total?: number | string | null;
    note?: string | null;
    line_items?: ReceiptReviewLineItem[];
  };
}
