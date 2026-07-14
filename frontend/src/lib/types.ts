export interface User {
  id: string;
  email: string;
}

export interface AccountConfig {
  id?: string;
  user_id: string;
  llm_provider: string;
  api_key: string;
  decode_model: string;
  embedding_model: string;
  deep_enrichment?: boolean;
}

export interface FileItem {
  id: string;
  filename: string;
  s3_key: string;
  upload_date: string;
  type: "csv" | "bill";
  is_hidden?: boolean;
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
  isError?: boolean;
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
  item_unit_subtotal_price: number | null; // pre-tax unit price actually paid (net of markdown)
  item_subtotal_price: number | null; // pre-tax line total (qty x unit)
  item_savings?: number | null; // how much this line was marked down, display only
  tax_amount: number | null; // tax for this item
  taxable: boolean | null;
  tax_rate: number | null;
  item_total_price: number | null; // post-tax line total (subtotal + tax)
  enriched_info: string | null;
}

export interface TransactionWithDetails extends TransactionListItem {
  enriched_info: string | null;
  content_hash?: string | null;
  source_csv_id?: string | null;
  source_bill_file_id?: string | null;
  details: TransactionDetailItem[];
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
    discounts?: ReceiptDiscount[]; // order-level coupons subtracted from the basket
    line_items?: ReceiptReviewLineItem[];
  };
}
