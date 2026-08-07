import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type {
  PendingTransaction,
  ReceiptReviewDraft,
  ReceiptReviewLineItem,
  TaxBreakdownEntry,
  TransactionListItem,
  TransactionWithDetails,
} from "../lib/types";

const log = createLogger("TransactionService");

export interface TransactionListParams {
  category?: string;
  startDate?: string;
  endDate?: string;
  q?: string;
}

export async function listTransactions(
  params: TransactionListParams = {}
): Promise<TransactionListItem[]> {
  const qs = new URLSearchParams();
  if (params.category) qs.set("category", params.category);
  if (params.startDate) qs.set("start_date", params.startDate);
  if (params.endDate) qs.set("end_date", params.endDate);
  if (params.q) qs.set("q", params.q);
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  const items = await apiJson<TransactionListItem[]>(`/transactions${suffix}`);
  log.info("Transactions loaded", { count: items.length });
  return items;
}

export async function getTransaction(id: string): Promise<TransactionWithDetails> {
  return apiJson<TransactionWithDetails>(`/transactions/${id}`);
}

export async function getReceiptReview(fileId: string): Promise<ReceiptReviewDraft> {
  return apiJson<ReceiptReviewDraft>(`/transactions/receipt-review/${fileId}`);
}

export interface ReceiptReviewPayload {
  date: string;
  time?: string | null;
  merchant_name: string;
  category: string;
  location?: string | null;
  total_amount?: number;
  discount_total?: number; // order-level coupons subtracted from the basket
  line_items: Required<ReceiptReviewLineItem>[];
  /**
   * Same note field the transaction editor writes. Omit to leave an existing
   * note untouched when re-verifying; send "" to clear it.
   */
  note?: string;
}

export async function verifyReceiptReview(
  fileId: string,
  review: ReceiptReviewPayload
): Promise<TransactionWithDetails> {
  log.info("Verifying receipt review", { fileId, lineItemCount: review.line_items.length });
  return apiJson<TransactionWithDetails>(`/transactions/receipt-review/${fileId}/verify`, {
    method: "POST",
    body: JSON.stringify(review),
    timeout: 30000,
  });
}

export interface TransactionUpdatePayload {
  trans_date?: string;
  description?: string;
  merchant_name?: string;
  amount?: number;
  category?: string;
  location?: string;
  note?: string | null;
  subtotal?: number;
  tax_total?: number;
  tax_breakdown?: TaxBreakdownEntry[];
}

export async function updateTransaction(
  id: string,
  changes: TransactionUpdatePayload
): Promise<TransactionWithDetails> {
  log.info("Updating transaction", { id, fields: Object.keys(changes) });
  // Editing merchant/category re-embeds the vector server-side, which can take
  // a few seconds — give it more room than the default 5s request timeout.
  return apiJson<TransactionWithDetails>(`/transactions/${id}`, {
    method: "PATCH",
    body: JSON.stringify(changes),
    timeout: 30000,
  });
}

export async function deleteTransaction(id: string): Promise<void> {
  log.info("Deleting transaction", { id });
  await apiJson(`/transactions/${id}`, { method: "DELETE", timeout: 15000 });
}

export interface TransactionDetailInput {
  item_description?: string | null;
  item_quantity?: number | null;
  // Sent back unchanged by the editor. The server rewrites every line on save,
  // so a field omitted here is a field erased.
  item_quantity_unit?: string | null;
  unit_quantity_subtotal?: number | null;
  item_savings?: number | null;
  tax_rate?: number | null;
}

export async function replaceTransactionDetails(
  id: string,
  details: TransactionDetailInput[]
): Promise<TransactionWithDetails> {
  log.info("Replacing line items", { id, count: details.length });
  // Replacing line items re-embeds the vector server-side — allow extra time.
  return apiJson<TransactionWithDetails>(`/transactions/${id}/details`, {
    method: "PUT",
    body: JSON.stringify({ details }),
    timeout: 30000,
  });
}

export interface TransactionConfirmResult {
  id: string;
  description: string;
  amount: number;
  trans_date: string;
  category: string;
  merchant_name: string | null;
}

export async function confirmTransaction(
  tx: PendingTransaction
): Promise<TransactionConfirmResult> {
  const payload = { ...tx, amount: Math.abs(tx.amount) };
  log.info("Confirming transaction", { description: payload.description, amount: payload.amount });
  return apiJson<TransactionConfirmResult>("/transactions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
