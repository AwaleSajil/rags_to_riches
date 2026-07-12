import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type {
  PendingTransaction,
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
