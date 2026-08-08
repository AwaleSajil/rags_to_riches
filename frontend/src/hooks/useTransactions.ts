import * as transactionService from "../services/transactionService";
import type { TransactionListItem } from "../lib/types";
import { useAsyncResource } from "./useAsyncResource";

const EMPTY: TransactionListItem[] = [];

export function useTransactions() {
  const { data, isLoading, hasLoaded, error, refresh } = useAsyncResource(
    transactionService.listTransactions,
    EMPTY,
    { label: "transactions" }
  );
  return { transactions: data, isLoading, hasLoaded, error, refresh };
}
