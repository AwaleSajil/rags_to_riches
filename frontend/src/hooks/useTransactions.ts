import { useState, useEffect, useCallback } from "react";
import * as transactionService from "../services/transactionService";
import { createLogger } from "../lib/logger";
import type { TransactionListItem } from "../lib/types";

const log = createLogger("useTransactions");

export function useTransactions() {
  const [transactions, setTransactions] = useState<TransactionListItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      setTransactions(await transactionService.listTransactions());
    } catch (e: any) {
      log.error("Failed to load transactions", e);
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { transactions, isLoading, error, refresh };
}
