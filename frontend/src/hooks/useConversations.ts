import { useState, useCallback } from "react";
import * as conversationService from "../services/conversationService";
import { createLogger } from "../lib/logger";
import type { Conversation } from "../lib/types";

const log = createLogger("useConversations");

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      setConversations(await conversationService.listConversations());
    } catch (e) {
      log.error("Failed to load conversations", e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    try {
      await conversationService.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      log.error("Failed to delete conversation", e);
    }
  }, []);

  return { conversations, isLoading, refresh, remove };
}
