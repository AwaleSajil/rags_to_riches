import { useCallback } from "react";
import * as conversationService from "../services/conversationService";
import { createLogger } from "../lib/logger";
import type { Conversation } from "../lib/types";
import { useAsyncResource } from "./useAsyncResource";

const log = createLogger("useConversations");
const EMPTY: Conversation[] = [];

export function useConversations() {
  // Not on mount: the drawer loads when it opens, not when the chat screen
  // mounts behind it.
  const { data, setData, isLoading, error, refresh } = useAsyncResource(
    conversationService.listConversations,
    EMPTY,
    { label: "conversations", immediate: false }
  );

  const remove = useCallback(
    async (id: string) => {
      // Removed from the list straight away rather than refetching — the row
      // vanishing under the user's finger is the point.
      try {
        await conversationService.deleteConversation(id);
        setData((prev) => prev.filter((c) => c.id !== id));
      } catch (e) {
        log.error("Failed to delete conversation", e);
      }
    },
    [setData]
  );

  // `error` is new here: a failed load used to be logged and nothing else, so
  // the drawer just sat empty as though there were no conversations.
  return { conversations: data, isLoading, error, refresh, remove };
}
