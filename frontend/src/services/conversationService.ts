import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type { Conversation, StoredMessage } from "../lib/types";

const log = createLogger("ConversationService");

export async function listConversations(): Promise<Conversation[]> {
  const res = await apiJson<{ conversations: Conversation[] }>("/conversations");
  log.info("Conversations loaded", { count: res.conversations.length });
  return res.conversations;
}

export async function createConversation(): Promise<Conversation> {
  return apiJson<Conversation>("/conversations", { method: "POST" });
}

export async function getConversationMessages(id: string): Promise<StoredMessage[]> {
  const res = await apiJson<{ messages: StoredMessage[] }>(`/conversations/${id}/messages`);
  return res.messages;
}

export async function deleteConversation(id: string): Promise<void> {
  await apiJson(`/conversations/${id}`, { method: "DELETE" });
}
