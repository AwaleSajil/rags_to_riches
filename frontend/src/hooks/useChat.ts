import { useState, useCallback } from "react";
import { streamChat } from "../services/chatService";
import * as conversationService from "../services/conversationService";
import { createLogger } from "../lib/logger";
import type { ChatMessage, StoredMessage, ToolEvent } from "../lib/types";

const log = createLogger("useChat");

function toChatMessage(m: StoredMessage): ChatMessage {
  return {
    role: m.role,
    content: m.content,
    charts: m.charts ?? undefined,
    images: m.images ?? undefined,
    pendingTransactions: m.pending_transactions ?? undefined,
  };
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentToolTraces, setCurrentToolTraces] = useState<ToolEvent[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      if (isStreaming) {
        log.warn("sendMessage called while already streaming - ignoring");
        return;
      }
      if (!text.trim()) {
        log.debug("sendMessage called with empty text - ignoring");
        return;
      }

      log.info("Sending message", { text: text.substring(0, 80), messageCount: messages.length });

      // Add user message
      const userMsg: ChatMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setCurrentToolTraces([]);

      let toolTraces: ToolEvent[] = [];

      try {
        await streamChat(text, {
          onConversation: (id) => {
            log.info("Conversation id (hook)", { id });
            setConversationId(id);
          },
          onToolStart: (data) => {
            log.info("Tool started (hook)", { name: data.name });
            const event: ToolEvent = {
              type: "tool_start",
              name: data.name,
              input: data.input,
            };
            toolTraces = [...toolTraces, event];
            setCurrentToolTraces([...toolTraces]);
          },
          onToolEnd: (data) => {
            log.info("Tool ended (hook)", { name: data.name, snippetLength: data.snippet?.length });
            const event: ToolEvent = {
              type: "tool_end",
              name: data.name,
              snippet: data.snippet,
            };
            toolTraces = [...toolTraces, event];
            setCurrentToolTraces([...toolTraces]);
          },
          onFinal: (data) => {
            log.info("Final response (hook)", {
              contentLength: data.content?.length,
              chartCount: data.charts?.length || 0,
              imageCount: data.images?.length || 0,
              toolTraceCount: toolTraces.length,
            });
            const assistantMsg: ChatMessage = {
              role: "assistant",
              content: data.content,
              charts: data.charts?.length ? data.charts : undefined,
              images: data.images?.length ? data.images : undefined,
              toolTraces: toolTraces.length ? [...toolTraces] : undefined,
              pendingTransactions: data.pendingTransactions?.length ? data.pendingTransactions : undefined,
            };
            setMessages((prev) => [...prev, assistantMsg]);
            setCurrentToolTraces([]);
          },
          onDone: () => {
            log.info("Stream done (hook) - setting isStreaming=false");
            setIsStreaming(false);
          },
          onError: (error) => {
            log.error("Stream error (hook)", { error });
            const errorMsg: ChatMessage = {
              role: "assistant",
              content: error || "Something went wrong. Please try again.",
              isError: true,
            };
            setMessages((prev) => [...prev, errorMsg]);
            setIsStreaming(false);
          },
        }, conversationId);
      } catch (e: any) {
        log.error("sendMessage exception", e);
        const errorMsg: ChatMessage = {
          role: "assistant",
          content: e?.message || "Network error. Please try again.",
          isError: true,
        };
        setMessages((prev) => [...prev, errorMsg]);
        setIsStreaming(false);
      }
    },
    [isStreaming, conversationId]
  );

  const loadConversation = useCallback(async (id: string) => {
    log.info("Loading conversation", { id });
    setConversationId(id);
    setCurrentToolTraces([]);
    try {
      const stored = await conversationService.getConversationMessages(id);
      setMessages(stored.map(toChatMessage));
    } catch (e) {
      log.error("Failed to load conversation", e);
      setMessages([]);
    }
  }, []);

  const newConversation = useCallback(() => {
    log.info("Starting new conversation");
    setConversationId(null);
    setMessages([]);
    setCurrentToolTraces([]);
  }, []);

  const clearMessages = useCallback(() => {
    log.info("Clearing all messages");
    setMessages([]);
    setCurrentToolTraces([]);
  }, []);

  return {
    messages,
    isStreaming,
    currentToolTraces,
    conversationId,
    sendMessage,
    loadConversation,
    newConversation,
    clearMessages,
  };
}
