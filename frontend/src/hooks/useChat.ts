import { useState, useCallback } from "react";
import { streamChat } from "../services/chatService";
import { capturePhoto, getCapture } from "../services/captureService";
import type { CaptureLocation, CaptureResult } from "../services/captureService";
import * as conversationService from "../services/conversationService";
import { createLogger } from "../lib/logger";
import type { ChatMessage, StoredMessage, ToolEvent } from "../lib/types";

const log = createLogger("useChat");

/** What the assistant says above a capture card. */
function captureIntro(kind: CaptureResult["kind"]): string {
  if (kind === "price_tag") return "That looks like a shelf price. Check the details and I'll compare it.";
  if (kind === "receipt") return "That looks like a receipt — open it to review and save.";
  return "";
}

function toChatMessage(m: StoredMessage): ChatMessage {
  return {
    role: m.role,
    content: m.content,
    charts: m.charts ?? undefined,
    images: m.images ?? undefined,
    pendingTransactions: m.pending_transactions ?? undefined,
  };
}

// Re-rendering per token means re-parsing the whole markdown body per token,
// which is the expensive part. Batching to ~20fps is indistinguishable from
// per-token and costs a fraction of the work.
const TOKEN_FLUSH_MS = 50;

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  // True once the answer itself starts arriving, so the screen can drop the
  // "Thinking..." indicator instead of showing it under live text.
  const [isReceivingText, setIsReceivingText] = useState(false);
  const [currentToolTraces, setCurrentToolTraces] = useState<ToolEvent[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string, billFileIds?: string[]) => {
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
      setIsReceivingText(false);
      setCurrentToolTraces([]);

      let toolTraces: ToolEvent[] = [];

      // --- live answer text ---------------------------------------------------
      // `streamed` is the full answer so far; the placeholder message it feeds is
      // always the last one, because nothing else can append during a turn.
      let streamed = "";
      let placeholderAdded = false;
      let flushTimer: ReturnType<typeof setTimeout> | null = null;

      const flushStreamed = () => {
        flushTimer = null;
        const snapshot = streamed;
        setMessages((prev) => {
          if (!prev.length) return prev;
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], content: snapshot };
          return next;
        });
      };

      const cancelFlush = () => {
        if (flushTimer) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }
      };

      try {
        await streamChat(text, {
          onConversation: (id) => {
            log.info("Conversation id (hook)", { id });
            setConversationId(id);
          },
          onToken: (chunk) => {
            streamed += chunk;
            if (!placeholderAdded) {
              placeholderAdded = true;
              setIsReceivingText(true);
              setMessages((prev) => [...prev, { role: "assistant", content: streamed }]);
              return;
            }
            if (!flushTimer) flushTimer = setTimeout(flushStreamed, TOKEN_FLUSH_MS);
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
            cancelFlush();
            const assistantMsg: ChatMessage = {
              role: "assistant",
              content: data.content,
              charts: data.charts?.length ? data.charts : undefined,
              images: data.images?.length ? data.images : undefined,
              toolTraces: toolTraces.length ? [...toolTraces] : undefined,
              pendingTransactions: data.pendingTransactions?.length ? data.pendingTransactions : undefined,
              pendingCorrections: data.pendingCorrections?.length ? data.pendingCorrections : undefined,
            };
            // Replaces the streamed placeholder rather than appending after it.
            // `data.content` is the authoritative text — fully stripped of the
            // marker blocks — so whatever streamed is corrected here, and the
            // saved message and the visible one cannot drift apart.
            setMessages((prev) =>
              placeholderAdded && prev.length
                ? [...prev.slice(0, -1), assistantMsg]
                : [...prev, assistantMsg]
            );
            setIsReceivingText(false);
            setCurrentToolTraces([]);
          },
          onDone: () => {
            log.info("Stream done (hook) - setting isStreaming=false");
            cancelFlush();
            setIsStreaming(false);
            setIsReceivingText(false);
          },
          onError: (error) => {
            log.error("Stream error (hook)", { error });
            cancelFlush();
            // Keep whatever had already streamed — it is real answer text, and
            // deleting it in front of the user reads as though more was lost
            // than actually was. Flush it first so it is not a stale snapshot.
            if (placeholderAdded) flushStreamed();
            const errorMsg: ChatMessage = {
              role: "assistant",
              content: error || "Something went wrong. Please try again.",
              isError: true,
            };
            setMessages((prev) => [...prev, errorMsg]);
            setIsStreaming(false);
            setIsReceivingText(false);
          },
        }, conversationId, billFileIds);
      } catch (e: any) {
        log.error("sendMessage exception", e);
        cancelFlush();
        if (placeholderAdded) flushStreamed();
        const errorMsg: ChatMessage = {
          role: "assistant",
          content: e?.message || "Network error. Please try again.",
          isError: true,
        };
        setMessages((prev) => [...prev, errorMsg]);
        setIsStreaming(false);
        setIsReceivingText(false);
      }
    },
    [isStreaming, conversationId]
  );

  /**
   * Take a photo into the conversation: show it, classify it, and answer with
   * the card that matches what it turned out to be.
   *
   * This does not go through /chat — there is no LLM conversation turn here,
   * just an upload and one vision call. It is presented as a turn because that
   * is how it reads to the user.
   */
  const sendPhoto = useCallback(
    async (
      photo: { uri: string; name: string; type: string },
      location?: CaptureLocation
      // Returns what the photo turned out to be so the screen can hand a
      // receipt straight to the review form, the way the Files tab does.
    ): Promise<CaptureResult | undefined> => {
      if (isStreaming) {
        // Say so rather than vanishing. This used to return silently, so a photo
        // taken while a previous turn was still streaming — or after one that
        // errored without clearing the flag — produced no upload, no card and
        // no message, which is indistinguishable from the app being broken.
        log.warn("sendPhoto called while busy - telling the user");
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "I'm still working on the last thing — try that photo again in a moment.",
            isError: true,
          },
        ]);
        return undefined;
      }
      log.info("Capturing photo into chat", { name: photo.name, located: !!location });

      setMessages((prev) => [
        ...prev,
        { role: "user", content: "", localImages: [photo.uri] },
      ]);
      setIsStreaming(true);

      try {
        const result = await capturePhoto(photo, location);
        log.info("Capture classified", { kind: result.kind, fileId: result.file_id });
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: captureIntro(result.kind), capture: result },
        ]);
        return result;
      } catch (e: any) {
        log.error("Capture failed", e);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: e?.message || "I couldn't read that photo. Try again with more light?",
            isError: true,
          },
        ]);
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming]
  );

  /**
   * Show a photo that was read elsewhere — the batch upload path in the Files
   * tab, or a stored photo tapped in the file list.
   *
   * Fetched rather than passed through navigation params, which can only carry
   * a file id. Ignores a photo already on screen so returning to this tab does
   * not stack duplicate cards.
   */
  const showCapture = useCallback(async (fileId: string) => {
    log.info("Loading capture into chat", { fileId });
    try {
      const result = await getCapture(fileId);
      setMessages((prev) =>
        prev.some((m) => m.capture?.file_id === fileId)
          ? prev
          : [...prev, { role: "assistant", content: captureIntro(result.kind), capture: result }]
      );
    } catch (e: any) {
      log.error("Could not load capture", e);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: e?.message || "I couldn't open that photo.",
          isError: true,
        },
      ]);
    }
  }, []);

  /** Replace a capture card in place once the user answers "which is this?". */
  const updateCapture = useCallback((fileId: string, result: CaptureResult) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.capture?.file_id === fileId
          ? { ...m, content: captureIntro(result.kind), capture: result }
          : m
      )
    );
  }, []);

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
    isReceivingText,
    currentToolTraces,
    conversationId,
    sendMessage,
    sendPhoto,
    showCapture,
    updateCapture,
    loadConversation,
    newConversation,
    clearMessages,
  };
}
