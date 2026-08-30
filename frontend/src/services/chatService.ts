import { Platform } from "react-native";
import { API_URL } from "./api";
import { getSupabase } from "../lib/supabase";
import { createLogger } from "../lib/logger";

const log = createLogger("ChatService");

export interface ChatEventCallbacks {
  onConversation?: (conversationId: string) => void;
  /** A piece of the answer as it is generated. Already stripped server-side of
   *  the ===MARKER=== blocks the UI must never show, so it is safe to render
   *  straight away. `onFinal` still replaces it with the authoritative text. */
  onToken?: (text: string) => void;
  onToolStart: (data: { name: string; input: string }) => void;
  onToolEnd: (data: { name: string; snippet: string }) => void;
  onFinal: (data: { content: string; charts: string[]; images: string[]; pendingTransactions?: any[]; pendingCorrections?: any[] }) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

function processSSEBuffer(
  buffer: string,
  callbacks: ChatEventCallbacks
): string {
  let boundary = buffer.indexOf("\n\n");
  while (boundary !== -1) {
    const frame = buffer.substring(0, boundary);
    buffer = buffer.substring(boundary + 2);

    const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
    const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));

    if (eventLine && dataLine) {
      const eventType = eventLine.substring(7);
      log.debug("SSE frame received", { eventType });
      try {
        const data = JSON.parse(dataLine.substring(6));

        switch (eventType) {
          case "conversation":
            log.info("Conversation id received", { id: data.conversation_id });
            callbacks.onConversation?.(data.conversation_id);
            break;
          // Deliberately not logged per token — one line per few characters
          // buries every other event in the stream.
          case "token":
            callbacks.onToken?.(data.text || "");
            break;
          case "tool_start":
            log.info("Tool started", { name: data.name, input: data.input?.substring(0, 100) });
            callbacks.onToolStart(data);
            break;
          case "tool_end":
            log.info("Tool ended", { name: data.name, snippetLength: data.snippet?.length });
            callbacks.onToolEnd(data);
            break;
          case "final":
            log.info("Final response received", {
              contentLength: data.content?.length,
              chartCount: data.charts?.length || 0,
            });
            callbacks.onFinal(data);
            break;
          case "done":
            log.info("Stream done");
            callbacks.onDone();
            break;
          case "error":
            log.error("Stream error event", { error: data.error });
            callbacks.onError(data.error || "Unknown error");
            break;
          default:
            log.warn("Unknown SSE event type", { eventType });
        }
      } catch (e) {
        log.warn("Malformed SSE frame - skipping", { frame: frame.substring(0, 200), error: e });
      }
    } else {
      log.debug("Incomplete SSE frame (no event/data line)", { frame: frame.substring(0, 100) });
    }

    boundary = buffer.indexOf("\n\n");
  }
  return buffer;
}

/**
 * Uses XMLHttpRequest for SSE on React Native (Android/iOS).
 * RN's fetch doesn't support ReadableStream, but XHR fires
 * onprogress with incremental responseText, giving us real-time streaming.
 */
function streamChatXHR(
  message: string,
  token: string | null,
  callbacks: ChatEventCallbacks,
  conversationId?: string | null,
  billFileIds?: string[] | null
): Promise<void> {
  log.info("XHR stream starting (mobile)", { messageLength: message.length, hasToken: !!token });
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/chat`);
    xhr.setRequestHeader("Content-Type", "application/json");
    if (token) {
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    }

    let lastIndex = 0;
    let progressCount = 0;
    let sseBuffer = "";

    xhr.onprogress = () => {
      const newText = xhr.responseText.substring(lastIndex);
      lastIndex = xhr.responseText.length;
      if (newText) {
        progressCount++;
        log.debug(`XHR onprogress #${progressCount}`, {
          chunkLength: newText.length,
          totalReceived: lastIndex,
        });
        sseBuffer += newText;
        sseBuffer = processSSEBuffer(sseBuffer, callbacks);
      }
    };

    xhr.onload = () => {
      log.info("XHR onload", { status: xhr.status, totalBytes: xhr.responseText.length });
      if (xhr.status >= 400) {
        log.error("XHR error response", { status: xhr.status });
        try {
          const error = JSON.parse(xhr.responseText);
          callbacks.onError(error.detail || `HTTP ${xhr.status}`);
        } catch {
          callbacks.onError(`HTTP ${xhr.status}`);
        }
        resolve();
        return;
      }
      // Process any remaining data
      const remaining = xhr.responseText.substring(lastIndex);
      if (remaining) {
        sseBuffer += remaining;
      }
      if (sseBuffer) {
        log.debug("Processing remaining XHR data", { remainingLength: sseBuffer.length });
        sseBuffer = processSSEBuffer(sseBuffer, callbacks);
      }
      // Ensure onDone fires
      if (!xhr.responseText.includes("event: done")) {
        log.warn("No 'done' event in stream - firing onDone manually");
        callbacks.onDone();
      }
      resolve();
    };

    xhr.onerror = () => {
      log.error("XHR network error");
      callbacks.onError("Network error");
      resolve();
    };

    xhr.ontimeout = () => {
      log.error("XHR request timed out");
      callbacks.onError("Request timed out");
      resolve();
    };

    log.debug("XHR sending request", { url: `${API_URL}/chat` });
    xhr.send(JSON.stringify({ message, conversation_id: conversationId ?? null, bill_file_ids: billFileIds ?? null }));
  });
}

export async function streamChat(
  message: string,
  callbacks: ChatEventCallbacks,
  conversationId?: string | null,
  /** Photos this turn is about. Stored with the message so the picture comes
   *  back on reload — the local file URI shown at the time does not survive,
   *  and a signed URL would have expired long before. */
  billFileIds?: string[] | null
): Promise<void> {
  let token: string | null = null;
  try {
    const supabase = await getSupabase();
    // Use refreshSession to ensure the token is valid before starting a stream
    // (streams can't retry on 401 mid-flight like regular requests)
    const { data: { session } } = await supabase.auth.refreshSession();
    token = session?.access_token ?? null;
  } catch (e) {
    log.warn("Failed to get token from Supabase session", e);
  }
  log.info("streamChat called", {
    platform: Platform.OS,
    messageLength: message.length,
    hasToken: !!token,
    message: message.substring(0, 80),
  });

  // React Native (Android/iOS) doesn't support fetch ReadableStream.
  // Use XMLHttpRequest which supports incremental onprogress events.
  if (Platform.OS !== "web") {
    log.info("Using XHR streaming (mobile platform)");
    return streamChatXHR(message, token, callbacks, conversationId, billFileIds);
  }

  // Web: use fetch + ReadableStream for true streaming
  log.info("Using fetch ReadableStream (web platform)");
  const url = `${API_URL}/chat`;
  log.debug("Fetch POST", { url });

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ message, conversation_id: conversationId ?? null, bill_file_ids: billFileIds ?? null }),
    });

    log.info("Fetch response received", { status: response.status, ok: response.ok });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ detail: response.statusText }));
      log.error("Chat fetch error", { status: response.status, detail: error.detail });
      callbacks.onError(error.detail || `HTTP ${response.status}`);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      log.error("ReadableStream not available on response body");
      callbacks.onError("Streaming not supported");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let chunkCount = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        log.info("ReadableStream done", { totalChunks: chunkCount });
        break;
      }

      chunkCount++;
      const decoded = decoder.decode(value, { stream: true });
      log.debug(`Stream chunk #${chunkCount}`, { chunkLength: decoded.length });
      buffer += decoded;
      buffer = processSSEBuffer(buffer, callbacks);
    }
  } catch (error) {
    log.error("streamChat fetch exception", error);
    callbacks.onError(error instanceof Error ? error.message : "Network error");
  }
}
