import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { API_URL } from "./api";
import { createLogger } from "../lib/logger";

const log = createLogger("ChatService");

export interface ChatEventCallbacks {
  onToolStart: (data: { name: string; input: string }) => void;
  onToolEnd: (data: { name: string; snippet: string }) => void;
  onFinal: (data: { content: string; charts: string[]; images: string[] }) => void;
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
  callbacks: ChatEventCallbacks
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
    xhr.send(JSON.stringify({ message }));
  });
}

export async function streamChat(
  message: string,
  callbacks: ChatEventCallbacks
): Promise<void> {
  const token = await AsyncStorage.getItem("access_token");
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
    return streamChatXHR(message, token, callbacks);
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
      body: JSON.stringify({ message }),
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
