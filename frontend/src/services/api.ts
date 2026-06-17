import { Platform } from "react-native";
import Constants from "expo-constants";
import { getSupabase } from "../lib/supabase";
import { createLogger } from "../lib/logger";

const log = createLogger("API");

function getApiUrl(): string {
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  log.debug("getApiUrl called", { envUrl, platform: Platform.OS });

  if (Platform.OS === "web") {
    // If an explicit API URL is set (local dev or separate Docker containers), use it.
    // Otherwise fall back to a relative path (single-container / HF Spaces).
    const url = envUrl || "/api/v1";
    log.info("Web platform API URL", { url });
    return url;
  }

  // Native: if explicitly set to a non-localhost URL, use it as-is
  if (envUrl && !envUrl.includes("localhost")) {
    log.info("Using env API URL (non-localhost)", { url: envUrl });
    return envUrl;
  }

  if (Platform.OS === "android") {
    // Android emulator uses 10.0.2.2 to reach the host machine
    const url = "http://10.0.2.2:8000/api/v1";
    log.info("Android emulator API URL", { url });
    return url;
  }

  // iOS: extract LAN IP from Expo's dev server hostUri
  const debuggerHost = Constants.expoConfig?.hostUri?.split(":")[0];
  if (debuggerHost) {
    const url = `http://${debuggerHost}:8000/api/v1`;
    log.info("iOS API URL from hostUri", { debuggerHost, url });
    return url;
  }

  const url = envUrl || "http://localhost:8000/api/v1";
  log.info("Fallback API URL", { url });
  return url;
}

export const API_URL = getApiUrl();

/**
 * Get the current access token from the Supabase session.
 * Uses getSession() for the local cache, or refreshSession() to force
 * a server-side refresh when the cached token has been rejected.
 */
async function getAccessToken(forceRefresh = false): Promise<string | null> {
  try {
    const supabase = await getSupabase();
    if (forceRefresh) {
      log.info("Forcing Supabase session refresh");
      const { data: { session }, error } = await supabase.auth.refreshSession();
      if (error) {
        log.warn("Session refresh failed", { error: error.message });
        return null;
      }
      return session?.access_token ?? null;
    }
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token ?? null;
  } catch (e) {
    log.warn("Failed to get token from Supabase session", e);
    return null;
  }
}

interface CustomRequestInit extends RequestInit {
  timeout?: number;
}

export async function apiFetch(
  path: string,
  options: CustomRequestInit = {}
): Promise<Response> {
  const token = await getAccessToken();
  const method = options.method || "GET";

  log.info(`${method} ${path}`, {
    hasToken: !!token,
    hasBody: !!options.body,
    isFormData: options.body instanceof FormData,
  });

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  // Don't set Content-Type for FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const url = `${API_URL}${path}`;
  log.debug("Full request URL", { url });

  const controller = new AbortController();
  const timeoutMs = options.timeout || 5000;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });

    log.info(`${method} ${path} -> ${res.status}`, {
      status: res.status,
      statusText: res.statusText,
      ok: res.ok,
    });

    clearTimeout(timeoutId);

    // On 401, try once with a refreshed token before giving up
    if (res.status === 401 && token) {
      log.warn("401 Unauthorized - attempting session refresh and retry");
      const freshToken = await getAccessToken(true);
      if (freshToken && freshToken !== token) {
        log.info("Got fresh token - retrying request");
        const retryHeaders = { ...headers, Authorization: `Bearer ${freshToken}` };
        const retryRes = await fetch(url, { ...options, headers: retryHeaders, signal: controller.signal });
        log.info(`${method} ${path} retry -> ${retryRes.status}`);
        return retryRes;
      }
      log.warn("Session refresh did not yield a new token");
    }

    return res;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      log.error(`${method} ${path} TIMEOUT (${timeoutMs}ms)`);
      throw new Error("Request timed out — is the backend server running?");
    }
    log.error(`${method} ${path} NETWORK ERROR`, error);
    throw error;
  }
}

export async function apiJson<T>(
  path: string,
  options: CustomRequestInit = {}
): Promise<T> {
  const res = await apiFetch(path, options);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    log.error(`apiJson error for ${path}`, {
      status: res.status,
      detail: error.detail,
    });
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  log.debug(`apiJson response for ${path}`, { responseKeys: Object.keys(data || {}) });
  return data;
}
