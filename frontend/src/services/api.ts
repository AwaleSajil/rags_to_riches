import { getSupabase } from "../lib/supabase";
import { createLogger } from "../lib/logger";
import { API_URL } from "../lib/apiUrl";

const log = createLogger("API");

// Re-exported so the many `import { API_URL } from "./api"` call sites keep
// working; the value itself is resolved in lib/apiUrl.ts, which imports nothing
// from here so that supabase.ts can use it without creating a require cycle.
export { API_URL };

/**
 * Get the current access token from the Supabase session.
 * Uses getSession() for the local cache, or refreshSession() to force
 * a server-side refresh when the cached token has been rejected.
 */
export async function getAccessToken(forceRefresh = false): Promise<string | null> {
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

  const isFormData = options.body instanceof FormData;

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  // Don't set Content-Type for FormData (the runtime sets the multipart boundary)
  if (!isFormData) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const url = `${API_URL}${path}`;
  log.debug("Full request URL", { url });

  // React Native's fetch aborts multipart file uploads when an AbortSignal is
  // attached — it fails immediately with "Network request failed". So we only
  // wire up the timeout/abort for non-FormData requests; uploads run untimed.
  const timeoutMs = options.timeout || 5000;
  const controller = isFormData ? null : new AbortController();
  const timeoutId = controller
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;
  const signal = controller?.signal;

  try {
    const res = await fetch(url, {
      ...options,
      headers,
      signal,
    });

    log.info(`${method} ${path} -> ${res.status}`, {
      status: res.status,
      statusText: res.statusText,
      ok: res.ok,
    });

    if (timeoutId) clearTimeout(timeoutId);

    // On 401, try once with a refreshed token before giving up
    if (res.status === 401 && token) {
      log.warn("401 Unauthorized - attempting session refresh and retry");
      const freshToken = await getAccessToken(true);
      if (freshToken && freshToken !== token) {
        log.info("Got fresh token - retrying request");
        const retryHeaders = { ...headers, Authorization: `Bearer ${freshToken}` };
        const retryRes = await fetch(url, { ...options, headers: retryHeaders, signal });
        log.info(`${method} ${path} retry -> ${retryRes.status}`);
        return retryRes;
      }
      log.warn("Session refresh did not yield a new token");
    }

    return res;
  } catch (error: any) {
    if (timeoutId) clearTimeout(timeoutId);
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
