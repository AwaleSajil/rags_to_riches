import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import Constants from "expo-constants";
import { createLogger } from "../lib/logger";

const log = createLogger("API");

function getApiUrl(): string {
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  log.debug("getApiUrl called", { envUrl, platform: Platform.OS });

  // If explicitly set to a non-localhost URL, use it as-is
  if (envUrl && !envUrl.includes("localhost")) {
    log.info("Using env API URL (non-localhost)", { url: envUrl });
    return envUrl;
  }

  if (Platform.OS === "web") {
    // In production, always use relative path (same origin serves both).
    // Only honour envUrl for local dev (localhost).
    const url = envUrl && envUrl.includes("localhost") ? envUrl : "/api/v1";
    log.info("Web platform API URL", { url });
    return url;
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

export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = await AsyncStorage.getItem("access_token");
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

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    log.info(`${method} ${path} -> ${res.status}`, {
      status: res.status,
      statusText: res.statusText,
      ok: res.ok,
    });

    if (res.status === 401) {
      log.warn("401 Unauthorized - clearing stored token");
      await AsyncStorage.removeItem("access_token");
      // Caller should handle redirect to login
    }

    return res;
  } catch (error) {
    log.error(`${method} ${path} NETWORK ERROR`, error);
    throw error;
  }
}

export async function apiJson<T>(
  path: string,
  options: RequestInit = {}
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
