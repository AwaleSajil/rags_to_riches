import { Platform } from "react-native";
import Constants from "expo-constants";
import { createLogger } from "./logger";

const log = createLogger("API");

/**
 * Where the backend lives, resolved once at startup.
 *
 * This lives in its own leaf module rather than in services/api.ts because
 * supabase.ts needs the URL (to fetch /public-config) while api.ts needs a
 * Supabase client (to read the access token). Having both in api.ts made the
 * two files import each other, and Metro warns that a require cycle can hand
 * back a partially-initialised module — here that would surface as an
 * `undefined` API_URL depending on which file the bundler reached first.
 */
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

  // Last resort. On a device "localhost" is the phone itself, so this only
  // works on web/simulator — warn loudly rather than failing opaquely later.
  const url = envUrl || "http://localhost:8000/api/v1";
  if (url.includes("localhost")) {
    log.warn(
      "Falling back to a localhost API URL on a native device — requests will " +
        "fail because localhost is the device itself. Set EXPO_PUBLIC_API_URL " +
        "to your machine's LAN IP (e.g. http://192.168.1.10:8000/api/v1).",
      { url }
    );
  } else {
    log.info("Fallback API URL", { url });
  }
  return url;
}

export const API_URL = getApiUrl();
