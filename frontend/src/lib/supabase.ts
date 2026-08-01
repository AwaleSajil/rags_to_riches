import "react-native-url-polyfill/auto";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
// From lib/apiUrl rather than services/api: api.ts imports this module for the
// access token, so importing it back would form a require cycle.
import { API_URL } from "./apiUrl";

let _client: SupabaseClient | null = null;
let _initPromise: Promise<SupabaseClient> | null = null;

function buildClient(url: string, anonKey: string): SupabaseClient {
  return createClient(url, anonKey, {
    auth: {
      storage: AsyncStorage,
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: false,
    },
  });
}

/**
 * Returns a ready-to-use Supabase client.
 * In local dev, uses EXPO_PUBLIC env vars (available immediately).
 * In Docker/production, fetches from /api/v1/public-config.
 */
export async function getSupabase(): Promise<SupabaseClient> {
  if (_client) return _client;

  if (!_initPromise) {
    _initPromise = (async () => {
      // Try env vars first (local dev with frontend .env)
      const envUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
      const envKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;

      if (envUrl && envKey) {
        _client = buildClient(envUrl, envKey);
        return _client;
      }

      // Fetch from backend (Docker / HF Spaces)
      const res = await fetch(`${API_URL}/public-config`);
      const config = await res.json();
      _client = buildClient(config.supabase_url, config.supabase_anon_key);
      return _client;
    })();
  }

  return _initPromise;
}

// Eagerly initialize if env vars are available (local dev)
const envUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const envKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if (envUrl && envKey) {
  _client = buildClient(envUrl, envKey);
}

/** @deprecated Use getSupabase() for Docker/production support */
export const supabase = _client as SupabaseClient;
