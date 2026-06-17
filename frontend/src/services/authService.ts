import { getSupabase } from "../lib/supabase";
import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type { User } from "../lib/types";

const log = createLogger("AuthService");

export interface AuthResult {
  user: User;
}

export async function login(
  email: string,
  password: string
): Promise<AuthResult> {
  log.info("Login attempt", { email });

  // Use Supabase JS directly for auth (handles token refresh)
  const supabase = await getSupabase();
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    log.error("Supabase login failed", { email, error: error.message });
    throw error;
  }
  if (!data.user || !data.session) {
    log.error("Login returned no user/session", { email });
    throw new Error("Login failed");
  }

  log.info("Supabase login successful", {
    userId: data.user.id,
    email: data.user.email,
    tokenExpiry: data.session.expires_at,
  });

  // Sync user to our User table via backend
  try {
    log.debug("Syncing user to backend User table...");
    await apiJson("/auth/login", {
      method: "POST",
    });
    log.debug("Backend user sync successful");
  } catch (e) {
    log.warn("Backend user sync failed (non-critical)", e);
    // Non-critical: backend sync for User table upsert
  }

  return {
    user: { id: data.user.id, email: data.user.email! },
  };
}

export async function register(
  email: string,
  password: string
): Promise<{ message: string }> {
  log.info("Register attempt", { email });

  const supabase = await getSupabase();
  const { data, error } = await supabase.auth.signUp({ email, password });

  if (error) {
    log.error("Supabase registration failed", { email, error: error.message });
    throw error;
  }
  if (!data.user) {
    log.error("Registration returned no user", { email });
    throw new Error("Signup failed");
  }

  log.info("Supabase registration successful", {
    userId: data.user.id,
    email: data.user.email,
  });

  // Sync user to User table
  try {
    log.debug("Syncing new user to backend...");
    await apiJson("/auth/register", {
      method: "POST",
    });
    log.debug("Backend registration sync successful");
  } catch (e) {
    log.warn("Backend registration sync failed (non-critical)", e);
    // Non-critical
  }

  return { message: "Account created successfully" };
}

export async function logout(): Promise<void> {
  log.info("Logout initiated");
  const supabase = await getSupabase();
  await supabase.auth.signOut();
  log.info("Supabase signOut complete");
}
