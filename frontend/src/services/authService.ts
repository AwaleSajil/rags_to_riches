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
  const normalizedEmail = email.trim().toLowerCase();
  log.info("Register attempt", { email: normalizedEmail });

  const supabase = await getSupabase();
  const { data, error } = await supabase.auth.signUp({
    email: normalizedEmail,
    password,
    options: {
      // The standalone Android app owns this scheme (declared in app.config.js).
      // Supabase must allow r2r://** as an Auth redirect URL before release.
      emailRedirectTo: "r2r://auth/callback",
    },
  });

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

  // With Confirm Email enabled, Supabase deliberately returns no session until
  // the recipient follows the verification link. There is no Bearer token yet,
  // so defer the User-table sync until their first verified sign-in.
  if (!data.session) {
    return {
      message:
        "Check your email to verify your account. The verification link will reopen R2R.",
    };
  }

  // This path only applies while email confirmation is disabled. Retaining it
  // keeps local development usable, but production must enable Confirm Email.
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

/**
 * Permanently delete this account and everything in it.
 *
 * Required to exist inside the app by both stores — Apple guideline 5.1.1(v)
 * and Play's account deletion policy — and required to delete the ACCOUNT, not
 * merely its contents.
 *
 * The server signs out afterwards by removing the auth user, so the local
 * session is already dead by the time this returns; signing out locally is
 * still done so the app does not sit on a token it will never be able to
 * refresh.
 */
export async function deleteAccount(): Promise<void> {
  log.info("Account deletion requested");
  // Generous: this clears object storage and cascades every table.
  await apiJson("/auth/account", { method: "DELETE", timeout: 30000 });
  log.info("Account deleted — clearing local session");
  try {
    const supabase = await getSupabase();
    await supabase.auth.signOut();
  } catch (e) {
    // The account is already gone; a failed local sign-out just means a dead
    // token in storage, which the next refresh discards anyway.
    log.warn("Local sign-out after deletion failed", e);
  }
}
