import * as Linking from "expo-linking";
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
      // Resolves per platform rather than being hardcoded: r2r://auth/callback
      // in a standalone build (the scheme is declared in app.config.js), and the
      // dev-server or deployed origin on web, where a r2r:// link is something
      // the browser cannot open at all. Every form this returns has to be on
      // Supabase's Auth redirect allow-list, or the link silently falls back to
      // site_url and the app never sees the tokens.
      emailRedirectTo: Linking.createURL("auth/callback"),
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

  // Supabase deliberately answers a signup for an existing address with a
  // success response and no email, so that nobody can probe this endpoint to
  // discover which addresses have accounts. The tell is an empty identities
  // array, which never happens on a genuine new signup. Without this branch the
  // caller is told to check an inbox that will never receive anything, and the
  // only way to find out otherwise is to read the auth logs.
  if (data.user.identities?.length === 0) {
    log.info("Signup for an address that already has an account", {
      email: normalizedEmail,
    });
    return {
      message: "An account already exists for this email. Try signing in instead.",
    };
  }

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

/**
 * Send a password-reset link to an address.
 *
 * Deliberately does not report whether the address has an account. Supabase
 * answers the same way either way, and surfacing the difference here would
 * hand back exactly the account-enumeration oracle that the signup path goes
 * out of its way to avoid.
 *
 * The link lands on the same /auth/callback route as email verification and
 * carries the same token fragment — the only difference is type=recovery,
 * which AuthProvider uses to route to the set-a-new-password screen instead
 * of dropping straight into the app.
 */
export async function requestPasswordReset(email: string): Promise<void> {
  const normalizedEmail = email.trim().toLowerCase();
  log.info("Password reset requested", { email: normalizedEmail });

  const supabase = await getSupabase();
  const { error } = await supabase.auth.resetPasswordForEmail(normalizedEmail, {
    redirectTo: Linking.createURL("auth/callback"),
  });

  if (error) {
    log.error("Password reset request failed", {
      email: normalizedEmail,
      error: error.message,
    });
    throw error;
  }
  log.info("Password reset email dispatched", { email: normalizedEmail });
}

/**
 * Set a new password using the session a recovery link created.
 *
 * Supabase revokes the other sessions on a password change, so anyone who had
 * hold of the account is signed out — which is the point of resetting it.
 */
export async function updatePassword(newPassword: string): Promise<void> {
  log.info("Password update requested");

  const supabase = await getSupabase();
  const { error } = await supabase.auth.updateUser({ password: newPassword });

  if (error) {
    log.error("Password update failed", { error: error.message });
    throw error;
  }
  log.info("Password updated");
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
