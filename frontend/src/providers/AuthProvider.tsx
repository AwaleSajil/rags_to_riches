import React, { createContext, useContext, useEffect, useState } from "react";
import * as Linking from "expo-linking";
import { getSupabase } from "../lib/supabase";
import * as authService from "../services/authService";
import { createLogger } from "../lib/logger";
import type { User } from "../lib/types";

const log = createLogger("AuthProvider");

interface AuthContextType {
  user: User | null;
  loading: boolean;
  // True once a recovery link has established a session but the password has
  // not been changed yet. The callback screen reads it to route to the
  // set-a-new-password screen rather than into the app.
  recoveryPending: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<string>;
  logout: () => Promise<void>;
  deleteAccount: () => Promise<void>;
  requestPasswordReset: (email: string) => Promise<void>;
  updatePassword: (newPassword: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  recoveryPending: false,
  login: async () => {},
  register: async () => "",
  logout: async () => {},
  deleteAccount: async () => {},
  requestPasswordReset: async () => {},
  updatePassword: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [recoveryPending, setRecoveryPending] = useState(false);

  useEffect(() => {
    log.info("Checking for existing session on mount...");

    let subscription: { unsubscribe: () => void } | null = null;
    let linkingSubscription: { remove: () => void } | null = null;
    let cancelled = false;

    async function bootstrap() {
      const supabase = await getSupabase();

      // Supabase sends access and refresh tokens back in the fragment of the
      // auth/callback URL after the recipient follows an emailed link. React
      // Native does not automatically consume that URL, so exchange it for the
      // normal persisted session ourselves.
      //
      // Signup confirmation and password recovery arrive here identically
      // apart from the type parameter. Ignoring it would be a real hole: a
      // recovery link would silently sign someone in with the old password
      // still valid, having "reset" nothing, which is worse than failing.
      const completeAuthCallback = async (url: string) => {
        const fragment = url.split("#", 2)[1] ?? "";
        const params = new URLSearchParams(fragment);
        const redirectError = params.get("error_description") ?? params.get("error");
        if (redirectError) {
          log.error("Email verification redirect failed", { redirectError });
          return;
        }

        const accessToken = params.get("access_token");
        const refreshToken = params.get("refresh_token");
        if (!accessToken || !refreshToken) return;

        const { error } = await supabase.auth.setSession({
          access_token: accessToken,
          refresh_token: refreshToken,
        });
        if (error) {
          log.error("Could not create session from emailed link", {
            error: error.message,
          });
          return;
        }

        // Only after a session actually exists — the reset screen has nothing
        // to update the password with otherwise.
        if (params.get("type") === "recovery") {
          log.info("Recovery link consumed — a new password is required");
          setRecoveryPending(true);
        }
      };

      // Register the listener BEFORE the network call. If the refresh below
      // fails we still want sign-in events to reach the app — previously the
      // subscription was set up after it, so a failed refresh left the app with
      // no way to ever learn about a login.
      const { data } = supabase.auth.onAuthStateChange((event, session) => {
        log.info("Auth state changed", {
          event,
          hasSession: !!session,
          userId: session?.user?.id,
          email: session?.user?.email,
        });

        if (session?.user) {
          setUser({ id: session.user.id, email: session.user.email! });
        } else {
          log.info("Session cleared - user logged out");
          setUser(null);
        }
      });
      subscription = data.subscription;
      if (cancelled) {
        subscription.unsubscribe();
        return;
      }

      linkingSubscription = Linking.addEventListener("url", ({ url }) => {
        void completeAuthCallback(url);
      });

      // Handles an emailed link that launches R2R from a fully-closed state,
      // before the event listener has been attached.
      const initialUrl = await Linking.getInitialURL();
      if (initialUrl) await completeAuthCallback(initialUrl);

      // Verify session on mount by refreshing with the server.
      // getSession() only returns the local cache which may be stale
      // (e.g. session revoked server-side). refreshSession() validates
      // and returns a fresh token, or null if the session is truly dead.
      try {
        const { data: { session }, error } = await supabase.auth.refreshSession();
        if (error) {
          log.warn("Session refresh failed on mount", { error: error.message });
        }
        if (session?.user) {
          log.info("Valid session found", {
            userId: session.user.id,
            email: session.user.email,
            tokenExpiry: session.expires_at,
          });
          setUser({ id: session.user.id, email: session.user.email! });
        } else {
          log.info("No valid session found");
        }
      } catch (e: any) {
        // A rejected refresh (device offline, DNS failure, captive portal)
        // must not strand the app: fall through to the login screen, which the
        // user can retry from, rather than an indefinite spinner.
        log.warn("Session refresh threw on mount — continuing unauthenticated", {
          error: e?.message ?? String(e),
        });
      }
    }

    bootstrap()
      .catch((e: any) => {
        // getSupabase() itself failed (bad/missing config). Nothing works
        // without a client, but showing login beats hanging on a spinner.
        log.error("Auth bootstrap failed", { error: e?.message ?? String(e) });
      })
      .finally(() => {
        // The single place loading is cleared, so every path above reaches it.
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      log.debug("Unsubscribing auth state listener");
      subscription?.unsubscribe();
      linkingSubscription?.remove();
    };
  }, []);

  const login = async (email: string, password: string) => {
    log.info("Login flow started", { email });
    const result = await authService.login(email, password);
    log.info("Login flow complete - setting user state", { userId: result.user.id });
    setUser(result.user);
  };

  const register = async (email: string, password: string) => {
    log.info("Register flow started", { email });
    const result = await authService.register(email, password);
    log.info("Register flow complete", { message: result.message });
    return result.message;
  };

  const logout = async () => {
    log.info("Logout flow started");
    await authService.logout();
    setUser(null);
    log.info("Logout flow complete - state cleared");
  };

  const requestPasswordReset = async (email: string) => {
    log.info("Password reset flow started", { email });
    await authService.requestPasswordReset(email);
  };

  const updatePassword = async (newPassword: string) => {
    log.info("Password update flow started");
    await authService.updatePassword(newPassword);
    // Cleared only on success, so a failed attempt leaves the user on the reset
    // screen with their recovery session intact rather than bouncing them into
    // the app with the old password still live.
    setRecoveryPending(false);
    log.info("Password update flow complete");
  };

  const deleteAccount = async () => {
    log.info("Account deletion flow started");
    await authService.deleteAccount();
    // Cleared here rather than left to onAuthStateChange: the auth user no
    // longer exists, so there may be no sign-out event to listen for.
    setUser(null);
    log.info("Account deletion flow complete - state cleared");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        recoveryPending,
        login,
        register,
        logout,
        deleteAccount,
        requestPasswordReset,
        updatePassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
