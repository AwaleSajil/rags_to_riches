import React, { createContext, useContext, useEffect, useState } from "react";
import { getSupabase } from "../lib/supabase";
import * as authService from "../services/authService";
import { createLogger } from "../lib/logger";
import type { User } from "../lib/types";

const log = createLogger("AuthProvider");

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<string>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  login: async () => {},
  register: async () => "",
  logout: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    log.info("Checking for existing session on mount...");

    let subscription: { unsubscribe: () => void } | null = null;
    let cancelled = false;

    async function bootstrap() {
      const supabase = await getSupabase();

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

  return (
    <AuthContext.Provider
      value={{ user, loading, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
