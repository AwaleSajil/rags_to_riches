import React, { useEffect } from "react";
import { StyleSheet, View } from "react-native";
import { Text, Button } from "react-native-paper";
import { Redirect, useRouter } from "expo-router";
import { useAuth } from "../../src/providers/AuthProvider";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { colors, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";

const log = createLogger("AuthCallback");

// Long enough to read, short enough not to feel like a stall. The Continue
// button below means nobody has to wait for it either way.
const AUTO_CONTINUE_MS = 2500;

/**
 * Where a verification link lands.
 *
 * AuthProvider does the real work — it reads the tokens out of the URL fragment
 * and calls setSession. This screen exists for two reasons.
 *
 * First, expo-router needs a route for /auth/callback to resolve to at all:
 * without one the link arrives at an "Unmatched Route" screen, and the guard in
 * _layout.tsx does not rescue it, because its redirect branches only fire on the
 * login screen and on the bare entry route. A verified user would sit on a 404
 * while holding a perfectly good session.
 *
 * Second, verification is otherwise completely silent. Someone arriving from
 * their email client gets dropped into the chat tab with no acknowledgement that
 * the thing they just did worked, which reads as a failure. So this says so.
 */
export default function AuthCallback() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading || !user) return;
    log.info("Verification complete — entering the app", { userId: user.id });
    const timer = setTimeout(() => router.replace("/(tabs)/chat"), AUTO_CONTINUE_MS);
    return () => clearTimeout(timer);
  }, [user, loading]);

  if (loading) {
    return <LoadingSpinner message="Verifying your email..." />;
  }

  // The fragment carried no usable tokens, or setSession rejected them — an
  // expired or already-spent link. Login is where that is recoverable, and it
  // is also where a wrong guess about what happened does the least damage.
  if (!user) {
    log.warn("Callback resolved without a session — sending to login");
    return <Redirect href="/login" />;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.check}>✓</Text>
      <Text style={styles.title}>Email verified</Text>
      <Text style={styles.subtitle}>You are signed in as {user.email}</Text>
      <Button
        mode="contained"
        onPress={() => router.replace("/(tabs)/chat")}
        style={styles.button}
      >
        Continue to R2R
      </Button>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.xl,
    backgroundColor: colors.background,
  },
  check: {
    fontSize: 56,
    lineHeight: 64,
    color: colors.success,
  },
  title: {
    marginTop: spacing.md,
    fontSize: 22,
    fontWeight: "700",
    color: colors.text,
  },
  subtitle: {
    marginTop: spacing.sm,
    fontSize: 14,
    color: colors.textSecondary,
    textAlign: "center",
  },
  button: {
    marginTop: spacing.xxl,
  },
});
