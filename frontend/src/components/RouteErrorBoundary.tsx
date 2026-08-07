import React from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import type { ErrorBoundaryProps } from "expo-router";
import { colors, spacing, typography } from "../styles/theme";
import { createLogger } from "../lib/logger";

const log = createLogger("ErrorBoundary");

/**
 * Last line of defence for a render-time throw.
 *
 * Without this, any uncaught error in a screen unmounts the whole tree and
 * leaves a white screen with no way back — the user's only option is to force
 * quit, and we never hear about it.
 *
 * Deliberately built from plain react-native primitives. When a child route
 * throws, this renders in place of the layout's output, so nothing here can
 * assume PaperProvider or SafeAreaProvider is mounted above it.
 */
export function RouteErrorBoundary({ error, retry }: ErrorBoundaryProps) {
  // Only reaches a terminal in development — createLogger is gated on __DEV__.
  // A production crash currently goes unrecorded; wiring a crash reporter
  // (Sentry et al) in here is what closes that gap.
  log.error("Unhandled render error", error);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Something went wrong</Text>
        <Text style={styles.body}>
          The screen couldn't be displayed. Your data is safe — nothing was lost.
        </Text>

        <Pressable
          onPress={() => {
            void retry();
          }}
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          accessibilityRole="button"
          accessibilityLabel="Try loading this screen again"
        >
          <Text style={styles.buttonLabel}>Try again</Text>
        </Pressable>

        {/* The message and stack are useful to a developer and meaningless —
            sometimes alarming — to everyone else. */}
        {__DEV__ && (
          <View style={styles.debug}>
            <Text style={styles.debugTitle}>{error.name}: {error.message}</Text>
            {!!error.stack && <Text style={styles.debugStack}>{error.stack}</Text>}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flexGrow: 1,
    justifyContent: "center",
    padding: spacing.xxl,
  },
  title: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  body: {
    ...typography.body1,
    color: colors.textSecondary,
    marginBottom: spacing.xxl,
  },
  button: {
    alignSelf: "flex-start",
    backgroundColor: colors.primary,
    borderRadius: 10,
    paddingHorizontal: spacing.xxl,
    // 48pt tall, so it stays a comfortable target on both platforms.
    paddingVertical: 14,
  },
  buttonPressed: {
    opacity: 0.8,
  },
  buttonLabel: {
    ...typography.subtitle2,
    color: "#fff",
  },
  debug: {
    marginTop: spacing.xxxl,
    padding: spacing.md,
    borderRadius: 8,
    backgroundColor: colors.surfaceSubtle,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  debugTitle: {
    ...typography.caption,
    color: colors.error,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  debugStack: {
    ...typography.caption,
    color: colors.textSecondary,
    fontFamily: "monospace",
  },
});
