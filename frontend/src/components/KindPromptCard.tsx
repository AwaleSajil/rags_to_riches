import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import { Text, Button } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import { setCaptureKind } from "../services/captureService";
import type { CaptureResult } from "../services/captureService";
import { createLogger } from "../lib/logger";

const log = createLogger("KindPromptCard");

interface Props {
  fileId: string;
  onResolved: (result: CaptureResult) => void;
}

/**
 * Shown when the vision model could not tell a receipt from a price tag.
 *
 * This is a deliberate one-tap question rather than a confident guess. The two
 * mistakes are not symmetrical: a price tag filed as a receipt invents spending
 * that never happened and corrupts every total, so asking is cheap insurance.
 */
export function KindPromptCard({ fileId, onResolved }: Props) {
  const [busy, setBusy] = useState<"receipt" | "price_tag" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const choose = async (kind: "receipt" | "price_tag") => {
    setBusy(kind);
    setError(null);
    try {
      const result = await setCaptureKind(fileId, kind);
      log.info("Capture kind resolved by user", { fileId, kind });
      onResolved(result);
    } catch (e: any) {
      log.error("Could not set capture kind", e);
      setError(e.message || "Could not save that choice");
      setBusy(null);
    }
  };

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <MaterialCommunityIcons name="help-circle-outline" size={20} color={colors.primary} />
        <Text style={styles.title}>Which is this?</Text>
      </View>
      <Text style={styles.body}>
        I couldn't tell whether that's a receipt or a shelf price. Rather than
        guess, which one is it?
      </Text>
      {error && <Text style={styles.error}>{error}</Text>}
      <View style={styles.actions}>
        <Button
          mode="contained"
          icon="tag-outline"
          onPress={() => choose("price_tag")}
          loading={busy === "price_tag"}
          disabled={busy !== null}
          style={styles.button}
        >
          Price tag
        </Button>
        <Button
          mode="outlined"
          icon="receipt"
          onPress={() => choose("receipt")}
          loading={busy === "receipt"}
          disabled={busy !== null}
          style={styles.button}
        >
          Receipt
        </Button>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surfaceSubtle,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    padding: spacing.md,
    marginTop: spacing.sm,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    marginBottom: spacing.xs,
  },
  title: {
    ...typography.body2,
    fontWeight: "700",
    color: colors.text,
  },
  body: {
    ...typography.body2,
    color: colors.textSecondary,
    marginBottom: spacing.md,
  },
  error: {
    ...typography.caption,
    color: colors.error,
    marginBottom: spacing.sm,
  },
  actions: {
    flexDirection: "row",
    gap: spacing.sm,
  },
  button: {
    flex: 1,
  },
});
