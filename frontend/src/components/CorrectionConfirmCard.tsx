import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import { Button, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, spacing, typography } from "../styles/theme";
import { applyCorrection } from "../services/correctionService";
import type { PendingCorrection } from "../services/correctionService";

interface Props {
  correction: PendingCorrection;
}

function show(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

/**
 * A fix the agent proposed, shown before anything is written.
 *
 * The agent cannot change the database; it fills in this card and the user
 * confirms, which is what turns "I noticed that size is wrong" into a fix
 * instead of a message someone else has to act on.
 *
 * Both values are shown deliberately. A correction presented as only its new
 * value asks the user to confirm a change they cannot see, and the whole reason
 * this exists is that a wrong size was invisible until it was read aloud.
 */
export function CorrectionConfirmCard({ correction }: Props) {
  const [status, setStatus] = useState<"pending" | "saving" | "done" | "dismissed">("pending");
  const [error, setError] = useState<string | null>(null);

  const columns = Object.keys(correction.changes);

  const confirm = async () => {
    setStatus("saving");
    setError(null);
    try {
      await applyCorrection(correction);
      setStatus("done");
    } catch (e: any) {
      setStatus("pending");
      setError(e?.message || "Could not apply that fix");
    }
  };

  if (status === "dismissed") {
    return (
      <View style={styles.card}>
        <View style={styles.header}>
          <MaterialCommunityIcons name="close-circle" size={18} color={colors.textTertiary} />
          <Text style={styles.title}>Left as it was</Text>
        </View>
      </View>
    );
  }

  if (status === "done") {
    return (
      <View style={styles.card}>
        <View style={styles.header}>
          <MaterialCommunityIcons name="check-circle" size={18} color={colors.success} />
          <Text style={styles.title}>Fixed</Text>
        </View>
        {columns.map((c) => (
          <Text key={c} style={styles.row}>
            {correction.labels[c] ?? c}: {show(correction.changes[c])}
          </Text>
        ))}
      </View>
    );
  }

  const busy = status === "saving";

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <MaterialCommunityIcons name="pencil-outline" size={18} color={colors.primary} />
        <Text style={styles.title}>Fix this?</Text>
      </View>

      {columns.map((c) => (
        <Text key={c} style={styles.row}>
          {correction.labels[c] ?? c}:{" "}
          <Text style={styles.was}>{show(correction.current?.[c])}</Text>
          {"  →  "}
          <Text style={styles.now}>{show(correction.changes[c])}</Text>
        </Text>
      ))}

      {!!correction.reason && <Text style={styles.reason}>{correction.reason}</Text>}
      {error && <Text style={styles.error}>{error}</Text>}

      <View style={styles.actions}>
        <Button mode="contained" onPress={confirm} loading={busy} disabled={busy} compact>
          Confirm
        </Button>
        <Button mode="text" onPress={() => setStatus("dismissed")} disabled={busy} compact>
          Leave it
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
  title: { ...typography.body2, fontWeight: "700", color: colors.text },
  row: { ...typography.body2, color: colors.text, marginBottom: 2 },
  was: { color: colors.textTertiary, textDecorationLine: "line-through" },
  now: { color: colors.text, fontWeight: "700" },
  reason: { ...typography.caption, color: colors.textSecondary, marginTop: spacing.xs },
  error: { ...typography.caption, color: colors.error, marginTop: spacing.xs },
  actions: { flexDirection: "row", alignItems: "center", gap: spacing.xs, marginTop: spacing.sm },
});
