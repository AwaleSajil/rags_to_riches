import React, { useState } from "react";
import { StyleSheet, View, KeyboardAvoidingView, Platform } from "react-native";
import { Text, TextInput, Button, Snackbar } from "react-native-paper";
import { SafeAreaView } from "react-native-safe-area-context";
import { Redirect, useRouter } from "expo-router";
import { useAuth } from "../../src/providers/AuthProvider";
import { GlassCard } from "../../src/components/GlassCard";
import { MIN_PASSWORD_LENGTH } from "../../src/lib/passwordPolicy";
import { colors, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";

const log = createLogger("ResetPassword");

/**
 * Set a new password after following a recovery link.
 *
 * Only reachable with a live recovery session — AuthProvider sets
 * recoveryPending when the callback fragment carries type=recovery, and the
 * callback screen routes here on the strength of it. Someone landing here any
 * other way has nothing to update, so they go to login.
 *
 * The confirm field is not ceremony: this is the one screen in the app where a
 * typo locks you out of the account you are in the middle of recovering, and
 * the field is masked so there is nothing to re-read.
 */
export default function ResetPassword() {
  const { user, loading, recoveryPending, updatePassword } = useAuth();
  const router = useRouter();

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });

  if (!loading && (!user || !recoveryPending)) {
    log.info("No recovery session — redirecting to login");
    return <Redirect href="/login" />;
  }

  const handleSubmit = async () => {
    if (password.length < MIN_PASSWORD_LENGTH) {
      setSnackbar({
        visible: true,
        message: `Use a password with at least ${MIN_PASSWORD_LENGTH} characters.`,
        error: true,
      });
      return;
    }
    if (password !== confirm) {
      setSnackbar({ visible: true, message: "The two passwords do not match.", error: true });
      return;
    }

    setIsSubmitting(true);
    try {
      await updatePassword(password);
      log.info("Password reset complete — entering the app");
      router.replace("/(tabs)/chat");
    } catch (e: any) {
      log.error("Password reset failed", { error: e.message });
      setSnackbar({ visible: true, message: e.message, error: true });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <View style={styles.content}>
          <Text style={styles.title}>Choose a new password</Text>
          <Text style={styles.subtitle}>
            {user?.email ? `For ${user.email}` : "Your link has been verified."}
          </Text>

          <GlassCard variant="elevated" style={styles.card}>
            <TextInput
              mode="outlined"
              label="New password"
              placeholder={`At least ${MIN_PASSWORD_LENGTH} characters`}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              style={styles.input}
              dense
            />
            <TextInput
              mode="outlined"
              label="Confirm new password"
              value={confirm}
              onChangeText={setConfirm}
              secureTextEntry
              style={styles.input}
              dense
            />
            <Button
              mode="contained"
              onPress={handleSubmit}
              loading={isSubmitting}
              disabled={isSubmitting || !password || !confirm}
              style={styles.submitButton}
            >
              Set new password
            </Button>
          </GlassCard>

          {/* Saying so up front beats a silent sign-out on their other device. */}
          <Text style={styles.note}>
            Changing your password signs you out everywhere else.
          </Text>
        </View>
      </KeyboardAvoidingView>

      <Snackbar
        visible={snackbar.visible}
        onDismiss={() => setSnackbar({ ...snackbar, visible: false })}
        duration={4000}
        style={{ backgroundColor: snackbar.error ? colors.error : colors.success }}
      >
        {snackbar.message}
      </Snackbar>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  content: { flex: 1, justifyContent: "center", padding: spacing.xl },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: colors.text,
    textAlign: "center",
  },
  subtitle: {
    marginTop: spacing.sm,
    fontSize: 14,
    color: colors.textSecondary,
    textAlign: "center",
  },
  card: { marginTop: spacing.xxl, padding: spacing.lg },
  input: { marginBottom: spacing.md },
  submitButton: { marginTop: spacing.sm },
  note: {
    marginTop: spacing.lg,
    fontSize: 12,
    color: colors.textTertiary,
    textAlign: "center",
  },
});
