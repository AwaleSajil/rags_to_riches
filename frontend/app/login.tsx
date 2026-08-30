import React, { useState } from "react";
import {
  StyleSheet,
  View,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Linking,
} from "react-native";
import { Text, TextInput, Button, Snackbar, SegmentedButtons } from "react-native-paper";
import { SafeAreaView } from "react-native-safe-area-context";
import { useAuth } from "../src/providers/AuthProvider";
import { GlassCard } from "../src/components/GlassCard";
import { colors, typography, spacing } from "../src/styles/theme";
import { LEGAL_URLS, API_URL } from "../src/lib/apiUrl";
import { MIN_PASSWORD_LENGTH } from "../src/lib/passwordPolicy";
import { createLogger } from "../src/lib/logger";

const log = createLogger("LoginScreen");

// This catches obvious typos before an unnecessary auth request. Supabase is
// still the authority for account creation and email ownership verification.
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function LoginScreen() {
  log.debug("LoginScreen rendered");
  const { login, register, requestPasswordReset } = useAuth();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });

  const handleSubmit = async () => {
    const normalizedEmail = email.trim().toLowerCase();
    if (!EMAIL_PATTERN.test(normalizedEmail)) {
      setSnackbar({ visible: true, message: "Enter a valid email address.", error: true });
      return;
    }
    if (mode === "register" && password.length < MIN_PASSWORD_LENGTH) {
      setSnackbar({
        visible: true,
        message: `Use a password with at least ${MIN_PASSWORD_LENGTH} characters.`,
        error: true,
      });
      return;
    }

    setIsSubmitting(true);
    try {
      if (mode === "login") {
        log.info("Login button pressed", { email: normalizedEmail });
        await login(normalizedEmail, password);
        log.info("Login successful from UI");
      } else {
        log.info("Register button pressed", { email: normalizedEmail });
        const msg = await register(normalizedEmail, password);
        log.info("Registration successful from UI", { message: msg });
        setSnackbar({ visible: true, message: msg, error: false });
      }
    } catch (e: any) {
      log.error(`${mode} failed from UI`, { error: e.message });
      setSnackbar({ visible: true, message: e.message, error: true });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleForgotPassword = async () => {
    const normalizedEmail = email.trim().toLowerCase();
    if (!EMAIL_PATTERN.test(normalizedEmail)) {
      setSnackbar({
        visible: true,
        message: "Enter your email address first, then tap Forgot password.",
        error: true,
      });
      return;
    }

    setIsSubmitting(true);
    try {
      log.info("Forgot password pressed", { email: normalizedEmail });
      await requestPasswordReset(normalizedEmail);
      // Worded so it says nothing about whether the account exists. Supabase
      // answers identically either way, and a more helpful message here would
      // turn this button into an account-enumeration oracle.
      setSnackbar({
        visible: true,
        message: "If that address has an account, a reset link is on its way.",
        error: false,
      });
    } catch (e: any) {
      log.error("Password reset request failed from UI", { error: e.message });
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
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* Hero Section */}
        <View style={styles.hero}>
          <Text style={styles.title}>R2R</Text>
          <Text style={styles.subtitle}>
            Your AI-powered finance analyst
          </Text>
        </View>

        {/* Auth Card */}
        <GlassCard variant="elevated" style={styles.card}>
          <SegmentedButtons
            value={mode}
            onValueChange={(value) => setMode(value as "login" | "register")}
            buttons={[
              { value: "login", label: "Sign In" },
              { value: "register", label: "Create Account" },
            ]}
            style={styles.segmented}
          />

          <TextInput
            mode="outlined"
            label="Email"
            placeholder="you@example.com"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            style={styles.input}
            outlineStyle={styles.outline}
            dense
          />
          <TextInput
            mode="outlined"
            label="Password"
            placeholder={
              mode === "register"
                ? `At least ${MIN_PASSWORD_LENGTH} characters`
                : "Enter your password"
            }
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            style={styles.input}
            outlineStyle={styles.outline}
            dense
          />
          <Button
            mode="contained"
            onPress={handleSubmit}
            loading={isSubmitting}
            disabled={isSubmitting || !email || !password}
            style={styles.submitButton}
            labelStyle={styles.submitButtonLabel}
          >
            {mode === "login" ? "Sign In" : "Create Account"}
          </Button>

          {mode === "login" && (
            <Button
              mode="text"
              onPress={handleForgotPassword}
              disabled={isSubmitting}
              style={styles.forgotButton}
              labelStyle={styles.forgotLabel}
            >
              Forgot password?
            </Button>
          )}
        </GlassCard>

        {/* Shown at the point of account creation, which is where consent to
            these terms is actually given. Both stores expect the policy to be
            reachable before someone hands over their financial records. */}
        <Text style={styles.legal}>
          By continuing you agree to our{" "}
          <Text style={styles.legalLink} onPress={() => Linking.openURL(LEGAL_URLS.terms)}>
            Terms of Service
          </Text>{" "}
          and{" "}
          <Text style={styles.legalLink} onPress={() => Linking.openURL(LEGAL_URLS.privacy)}>
            Privacy Policy
          </Text>
          .
        </Text>

        <Text style={styles.footer}>Secured by Supabase Auth</Text>

        {/* Only ever rendered by a Metro-served build — `expo export` compiles
            __DEV__ to false, so the deployed bundle cannot show this. The two
            builds are otherwise pixel-identical on a phone, which made it
            impossible to tell whether a signup was hitting local code or the
            stale deployed one, and cost an afternoon of chasing the wrong bug. */}
        {__DEV__ && (
          <Text style={styles.devBadge}>DEV BUILD · {API_URL}</Text>
        )}
      </ScrollView>

      <Snackbar
        visible={snackbar.visible}
        onDismiss={() => setSnackbar({ ...snackbar, visible: false })}
        duration={4000}
        style={{
          backgroundColor: snackbar.error ? colors.error : colors.success,
        }}
      >
        {snackbar.message}
      </Snackbar>
    </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  flex: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing.xl,
    paddingBottom: 40,
    flexGrow: 1,
    justifyContent: "center",
  },
  hero: {
    alignItems: "center",
    paddingBottom: spacing.xxxl,
  },
  title: {
    ...typography.h1,
    color: colors.primary,
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body2,
    color: colors.textSecondary,
    textAlign: "center",
  },
  card: {
    marginBottom: spacing.xxl,
  },
  segmented: {
    marginBottom: spacing.xl,
  },
  input: {
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
  },
  outline: {
    borderRadius: 10,
    borderColor: colors.border,
  },
  submitButton: {
    borderRadius: 10,
    backgroundColor: colors.primary,
    marginTop: spacing.sm,
  },
  submitButtonLabel: {
    fontWeight: "600",
    paddingVertical: spacing.xs,
  },
  forgotButton: {
    marginTop: spacing.xs,
  },
  forgotLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  devBadge: {
    marginTop: spacing.sm,
    textAlign: "center",
    fontSize: 11,
    fontWeight: "700",
    color: colors.error,
  },
  footer: {
    ...typography.caption,
    color: colors.textTertiary,
    textAlign: "center",
  },
  legal: {
    ...typography.caption,
    color: colors.textTertiary,
    textAlign: "center",
    marginTop: spacing.lg,
    marginBottom: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  legalLink: {
    color: colors.primary,
    textDecorationLine: "underline",
  },
});
