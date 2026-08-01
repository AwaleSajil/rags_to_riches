import React, { useEffect } from "react";
import { Stack, useRouter, useSegments } from "expo-router";
import { Platform } from "react-native";
import { PaperProvider } from "react-native-paper";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { AuthProvider, useAuth } from "../src/providers/AuthProvider";
import { ChatSessionProvider } from "../src/providers/ChatSessionProvider";
import { theme, colors } from "../src/styles/theme";
import { LoadingSpinner } from "../src/components/LoadingSpinner";
import { createLogger } from "../src/lib/logger";

const log = createLogger("Navigation");

// Without this the Stack's first declared child (transaction/[id]) can win the
// initial route, so the app boots into a detail screen with no id and sits on
// its "Loading transaction..." spinner. index.tsx is the real entry point — it
// redirects to login or the tabs once auth resolves.
export const unstable_settings = {
  initialRouteName: "index",
};

function RootLayoutNav() {
  const { user, loading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    log.info("Navigation guard check", {
      loading,
      hasUser: !!user,
      userId: user?.id,
      email: user?.email,
      segments: segments.join("/"),
      platform: Platform.OS,
    });

    if (loading) return;

    // Every screen except login needs a session — including transaction/[id]
    // and receipt-review/[fileId], which sit outside the (tabs) group. Guarding
    // only (tabs) left those reachable while logged out, where they mount, fire
    // requests that 401, and sit on a spinner instead of bouncing to login.
    const onLoginScreen = segments[0] === "login";

    if (!user && !onLoginScreen) {
      log.info("Redirecting to /login (unauthenticated)", { from: segments.join("/") });
      router.replace("/login");
    } else if (user && (onLoginScreen || segments[0] === undefined)) {
      log.info("Redirecting to /(tabs)/chat (authenticated user on entry screen)");
      router.replace("/(tabs)/chat");
    }
  }, [user, loading, segments]);

  if (loading) {
    log.debug("Auth loading - showing spinner");
    return <LoadingSpinner message="Loading..." />;
  }

  // Root stack. The (tabs) group renders its own headers, so only the
  // pushed transaction detail screen shows the native stack header + Back.
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen
        name="transaction/[id]"
        options={{
          headerShown: true,
          title: "Transaction",
          headerStyle: { backgroundColor: colors.surface },
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: "700" },
        }}
      />
      <Stack.Screen
        name="receipt-review/[fileId]"
        options={{
          headerShown: true,
          title: "Review receipt",
          headerStyle: { backgroundColor: colors.surface },
          headerTintColor: colors.text,
          headerTitleStyle: { fontWeight: "700" },
        }}
      />
    </Stack>
  );
}

log.info("App starting", { platform: Platform.OS });

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <AuthProvider>
        <PaperProvider theme={theme}>
          <ChatSessionProvider>
            <RootLayoutNav />
          </ChatSessionProvider>
        </PaperProvider>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
