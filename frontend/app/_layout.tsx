import React, { useEffect } from "react";
import { Slot, useRouter, useSegments } from "expo-router";
import { Platform } from "react-native";
import { PaperProvider } from "react-native-paper";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { AuthProvider, useAuth } from "../src/providers/AuthProvider";
import { ChatSessionProvider } from "../src/providers/ChatSessionProvider";
import { theme } from "../src/styles/theme";
import { LoadingSpinner } from "../src/components/LoadingSpinner";
import { createLogger } from "../src/lib/logger";

const log = createLogger("Navigation");

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

    const inAuthGroup = segments[0] === "(tabs)";

    if (!user && inAuthGroup) {
      log.info("Redirecting to /login (unauthenticated user in auth group)");
      router.replace("/login");
    } else if (user && !inAuthGroup && (segments[0] === "login" || segments[0] === undefined)) {
      log.info("Redirecting to /(tabs)/chat (authenticated user outside tabs)");
      router.replace("/(tabs)/chat");
    }
  }, [user, loading, segments]);

  if (loading) {
    log.debug("Auth loading - showing spinner");
    return <LoadingSpinner message="Loading..." />;
  }

  return <Slot />;
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
