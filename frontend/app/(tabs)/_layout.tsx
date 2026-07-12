import React from "react";
import { Tabs } from "expo-router";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { View } from "react-native";
import { Text, IconButton } from "react-native-paper";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useAuth } from "../../src/providers/AuthProvider";
import { useChatSession } from "../../src/providers/ChatSessionProvider";
import { colors } from "../../src/styles/theme";

// Sessions drawer (☰) + new chat (＋), shown in the Chat tab's native header.
function ChatHeaderLeft() {
  const { handlers } = useChatSession();
  return (
    <View style={{ flexDirection: "row", marginLeft: 4 }}>
      <IconButton
        icon="menu"
        size={22}
        iconColor={colors.text}
        onPress={() => handlers.onMenu?.()}
        style={{ margin: 0 }}
      />
      <IconButton
        icon="plus"
        size={22}
        iconColor={colors.text}
        onPress={() => handlers.onNewChat?.()}
        style={{ margin: 0 }}
      />
    </View>
  );
}

export default function TabLayout() {
  const { user, logout } = useAuth();

  const displayName = user?.email?.split("@")[0] || "";
  const insets = useSafeAreaInsets();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textSecondary,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.divider,
          borderTopWidth: 1,
          height: 60 + insets.bottom,
          paddingBottom: insets.bottom + 4,
          paddingTop: 4,
          elevation: 8,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: "600",
        },
        headerStyle: {
          backgroundColor: colors.surface,
          borderBottomWidth: 1,
          borderBottomColor: colors.surfaceBorder,
          elevation: 0,
        },
        headerTintColor: colors.text,
        headerTitleStyle: { fontWeight: "700" },
        headerRight: () => (
          <View style={{ marginRight: 8, flexDirection: "row", alignItems: "center" }}>
            <Text style={{ color: colors.textSecondary, fontSize: 13, marginRight: 4 }}>
              {displayName}
            </Text>
            <IconButton
              icon="logout"
              size={20}
              iconColor={colors.textSecondary}
              onPress={logout}
              style={{ margin: 0 }}
            />
          </View>
        ),
      }}
    >
      <Tabs.Screen
        name="chat"
        options={{
          title: "Chat",
          headerTitle: "R2R",
          headerLeft: () => <ChatHeaderLeft />,
          tabBarIcon: ({ color, size, focused }) => (
            <MaterialCommunityIcons
              name={focused ? "message-text" : "message-text-outline"}
              size={size}
              color={color}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="ingest"
        options={{
          title: "Files",
          headerTitle: "Your Files",
          tabBarIcon: ({ color, size, focused }) => (
            <MaterialCommunityIcons
              name={focused ? "folder" : "folder-outline"}
              size={size}
              color={color}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="config"
        options={{
          title: "Settings",
          headerTitle: "Settings",
          tabBarIcon: ({ color, size, focused }) => (
            <MaterialCommunityIcons
              name={focused ? "cog" : "cog-outline"}
              size={size}
              color={color}
            />
          ),
        }}
      />
    </Tabs>
  );
}
