import React from "react";
import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, IconButton, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, spacing, typography } from "../styles/theme";
import type { Conversation } from "../lib/types";

interface ConversationDrawerProps {
  visible: boolean;
  conversations: Conversation[];
  activeId: string | null;
  isLoading?: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

function fmtDate(s?: string): string {
  if (!s) return "";
  const d = new Date(s);
  if (isNaN(d.getTime())) return "";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`;
}

export function ConversationDrawer({
  visible,
  conversations,
  activeId,
  isLoading,
  onClose,
  onSelect,
  onNew,
  onDelete,
}: ConversationDrawerProps) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.root}>
        <View style={styles.panel}>
          <View style={styles.header}>
            <Text style={styles.title}>Chats</Text>
            <IconButton icon="close" size={22} onPress={onClose} />
          </View>

          <Pressable
            style={styles.newBtn}
            onPress={() => {
              onNew();
              onClose();
            }}
          >
            <MaterialCommunityIcons name="plus" size={20} color={colors.primary} />
            <Text style={styles.newBtnText}>New chat</Text>
          </Pressable>

          {isLoading && <ActivityIndicator style={{ marginTop: 20 }} color={colors.primary} />}

          <ScrollView style={{ flex: 1 }}>
            {conversations.length === 0 && !isLoading && (
              <Text style={styles.empty}>No chats yet.</Text>
            )}
            {conversations.map((c) => {
              const active = c.id === activeId;
              return (
                <Pressable
                  key={c.id}
                  style={[styles.row, active && styles.rowActive]}
                  onPress={() => {
                    onSelect(c.id);
                    onClose();
                  }}
                >
                  <MaterialCommunityIcons
                    name="message-text-outline"
                    size={18}
                    color={active ? colors.primary : colors.textSecondary}
                  />
                  <View style={styles.rowText}>
                    <Text
                      style={[styles.rowTitle, active && { color: colors.primary }]}
                      numberOfLines={1}
                    >
                      {c.title || "New chat"}
                    </Text>
                    <Text style={styles.rowDate}>{fmtDate(c.updated_at)}</Text>
                  </View>
                  <IconButton
                    icon="trash-can-outline"
                    size={16}
                    iconColor={colors.textTertiary}
                    onPress={() => onDelete(c.id)}
                  />
                </Pressable>
              );
            })}
          </ScrollView>
        </View>

        <Pressable style={styles.backdrop} onPress={onClose} />
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    flexDirection: "row",
  },
  panel: {
    width: "82%",
    maxWidth: 340,
    backgroundColor: colors.background,
    paddingTop: 44,
  },
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingLeft: spacing.lg,
    paddingRight: spacing.xs,
  },
  title: {
    ...typography.h3,
    color: colors.text,
  },
  newBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    marginHorizontal: spacing.md,
    marginVertical: spacing.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    backgroundColor: colors.surface,
  },
  newBtnText: {
    ...typography.subtitle2,
    color: colors.primary,
  },
  empty: {
    ...typography.body2,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: 30,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingLeft: spacing.md,
    paddingRight: spacing.xs,
    paddingVertical: 4,
    marginHorizontal: spacing.sm,
    borderRadius: 10,
  },
  rowActive: {
    backgroundColor: colors.primaryLight,
  },
  rowText: {
    flex: 1,
    paddingVertical: spacing.sm,
  },
  rowTitle: {
    ...typography.subtitle2,
    color: colors.text,
  },
  rowDate: {
    ...typography.caption,
    color: colors.textTertiary,
  },
});
