import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import type { TransactionListItem } from "../lib/types";

interface TransactionRowProps {
  transaction: TransactionListItem;
  onPress: (tx: TransactionListItem) => void;
}

function formatDay(dateStr: string | null): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function TransactionRow({ transaction, onPress }: TransactionRowProps) {
  const isBill = transaction.source === "bill";
  const iconName = isBill ? "receipt-text-outline" : "bank-outline";
  const amount = transaction.amount ?? 0;

  return (
    <Pressable
      style={({ pressed }) => [styles.container, pressed && styles.pressed]}
      onPress={() => onPress(transaction)}
    >
      <View style={styles.iconCircle}>
        <MaterialCommunityIcons name={iconName} size={20} color={colors.primary} />
      </View>
      <View style={styles.textContainer}>
        <Text style={styles.merchant} numberOfLines={1}>
          {transaction.merchant_name || transaction.description || "Unknown"}
        </Text>
        <Text style={styles.meta} numberOfLines={1}>
          {formatDay(transaction.trans_date)}
          {transaction.category ? ` · ${transaction.category}` : ""}
        </Text>
      </View>
      <Text style={styles.amount}>${amount.toFixed(2)}</Text>
      <MaterialCommunityIcons
        name="chevron-right"
        size={20}
        color={colors.textTertiary}
      />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    borderRadius: 12,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginBottom: spacing.sm,
  },
  pressed: {
    opacity: 0.6,
  },
  iconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.primaryLight,
    marginRight: spacing.md,
  },
  textContainer: {
    flex: 1,
  },
  merchant: {
    ...typography.subtitle2,
    color: colors.text,
  },
  meta: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  amount: {
    ...typography.subtitle2,
    color: colors.text,
    marginRight: spacing.xs,
  },
});
