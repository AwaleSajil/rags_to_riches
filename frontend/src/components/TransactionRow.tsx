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
  // PostgreSQL DATE values have no timezone. Parsing YYYY-MM-DD directly as
  // UTC shifts them to the previous local day in timezones west of Greenwich.
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(dateStr) ? `${dateStr}T00:00:00` : dateStr);
  if (isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function TransactionRowComponent({ transaction, onPress }: TransactionRowProps) {
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
        {transaction.enriched_info ? (
          <View style={styles.enrichedBadge}>
            <Text style={styles.enrichedText}>Enriched</Text>
          </View>
        ) : null}
        {(transaction.linked_transaction_ids?.length ?? 0) > 0 ? (
          <Text style={styles.linkedText}>Linked source{transaction.linked_transaction_ids!.length > 1 ? "s" : ""}</Text>
        ) : null}
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

// The list re-renders on every keystroke in the search box and on every refresh
// state change. Without this, each of those re-renders every mounted row even
// though nothing about the row changed.
export const TransactionRow = React.memo(TransactionRowComponent);

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
  enrichedBadge: {
    marginTop: 3,
  },
  enrichedText: {
    ...typography.caption,
    color: colors.success,
    fontWeight: "700",
  },
  linkedText: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: 3,
  },
  amount: {
    ...typography.subtitle2,
    color: colors.text,
    marginRight: spacing.xs,
  },
});
