import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import type { TransactionListItem } from "../lib/types";
import { formatDate, money } from "../lib/format";
import { transactionVisual } from "../lib/sourceVisual";

interface TransactionRowProps {
  transaction: TransactionListItem;
  onPress: (tx: TransactionListItem) => void;
}

function TransactionRowComponent({ transaction, onPress }: TransactionRowProps) {
  const isBill = transaction.source === "bill";
  const visual = transactionVisual(transaction.source);
  const amount = transaction.amount ?? 0;
  const isLinked = (transaction.linked_transaction_ids?.length ?? 0) > 0;

  return (
    <Pressable
      style={({ pressed }) => [styles.container, pressed && styles.pressed]}
      onPress={() => onPress(transaction)}
    >
      {/* Where this transaction came from. The glyph alone used to carry it —
          two outline icons the same size and colour in the same circle — which
          is not a difference you can see while scrolling. Colour does the work
          now, and the badge below says it in words. Shared with the Files tab
          so one receipt looks the same in both. */}
      <View style={[styles.iconCircle, { backgroundColor: visual.background }]}>
        <MaterialCommunityIcons name={visual.icon} size={20} color={visual.color} />
      </View>
      <View style={styles.textContainer}>
        <Text style={styles.merchant} numberOfLines={1}>
          {transaction.merchant_name || transaction.description || "Unknown"}
        </Text>
        <Text style={styles.meta} numberOfLines={1}>
          {formatDate(transaction.trans_date)}
          {transaction.category ? ` · ${transaction.category}` : ""}
        </Text>
        {/* One horizontal strip rather than a stacked line each, so adding the
            source badge does not make every row taller. */}
        {(isBill || transaction.enriched_info || isLinked) ? (
          <View style={styles.badgeRow}>
            {isBill ? (
              <View style={styles.receiptBadge}>
                <MaterialCommunityIcons name="camera-outline" size={11} color={visual.color} />
                <Text style={styles.receiptBadgeText}>{visual.label}</Text>
              </View>
            ) : null}
            {transaction.enriched_info ? (
              <Text style={styles.enrichedText}>Enriched</Text>
            ) : null}
            {isLinked ? (
              // "Linked source" said nothing a user could act on. This says
              // which of the two they are looking at and that the other still
              // exists — the detail screen explains the rest.
              <View style={styles.linkedBadge}>
                <MaterialCommunityIcons
                  name="link-variant"
                  size={11}
                  color={colors.textSecondary}
                />
                <Text style={styles.linkedText}>
                  {isBill ? "Also on your statement" : "Receipt on file"}
                </Text>
              </View>
            ) : null}
          </View>
        ) : null}
      </View>
      <Text style={styles.amount}>{money(amount)}</Text>
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
  badgeRow: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: spacing.xs,
    marginTop: 3,
  },
  receiptBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 6,
    backgroundColor: colors.primaryFaded,
  },
  receiptBadgeText: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: "700",
  },
  enrichedText: {
    ...typography.caption,
    color: colors.success,
    fontWeight: "700",
  },
  linkedBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
  },
  linkedText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  amount: {
    ...typography.subtitle2,
    color: colors.text,
    marginRight: spacing.xs,
  },
});
