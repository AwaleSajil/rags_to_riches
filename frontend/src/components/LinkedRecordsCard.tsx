/**
 * "This purchase is also recorded somewhere else."
 *
 * A purchase can arrive twice: once as a line on a bank statement, once as a
 * photographed receipt. They are matched on merchant, date and amount and then
 * LINKED — never merged, because neither is safe to discard. The statement is
 * what the bank says; the receipt is what was actually bought.
 *
 * The transactions list already collapses a linked pair into one row, showing
 * the receipt because it carries the line items. That happened silently, so a
 * user who imported a statement and then photographed the receipt saw one entry
 * where they knew there were two, with no way to tell whether the other had
 * been absorbed, replaced, or lost. This says it, and lets them go and look.
 */

import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import { colors, spacing, typography } from "../styles/theme";
import { formatDate, money } from "../lib/format";
import { transactionVisual } from "../lib/sourceVisual";
import type { LinkedTransaction } from "../lib/types";

interface LinkedRecordsCardProps {
  /** The record being viewed — decides which side of the pair to describe. */
  viewingSource: string | null;
  viewingDetailCount: number;
  linked: LinkedTransaction[];
  onOpen: (id: string) => void;
}

/**
 * One line saying why both exist and which is worth reading.
 *
 * Deliberately concrete about the difference rather than just "linked": the
 * useful fact is that one side itemises the shopping and the other does not.
 */
function summarise(
  viewingSource: string | null,
  viewingDetailCount: number,
  other: LinkedTransaction
): string {
  const viewingIsReceipt = viewingSource === "bill";
  const otherIsReceipt = other.source === "bill";

  if (viewingIsReceipt && !otherIsReceipt) {
    return viewingDetailCount > 0
      ? `Your bank statement has this purchase too. This receipt is the fuller record — it lists all ${viewingDetailCount} items.`
      : "Your bank statement has this purchase too. Both are kept; neither was replaced.";
  }
  if (!viewingIsReceipt && otherIsReceipt) {
    return other.detail_count > 0
      ? `You photographed the receipt for this. That copy lists all ${other.detail_count} items — it is the fuller record.`
      : "You photographed the receipt for this purchase as well.";
  }
  // csv_csv: the same purchase appearing in two statement exports.
  return "This purchase appears in another statement you imported. Both are kept, and it is only counted once.";
}

function LinkedRecordsCardComponent({
  viewingSource,
  viewingDetailCount,
  linked,
  onOpen,
}: LinkedRecordsCardProps) {
  if (!linked.length) return null;

  return (
    <View>
      <View style={styles.headerRow}>
        <MaterialCommunityIcons name="link-variant" size={16} color={colors.primary} />
        <Text style={styles.heading}>
          Also recorded {linked.length > 1 ? `in ${linked.length} places` : "elsewhere"}
        </Text>
      </View>

      <Text style={styles.explanation}>
        {summarise(viewingSource, viewingDetailCount, linked[0])}
      </Text>

      {linked.map((other) => {
        const visual = transactionVisual(other.source);
        return (
          <Pressable
            key={other.id}
            onPress={() => onOpen(other.id)}
            style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
          >
            <View style={[styles.icon, { backgroundColor: visual.background }]}>
              <MaterialCommunityIcons name={visual.icon} size={18} color={visual.color} />
            </View>
            <View style={styles.rowText}>
              <Text style={styles.rowTitle} numberOfLines={1}>
                {other.merchant_name || "Unknown"}
              </Text>
              <Text style={styles.rowMeta} numberOfLines={1}>
                {[
                  visual.label,
                  formatDate(other.trans_date),
                  other.detail_count > 0
                    ? `${other.detail_count} item${other.detail_count === 1 ? "" : "s"}`
                    : "no items",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </Text>
            </View>
            <Text style={styles.rowAmount}>{money(other.amount)}</Text>
            <MaterialCommunityIcons
              name="chevron-right"
              size={20}
              color={colors.textTertiary}
            />
          </Pressable>
        );
      })}

      {/* The thing a user worries about the moment they learn there are two. */}
      <Text style={styles.footnote}>
        Counted once in your totals, not twice.
      </Text>
    </View>
  );
}

export const LinkedRecordsCard = React.memo(LinkedRecordsCardComponent);

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
  },
  heading: {
    ...typography.subtitle1,
    color: colors.text,
  },
  explanation: {
    ...typography.body2,
    color: colors.textSecondary,
    marginTop: 2,
    marginBottom: spacing.sm,
    lineHeight: 19,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingVertical: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  rowPressed: {
    opacity: 0.6,
  },
  icon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  rowText: {
    flex: 1,
    minWidth: 0,
  },
  rowTitle: {
    ...typography.body2,
    color: colors.text,
    fontWeight: "600",
  },
  rowMeta: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  rowAmount: {
    ...typography.body2,
    color: colors.text,
  },
  footnote: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.xs,
  },
});
