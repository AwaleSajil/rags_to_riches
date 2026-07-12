import React, { useCallback, useState } from "react";
import { StyleSheet, View, ScrollView } from "react-native";
import { Text, Chip, Divider } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useLocalSearchParams, useFocusEffect } from "expo-router";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import * as transactionService from "../../src/services/transactionService";
import { colors, typography, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";
import type { TransactionWithDetails, TransactionDetailItem } from "../../src/lib/types";

const log = createLogger("TransactionDetail");

function money(n: number | null | undefined): string {
  return `$${(n ?? 0).toFixed(2)}`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function formatQty(qty: number | null): string {
  if (qty == null) return "";
  return Number.isInteger(qty) ? String(qty) : String(qty);
}

function LineItem({ item }: { item: TransactionDetailItem }) {
  const qty = item.item_quantity;
  const unit = item.item_unit_price;
  const showQtyLine = qty != null && unit != null;
  const isTaxable = item.taxable === true || (item.tax_rate ?? 0) > 0;

  return (
    <View style={styles.lineItem}>
      <View style={styles.lineItemMain}>
        <Text style={styles.lineItemDesc} numberOfLines={2}>
          {item.item_description || "Item"}
        </Text>
        {showQtyLine && (
          <Text style={styles.lineItemSub}>
            {formatQty(qty)} × {money(unit)}
          </Text>
        )}
      </View>
      <View style={styles.lineItemRight}>
        <Text style={styles.lineItemTotal}>{money(item.item_total_price)}</Text>
        <Chip
          compact
          style={[styles.taxChip, isTaxable ? styles.taxChipOn : styles.taxChipOff]}
          textStyle={styles.taxChipText}
        >
          {isTaxable ? `Tax ${item.tax_rate ?? 0}%` : "Exempt"}
        </Chip>
      </View>
    </View>
  );
}

export default function TransactionDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [transaction, setTransaction] = useState<TransactionWithDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!id) return;
    setIsLoading(true);
    setError(null);
    try {
      const tx = await transactionService.getTransaction(id);
      setTransaction(tx);
    } catch (e: any) {
      log.error("Failed to load transaction", e);
      setError(e.message || "Failed to load transaction");
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  if (isLoading && !transaction) {
    return <LoadingSpinner message="Loading transaction..." />;
  }

  if (error || !transaction) {
    return (
      <View style={styles.centered}>
        <MaterialCommunityIcons name="alert-circle-outline" size={40} color={colors.error} />
        <Text style={styles.errorText}>{error || "Transaction not found"}</Text>
      </View>
    );
  }

  const tx = transaction;
  const hasTax =
    tx.subtotal != null ||
    tx.tax_total != null ||
    (tx.tax_breakdown != null && tx.tax_breakdown.length > 0);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header card */}
        <GlassCard variant="elevated" style={styles.headerCard}>
          <Text style={styles.merchant}>
            {tx.merchant_name || tx.description || "Unknown"}
          </Text>
          <Text style={styles.date}>{formatDate(tx.trans_date)}</Text>
          <Text style={styles.amount}>{money(tx.amount)}</Text>
          <View style={styles.headerMetaRow}>
            {tx.category && (
              <Chip
                compact
                style={styles.categoryChip}
                textStyle={styles.categoryChipText}
                icon="tag-outline"
              >
                {tx.category}
              </Chip>
            )}
            {tx.source && (
              <Chip
                compact
                style={styles.sourceChip}
                textStyle={styles.sourceChipText}
                icon={tx.source === "bill" ? "receipt-text-outline" : "bank-outline"}
              >
                {tx.source === "bill" ? "Receipt" : "CSV"}
              </Chip>
            )}
          </View>
          {tx.location ? (
            <View style={styles.locationRow}>
              <MaterialCommunityIcons
                name="map-marker-outline"
                size={16}
                color={colors.textSecondary}
              />
              <Text style={styles.location}>{tx.location}</Text>
            </View>
          ) : null}
        </GlassCard>

        {/* Tax breakdown card */}
        {hasTax && (
          <GlassCard style={styles.card}>
            <Text style={styles.sectionTitle}>Totals</Text>
            {tx.subtotal != null && (
              <View style={styles.totalsRow}>
                <Text style={styles.totalsLabel}>Subtotal</Text>
                <Text style={styles.totalsValue}>{money(tx.subtotal)}</Text>
              </View>
            )}
            {tx.tax_breakdown?.map((t, i) => (
              <View key={i} style={styles.totalsRow}>
                <Text style={styles.totalsLabel}>
                  {t.label || "Tax"}
                  {t.rate != null ? ` (${t.rate}%)` : ""}
                </Text>
                <Text style={styles.totalsValue}>{money(t.amount)}</Text>
              </View>
            ))}
            {tx.tax_total != null && !tx.tax_breakdown?.length && (
              <View style={styles.totalsRow}>
                <Text style={styles.totalsLabel}>Tax</Text>
                <Text style={styles.totalsValue}>{money(tx.tax_total)}</Text>
              </View>
            )}
            <Divider style={styles.divider} />
            <View style={styles.totalsRow}>
              <Text style={styles.totalsLabelBold}>Total</Text>
              <Text style={styles.totalsValueBold}>{money(tx.amount)}</Text>
            </View>
          </GlassCard>
        )}

        {/* Line items */}
        {tx.details.length > 0 && (
          <GlassCard style={styles.card}>
            <Text style={styles.sectionTitle}>
              Line items ({tx.details.length})
            </Text>
            {tx.details.map((item, i) => (
              <View key={item.id}>
                {i > 0 && <Divider style={styles.itemDivider} />}
                <LineItem item={item} />
              </View>
            ))}
          </GlassCard>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scrollContent: {
    padding: spacing.lg,
    paddingBottom: 40,
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
    padding: spacing.xl,
    gap: spacing.md,
  },
  errorText: {
    ...typography.body1,
    color: colors.textSecondary,
    textAlign: "center",
  },
  headerCard: {
    marginBottom: spacing.lg,
    alignItems: "center",
  },
  merchant: {
    ...typography.h2,
    color: colors.text,
    textAlign: "center",
  },
  date: {
    ...typography.body2,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
  amount: {
    ...typography.h1,
    color: colors.primary,
    marginTop: spacing.md,
  },
  headerMetaRow: {
    flexDirection: "row",
    gap: spacing.sm,
    marginTop: spacing.md,
    flexWrap: "wrap",
    justifyContent: "center",
  },
  categoryChip: {
    backgroundColor: colors.primaryLight,
  },
  categoryChipText: {
    ...typography.caption,
    color: colors.primaryDark,
  },
  sourceChip: {
    backgroundColor: colors.surfaceSubtle,
  },
  sourceChipText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  locationRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    marginTop: spacing.md,
    paddingHorizontal: spacing.md,
  },
  location: {
    ...typography.caption,
    color: colors.textSecondary,
    flexShrink: 1,
    textAlign: "center",
  },
  card: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    ...typography.subtitle1,
    color: colors.text,
    marginBottom: spacing.md,
  },
  totalsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: spacing.xs,
  },
  totalsLabel: {
    ...typography.body2,
    color: colors.textSecondary,
    flexShrink: 1,
    marginRight: spacing.sm,
  },
  totalsValue: {
    ...typography.body2,
    color: colors.text,
  },
  totalsLabelBold: {
    ...typography.subtitle2,
    color: colors.text,
  },
  totalsValueBold: {
    ...typography.subtitle1,
    color: colors.text,
  },
  divider: {
    marginVertical: spacing.sm,
    backgroundColor: colors.divider,
  },
  itemDivider: {
    backgroundColor: colors.divider,
  },
  lineItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.md,
    gap: spacing.md,
  },
  lineItemMain: {
    flex: 1,
  },
  lineItemDesc: {
    ...typography.body2,
    color: colors.text,
  },
  lineItemSub: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: 2,
  },
  lineItemRight: {
    alignItems: "flex-end",
    gap: spacing.xs,
  },
  lineItemTotal: {
    ...typography.subtitle2,
    color: colors.text,
  },
  taxChip: {
    height: 22,
  },
  taxChipOn: {
    backgroundColor: "#fef3c7",
  },
  taxChipOff: {
    backgroundColor: colors.surfaceSubtle,
  },
  taxChipText: {
    ...typography.caption,
    fontSize: 10,
    lineHeight: 12,
    marginVertical: 0,
  },
});
