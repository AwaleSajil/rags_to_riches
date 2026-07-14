import React, { useCallback, useMemo, useState } from "react";
import { StyleSheet, View, ScrollView, Pressable, RefreshControl } from "react-native";
import { Text, Searchbar, Chip, Badge } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useRouter, useFocusEffect } from "expo-router";
import { TransactionRow } from "../../src/components/TransactionRow";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { useTransactions } from "../../src/hooks/useTransactions";
import { colors, typography, spacing } from "../../src/styles/theme";
import type { TransactionListItem } from "../../src/lib/types";

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function monthLabel(dateStr: string | null): string {
  if (!dateStr) return "Unknown date";
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(dateStr) ? `${dateStr}T00:00:00` : dateStr);
  if (isNaN(d.getTime())) return "Unknown date";
  return `${MONTH_NAMES[d.getMonth()]} ${d.getFullYear()}`;
}

export default function TransactionsScreen() {
  const { transactions, isLoading, refresh } = useTransactions();
  const router = useRouter();

  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string>("all");
  const [collapsedMonths, setCollapsedMonths] = useState<Record<string, boolean>>({});

  // Refresh whenever the tab regains focus (e.g. returning after an edit/delete).
  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh])
  );

  // Distinct categories present in the data → filter chips.
  const categories = useMemo(() => {
    const set = new Set<string>();
    for (const t of transactions) {
      if (t.category) set.add(t.category);
    }
    return Array.from(set).sort();
  }, [transactions]);

  // Filter (category + text) → sort newest-first → group into month buckets.
  const groupedTransactions = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = transactions
      .filter((t) => category === "all" || t.category === category)
      .filter(
        (t) =>
          !q ||
          (t.merchant_name || "").toLowerCase().includes(q) ||
          (t.description || "").toLowerCase().includes(q)
      )
      .sort((a, b) => (b.trans_date || "").localeCompare(a.trans_date || ""));

    // A link means two source rows describe one real-world purchase. Collapse
    // each linked component in the list, preferring the receipt because it has
    // the reviewed items and tax breakdown. Neither source is deleted.
    const byId = new Map(filtered.map((transaction) => [transaction.id, transaction]));
    const visited = new Set<string>();
    const visibleTransactions: TransactionListItem[] = [];
    for (const transaction of filtered) {
      if (visited.has(transaction.id)) continue;
      const component: TransactionListItem[] = [];
      const queue = [transaction.id];
      while (queue.length) {
        const currentId = queue.pop()!;
        if (visited.has(currentId)) continue;
        visited.add(currentId);
        const current = byId.get(currentId);
        if (!current) continue;
        component.push(current);
        for (const linkedId of current.linked_transaction_ids || []) {
          if (!visited.has(linkedId) && byId.has(linkedId)) queue.push(linkedId);
        }
      }
      component.sort((left, right) => {
        const leftReceipt = left.source === "bill" ? 1 : 0;
        const rightReceipt = right.source === "bill" ? 1 : 0;
        return rightReceipt - leftReceipt;
      });
      visibleTransactions.push(component[0]);
    }

    const groups: { month: string; items: TransactionListItem[] }[] = [];
    const index: Record<string, TransactionListItem[]> = {};
    for (const t of visibleTransactions) {
      const label = monthLabel(t.trans_date);
      if (!index[label]) {
        index[label] = [];
        groups.push({ month: label, items: index[label] });
      }
      index[label].push(t);
    }
    return groups;
  }, [transactions, category, search]);

  const toggleMonth = (month: string) =>
    setCollapsedMonths((prev) => ({ ...prev, [month]: !(prev[month] ?? true) }));

  const openTransaction = (tx: TransactionListItem) => {
    router.push(`/transaction/${tx.id}`);
  };

  if (isLoading && transactions.length === 0) {
    return <LoadingSpinner message="Loading transactions..." />;
  }

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.primary} />
        }
      >
        <View style={styles.header}>
          <View style={styles.titleRow}>
            <Text style={styles.title}>Transactions</Text>
            {transactions.length > 0 && (
              <Badge style={styles.count}>{transactions.length}</Badge>
            )}
          </View>
        </View>

        {transactions.length === 0 ? (
          <Text style={styles.emptyText}>
            No transactions yet. Upload a CSV or receipt in the Files tab to get started.
          </Text>
        ) : (
          <>
            <Searchbar
              placeholder="Search merchant or description"
              value={search}
              onChangeText={setSearch}
              style={styles.searchbar}
              inputStyle={styles.searchInput}
              icon="magnify"
            />

            {/* Category filter chips */}
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.filterRow}
            >
              {[{ key: "all", label: "All" }, ...categories.map((c) => ({ key: c, label: c }))].map(
                (chip) => (
                  <Chip
                    key={chip.key}
                    selected={category === chip.key}
                    onPress={() => setCategory(chip.key)}
                    style={[styles.filterChip, category === chip.key && styles.filterChipActive]}
                    showSelectedCheck={false}
                    compact
                  >
                    {chip.label}
                  </Chip>
                )
              )}
            </ScrollView>

            {/* Collapsible month groups */}
            {groupedTransactions.length === 0 ? (
              <Text style={styles.emptyText}>No transactions match your search.</Text>
            ) : (
              groupedTransactions.map(({ month, items }) => {
                // Keep the list compact on entry; users expand only the month
                // they want to inspect.
                const collapsed = collapsedMonths[month] ?? true;
                // Spending is stored as a positive amount. Negative entries are
                // credit-card payments/refunds and should not reduce the month's
                // total cost shown in this header.
                const monthTotal = items.reduce(
                  (sum, t) => sum + Math.max(t.amount ?? 0, 0),
                  0
                );
                return (
                  <View key={month} style={styles.monthGroup}>
                    <Pressable style={styles.monthHeader} onPress={() => toggleMonth(month)}>
                      <MaterialCommunityIcons
                        name={collapsed ? "chevron-right" : "chevron-down"}
                        size={22}
                        color={colors.textSecondary}
                      />
                      <Text style={styles.monthTitle}>{month}</Text>
                      <Text style={styles.monthTotal}>${monthTotal.toFixed(2)}</Text>
                      <Badge style={styles.monthBadge}>{items.length}</Badge>
                    </Pressable>
                    {!collapsed &&
                      items.map((tx) => (
                        <TransactionRow key={tx.id} transaction={tx} onPress={openTransaction} />
                      ))}
                  </View>
                );
              })
            )}
          </>
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
  header: {
    marginBottom: spacing.lg,
  },
  titleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  title: {
    ...typography.h3,
    color: colors.text,
  },
  count: {
    backgroundColor: colors.primary,
  },
  searchbar: {
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    borderRadius: 12,
    elevation: 0,
  },
  searchInput: {
    fontSize: 14,
    minHeight: 0,
  },
  filterRow: {
    gap: spacing.sm,
    marginBottom: spacing.lg,
    paddingRight: spacing.lg,
  },
  filterChip: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  filterChipActive: {
    backgroundColor: colors.primaryLight,
    borderColor: colors.primary,
  },
  monthGroup: {
    marginBottom: spacing.sm,
  },
  monthHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.sm,
    gap: spacing.xs,
  },
  monthTitle: {
    ...typography.subtitle2,
    color: colors.textSecondary,
    flex: 1,
  },
  monthTotal: {
    ...typography.caption,
    color: colors.textSecondary,
    marginRight: spacing.sm,
  },
  monthBadge: {
    backgroundColor: colors.textTertiary,
  },
  emptyText: {
    ...typography.body2,
    color: colors.textSecondary,
  },
});
