import React, { useCallback, useMemo, useState } from "react";
import { StyleSheet, View, ScrollView, Pressable, RefreshControl, SectionList } from "react-native";
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

// One shared empty array, so collapsing a month doesn't hand SectionList a new
// `data` identity every render and re-render the whole list.
const NO_ROWS: TransactionListItem[] = [];

type MonthSection = {
  title: string;
  count: number;
  total: number;
  collapsed: boolean;
  data: TransactionListItem[];
};

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
  const monthGroups = useMemo(() => {
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
    // Spending is stored as a positive amount. Negative entries are credit-card
    // payments/refunds and should not reduce the month's total cost.
    return groups.map(({ month, items }) => ({
      month,
      items,
      total: items.reduce((sum, t) => sum + Math.max(t.amount ?? 0, 0), 0),
    }));
  }, [transactions, category, search]);

  // Collapsing swaps a section's rows for an empty array rather than rebuilding
  // the groups, so toggling a month never re-runs the filtering and link
  // resolution above.
  const sections = useMemo<MonthSection[]>(
    () =>
      monthGroups.map((group) => {
        // Keep the list compact on entry; users expand only the month they
        // want to inspect.
        const collapsed = collapsedMonths[group.month] ?? true;
        return {
          title: group.month,
          count: group.items.length,
          total: group.total,
          collapsed,
          data: collapsed ? NO_ROWS : group.items,
        };
      }),
    [monthGroups, collapsedMonths]
  );

  const toggleMonth = useCallback(
    (month: string) =>
      setCollapsedMonths((prev) => ({ ...prev, [month]: !(prev[month] ?? true) })),
    []
  );

  const openTransaction = useCallback(
    (tx: TransactionListItem) => {
      router.push(`/transaction/${tx.id}`);
    },
    [router]
  );

  const keyExtractor = useCallback((tx: TransactionListItem) => tx.id, []);

  const renderItem = useCallback(
    ({ item }: { item: TransactionListItem }) => (
      <TransactionRow transaction={item} onPress={openTransaction} />
    ),
    [openTransaction]
  );

  const renderSectionHeader = useCallback(
    ({ section }: { section: MonthSection }) => (
      <Pressable style={styles.monthHeader} onPress={() => toggleMonth(section.title)}>
        <MaterialCommunityIcons
          name={section.collapsed ? "chevron-right" : "chevron-down"}
          size={22}
          color={colors.textSecondary}
        />
        <Text style={styles.monthTitle}>{section.title}</Text>
        <Text style={styles.monthTotal}>${section.total.toFixed(2)}</Text>
        <Badge style={styles.monthBadge}>{section.count}</Badge>
      </Pressable>
    ),
    [toggleMonth]
  );

  // Replaces the `marginBottom` the wrapping <View> used to give each group.
  const renderSectionFooter = useCallback(() => <View style={styles.sectionGap} />, []);

  if (isLoading && transactions.length === 0) {
    return <LoadingSpinner message="Loading transactions..." />;
  }

  // Passed as an ELEMENT, not a function. An inline `() => <.../>` is a new
  // component type on every render, so SectionList would unmount and remount
  // this whole block — and the Searchbar would lose focus on every keystroke.
  const listHeader = (
    <>
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

          {/* Not SectionList's ListEmptyComponent: that also fires when every
              month happens to be collapsed, which is the normal opening state. */}
          {sections.length === 0 && (
            <Text style={styles.emptyText}>No transactions match your search.</Text>
          )}
        </>
      )}
    </>
  );

  return (
    <View style={styles.container}>
      {/* Collapsible month groups. Virtualized, so a year of statements mounts
          only the rows on screen instead of every row at once. */}
      <SectionList
        sections={sections}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        renderSectionHeader={renderSectionHeader}
        renderSectionFooter={renderSectionFooter}
        ListHeaderComponent={listHeader}
        contentContainerStyle={styles.scrollContent}
        // The month headers have no background of their own, so sticking them
        // would let rows scroll through the text.
        stickySectionHeadersEnabled={false}
        keyboardShouldPersistTaps="handled"
        initialNumToRender={12}
        maxToRenderPerBatch={12}
        windowSize={11}
        refreshControl={
          <RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.primary} />
        }
      />
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
  sectionGap: {
    height: spacing.sm,
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
