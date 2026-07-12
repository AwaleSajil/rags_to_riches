import React, { useCallback, useState } from "react";
import {
  StyleSheet,
  View,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import {
  Text,
  Chip,
  Divider,
  Button,
  TextInput,
  Dialog,
  Portal,
  Snackbar,
} from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useLocalSearchParams, useFocusEffect, useRouter } from "expo-router";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import * as transactionService from "../../src/services/transactionService";
import type { TransactionUpdatePayload } from "../../src/services/transactionService";
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

interface FormState {
  merchant_name: string;
  description: string;
  trans_date: string;
  amount: string;
  category: string;
  location: string;
  subtotal: string;
  tax_total: string;
}

function toForm(tx: TransactionWithDetails): FormState {
  const numStr = (n: number | null | undefined) => (n == null ? "" : String(n));
  return {
    merchant_name: tx.merchant_name ?? "",
    description: tx.description ?? "",
    trans_date: tx.trans_date ?? "",
    amount: numStr(tx.amount),
    category: tx.category ?? "",
    location: tx.location ?? "",
    subtotal: numStr(tx.subtotal),
    tax_total: numStr(tx.tax_total),
  };
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
            {qty} × {money(unit)}
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
  const router = useRouter();
  const [transaction, setTransaction] = useState<TransactionWithDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [isEditing, setIsEditing] = useState(false);
  const [form, setForm] = useState<FormState | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });

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

  // Only auto-load when not mid-edit (avoid clobbering the form on focus).
  useFocusEffect(
    useCallback(() => {
      if (!isEditing) load();
    }, [load, isEditing])
  );

  const startEdit = () => {
    if (!transaction) return;
    setForm(toForm(transaction));
    setIsEditing(true);
  };

  const cancelEdit = () => {
    setIsEditing(false);
    setForm(null);
  };

  const setField = (key: keyof FormState, value: string) =>
    setForm((prev) => (prev ? { ...prev, [key]: value } : prev));

  const handleSave = async () => {
    if (!transaction || !form || !id) return;
    const orig = toForm(transaction);
    const changes: TransactionUpdatePayload = {};

    // Text fields
    if (form.merchant_name !== orig.merchant_name) changes.merchant_name = form.merchant_name.trim();
    if (form.description !== orig.description) changes.description = form.description.trim();
    if (form.category !== orig.category) changes.category = form.category.trim();
    if (form.location !== orig.location) changes.location = form.location.trim();

    // Date (YYYY-MM-DD)
    if (form.trans_date !== orig.trans_date) {
      const dt = form.trans_date.trim();
      if (!/^\d{4}-\d{2}-\d{2}$/.test(dt)) {
        setSnackbar({ visible: true, message: "Date must be YYYY-MM-DD", error: true });
        return;
      }
      changes.trans_date = dt;
    }

    // Amount (required, > 0)
    if (form.amount !== orig.amount) {
      const n = parseFloat(form.amount);
      if (isNaN(n) || n <= 0) {
        setSnackbar({ visible: true, message: "Amount must be a positive number", error: true });
        return;
      }
      changes.amount = n;
    }

    // Optional numerics — only send when non-empty
    for (const key of ["subtotal", "tax_total"] as const) {
      if (form[key] !== orig[key] && form[key].trim() !== "") {
        const n = parseFloat(form[key]);
        if (isNaN(n) || n < 0) {
          setSnackbar({ visible: true, message: `${key} must be a number`, error: true });
          return;
        }
        changes[key] = n;
      }
    }

    if (Object.keys(changes).length === 0) {
      cancelEdit();
      return;
    }

    setIsSaving(true);
    try {
      const updated = await transactionService.updateTransaction(id, changes);
      setTransaction(updated);
      setIsEditing(false);
      setForm(null);
      setSnackbar({ visible: true, message: "Transaction updated", error: false });
    } catch (e: any) {
      log.error("Failed to update transaction", e);
      setSnackbar({ visible: true, message: e.message || "Update failed", error: true });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!id) return;
    setConfirmDelete(false);
    setIsDeleting(true);
    try {
      await transactionService.deleteTransaction(id);
      // Pop back to the list, which refreshes on focus.
      router.back();
    } catch (e: any) {
      log.error("Failed to delete transaction", e);
      setIsDeleting(false);
      setSnackbar({ visible: true, message: e.message || "Delete failed", error: true });
    }
  };

  if (isLoading && !transaction) {
    return <LoadingSpinner message="Loading transaction..." />;
  }

  if (error || !transaction) {
    return (
      <View style={styles.centered}>
        <MaterialCommunityIcons name="alert-circle-outline" size={40} color={colors.error} />
        <Text style={styles.errorText}>{error || "Transaction not found"}</Text>
        <Button mode="outlined" onPress={load}>
          Retry
        </Button>
      </View>
    );
  }

  const tx = transaction;
  const hasTax =
    tx.subtotal != null ||
    tx.tax_total != null ||
    (tx.tax_breakdown != null && tx.tax_breakdown.length > 0);

  // -------- Edit mode --------
  if (isEditing && form) {
    return (
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
          <GlassCard style={styles.card}>
            <Text style={styles.sectionTitle}>Edit transaction</Text>
            <TextInput
              mode="outlined"
              label="Merchant"
              value={form.merchant_name}
              onChangeText={(v) => setField("merchant_name", v)}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Description"
              value={form.description}
              onChangeText={(v) => setField("description", v)}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Date (YYYY-MM-DD)"
              value={form.trans_date}
              onChangeText={(v) => setField("trans_date", v)}
              autoCapitalize="none"
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Amount"
              value={form.amount}
              onChangeText={(v) => setField("amount", v)}
              keyboardType="decimal-pad"
              left={<TextInput.Affix text="$" />}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Category"
              value={form.category}
              onChangeText={(v) => setField("category", v)}
              style={styles.input}
            />
            <TextInput
              mode="outlined"
              label="Location"
              value={form.location}
              onChangeText={(v) => setField("location", v)}
              style={styles.input}
            />
            <View style={styles.inputRow}>
              <TextInput
                mode="outlined"
                label="Subtotal"
                value={form.subtotal}
                onChangeText={(v) => setField("subtotal", v)}
                keyboardType="decimal-pad"
                style={[styles.input, styles.inputHalf]}
              />
              <TextInput
                mode="outlined"
                label="Tax total"
                value={form.tax_total}
                onChangeText={(v) => setField("tax_total", v)}
                keyboardType="decimal-pad"
                style={[styles.input, styles.inputHalf]}
              />
            </View>

            <View style={styles.editActions}>
              <Button
                mode="outlined"
                onPress={cancelEdit}
                disabled={isSaving}
                style={styles.actionButton}
              >
                Cancel
              </Button>
              <Button
                mode="contained"
                onPress={handleSave}
                loading={isSaving}
                disabled={isSaving}
                style={styles.actionButton}
              >
                Save
              </Button>
            </View>
          </GlassCard>
        </ScrollView>
        <Snackbar
          visible={snackbar.visible}
          onDismiss={() => setSnackbar({ ...snackbar, visible: false })}
          duration={4000}
          style={{ backgroundColor: snackbar.error ? colors.error : colors.success }}
        >
          {snackbar.message}
        </Snackbar>
      </KeyboardAvoidingView>
    );
  }

  // -------- Read-only mode --------
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

        {/* Actions */}
        <View style={styles.topActions}>
          <Button
            mode="outlined"
            icon="pencil-outline"
            onPress={startEdit}
            style={styles.actionButton}
          >
            Edit
          </Button>
          <Button
            mode="outlined"
            icon="trash-can-outline"
            textColor={colors.error}
            onPress={() => setConfirmDelete(true)}
            loading={isDeleting}
            disabled={isDeleting}
            style={[styles.actionButton, styles.deleteButton]}
          >
            Delete
          </Button>
        </View>

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
            <Text style={styles.sectionTitle}>Line items ({tx.details.length})</Text>
            {tx.details.map((item, i) => (
              <View key={item.id}>
                {i > 0 && <Divider style={styles.itemDivider} />}
                <LineItem item={item} />
              </View>
            ))}
          </GlassCard>
        )}
      </ScrollView>

      {/* Delete confirmation */}
      <Portal>
        <Dialog
          visible={confirmDelete}
          onDismiss={() => setConfirmDelete(false)}
          style={{ borderRadius: 16 }}
        >
          <Dialog.Title>Delete transaction</Dialog.Title>
          <Dialog.Content>
            <Text>
              Permanently delete{" "}
              <Text style={{ fontWeight: "700" }}>
                {tx.merchant_name || tx.description || "this transaction"}
              </Text>
              ? This removes it and its line items from the database and the search index.
            </Text>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setConfirmDelete(false)} textColor={colors.textSecondary}>
              Cancel
            </Button>
            <Button
              onPress={handleDelete}
              textColor="#ffffff"
              mode="contained"
              buttonColor={colors.error}
              style={{ borderRadius: 8 }}
            >
              Delete
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>

      <Snackbar
        visible={snackbar.visible}
        onDismiss={() => setSnackbar({ ...snackbar, visible: false })}
        duration={4000}
        style={{ backgroundColor: snackbar.error ? colors.error : colors.success }}
      >
        {snackbar.message}
      </Snackbar>
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
  topActions: {
    flexDirection: "row",
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  actionButton: {
    flex: 1,
    borderRadius: 10,
  },
  deleteButton: {
    borderColor: colors.error,
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
  input: {
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
  },
  inputRow: {
    flexDirection: "row",
    gap: spacing.md,
  },
  inputHalf: {
    flex: 1,
  },
  editActions: {
    flexDirection: "row",
    gap: spacing.md,
    marginTop: spacing.sm,
  },
});
