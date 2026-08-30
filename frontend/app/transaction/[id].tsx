import React, { useCallback, useEffect, useState } from "react";
import {
  StyleSheet,
  View,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Pressable,
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
  Checkbox,
} from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useLocalSearchParams, useFocusEffect, useRouter } from "expo-router";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { FilePreviewModal } from "../../src/components/FilePreviewModal";
import { LinkedRecordsCard } from "../../src/components/LinkedRecordsCard";
import * as transactionService from "../../src/services/transactionService";
import * as fileService from "../../src/services/fileService";
import type {
  TransactionUpdatePayload,
  TransactionDetailInput,
} from "../../src/services/transactionService";
import { colors, typography, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";
import { formatDateLong, money, numberText, parseNumber } from "../../src/lib/format";
import {
  amountsOf,
  emptyLineItem,
  rollUp,
  validateLineItems,
  type LineItemField,
  type LineItemForm,
} from "../../src/lib/receiptMath";
import { LineItemEditor } from "../../src/components/LineItemEditor";
import { transactionVisual } from "../../src/lib/sourceVisual";
import type { FileItem, TransactionWithDetails, TransactionDetailItem } from "../../src/lib/types";

const log = createLogger("TransactionDetail");

// Deep enrichment can take up to ~a minute (a sequential web search per line
// item + an LLM pass), so poll well past that before giving up.
const MAX_ENRICHMENT_POLLS = 12;

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
  return {
    merchant_name: tx.merchant_name ?? "",
    description: tx.description ?? "",
    trans_date: tx.trans_date ?? "",
    amount: numberText(tx.amount),
    category: tx.category ?? "",
    location: tx.location ?? "",
    subtotal: numberText(tx.subtotal),
    tax_total: numberText(tx.tax_total),
  };
}

function toDetailForm(d: TransactionDetailItem): LineItemForm {
  return {
    key: d.id,
    description: d.item_description ?? "",
    quantity: numberText(d.item_quantity),
    unitPrice: numberText(d.unit_quantity_subtotal),
    savings: numberText(d.item_savings),
    taxRate: numberText(d.tax_rate),
    // Not shown in the form, carried so a save does not wipe it.
    quantityUnit: d.item_quantity_unit ?? "",
    sizeValue: d.size_value != null ? String(d.size_value) : "",
    sizeUnit: d.size_unit ?? "",
  };
}

// Canonical string of the editable detail fields, for change detection.
function detailSnapshot(rows: LineItemForm[]): string {
  return rows
    .map(
      (r) =>
        `${r.description.trim()}|${r.quantity.trim()}|${r.unitPrice.trim()}|${r.savings.trim()}|${r.taxRate.trim()}`
    )
    .join("~");
}

function LineItem({ item }: { item: TransactionDetailItem }) {
  const qty = item.item_quantity;
  const unit = item.unit_quantity_subtotal;
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
        {isTaxable && (item.tax_amount ?? 0) > 0 && (
          <Text style={styles.lineItemSub}>
            {money(item.item_subtotal_price)} + {money(item.tax_amount)} tax
          </Text>
        )}
        {(item.item_savings ?? 0) > 0 && (
          <Text style={styles.lineItemSavings}>Saved {money(item.item_savings)}</Text>
        )}
        {item.enriched_info ? (
          <Text style={styles.enrichedInfo}>{item.enriched_info}</Text>
        ) : null}
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
  const [detailForms, setDetailForms] = useState<LineItemForm[]>([]);
  const [origDetailSnapshot, setOrigDetailSnapshot] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  // Off by default: keeping the photo is the safer of the two, and the receipt
  // is often the only remaining copy once the paper is in the bin.
  const [alsoDeleteReceipt, setAlsoDeleteReceipt] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });
  const [enrichmentRefreshes, setEnrichmentRefreshes] = useState(0);
  const [editingNote, setEditingNote] = useState(false);
  const [noteDraft, setNoteDraft] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  // The BillFile row behind this transaction, fetched on demand. Non-null while
  // the photo preview is open.
  const [receiptFile, setReceiptFile] = useState<FileItem | null>(null);
  const [receiptLoading, setReceiptLoading] = useState(false);

  const load = useCallback(async () => {
    if (!id) {
      // isLoading starts true, so returning early here used to leave the screen
      // on "Loading transaction..." forever. Surface it instead — this is only
      // reachable via a bad deep link or a stray navigation.
      log.warn("Transaction screen opened without an id");
      setIsLoading(false);
      setError("No transaction was specified.");
      return;
    }
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

  // Deep enrichment runs after receipt verification as a background task that
  // web-searches the merchant and every line item *sequentially* before an LLM
  // pass, so it can take up to a minute for a long receipt. Poll with backoff
  // over a wide-enough window (instead of a fixed 10s) so the descriptions
  // appear without the user having to leave and re-open the screen. Stops the
  // moment everything is enriched.
  useEffect(() => {
    if (!transaction || isEditing || transaction.source !== "bill") return;
    if (enrichmentRefreshes >= MAX_ENRICHMENT_POLLS) return;
    const complete = Boolean(transaction.enriched_info) && transaction.details.every((d) => Boolean(d.enriched_info));
    if (complete) return;
    // 2.5s, 5s, 7.5s, then 10s each — ~100s total across MAX_ENRICHMENT_POLLS.
    const delay = Math.min(2500 + enrichmentRefreshes * 2500, 10000);
    const timer = setTimeout(() => {
      setEnrichmentRefreshes((count) => count + 1);
      load();
    }, delay);
    return () => clearTimeout(timer);
  }, [transaction, isEditing, enrichmentRefreshes, load]);

  const startEdit = () => {
    if (!transaction) return;
    setForm(toForm(transaction));
    const rows = transaction.details.map(toDetailForm);
    setDetailForms(rows);
    setOrigDetailSnapshot(detailSnapshot(rows));
    setIsEditing(true);
  };

  const cancelEdit = () => {
    setIsEditing(false);
    setForm(null);
    setDetailForms([]);
  };

  const setField = (key: keyof FormState, value: string) =>
    setForm((prev) => (prev ? { ...prev, [key]: value } : prev));

  const setDetailField = (key: string, field: LineItemField, value: string) =>
    setDetailForms((prev) =>
      prev.map((r) => (r.key === key ? { ...r, [field]: value } : r))
    );

  const addDetailRow = () => setDetailForms((prev) => [...prev, emptyLineItem()]);

  const removeDetailRow = (key: string) =>
    setDetailForms((prev) => prev.filter((r) => r.key !== key));

  const handleSave = async () => {
    if (!transaction || !form || !id) return;
    const orig = toForm(transaction);
    const changes: TransactionUpdatePayload = {};
    // When line items exist, the header subtotal/tax/amount are derived from
    // them on the server, so we don't edit those fields directly here.
    const hasLineItems = detailForms.length > 0;

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

    if (!hasLineItems) {
      // Amount (required, > 0)
      if (form.amount !== orig.amount) {
        const n = parseNumber(form.amount);
        if (!Number.isFinite(n) || n <= 0) {
          setSnackbar({ visible: true, message: "Amount must be a positive number", error: true });
          return;
        }
        changes.amount = n;
      }

      // Optional numerics — only send when non-empty
      for (const key of ["subtotal", "tax_total"] as const) {
        if (form[key] !== orig[key] && form[key].trim() !== "") {
          const n = parseNumber(form[key]);
          if (!Number.isFinite(n) || n < 0) {
            setSnackbar({ visible: true, message: `${key} must be a number`, error: true });
            return;
          }
          changes[key] = n;
        }
      }
    }

    // --- Line items ---
    const detailsChanged = detailSnapshot(detailForms) !== origDetailSnapshot;
    let detailsPayload: TransactionDetailInput[] = [];
    if (detailsChanged) {
      // Same rules as the receipt review screen — see validateLineItems. This
      // screen used to accept a negative quantity and a blank description that
      // the other one rejected, for the same row in the same table.
      const validated = validateLineItems(detailForms);
      if (!validated.ok) {
        setSnackbar({ visible: true, message: validated.message, error: true });
        return;
      }
      detailsPayload = validated.rows.map((r) => ({
        item_description: r.description,
        item_quantity: r.quantity,
        item_quantity_unit: r.quantityUnit,
        size_value: r.sizeValue,
        size_unit: r.sizeUnit,
        unit_quantity_subtotal: r.unitPrice,
        item_savings: r.savings,
        tax_rate: r.taxRate,
      }));
    }

    if (Object.keys(changes).length === 0 && !detailsChanged) {
      cancelEdit();
      return;
    }

    setIsSaving(true);
    try {
      let updated = transaction;
      if (Object.keys(changes).length > 0) {
        updated = await transactionService.updateTransaction(id, changes);
      }
      if (detailsChanged) {
        updated = await transactionService.replaceTransactionDetails(id, detailsPayload);
      }
      setTransaction(updated);
      setIsEditing(false);
      setForm(null);
      setDetailForms([]);
      setSnackbar({ visible: true, message: "Transaction updated", error: false });
    } catch (e: any) {
      log.error("Failed to update transaction", e);
      setSnackbar({ visible: true, message: e.message || "Update failed", error: true });
    } finally {
      setIsSaving(false);
    }
  };

  /** Fetch the BillFile row so the preview can sign a URL for its s3_key. */
  const openReceipt = async () => {
    const fileId = transaction?.source_bill_file_id;
    if (!fileId || receiptLoading) return;
    setReceiptLoading(true);
    try {
      setReceiptFile(await fileService.getFile(fileId));
    } catch (e: any) {
      log.error("Failed to load the receipt photo", e);
      setSnackbar({
        visible: true,
        message: e.message || "Could not open the receipt photo",
        error: true,
      });
    } finally {
      setReceiptLoading(false);
    }
  };

  const startEditNote = () => {
    setNoteDraft(transaction?.note ?? "");
    setEditingNote(true);
  };

  const saveNote = async () => {
    if (!id || !transaction) return;
    const next = noteDraft.trim();
    if (next === (transaction.note ?? "")) {
      setEditingNote(false);
      return;
    }
    setSavingNote(true);
    try {
      // Sending "" clears the note; the server re-indexes the vector so the
      // note becomes (or stops being) searchable.
      const updated = await transactionService.updateTransaction(id, { note: next });
      setTransaction(updated);
      setEditingNote(false);
      setSnackbar({ visible: true, message: next ? "Note saved" : "Note removed", error: false });
    } catch (e: any) {
      log.error("Failed to save note", e);
      setSnackbar({ visible: true, message: e.message || "Could not save note", error: true });
    } finally {
      setSavingNote(false);
    }
  };

  const handleDelete = async () => {
    if (!id) return;
    setConfirmDelete(false);
    setIsDeleting(true);
    try {
      const billFileId = alsoDeleteReceipt ? transaction?.source_bill_file_id : null;
      if (billFileId) {
        // Deleting the RECEIPT, not the transaction: the cascade runs from
        // BillFile down, so this takes the transaction and its line items with
        // it, and removes the stored photo too. Deleting the transaction first
        // would leave nothing for this call to find.
        await fileService.deleteFile(billFileId, "bill");
      } else {
        await transactionService.deleteTransaction(id);
      }
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
  const sourceLook = transactionVisual(tx.source);
  const hasTax =
    tx.subtotal != null ||
    tx.tax_total != null ||
    (tx.tax_breakdown != null && tx.tax_breakdown.length > 0) ||
    (tx.discount_total ?? 0) > 0 ||
    (tx.savings_total ?? 0) > 0;

  // -------- Edit mode --------
  if (isEditing && form) {
    const hasLineItems = detailForms.length > 0;
    // An order-level discount lives on the header (not the line items) and can't
    // be edited here; carry it through so the preview total matches what the
    // server will store (subtotal + tax - discount).
    const orderDiscount = tx.discount_total ?? 0;
    // Header totals are derived from the line items when they exist.
    const liveTotals = rollUp(detailForms.map(amountsOf), orderDiscount);
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
            {!hasLineItems && (
              <TextInput
                mode="outlined"
                label="Amount"
                value={form.amount}
                onChangeText={(v) => setField("amount", v)}
                keyboardType="decimal-pad"
                left={<TextInput.Affix text="$" />}
                style={styles.input}
              />
            )}
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
            {!hasLineItems && (
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
            )}
          </GlassCard>

          {/* Line-item editor */}
          <GlassCard style={styles.card}>
            <LineItemEditor
              rows={detailForms}
              onChange={setDetailField}
              onAdd={addDetailRow}
              onRemove={removeDetailRow}
              disabled={isSaving}
            />

            {hasLineItems && (
              <>
                <Divider style={styles.divider} />
                <View style={styles.totalsRow}>
                  <Text style={styles.totalsLabel}>Subtotal</Text>
                  <Text style={styles.totalsValue}>{money(liveTotals.subtotal)}</Text>
                </View>
                <View style={styles.totalsRow}>
                  <Text style={styles.totalsLabel}>Tax</Text>
                  <Text style={styles.totalsValue}>{money(liveTotals.tax)}</Text>
                </View>
                {orderDiscount > 0 && (
                  <View style={styles.totalsRow}>
                    <Text style={styles.totalsLabel}>Discount</Text>
                    <Text style={styles.totalsValue}>−{money(orderDiscount)}</Text>
                  </View>
                )}
                <View style={styles.totalsRow}>
                  <Text style={styles.totalsLabelBold}>Total</Text>
                  <Text style={styles.totalsValueBold}>{money(liveTotals.total)}</Text>
                </View>
                <Text style={styles.derivedNote}>
                  {orderDiscount > 0
                    ? "Header subtotal and tax come from these line items; the order discount is applied to the total."
                    : "Header subtotal, tax, and total are computed from these line items."}
                </Text>
              </>
            )}
          </GlassCard>

          <GlassCard style={styles.editActionsCard}>
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
          <Text style={styles.date}>{formatDateLong(tx.trans_date)}</Text>
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
                icon={sourceLook.icon}
              >
                {sourceLook.label}
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
          {tx.enriched_info ? (
            <Text style={styles.merchantInfo}>{tx.enriched_info}</Text>
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

        {/* Both records of one purchase, when a statement line and a receipt
            were matched to each other. The list collapses them into a single
            row, so this is the only place that says the other one exists. */}
        {tx.linked_transactions && tx.linked_transactions.length > 0 ? (
          <GlassCard style={styles.card}>
            <LinkedRecordsCard
              viewingSource={tx.source}
              viewingDetailCount={tx.details.length}
              linked={tx.linked_transactions}
              // replace, not push: paging back and forth between two records of
              // the same purchase should not build a stack you have to unwind.
              onOpen={(id) => router.replace(`/transaction/${id}`)}
            />
          </GlassCard>
        ) : null}

        {/* Source receipt. Tapping shows the PHOTO — that is what someone
            coming back to a transaction wants to see, and what the label
            promises. Correcting the scan is a deliberate second step from
            inside the preview, because it rewrites the transaction's line
            items rather than just looking at it. */}
        {tx.source_bill_file_id ? (
          <Pressable
            onPress={openReceipt}
            disabled={receiptLoading}
            style={({ pressed }) => pressed && styles.receiptCardPressed}
          >
            <GlassCard style={styles.card}>
              <View style={styles.receiptRow}>
                <View style={[styles.receiptIcon, { backgroundColor: sourceLook.background }]}>
                  <MaterialCommunityIcons
                    name={sourceLook.icon}
                    size={22}
                    color={sourceLook.color}
                  />
                </View>
                <View style={styles.receiptText}>
                  <Text style={styles.receiptTitle}>Receipt photo</Text>
                  <Text style={styles.receiptSubtitle}>
                    {receiptLoading ? "Opening…" : "Tap to view the original"}
                  </Text>
                </View>
                <MaterialCommunityIcons
                  name="chevron-right"
                  size={22}
                  color={colors.textTertiary}
                />
              </View>
            </GlassCard>
          </Pressable>
        ) : null}

        {/* Notes card — add/edit while viewing; the note is also indexed for search */}
        <GlassCard style={styles.card}>
          <View style={styles.notesHeader}>
            <Text style={styles.sectionTitle}>Note</Text>
            {!editingNote && (
              <Button
                mode="text"
                compact
                icon={tx.note ? "pencil-outline" : "plus"}
                onPress={startEditNote}
              >
                {tx.note ? "Edit" : "Add note"}
              </Button>
            )}
          </View>
          {editingNote ? (
            <>
              <TextInput
                mode="outlined"
                placeholder="Add a note — e.g. “split with Alex”, “reimburse from work”…"
                value={noteDraft}
                onChangeText={setNoteDraft}
                multiline
                numberOfLines={3}
                autoFocus
                style={styles.noteInput}
              />
              <View style={styles.noteActions}>
                <Button
                  mode="outlined"
                  onPress={() => setEditingNote(false)}
                  disabled={savingNote}
                  style={styles.actionButton}
                >
                  Cancel
                </Button>
                <Button
                  mode="contained"
                  onPress={saveNote}
                  loading={savingNote}
                  disabled={savingNote}
                  style={styles.actionButton}
                >
                  Save
                </Button>
              </View>
            </>
          ) : tx.note ? (
            <Text style={styles.noteText}>{tx.note}</Text>
          ) : (
            <Text style={styles.notePlaceholder}>
              No note yet. Notes are searchable across your transactions.
            </Text>
          )}
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
            {(tx.discount_total ?? 0) > 0 && (
              <View style={styles.totalsRow}>
                <Text style={styles.totalsLabel}>Discount</Text>
                <Text style={styles.totalsValue}>−{money(tx.discount_total)}</Text>
              </View>
            )}
            <Divider style={styles.divider} />
            <View style={styles.totalsRow}>
              <Text style={styles.totalsLabelBold}>Total</Text>
              <Text style={styles.totalsValueBold}>{money(tx.amount)}</Text>
            </View>
            {(tx.savings_total ?? 0) > 0 && (
              <View style={styles.totalsRow}>
                <Text style={styles.savingsLabel}>You saved</Text>
                <Text style={styles.savingsValue}>{money(tx.savings_total)}</Text>
              </View>
            )}
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

      {/* The photo itself, zoomable. Its "Review receipt" action is the way to
          the correction form — a deliberate second tap, since saving there
          rewrites this transaction's line items. */}
      <FilePreviewModal
        file={receiptFile}
        onClose={() => setReceiptFile(null)}
        onReviewReceipt={(file) => {
          setReceiptFile(null);
          router.push({
            pathname: "/receipt-review/[fileId]",
            params: { fileId: file.id, fromTransaction: "1" },
          });
        }}
      />

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
            {/* Deleting a transaction leaves the receipt it came from behind —
                the cascade runs the other way. Said out loud, because the photo
                then sits in Files counting towards nothing, which is a confusing
                state to arrive at silently. */}
            {!!tx.source_bill_file_id && (
              <>
                <Text style={styles.deleteHint}>
                  The receipt photo is kept in Files so you can review it again.
                  It won't count towards your spending until you do.
                </Text>
                <Checkbox.Item
                  label="Also delete the receipt photo"
                  position="leading"
                  status={alsoDeleteReceipt ? "checked" : "unchecked"}
                  onPress={() => setAlsoDeleteReceipt((v) => !v)}
                  labelStyle={styles.deleteCheckboxLabel}
                  style={styles.deleteCheckbox}
                />
              </>
            )}
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
  merchantInfo: {
    ...typography.body2,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.md,
  },
  topActions: {
    flexDirection: "row",
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  notesHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  receiptCardPressed: {
    opacity: 0.7,
  },
  receiptRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
  },
  receiptIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  receiptText: {
    flex: 1,
  },
  receiptTitle: {
    ...typography.body1,
    color: colors.text,
    fontWeight: "600",
  },
  receiptSubtitle: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 2,
  },
  noteText: {
    ...typography.body2,
    color: colors.text,
    lineHeight: 20,
  },
  notePlaceholder: {
    ...typography.body2,
    color: colors.textSecondary,
  },
  noteInput: {
    backgroundColor: colors.surface,
    marginTop: spacing.xs,
  },
  noteActions: {
    flexDirection: "row",
    gap: spacing.md,
    marginTop: spacing.md,
  },
  actionButton: {
    flex: 1,
    borderRadius: 10,
  },
  deleteHint: {
    ...typography.body2,
    color: colors.textSecondary,
    marginTop: spacing.md,
    lineHeight: 20,
  },
  deleteCheckbox: {
    paddingHorizontal: 0,
    marginTop: spacing.xs,
    marginLeft: -spacing.sm,
  },
  deleteCheckboxLabel: {
    ...typography.body2,
    color: colors.text,
    textAlign: "left",
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
  lineItemSavings: {
    ...typography.caption,
    color: colors.success,
    marginTop: 2,
  },
  savingsLabel: {
    ...typography.body2,
    color: colors.success,
    flexShrink: 1,
    marginRight: spacing.sm,
  },
  savingsValue: {
    ...typography.body2,
    color: colors.success,
    fontWeight: "600",
  },
  enrichedInfo: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: spacing.xs,
    lineHeight: 18,
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
  editActionsCard: {
    marginBottom: spacing.lg,
    paddingVertical: spacing.md,
  },
  derivedNote: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.sm,
    fontStyle: "italic",
  },
});
