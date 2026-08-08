import React, { useCallback, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { Banner, Button, Dialog, Divider, Portal, Snackbar, Text, TextInput } from "react-native-paper";
import { useFocusEffect, useLocalSearchParams, useRouter } from "expo-router";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import * as transactionService from "../../src/services/transactionService";
import * as fileService from "../../src/services/fileService";
import { colors, spacing } from "../../src/styles/theme";
import { money, numberText, parseNumber } from "../../src/lib/format";
import {
  amountsOf,
  emptyLineItem,
  newLineItemKey,
  rollUp,
  validateLineItems,
  type LineItemField,
  type LineItemForm,
} from "../../src/lib/receiptMath";
import { LineItemEditor } from "../../src/components/LineItemEditor";

const normalizeTime = (value: string) => {
  const raw = value.trim();
  if (!raw) return "";
  const twentyFourHour = raw.match(/^([01]?\d|2[0-3]):([0-5]\d)$/);
  if (twentyFourHour) return `${twentyFourHour[1].padStart(2, "0")}:${twentyFourHour[2]}`;
  const twelveHour = raw.match(/^(\d{1,2}):([0-5]\d)\s*([AaPp][Mm])$/);
  if (!twelveHour) return raw;
  let hour = Number(twelveHour[1]);
  if (hour < 1 || hour > 12) return raw;
  if (twelveHour[3].toLowerCase() === "pm" && hour !== 12) hour += 12;
  if (twelveHour[3].toLowerCase() === "am" && hour === 12) hour = 0;
  return `${String(hour).padStart(2, "0")}:${twelveHour[2]}`;
};

export default function ReceiptReviewScreen() {
  const {
    fileId,
    remaining = "",
    // Set when arriving from a transaction that already exists, rather than
    // from a freshly-scanned photo. Same form, but two things it says are
    // wrong in that case: this is not a first review, and discarding no longer
    // throws away just a photo — it takes the transaction with it.
    fromTransaction,
  } = useLocalSearchParams<{ fileId: string; remaining?: string; fromTransaction?: string }>();
  const isReReview = fromTransaction === "1";
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState("");
  const [merchant, setMerchant] = useState("");
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [category, setCategory] = useState("Uncategorized");
  const [location, setLocation] = useState("");
  const [note, setNote] = useState("");
  const [extractedTotal, setExtractedTotal] = useState("");
  const [orderDiscount, setOrderDiscount] = useState("");
  const [items, setItems] = useState<LineItemForm[]>([]);
  const [totalChoiceVisible, setTotalChoiceVisible] = useState(false);
  const [dismissVisible, setDismissVisible] = useState(false);
  const [dismissing, setDismissing] = useState(false);
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });
  // Id of the transaction this receipt turned out to duplicate. Set only after
  // verifying, because the match is on the confirmed contents, not the photo.
  const [duplicateOf, setDuplicateOf] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!fileId) {
      // Same trap as transaction/[id]: `loading` starts true, so an early
      // return here would strand the screen on its spinner.
      setLoading(false);
      setError("No receipt was specified.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const draft = await transactionService.getReceiptReview(fileId);
      const extracted = draft.extracted || {};
      setFilename(draft.filename || "Receipt");
      setMerchant(extracted.merchant_name || "");
      setDate(extracted.date || new Date().toISOString().slice(0, 10));
      setTime(normalizeTime(extracted.time || ""));
      setCategory(extracted.category || "Uncategorized");
      setLocation(extracted.location || "");
      setExtractedTotal(numberText(extracted.total_amount));
      // Order-level coupons are subtracted from the basket; sum them into one
      // editable field. (Per-item markdowns are already netted into unit prices.)
      //
      // Two shapes arrive here. Raw OCR produces `discounts: [{label, amount}]`.
      // A receipt that has already been verified has its REVIEW written back
      // over the draft (see _verify_receipt_row), and that carries the single
      // `discount_total` this form submits. Reading only the first meant
      // re-opening a verified receipt showed an empty coupon field, and saving
      // again silently dropped the discount and changed the stored total.
      const discountSum =
        extracted.discount_total != null
          ? parseNumber(numberText(extracted.discount_total, "0")) || 0
          : (extracted.discounts || []).reduce(
              (sum, d) => sum + (parseNumber(numberText(d.amount, "0")) || 0),
              0,
            );
      setOrderDiscount(discountSum > 0 ? String(Number(discountSum.toFixed(2))) : "");
      setItems((extracted.line_items || []).map((item) => ({
        key: newLineItemKey(),
        description: item.item_description || "",
        quantity: numberText(item.item_quantity, "1"),
        unitPrice: numberText(item.item_unit_price),
        taxRate: numberText(item.tax_rate, "0"),
        savings: numberText(item.item_savings, "0"),
      })));
    } catch (e: any) {
      setError(e.message || "Could not load the receipt review");
    } finally {
      setLoading(false);
    }
  }, [fileId]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const changeItem = (key: string, field: LineItemField, value: string) =>
    setItems((current) => current.map((item) => item.key === key ? { ...item, [field]: value } : item));
  const addItem = () => setItems((current) => [...current, emptyLineItem()]);
  const removeItem = (key: string) =>
    setItems((current) => current.filter((row) => row.key !== key));

  const discountAmount = Math.max(0, parseNumber(orderDiscount) || 0);
  // Item markdowns are already inside unitPrice; only the order-level coupon
  // is subtracted. See src/lib/receiptMath.ts, which mirrors the server.
  const totals = rollUp(items.map(amountsOf), discountAmount);

  const verify = async (chosenTotal?: number) => {
    if (!fileId) return;
    if (!merchant.trim()) {
      setSnackbar({ visible: true, message: "Merchant is required", error: true });
      return;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date.trim())) {
      setSnackbar({ visible: true, message: "Date must be YYYY-MM-DD", error: true });
      return;
    }
    const receiptTime = normalizeTime(time);
    if (receiptTime && !/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(receiptTime)) {
      setSnackbar({ visible: true, message: "Time must be HH:MM, e.g. 14:30", error: true });
      return;
    }
    const validated = validateLineItems(items);
    if (!validated.ok) {
      setSnackbar({ visible: true, message: validated.message, error: true });
      return;
    }
    const lineItems = validated.rows.map((row) => ({
      item_description: row.description,
      item_quantity: row.quantity,
      item_unit_price: row.unitPrice,
      item_savings: row.savings,
      tax_rate: row.taxRate,
    }));
    if (!lineItems.length) {
      setSnackbar({ visible: true, message: "Add at least one receipt item", error: true });
      return;
    }
    const receiptTotal = parseNumber(extractedTotal);
    if (chosenTotal == null && Number.isFinite(receiptTotal) && Math.abs(receiptTotal - totals.total) > 0.02) {
      setTotalChoiceVisible(true);
      return;
    }
    setSaving(true);
    try {
      const totalAmount = chosenTotal ?? (Number.isFinite(receiptTotal) ? receiptTotal : totals.total);
      const transaction = await transactionService.verifyReceiptReview(fileId, {
        date: date.trim(), time: receiptTime || null,
        merchant_name: merchant.trim(), category: category.trim() || "Uncategorized",
        location: location.trim() || null, total_amount: totalAmount,
        discount_total: discountAmount > 0 ? Number(discountAmount.toFixed(2)) : undefined,
        line_items: lineItems,
        // Sent only when written. This form does not load an existing note, so
        // passing a blank one on re-verification would clear a note added later
        // from the transaction screen. Clearing stays that screen's job.
        ...(note.trim() ? { note: note.trim() } : {}),
      });
      const queued = remaining.split(",").filter(Boolean);
      if (transaction.is_duplicate) {
        // Nothing was written: this receipt matched one already recorded. Say so
        // and stay put, because silently opening the earlier transaction reads
        // as "saved" and the same receipt gets re-uploaded again next time.
        setSnackbar({
          visible: true,
          message: "Already recorded — this matches a receipt you've saved before.",
          // Not an error: dedup working is the correct outcome, not a failure.
          error: false,
        });
        setDuplicateOf(transaction.id);
        return;
      }
      if (queued.length) {
        router.replace({ pathname: "/receipt-review/[fileId]", params: { fileId: queued[0], remaining: queued.slice(1).join(",") } });
      } else {
        router.replace(`/transaction/${transaction.id}`);
      }
    } catch (e: any) {
      setSnackbar({ visible: true, message: e.message || "Could not verify receipt", error: true });
    } finally {
      setSaving(false);
    }
  };

  /** Move to the next queued receipt, or leave if this was the last one. */
  const leaveReview = () => {
    const queued = remaining.split(",").filter(Boolean);
    if (queued.length) {
      router.replace({
        pathname: "/receipt-review/[fileId]",
        params: { fileId: queued[0], remaining: queued.slice(1).join(","), fromTransaction },
      });
      return;
    }
    if (isReReview) {
      // Going back would land on the transaction detail screen for a
      // transaction the cascade just deleted, which renders as "Transaction
      // not found". The list is the nearest place that still exists.
      router.replace("/(tabs)/transactions");
      return;
    }
    router.back();
  };

  /**
   * Throw the receipt away for good.
   *
   * A receipt is stored the moment it is read, before review, so that a photo
   * taken and then interrupted is not lost — the paper is usually already in
   * the bin. The cost of that choice is a stored receipt with no way to say "I
   * did not want this", which left unreviewed photos sitting in Files forever.
   *
   * Deletion is the ordinary file delete: it removes the stored image, and
   * Transaction, TransactionDetail and PriceObservation all cascade off
   * BillFile, so a receipt already verified takes its transaction with it.
   */
  const dismiss = async () => {
    if (!fileId) return;
    setDismissing(true);
    try {
      await fileService.deleteFile(fileId, "bill");
      setDismissVisible(false);
      leaveReview();
    } catch (e: any) {
      // Stay put and say so. Navigating away from a receipt that is still
      // stored would read as "dismissed" and it would turn up again later.
      setDismissVisible(false);
      setSnackbar({
        visible: true,
        message: e.message || "Could not discard that receipt",
        error: true,
      });
    } finally {
      setDismissing(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading extracted receipt…" />;
  if (error) return (
    <View style={styles.centered}>
      <Text style={styles.error}>{error}</Text>
      <Button mode="contained" onPress={load}>Retry</Button>
    </View>
  );

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {duplicateOf && (
          <Banner
            visible
            icon="content-duplicate"
            actions={[
              { label: "View the original", onPress: () => router.replace(`/transaction/${duplicateOf}`) },
              { label: "Back to files", onPress: () => router.back() },
            ]}
          >
            This receipt is already recorded. Nothing was saved, so your totals
            haven't been counted twice.
          </Banner>
        )}
        <GlassCard style={styles.card}>
          <Text variant="titleLarge" style={styles.title}>
            {isReReview ? "Review this receipt" : "Review extracted receipt"}
          </Text>
          <Text style={styles.hint}>
            {isReReview
              ? `${filename}. Saving updates the transaction this receipt created — it will not add a second one.`
              : `${filename}. Check the OCR result and correct anything before verifying.`}
          </Text>
          <TextInput mode="outlined" label="Merchant" value={merchant} onChangeText={setMerchant} style={styles.input} />
          <TextInput mode="outlined" label="Date (YYYY-MM-DD)" value={date} onChangeText={setDate} autoCapitalize="none" style={styles.input} />
          <TextInput mode="outlined" label="Time (HH:MM, optional)" value={time} onChangeText={setTime} autoCapitalize="none" style={styles.input} />
          <TextInput mode="outlined" label="Category" value={category} onChangeText={setCategory} style={styles.input} />
          <TextInput mode="outlined" label="Location (optional)" value={location} onChangeText={setLocation} style={styles.input} />
          <TextInput
            mode="outlined"
            label="Note (optional)"
            placeholder="What was this for?"
            value={note}
            onChangeText={setNote}
            multiline
            numberOfLines={2}
            style={styles.input}
          />
        </GlassCard>

        <GlassCard style={styles.card}>
          <LineItemEditor
            rows={items}
            onChange={changeItem}
            onAdd={addItem}
            onRemove={removeItem}
            disabled={saving}
          />
          <Divider style={styles.divider} />
          <TextInput
            mode="outlined"
            label="Order discount / coupon ($, optional)"
            value={orderDiscount}
            onChangeText={setOrderDiscount}
            keyboardType="decimal-pad"
            style={styles.input}
            dense
          />
          <Text style={styles.discountHint}>Only whole-basket coupons (e.g. “$5 off $50”). Per-item markdowns are already in the Unit $ above.</Text>
          <Text style={styles.total}>Subtotal {money(totals.subtotal)}   Tax {money(totals.tax)}</Text>
          {discountAmount > 0 && (
            <Text style={styles.total}>Discount −{money(discountAmount)}</Text>
          )}
          <Text variant="titleMedium" style={styles.total}>Total {money(totals.total)}</Text>
          {totals.savings > 0 && (
            <Text style={styles.savingsSummary}>You saved {money(totals.savings)}</Text>
          )}
          {Number.isFinite(parseNumber(extractedTotal)) && (
            <Text style={styles.extractedTotal}>Receipt total: {money(parseNumber(extractedTotal))}</Text>
          )}
        </GlassCard>

        <Button mode="contained" icon="check-circle-outline" onPress={() => verify()} loading={saving} disabled={saving || dismissing} style={styles.verify}>
          {isReReview ? "Save changes" : "Verify receipt"}
        </Button>
        <Text style={styles.footer}>
          {isReReview
            ? "Saving replaces this receipt's line items and re-indexes it for search."
            : "Verification saves this receipt as a transaction and adds it to search."}
        </Text>
        {/* Secondary and low-contrast on purpose: verifying is what almost
            everyone came here to do, and this one cannot be undone. */}
        <Button
          mode="text"
          icon="trash-can-outline"
          textColor={colors.error}
          onPress={() => setDismissVisible(true)}
          disabled={saving || dismissing}
          style={styles.dismiss}
        >
          {isReReview ? "Delete receipt and transaction" : "Discard this receipt"}
        </Button>
      </ScrollView>
      <Portal>
        <Dialog visible={totalChoiceVisible} onDismiss={() => setTotalChoiceVisible(false)} style={{ borderRadius: 8 }}>
          <Dialog.Title>Totals don’t match</Dialog.Title>
          <Dialog.Content>
            <Text>
              The receipt says {money(parseNumber(extractedTotal))}, but the line items calculate to {money(totals.total)}. Which total should be saved?
            </Text>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setTotalChoiceVisible(false)}>Edit items</Button>
            <Button onPress={() => { setTotalChoiceVisible(false); verify(totals.total); }}>Use computed</Button>
            <Button mode="contained" onPress={() => { setTotalChoiceVisible(false); verify(parseNumber(extractedTotal)); }}>Use receipt</Button>
          </Dialog.Actions>
        </Dialog>
        <Dialog visible={dismissVisible} onDismiss={() => setDismissVisible(false)} style={{ borderRadius: 8 }}>
          <Dialog.Title>
            {isReReview ? "Delete receipt and transaction?" : "Discard this receipt?"}
          </Dialog.Title>
          <Dialog.Content>
            {/* Said explicitly when arriving from a transaction. The cascade
                runs from BillFile down, so what reads as "throw away a photo"
                also deletes a recorded transaction and its line items — and
                the user got here from that transaction, not from Files. */}
            <Text>
              {isReReview
                ? "This deletes the photo AND the transaction it created, including its line items. Your spending totals will change. This cannot be undone."
                : "The photo and everything read from it are deleted for good. Nothing from this receipt will count towards your spending."}
            </Text>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setDismissVisible(false)} disabled={dismissing}>Keep it</Button>
            <Button
              mode="contained"
              buttonColor={colors.error}
              textColor="#ffffff"
              onPress={dismiss}
              loading={dismissing}
              disabled={dismissing}
            >
              Discard
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
      <Snackbar visible={snackbar.visible} onDismiss={() => setSnackbar({ ...snackbar, visible: false })} style={{ backgroundColor: snackbar.error ? colors.error : colors.success }}>{snackbar.message}</Snackbar>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: 40 },
  centered: { flex: 1, justifyContent: "center", alignItems: "center", padding: spacing.lg, gap: spacing.md },
  error: { color: colors.error, textAlign: "center" },
  card: { marginBottom: spacing.md }, title: { fontWeight: "700", marginBottom: 4 },
  hint: { color: colors.textSecondary, marginBottom: spacing.md, lineHeight: 20 }, input: { marginBottom: spacing.sm }, itemsTitle: { fontWeight: "700" }, divider: { marginVertical: spacing.sm }, itemTop: { flexDirection: "row", alignItems: "center" }, numbers: { flexDirection: "row", gap: spacing.xs }, number: { flex: 1 },
  total: { textAlign: "right", marginTop: 4, fontWeight: "600" }, verify: { marginTop: spacing.sm },
  dismiss: { marginTop: spacing.xs, alignSelf: "center" },
  extractedTotal: { textAlign: "right", marginTop: 4, color: colors.textSecondary },
  discountHint: { color: colors.textSecondary, fontSize: 12, marginTop: -2, marginBottom: spacing.sm },
  savingsSummary: { textAlign: "right", marginTop: 2, color: colors.success, fontWeight: "600" },
  footer: { color: colors.textSecondary, textAlign: "center", marginTop: spacing.sm },
});
