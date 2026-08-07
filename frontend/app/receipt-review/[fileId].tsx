import React, { useCallback, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { Banner, Button, Dialog, Divider, IconButton, Portal, Snackbar, Text, TextInput } from "react-native-paper";
import { useFocusEffect, useLocalSearchParams, useRouter } from "expo-router";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import * as transactionService from "../../src/services/transactionService";
import { colors, spacing } from "../../src/styles/theme";

type ItemForm = { key: string; description: string; quantity: string; unitPrice: string; taxRate: string; savings: string };
let keySequence = 0;
const newKey = () => `item-${Date.now()}-${keySequence++}`;
// OCR frequently returns money as "$4.99" or numbers as strings. Accept
// those familiar receipt formats instead of rejecting an otherwise valid edit.
const parseNumber = (value: string) => {
  const normalized = value.trim().replace(/[$,%\s,]/g, "");
  return normalized === "" ? Number.NaN : Number(normalized);
};
const numberText = (value: unknown, fallback = "") => {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string") {
    const parsed = parseNumber(value);
    if (Number.isFinite(parsed)) return String(parsed);
  }
  return fallback;
};
const money = (n: number) => `$${n.toFixed(2)}`;
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
  const { fileId, remaining = "" } = useLocalSearchParams<{ fileId: string; remaining?: string }>();
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
  const [items, setItems] = useState<ItemForm[]>([]);
  const [totalChoiceVisible, setTotalChoiceVisible] = useState(false);
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
      const discountSum = (extracted.discounts || []).reduce(
        (sum, d) => sum + (parseNumber(numberText(d.amount, "0")) || 0),
        0,
      );
      setOrderDiscount(discountSum > 0 ? String(Number(discountSum.toFixed(2))) : "");
      setItems((extracted.line_items || []).map((item) => ({
        key: newKey(),
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

  const changeItem = (key: string, field: keyof Omit<ItemForm, "key">, value: string) =>
    setItems((current) => current.map((item) => item.key === key ? { ...item, [field]: value } : item));

  const discountAmount = Math.max(0, parseNumber(orderDiscount) || 0);
  const itemsRollup = items.reduce((sum, item) => {
    const quantity = parseNumber(item.quantity) || 0;
    const unitPrice = parseNumber(item.unitPrice) || 0;
    const rate = parseNumber(item.taxRate) || 0;
    const subtotal = quantity * unitPrice;
    const tax = subtotal * rate / 100;
    return {
      subtotal: sum.subtotal + subtotal,
      tax: sum.tax + tax,
      itemSavings: sum.itemSavings + Math.max(0, parseNumber(item.savings) || 0),
    };
  }, { subtotal: 0, tax: 0, itemSavings: 0 });
  const totals = {
    subtotal: itemsRollup.subtotal,
    tax: itemsRollup.tax,
    // Item markdowns are already inside unitPrice; only the order-level coupon is
    // subtracted here.
    total: itemsRollup.subtotal + itemsRollup.tax - discountAmount,
    savings: itemsRollup.itemSavings + discountAmount,
  };

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
    const lineItems = [] as { item_description: string; item_quantity: number; item_unit_price: number; item_savings: number; tax_rate: number }[];
    for (const [index, item] of items.entries()) {
      if (!item.description.trim() && !item.unitPrice.trim()) continue;
      const quantity = parseNumber(item.quantity);
      const unitPrice = parseNumber(item.unitPrice);
      const taxRate = parseNumber(item.taxRate || "0");
      const savings = parseNumber(item.savings || "0");
      if (!item.description.trim() || !Number.isFinite(quantity) || quantity < 0 || !Number.isFinite(unitPrice) || unitPrice < 0 || !Number.isFinite(taxRate) || taxRate < 0 || !Number.isFinite(savings) || savings < 0) {
        setSnackbar({
          visible: true,
          message: `Item ${index + 1}: enter a description and non-negative quantity, price, tax, and savings`,
          error: true,
        });
        return;
      }
      lineItems.push({ item_description: item.description.trim(), item_quantity: quantity, item_unit_price: unitPrice, item_savings: savings, tax_rate: taxRate });
    }
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
          <Text variant="titleLarge" style={styles.title}>Review extracted receipt</Text>
          <Text style={styles.hint}>{filename}. Check the OCR result and correct anything before verifying.</Text>
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
          <View style={styles.itemHeader}>
            <Text variant="titleMedium" style={styles.itemsTitle}>Line items</Text>
            <Button mode="text" icon="plus" onPress={() => setItems((current) => [...current, { key: newKey(), description: "", quantity: "1", unitPrice: "", taxRate: "0", savings: "0" }])}>Add</Button>
          </View>
          {items.map((item, index) => (
            <View key={item.key} style={styles.item}>
              {index > 0 && <Divider style={styles.divider} />}
              <View style={styles.itemTop}>
                <TextInput mode="outlined" label="Description" value={item.description} onChangeText={(value) => changeItem(item.key, "description", value)} style={styles.description} dense />
                <IconButton icon="trash-can-outline" iconColor={colors.error} onPress={() => setItems((current) => current.filter((row) => row.key !== item.key))} />
              </View>
              <View style={styles.numbers}>
                <TextInput mode="outlined" label="Qty" value={item.quantity} onChangeText={(value) => changeItem(item.key, "quantity", value)} keyboardType="decimal-pad" style={styles.number} dense />
                <TextInput mode="outlined" label="Unit $" value={item.unitPrice} onChangeText={(value) => changeItem(item.key, "unitPrice", value)} keyboardType="decimal-pad" style={styles.number} dense />
                <TextInput mode="outlined" label="Tax %" value={item.taxRate} onChangeText={(value) => changeItem(item.key, "taxRate", value)} keyboardType="decimal-pad" style={styles.number} dense />
                <TextInput mode="outlined" label="Saved $" value={item.savings} onChangeText={(value) => changeItem(item.key, "savings", value)} keyboardType="decimal-pad" style={styles.number} dense />
              </View>
              <Text style={styles.itemSavingsHint}>Unit $ is the price you paid; Saved $ is the markdown on this item (0 if none).</Text>
            </View>
          ))}
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

        <Button mode="contained" icon="check-circle-outline" onPress={() => verify()} loading={saving} disabled={saving} style={styles.verify}>Verify receipt</Button>
        <Text style={styles.footer}>Verification saves this receipt as a transaction and adds it to search.</Text>
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
  hint: { color: colors.textSecondary, marginBottom: spacing.md, lineHeight: 20 }, input: { marginBottom: spacing.sm },
  itemHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" }, itemsTitle: { fontWeight: "700" },
  item: { paddingTop: spacing.sm }, divider: { marginVertical: spacing.sm }, itemTop: { flexDirection: "row", alignItems: "center" },
  description: { flex: 1 }, numbers: { flexDirection: "row", gap: spacing.xs }, number: { flex: 1 },
  total: { textAlign: "right", marginTop: 4, fontWeight: "600" }, verify: { marginTop: spacing.sm },
  extractedTotal: { textAlign: "right", marginTop: 4, color: colors.textSecondary },
  itemSavingsHint: { color: colors.textSecondary, marginTop: 4, fontSize: 11 },
  discountHint: { color: colors.textSecondary, fontSize: 12, marginTop: -2, marginBottom: spacing.sm },
  savingsSummary: { textAlign: "right", marginTop: 2, color: colors.success, fontWeight: "600" },
  footer: { color: colors.textSecondary, textAlign: "center", marginTop: spacing.sm },
});
