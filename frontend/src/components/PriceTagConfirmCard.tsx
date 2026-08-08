import React, { useState } from "react";
import { StyleSheet, View, Pressable } from "react-native";
import { Text, Button, TextInput, Checkbox } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import { money, parseNumber } from "../lib/format";
import type { PriceTagDraft } from "../services/captureService";
import type { PriceComparison } from "../services/priceService";

interface Props {
  fileId: string;
  /** Every tag the photo showed. A shelf shot normally has several. */
  tags: PriceTagDraft[];
  onConfirm: (
    draft: PriceTagDraft & { tag_index: number }
  ) => Promise<PriceComparison | null>;
  /** Throw the photo away for real — storage object and BillFile row.
   *  Discard used to be local UI state only, so a dismissed tag still sat in the
   *  Files tab as if it had been kept. */
  onDiscard: () => Promise<void>;
  /** Ask the agent about everything just confirmed — ONCE, for the whole photo.
   *  Asking per tag started a chat turn per tag; they raced on one conversation
   *  and each answer inherited the previous one's context. */
  onAsk?: (drafts: (PriceTagDraft & { tag_index: number })[], fileId?: string) => void;
  /** Resolved place name for where the photo was taken, e.g. "Main St,
   *  Norwalk". Saved with every price on this photo — a shelf tag carries no
   *  address, so this is the only thing that says WHICH branch it was. */
  place?: string | null;
}

/** "12 OZ" -> { size_value: 12, size_unit: "oz" }. Returns {} when
 *  the text holds no usable number, so a blank field clears nothing. */
function splitSize(text: string): {
  size_value?: number | null;
  size_unit?: string | null;
} {
  const match = text.trim().match(/^([\d.]+)\s*([a-zA-Z]+)?$/);
  if (!match) return {};
  const quantity = parseFloat(match[1]);
  if (!isFinite(quantity) || quantity <= 0) return {};
  return { size_value: quantity, size_unit: match[2]?.toLowerCase() ?? null };
}

/** One tag's editable state. */
interface Row {
  selected: boolean;
  item: string;
  price: string;
  size: string;
  merchant: string;
  result: PriceComparison | null;
  saved: boolean;
  error: string | null;
}

function toRow(tag: PriceTagDraft, index: number, total: number): Row {
  return {
    // The first tag is pre-selected because a single-tag photo is the common
    // case and one tap should finish it. On a shelf shot the rest are left off:
    // the user framed the photo for something, and saving prices they did not
    // ask about fills their history with items they never looked at.
    selected: index === 0,
    item: tag.item_description ?? "",
    price: tag.item_subtotal_price != null ? String(tag.item_subtotal_price) : "",
    size:
      tag.size_value != null
        ? `${tag.size_value}${tag.size_unit ? " " + tag.size_unit : ""}`
        : "",
    merchant: tag.merchant_name ?? "",
    result: null,
    saved: false,
    error: null,
  };
}

type Status = "editing" | "saving" | "done" | "cancelled";

/**
 * The last stop before a price observation is written.
 *
 * Everything is editable because OCR on a cluttered shelf tag is the weakest
 * link in the chain — an unnoticed wrong size silently corrupts every future
 * unit-price comparison for this item. A wide shelf shot makes that worse, not
 * better: the tags are smaller and more angled, so checking matters more.
 */
export function PriceTagConfirmCard({ fileId, tags, onConfirm, onDiscard, onAsk, place }: Props) {
  // A photo the model could not read arrives with no tags at all — and the user
  // has just told us it IS a price tag. One empty row lets them type it in.
  // Without this the card rendered a heading, no fields, and a "Pick at least
  // one price to check" error about a list that did not exist.
  const [rows, setRows] = useState<Row[]>(() =>
    (tags.length ? tags : [{}]).map((tag, i) => toRow(tag, i, tags.length))
  );
  const [status, setStatus] = useState<Status>("editing");
  const [error, setError] = useState<string | null>(null);
  // Tracked apart from `status` because discarding is reachable from BOTH the
  // form and the saved state, and folding it into the status would make those
  // two states mutually exclusive with it.
  const [discarding, setDiscarding] = useState(false);

  const multiple = tags.length > 1;
  const chosen = rows.filter((r) => r.selected);

  const patch = (index: number, change: Partial<Row>) =>
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, ...change } : r)));

  const handleConfirm = async () => {
    if (!chosen.length) {
      setError("Pick at least one price to check.");
      return;
    }
    for (const [index, row] of rows.entries()) {
      if (!row.selected) continue;
      if (!row.item.trim()) {
        setError(`Tag ${index + 1} needs an item name.`);
        return;
      }
      const parsed = parseFloat(row.price);
      if (!isFinite(parsed) || parsed < 0) {
        setError(`Tag ${index + 1} needs the price shown on the tag.`);
        return;
      }
    }

    setStatus("saving");
    setError(null);

    const asked: (PriceTagDraft & { tag_index: number })[] = [];

    // Saved one at a time rather than in parallel: each one embeds, enriches and
    // searches, and firing five of those at once is a good way to hit an
    // embedding rate limit and lose the lot.
    for (const [index, row] of rows.entries()) {
      if (!row.selected) continue;
      try {
        const draft = {
          ...(tags[index] ?? {}),
          tag_index: index,
          item_description: row.item.trim(),
          item_subtotal_price: parseFloat(row.price),
          // Split back into a number and its unit: the tag printed them together
          // ("12 OZ") and the row stores them apart so a shelf tag and a receipt
          // line hold the same shape.
          ...splitSize(row.size),
          merchant_name: row.merchant.trim() || null,
          location: place || null,
        };
        const comparison = await onConfirm(draft);
        asked.push(draft);
        patch(index, { result: comparison, saved: true, error: null });
      } catch (e: any) {
        // One failure must not discard the ones that worked.
        patch(index, { error: e?.message || "Could not save that price", saved: false });
      }
    }
    setStatus("done");
    // One question covering everything that saved, so the agent answers about
    // all of it in a single reply instead of racing itself.
    if (asked.length) onAsk?.(asked, fileId);
  };

  const handleDiscard = async () => {
    setDiscarding(true);
    setError(null);
    try {
      await onDiscard();
      setStatus("cancelled");
    } catch (e: any) {
      // Say so rather than showing "Discarded" over a photo that is still there.
      setError(e?.message || "Could not discard that photo");
    } finally {
      setDiscarding(false);
    }
  };

  if (status === "cancelled") {
    return (
      <View style={styles.card}>
        <View style={styles.header}>
          <MaterialCommunityIcons name="close-circle" size={20} color={colors.textTertiary} />
          <Text style={styles.title}>Discarded</Text>
        </View>
      </View>
    );
  }

  if (status === "done") {
    return (
      <View style={styles.card}>
        {rows.map((row, i) =>
          !row.selected ? null : (
            <View key={i} style={i > 0 ? styles.savedBlock : undefined}>
              <View style={styles.header}>
                <MaterialCommunityIcons
                  name={row.error ? "alert-circle" : "check-circle"}
                  size={20}
                  color={row.error ? colors.error : colors.success}
                />
                <Text style={styles.title}>
                  {row.item}
                  {row.price ? ` · ${money(parseNumber(row.price))}` : ""}
                </Text>
              </View>
              {/* Deliberately no comparison here. The agent answers in the very
                  next message with the same evidence and can actually reason
                  about it; printing a second, blunter version above it just made
                  the user read the same thing twice. */}
              {row.error && <Text style={styles.error}>{row.error}</Text>}
            </View>
          )
        )}
        {/* Reachable AFTER saving too. Changing your mind once you have seen the
            answer is the commonest reason to want the photo gone, and until now
            this state had no way out — the only Discard was on the form you had
            already left. Deleting the photo takes the observations it created
            with it via ON DELETE CASCADE. */}
        {error && <Text style={styles.error}>{error}</Text>}
        <View style={styles.actions}>
          <Button
            mode="text"
            onPress={handleDiscard}
            loading={discarding}
            disabled={discarding}
            compact
          >
            Discard this photo
          </Button>
        </View>
      </View>
    );
  }

  const busy = status === "saving" || discarding;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <MaterialCommunityIcons name="tag-outline" size={20} color={colors.primary} />
        <Text style={styles.title}>
          {multiple ? `${tags.length} prices in this photo` : "Check this price"}
        </Text>
      </View>
      {multiple && (
        <Text style={styles.hintPlain}>Pick the ones you want to check.</Text>
      )}
      {place && <Text style={styles.hintPlain}>📍 {place}</Text>}

      {rows.map((row, index) => (
        <View key={index} style={multiple ? styles.tagBlock : undefined}>
          {multiple && (
            <Pressable
              onPress={() => !busy && patch(index, { selected: !row.selected })}
              style={styles.tagHeader}
            >
              <Checkbox
                status={row.selected ? "checked" : "unchecked"}
                onPress={() => !busy && patch(index, { selected: !row.selected })}
                disabled={busy}
              />
              <Text style={styles.tagLabel} numberOfLines={1}>
                {row.item || `Tag ${index + 1}`}
                {row.price ? ` · $${row.price}` : ""}
                {row.size ? ` · ${row.size}` : ""}
              </Text>
            </Pressable>
          )}

          {/* Only what you are actually saving opens for editing — a shelf photo
              with eight tags would otherwise be a wall of inputs. */}
          {(!multiple || row.selected) && (
            <>
              <TextInput
                mode="outlined"
                label="Item"
                value={row.item}
                onChangeText={(v) => patch(index, { item: v })}
                dense
                style={styles.input}
                disabled={busy}
              />
              <View style={styles.row}>
                <TextInput
                  mode="outlined"
                  label="Price"
                  value={row.price}
                  onChangeText={(v) => patch(index, { price: v })}
                  keyboardType="decimal-pad"
                  left={<TextInput.Affix text="$" />}
                  dense
                  style={[styles.input, styles.flex]}
                  disabled={busy}
                />
                <TextInput
                  mode="outlined"
                  label="Size"
                  placeholder="12 oz"
                  value={row.size}
                  onChangeText={(v) => patch(index, { size: v })}
                  dense
                  style={[styles.input, styles.flex]}
                  disabled={busy}
                />
              </View>
              <TextInput
                mode="outlined"
                label="Store"
                value={row.merchant}
                onChangeText={(v) => patch(index, { merchant: v })}
                dense
                style={styles.input}
                disabled={busy}
              />

              {/* The tag's own per-unit figure beats anything we compute: the
                  store did the arithmetic for the exact package on the shelf. */}
              {tags[index]?.unit_price_display && (
                <Text style={styles.unitPrice}>
                  {tags[index]?.unit_price_display} (printed on the tag)
                </Text>
              )}
              {tags[index]?.size_unknown && (
                <Text style={styles.hint}>
                  No size found — I'll compare this against the same item only, not per unit.
                </Text>
              )}
              {/* Shown verbatim rather than as parsed badges. The tag's own
                  wording is what the user can check against the shelf in front
                  of them, and it is what the agent later reasons from. */}
              {tags[index]?.item_qualitative_description && (
                <Text style={styles.observedContext} numberOfLines={3}>
                  {tags[index]?.item_qualitative_description}
                </Text>
              )}
            </>
          )}
        </View>
      ))}

      {error && <Text style={styles.error}>{error}</Text>}

      <View style={styles.actions}>
        <Button
          mode="contained"
          onPress={handleConfirm}
          loading={busy}
          disabled={busy}
          style={styles.flex}
        >
          {multiple && chosen.length !== 1 ? `Compare ${chosen.length}` : "Compare"}
        </Button>
        <Button
          mode="text"
          onPress={handleDiscard}
          loading={discarding}
          disabled={busy}
        >
          Discard
        </Button>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surfaceSubtle,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    padding: spacing.md,
    marginTop: spacing.sm,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  title: { ...typography.body2, fontWeight: "700", color: colors.text, flexShrink: 1 },
  body: { ...typography.body2, color: colors.textSecondary },
  input: { backgroundColor: colors.surface, marginBottom: spacing.xs },
  row: { flexDirection: "row", gap: spacing.xs },
  flex: { flex: 1 },
  tagBlock: {
    borderTopWidth: 1,
    borderTopColor: colors.surfaceBorder,
    paddingTop: spacing.xs,
    marginTop: spacing.xs,
  },
  tagHeader: { flexDirection: "row", alignItems: "center", gap: 2 },
  tagLabel: { ...typography.body2, color: colors.text, flexShrink: 1 },
  savedBlock: {
    borderTopWidth: 1,
    borderTopColor: colors.surfaceBorder,
    paddingTop: spacing.sm,
    marginTop: spacing.sm,
  },
  observedContext: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: spacing.xs,
    marginBottom: spacing.xs,
    fontStyle: "italic",
  },
  unitPrice: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 2,
    marginBottom: spacing.xs,
  },
  hint: { ...typography.caption, color: colors.warning, marginBottom: spacing.xs },
  hintPlain: { ...typography.caption, color: colors.textSecondary, marginBottom: spacing.xs },
  error: { ...typography.caption, color: colors.error, marginBottom: spacing.xs },
  actions: { flexDirection: "row", alignItems: "center", gap: spacing.xs, marginTop: spacing.xs },
});
