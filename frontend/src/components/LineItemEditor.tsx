/**
 * Editing a receipt's line items — the same four numbers, wherever you are.
 *
 * This existed twice: once on the receipt review screen and once inside the
 * transaction editor. Identical inputs and layout, different field names, and
 * — the part that mattered — different validation, so a row with a negative
 * quantity was rejected on one screen and saved on the other. The rules now
 * live with the arithmetic in lib/receiptMath; this is just their form.
 *
 * The list is rendered here; the surrounding card and the totals footer stay
 * with each screen, because those genuinely differ (the review screen shows a
 * compact summary, the transaction editor shows a full breakdown with the
 * order-level discount).
 */

import React from "react";
import { StyleSheet, View } from "react-native";
import { Button, Divider, IconButton, Text, TextInput } from "react-native-paper";

import { colors, spacing, typography } from "../styles/theme";
import { money } from "../lib/format";
import {
  amountsOf,
  lineTotals,
  type LineItemField,
  type LineItemForm,
} from "../lib/receiptMath";

interface LineItemEditorProps {
  rows: LineItemForm[];
  onChange: (key: string, field: LineItemField, value: string) => void;
  onAdd: () => void;
  onRemove: (key: string) => void;
  disabled?: boolean;
  /** Heading above the list. */
  title?: string;
}

const NUMERIC_FIELDS: { field: LineItemField; label: string }[] = [
  { field: "quantity", label: "Qty" },
  { field: "unitPrice", label: "Unit $" },
  { field: "taxRate", label: "Tax %" },
  { field: "savings", label: "Saved $" },
];

function LineItemEditorComponent({
  rows,
  onChange,
  onAdd,
  onRemove,
  disabled = false,
  title = "Line items",
}: LineItemEditorProps) {
  return (
    <>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <Button mode="text" icon="plus" compact onPress={onAdd} disabled={disabled}>
          Add
        </Button>
      </View>

      {rows.length === 0 ? (
        <Text style={styles.empty}>No line items. Tap “Add” to create one.</Text>
      ) : (
        rows.map((row, index) => {
          const totals = lineTotals(amountsOf(row));
          return (
            <View key={row.key} style={styles.row}>
              {index > 0 && <Divider style={styles.divider} />}
              <View style={styles.top}>
                <TextInput
                  mode="outlined"
                  label="Description"
                  value={row.description}
                  onChangeText={(value) => onChange(row.key, "description", value)}
                  style={styles.description}
                  dense
                />
                <IconButton
                  icon="trash-can-outline"
                  iconColor={colors.error}
                  size={20}
                  onPress={() => onRemove(row.key)}
                  disabled={disabled}
                  style={styles.delete}
                />
              </View>
              <View style={styles.numbers}>
                {NUMERIC_FIELDS.map(({ field, label }) => (
                  <TextInput
                    key={field}
                    mode="outlined"
                    label={label}
                    value={row[field]}
                    onChangeText={(value) => onChange(row.key, field, value)}
                    keyboardType="decimal-pad"
                    style={styles.number}
                    dense
                  />
                ))}
              </View>
              <Text style={styles.computed}>
                {money(totals.subtotal)}
                {totals.tax > 0 ? ` + ${money(totals.tax)} tax` : ""} = {money(totals.total)}
              </Text>
            </View>
          );
        })
      )}

      {rows.length > 0 ? (
        <Text style={styles.hint}>
          Unit $ is the price you paid; Saved $ is the markdown on this item (0 if none).
        </Text>
      ) : null}
    </>
  );
}

// Each keystroke re-renders the parent screen, and without this every row's
// four inputs re-render with it.
export const LineItemEditor = React.memo(LineItemEditorComponent);

const styles = StyleSheet.create({
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  title: {
    ...typography.subtitle1,
    color: colors.text,
  },
  empty: {
    ...typography.body2,
    color: colors.textSecondary,
    paddingVertical: spacing.sm,
  },
  row: {
    marginBottom: spacing.xs,
  },
  divider: {
    marginVertical: spacing.sm,
    backgroundColor: colors.divider,
  },
  top: {
    flexDirection: "row",
    alignItems: "center",
  },
  description: {
    flex: 1,
  },
  delete: {
    margin: 0,
  },
  numbers: {
    flexDirection: "row",
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  number: {
    flex: 1,
  },
  computed: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 4,
    textAlign: "right",
  },
  hint: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.xs,
  },
});
