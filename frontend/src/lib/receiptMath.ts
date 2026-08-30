/**
 * What a receipt's line items add up to.
 *
 * This mirrors the server, which is the source of truth: see
 * `_prepare_detail_rows` and `_header_totals_from_details` in
 * backend/services/transaction_service.py. Both editing screens preview the
 * totals live, so the arithmetic existed three times and the two discount rules
 * — the subtle part — had to be remembered separately in each:
 *
 *   - An ITEM MARKDOWN (`savings`) is already inside the unit price. It is
 *     informational, shown so the user can see what they saved, and must never
 *     be subtracted again.
 *   - An ORDER COUPON (`orderDiscount`) comes off the whole basket and IS
 *     subtracted from the total.
 *
 * Subtracting the markdowns too undercounts every receipt that has them, and it
 * is the kind of error that looks plausible on screen.
 */

import { parseNumber } from "./format";

/** Form fields or stored numbers — both screens hold these as text. */
export type Amount = string | number | null | undefined;

export interface LineItemAmounts {
  quantity: Amount;
  unitPrice: Amount;
  /** Percent, e.g. 8.25. Zero or blank means the item is not taxed. */
  taxRate?: Amount;
  /** Item markdown. Informational — see the note above. */
  savings?: Amount;
}

export interface LineTotals {
  subtotal: number;
  tax: number;
  total: number;
}

export interface ReceiptTotals extends LineTotals {
  /** Item markdowns plus the order coupon: what the receipt says was saved. */
  savings: number;
}

/** A carried size, or null. Never rejects — see the note at the call site. */
function sizeNumber(value: Amount): number | null {
  const raw = typeof value === "string" ? value.trim() : value;
  if (raw === "" || raw === null || raw === undefined) return null;
  const parsed = parseNumber(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function num(value: Amount, fallback: number): number {
  const parsed = parseNumber(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/**
 * One line's subtotal, tax and post-tax total.
 *
 * A blank quantity counts as 1 and a blank price as 0, matching what the server
 * stores for a null — so the preview shows what will actually be saved while a
 * half-filled row is being typed.
 */
export function lineTotals(item: LineItemAmounts): LineTotals {
  const subtotal = num(item.quantity, 1) * num(item.unitPrice, 0);
  const rate = num(item.taxRate, 0);
  const tax = rate > 0 ? (subtotal * rate) / 100 : 0;
  return { subtotal, tax, total: subtotal + tax };
}

/**
 * Roll line items up into the header totals.
 *
 * `orderDiscount` is a basket-level coupon; anything negative is treated as
 * none, since a coupon that adds money is a typo rather than a charge.
 */
export function rollUp(items: LineItemAmounts[], orderDiscount: Amount = 0): ReceiptTotals {
  const discount = Math.max(0, num(orderDiscount, 0));
  const summed = items.reduce(
    (acc, item) => {
      const line = lineTotals(item);
      return {
        subtotal: acc.subtotal + line.subtotal,
        tax: acc.tax + line.tax,
        savings: acc.savings + Math.max(0, num(item.savings, 0)),
      };
    },
    { subtotal: 0, tax: 0, savings: 0 }
  );
  return {
    subtotal: summed.subtotal,
    tax: summed.tax,
    total: summed.subtotal + summed.tax - discount,
    savings: summed.savings + discount,
  };
}

// --- the editable form row ----------------------------------------------------
//
// Both editing screens held their own shape for this — `ItemForm` on the review
// screen, `DetailFormState` on the transaction screen — with different field
// names for the same four numbers. They also validated them differently: the
// review screen demanded a description and rejected negatives, while the
// transaction screen accepted both, so the same line was valid on one screen
// and not the other. One shape and one rule set.

/** A line item as the user is typing it. All strings, because inputs are. */
export interface LineItemForm {
  key: string;
  description: string;
  quantity: string;
  unitPrice: string;
  taxRate: string;
  savings: string;
  /**
   * What `quantity` counts (each / lb / oz…). Not edited in either form, but
   * carried through: the server replaces every line item wholesale, so a field
   * the form drops is deleted from the row.
   */
  quantityUnit?: string;
  /**
   * How much is in ONE unit — 5 and "lb" for a 5 lb bag. Carried for the same
   * reason as `quantityUnit`, and it matters more: this is the confirmed size
   * a per-unit price comparison rests on, and when it is missing the size has
   * to be guessed back out of the abbreviated description ("+RED POTA 5L US#"
   * read as five litres).
   */
  sizeValue?: string;
  sizeUnit?: string;
}

/** The fields a form actually edits — the rest are carried, not typed into. */
export type LineItemField = Exclude<
  keyof LineItemForm,
  "key" | "quantityUnit" | "sizeValue" | "sizeUnit"
>;

let keySequence = 0;

/** Row identity for React. Not the database id — a new row has none yet. */
export function newLineItemKey(): string {
  return `line-${Date.now()}-${keySequence++}`;
}

export function emptyLineItem(): LineItemForm {
  return {
    key: newLineItemKey(),
    description: "",
    quantity: "1",
    unitPrice: "",
    taxRate: "0",
    savings: "0",
  };
}

/** The amounts view of a row, for lineTotals / rollUp. */
export function amountsOf(row: LineItemForm): LineItemAmounts {
  return {
    quantity: row.quantity,
    unitPrice: row.unitPrice,
    taxRate: row.taxRate,
    savings: row.savings,
  };
}

/** A row that has passed validation: numbers are numbers, and non-negative. */
export interface ValidLineItem {
  description: string;
  quantity: number;
  unitPrice: number;
  taxRate: number;
  savings: number;
  quantityUnit: string | null;
  sizeValue: number | null;
  sizeUnit: string | null;
}

export type ValidationResult =
  | { ok: true; rows: ValidLineItem[] }
  | { ok: false; message: string };

/**
 * The one definition of a usable line item.
 *
 * Wholly blank rows are dropped rather than rejected — an empty row is someone
 * who tapped Add and changed their mind, not a mistake to scold them for. A
 * row with anything in it must be complete and non-negative, because a
 * negative quantity or price silently corrupts the transaction total and every
 * per-unit price comparison built on it afterwards.
 */
export function validateLineItems(rows: LineItemForm[]): ValidationResult {
  const valid: ValidLineItem[] = [];

  for (const [index, row] of rows.entries()) {
    const description = row.description.trim();
    const isBlank =
      !description &&
      !row.quantity.trim() &&
      !row.unitPrice.trim() &&
      !row.savings.trim();
    if (isBlank) continue;

    const label = description || `Item ${index + 1}`;
    if (!description) {
      return { ok: false, message: `Item ${index + 1} needs a description.` };
    }

    const numbers: Record<string, number> = {};
    for (const [field, name, fallback] of [
      ["quantity", "Quantity", "1"],
      ["unitPrice", "Unit price", ""],
      ["taxRate", "Tax rate", "0"],
      ["savings", "Savings", "0"],
    ] as const) {
      const raw = row[field].trim() || fallback;
      const value = parseNumber(raw);
      if (!Number.isFinite(value)) {
        return { ok: false, message: `${name} must be a number (${label}).` };
      }
      if (value < 0) {
        return { ok: false, message: `${name} cannot be negative (${label}).` };
      }
      numbers[field] = value;
    }

    valid.push({
      description,
      quantity: numbers.quantity,
      unitPrice: numbers.unitPrice,
      taxRate: numbers.taxRate,
      savings: numbers.savings,
      quantityUnit: row.quantityUnit?.trim() || null,
      // Carried, not validated: neither form offers a size field, so the only
      // value here is one the vision pass or an earlier save put there. A blank
      // is a genuine "unknown", which is what the column already means.
      sizeValue: sizeNumber(row.sizeValue),
      sizeUnit: row.sizeUnit?.trim().toLowerCase() || null,
    });
  }

  return { ok: true, rows: valid };
}
