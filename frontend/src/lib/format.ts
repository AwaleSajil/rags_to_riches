/**
 * Turning values into the strings the user reads, and reading them back.
 *
 * These were defined separately on each screen, which is how the app ended up
 * accepting "$4.99" on the receipt review form and rejecting it as "must be a
 * number" on the transaction edit form — the same value, two parsers.
 */

/** "$4.99". A missing amount is $0.00, never "$NaN". */
export function money(n: number | null | undefined): string {
  return `$${(n ?? 0).toFixed(2)}`;
}

/**
 * Read a number the way a person typed it, or OCR produced it.
 *
 * Vision extraction returns money as "$4.99" and rates as "8.25%", and users
 * type thousands separators. Rejecting those is pedantry — the value is not
 * ambiguous. Returns NaN for anything genuinely unreadable, so callers check
 * with Number.isFinite rather than getting a silent 0.
 */
export function parseNumber(value: string | number | null | undefined): number {
  if (typeof value === "number") return value;
  if (value == null) return Number.NaN;
  const normalized = value.trim().replace(/[$,%\s]/g, "");
  return normalized === "" ? Number.NaN : Number(normalized);
}

/** A number as text for a form field: "" when there is nothing to show. */
export function numberText(value: unknown, fallback = ""): string {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string") {
    const parsed = parseNumber(value);
    if (Number.isFinite(parsed)) return String(parsed);
  }
  return fallback;
}

/**
 * Parse a stored date without moving it a day.
 *
 * PostgreSQL DATE values carry no timezone. `new Date("2026-08-07")` parses as
 * UTC midnight, which is the 6th anywhere west of Greenwich — so a receipt from
 * this morning shows yesterday's date. Appending a time forces local parsing.
 */
function toLocalDate(dateStr: string): Date {
  return new Date(/^\d{4}-\d{2}-\d{2}$/.test(dateStr) ? `${dateStr}T00:00:00` : dateStr);
}

function format(
  dateStr: string | null | undefined,
  options: Intl.DateTimeFormatOptions,
  empty: string
): string {
  if (!dateStr) return empty;
  const d = toLocalDate(dateStr);
  // An unparseable date is shown as stored rather than as "Invalid Date": the
  // raw value is at least a clue about what went wrong.
  if (isNaN(d.getTime())) return dateStr;
  // Locale is deliberately the device's, not a fixed one.
  return d.toLocaleDateString(undefined, options);
}

/** "Aug 7, 2026" — lists and confirmation cards. */
export function formatDate(dateStr: string | null | undefined, empty = ""): string {
  return format(dateStr, { month: "short", day: "numeric", year: "numeric" }, empty);
}

/** "Fri, August 7, 2026" — the detail screen, where there is room for it. */
export function formatDateLong(dateStr: string | null | undefined, empty = "—"): string {
  return format(
    dateStr,
    { weekday: "short", month: "long", day: "numeric", year: "numeric" },
    empty
  );
}
