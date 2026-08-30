/**
 * What each kind of record looks like — decided once, for every list.
 *
 * The same receipt used to appear as an amber `file-image` in Files and an
 * indigo `receipt-text` in Transactions, so nothing tied the two views of one
 * thing together. Worse, in Transactions the receipt and CSV icons were the
 * same size and colour in the same circle, differing only in glyph — a
 * distinction nobody can make while scrolling, which is the whole job of a
 * marker.
 *
 * Colour carries it, because colour is what the eye catches in a list. Photos
 * are tinted, bank data is neutral: the photos are the rows a user acts on.
 */

import type { MaterialCommunityIcons } from "@expo/vector-icons";
import type React from "react";

import { colors } from "../styles/theme";
import type { FileItem } from "./types";

type IconName = React.ComponentProps<typeof MaterialCommunityIcons>["name"];

export type SourceKind = "receipt" | "price_tag" | "csv";

export interface SourceVisual {
  icon: IconName;
  /** Glyph colour, and the colour of any badge text beside it. */
  color: string;
  /** The icon circle behind it. */
  background: string;
  /** Short human name, for a badge or a chip. */
  label: string;
}

const VISUALS: Record<SourceKind, SourceVisual> = {
  receipt: {
    icon: "receipt-text",
    color: colors.primary,
    background: colors.primaryLight,
    label: "Receipt",
  },
  price_tag: {
    icon: "tag-outline",
    color: colors.secondary,
    background: "#f3e8ff",
    label: "Price tag",
  },
  csv: {
    icon: "bank-outline",
    color: colors.textSecondary,
    background: colors.surfaceSubtle,
    label: "CSV",
  },
};

export function sourceVisual(kind: SourceKind): SourceVisual {
  return VISUALS[kind];
}

/** A row in the Files tab. Photos carry `kind`; CSVs do not. */
export function fileVisual(file: Pick<FileItem, "type" | "kind">): SourceVisual {
  if (file.type === "csv") return VISUALS.csv;
  // Bill rows written before migration 014 have no kind. They are receipts —
  // price tags did not exist yet — so that is the safe assumption.
  return VISUALS[file.kind === "price_tag" ? "price_tag" : "receipt"];
}

/**
 * A row in the Transactions tab.
 *
 * A price tag is deliberately absent: it is a sighting, not a purchase, and
 * never becomes a transaction. Anything here is a receipt or a bank CSV.
 */
export function transactionVisual(source: string | null | undefined): SourceVisual {
  return VISUALS[source === "bill" ? "receipt" : "csv"];
}
