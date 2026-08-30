import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type { PriceTagDraft } from "./captureService";

const log = createLogger("PriceService");

export interface PriceObservation {
  id: string;
  /** The photo this price came from, AFTER it was stored. Not the capture
   *  handle the card was holding — confirming turns one into the other. */
  bill_file_id?: string | null;
  item_description: string;
  brand_name?: string | null;
  size_value?: number | null;
  size_unit?: string | null;
  unit_quantity_subtotal?: number | null;
  unit_price_unit?: string | null;
  item_subtotal_price?: number | null;
  merchant_name?: string | null;
  location?: string | null;
  /** What the photo showed beyond the price, in the tag's own words. */
  item_qualitative_description?: string | null;
  /** What the user said about it in chat. Not embedded — an occasion is not
   *  product identity. */
  note?: string | null;
  created_at: string;
  /** What the user has paid for this before. Null when it could not be worked out. */
  comparison?: PriceComparison | null;
}

/** Evidence about a price — deliberately not a verdict. */
export interface PriceComparison {
  item: string;
  size: string | null;
  shelf_price: number | null;
  purchases: {
    /** True when this cleared the measured match floor. False means "closest
     *  thing found", which is worth showing and not worth averaging. */
    confident: boolean;
    score: number;
    /** How much more (+) or less (-) the shelf price is per unit than this
     *  purchase. Null when the two sizes aren't in the same unit family. */
    vs_shelf_percent: number | null;
    description: string;
    date: string;
    merchant: string | null;
    paid_per_unit: number | null;
    quantity_unit: string | null;
    unit_price_display: string | null;
    was_on_offer: boolean;
    caveats: string[];
  }[];
  baseline: { typical: number; low: number; high: number; count: number } | null;
  /** The shelf price expressed per unit, e.g. "$3.49/gal". */
  shelf_unit_price: string | null;
  /** Best like-for-like number available when there is no confident baseline. */
  closest_comparable: {
    description: string;
    date: string;
    merchant: string | null;
    their_unit_price: string | null;
    percent: number;
    confident: boolean;
    /** 'paid' = a purchase of yours. 'seen' = a shelf price you photographed
     *  but did not buy — real evidence, but never call it something you paid. */
    kind: "paid" | "seen";
  } | null;
  comparison: {
    shelf_per_unit: number;
    typical_paid_per_unit: number;
    percent: number;
    based_on: number;
    /** The unit both figures are expressed in, e.g. "lb". */
    unit: string | null;
  } | null;
  /** Why the answer may be weak. Always shown — a comparison without its
   *  caveats is how a clearance price gets read as the going rate. */
  cautions: string[];
}

/**
 * Save a confirmed shelf price.
 *
 * Note what this is NOT: a transaction. Seeing a price is not spending money,
 * so observations live in their own table and never touch spending totals.
 */
export async function savePriceObservation(
  fileId: string,
  draft: PriceTagDraft & { tag_index?: number }
): Promise<PriceObservation> {
  log.info("Saving price observation", {
    fileId,
    item: draft.item_description,
    price: draft.item_subtotal_price,
  });
  return apiJson<PriceObservation>("/price-observations", {
    method: "POST",
    // bill_file_id, not source_bill_file_id — the server's schema ignores unknown
    // fields, so the wrong name silently saved every observation with no link
    // back to the photo it was read from.
    body: JSON.stringify({ ...draft, bill_file_id: fileId }),
    // Server-side matching against purchase history can take a moment.
    timeout: 30000,
  });
}
