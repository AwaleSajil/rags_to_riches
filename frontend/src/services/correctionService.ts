import { apiJson } from "./api";
import { createLogger } from "../lib/logger";

const log = createLogger("CorrectionService");

/** A fix the agent proposed, awaiting the user's confirmation. */
export interface PendingCorrection {
  table: string;
  row_id: string;
  /** column -> new value */
  changes: Record<string, string | number | null>;
  /** column -> value stored right now, so the card can show the change. */
  current: Record<string, string | number | null>;
  /** column -> human label, e.g. size_value -> "Package size" */
  labels: Record<string, string>;
  reason?: string;
}

/**
 * Apply a correction the user confirmed.
 *
 * Goes out as a normal authenticated request, which is the whole point: the
 * agent proposes and cannot write, so the update happens on the user's own
 * request under their own permissions. There is deliberately no delete
 * counterpart.
 */
export async function applyCorrection(
  correction: PendingCorrection
): Promise<void> {
  log.info("Applying correction", {
    table: correction.table,
    rowId: correction.row_id,
    columns: Object.keys(correction.changes),
  });
  await apiJson("/corrections", {
    method: "POST",
    body: JSON.stringify({
      table: correction.table,
      row_id: correction.row_id,
      changes: correction.changes,
    }),
    timeout: 20000,
  });
}
