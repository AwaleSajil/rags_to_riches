import { Platform } from "react-native";
import * as FileSystem from "expo-file-system/legacy";
import { apiFetch, getAccessToken, API_URL } from "./api";
import { createLogger } from "../lib/logger";
import { compressImage } from "../lib/compressImage";
import type { CaptureKind } from "./fileService";

const log = createLogger("CaptureService");

/** What the vision pass read off a shelf/price tag. */
export interface PriceTagDraft {
  item_description?: string;
  brand_name?: string | null;

  /**
   * Shaped like a receipt line, because the two get compared. A tag reading
   * "$4.29 / 12 OZ" is quantity 12, unit 'oz', unit_quantity_subtotal 0.3575,
   * item_subtotal_price 4.29.
   */
  /** The package the tag prices: "$4.29 / 12 OZ" is size_value 12, size_unit
   *  'oz'. Named for what it holds — nothing is bought from a shelf, so there
   *  was never a count to store here. */
  size_value?: number | null;
  size_unit?: string | null;
  /** The per-unit price the tag PRINTS. Never recomputed when present — the
   *  store did the pack-size arithmetic for the package on the shelf. */
  unit_quantity_subtotal?: number | null;
  /** What that printed figure is PER — regularly not the package unit. A
   *  one-gallon jug at $3.49 is commonly tagged "$0.87 PER QUART". */
  unit_price_unit?: string | null;
  /** What the shopper pays for the package. */
  item_subtotal_price?: number | null;

  merchant_name?: string | null;

  /**
   * Everything the photo shows that isn't a number, in the tag's own words:
   * "2 for $5 with card", "CLEARANCE", "Sale ends 8/15", "best before 08/05".
   * Deliberately not parsed into flags and dates — a model reading a shelf sign
   * is guessing at structure, and a wrong end date turns a limited offer into
   * the item's normal price.
   */
  item_qualitative_description?: string | null;

  /** Where the photo was taken, resolved to a name on the device. Attached at
   *  confirm time from the capture, not read off the tag. */
  location?: string | null;

  // Filled in server-side by capture_service.enrich_price_tag_draft.
  unit_price_display?: string | null;
  /** True when the tag gave no size; the comparison is same-item-only. */
  size_unknown?: boolean;
}

export interface CaptureResult {
  file_id: string;
  /** "processing" while the vision pass is still reading the photo. */
  kind: CaptureKind | "processing";
  draft: PriceTagDraft & Record<string, any>;
  /** An already-resolved place name, e.g. "Main St, Norwalk". */
  location?: string | null;
  /** Set when extraction failed but the photo was stored; user classifies by hand. */
  error?: string;
}

/**
 * Where the photo was taken, resolved to a name ON THE DEVICE.
 *
 * A GPS fix plus reverse geocoding happens here; only the resulting label is
 * sent. No coordinate reaches the server or the database — the store-location
 * table that used to hold them was dropped.
 */
export type CaptureLocation = string;

/**
 * Upload one photo and get back what it is plus the extracted draft.
 *
 * Classification and extraction run inline on the server (one vision call), so
 * this is slower than a normal request — the caller shows a typing indicator
 * rather than a spinner, because it reads as the assistant thinking.
 */
export async function capturePhoto(
  source: { uri: string; name: string; type: string },
  location?: CaptureLocation
): Promise<CaptureResult> {
  // Shrunk before it leaves the phone. The extractor is unaffected — see the
  // size chosen in compressImage — and a shelf photo taken on cellular is the
  // slowest upload in the app, with the user standing in the aisle waiting.
  const photo = await compressImage(source);

  log.info("Capturing photo", { name: photo.name, located: !!location });
  const stored = Platform.OS === "web"
    ? await captureWeb(photo, location)
    : await captureNative(photo, location);

  // The upload returns as soon as the photo is saved; reading it takes another
  // ~15s server-side. Waiting for that inside the upload request is what broke
  // this on mobile — the phone dropped the connection long before the reply,
  // so the work completed and the answer reached nobody.
  if (stored.kind !== "processing") return stored;
  return { ...(await waitForCapture(stored.file_id)), location: stored.location };
}

const POLL_INTERVAL_MS = 1500;
// ~60s. Beyond this the vision call has almost certainly failed, and asking
// "receipt or price tag?" beats spinning forever.
const MAX_POLLS = 40;

/** Poll until the photo has been read, or give up and let the user classify it. */
async function waitForCapture(fileId: string): Promise<CaptureResult> {
  for (let attempt = 0; attempt < MAX_POLLS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    try {
      const result = await getCapture(fileId);
      if (result.kind !== "processing") {
        log.info("Capture ready", { fileId, kind: result.kind, attempt });
        return result;
      }
    } catch (e) {
      // A dropped poll is not a failed capture — the next one may well succeed.
      log.warn("Capture poll failed, retrying", { fileId, attempt });
    }
  }
  log.warn("Capture never finished reading", { fileId });
  return { file_id: fileId, kind: "unknown", draft: {} };
}

// Vision extraction is a live LLM call; allow well past a normal request.
const CAPTURE_TIMEOUT_MS = 90000;

async function captureWeb(
  photo: { uri: string; name: string; type: string },
  location?: CaptureLocation
): Promise<CaptureResult> {
  const body = new FormData();
  const blob = await (await fetch(photo.uri)).blob();
  body.append("file", new File([blob], photo.name, { type: photo.type }));
  if (location) body.append("location", location);

  const res = await apiFetch("/captures", { method: "POST", body, timeout: CAPTURE_TIMEOUT_MS });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Capture failed: HTTP ${res.status}`);
  }
  return res.json();
}

async function captureNative(
  photo: { uri: string; name: string; type: string },
  location?: CaptureLocation
): Promise<CaptureResult> {
  // React Native's fetch fails multipart file bodies with "Network request
  // failed", so uploads go through expo-file-system — same reason as
  // fileService.uploadFilesNative.
  const token = await getAccessToken();
  const parameters: Record<string, string> = {};
  if (location) parameters.location = location;

  // uploadAsync takes no timeout of its own, so a stalled connection leaves this
  // promise pending forever: the photo bubble sits there with a typing indicator
  // and no card ever arrives, which is indistinguishable from the app doing
  // nothing. Classifying a photo is a live vision call and legitimately takes
  // ~15s, so the bound is generous — it exists to guarantee an ANSWER, not to
  // cut the work short.
  const upload = FileSystem.uploadAsync(`${API_URL}/captures`, photo.uri, {
    httpMethod: "POST",
    uploadType: FileSystem.FileSystemUploadType.MULTIPART,
    fieldName: "file",
    mimeType: photo.type,
    parameters,
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });

  let timer: ReturnType<typeof setTimeout> | undefined;
  const res = await Promise.race([
    upload,
    new Promise<never>((_, reject) => {
      timer = setTimeout(
        () => reject(new Error("Reading that photo took too long. It may still have saved — check the Files tab.")),
        CAPTURE_TIMEOUT_MS
      );
    }),
  ]).finally(() => clearTimeout(timer));

  log.info("Capture upload finished (native)", { status: res.status });

  if (res.status < 200 || res.status >= 300) {
    let detail = `Capture failed: HTTP ${res.status}`;
    try {
      detail = JSON.parse(res.body).detail || detail;
    } catch {
      /* non-JSON body */
    }
    log.error("Capture failed (native)", { status: res.status });
    throw new Error(detail);
  }
  return JSON.parse(res.body);
}

/**
 * Re-open a photo that was read earlier.
 *
 * The batch upload path reads photos in a subprocess and reports back only a
 * file id, so a price tag from the Files tab needs its draft fetched before the
 * confirm card can be shown.
 */
export async function getCapture(fileId: string): Promise<CaptureResult> {
  log.info("Loading stored capture", { fileId });
  const res = await apiFetch(`/captures/${fileId}`, { method: "GET", timeout: 15000 });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Throw away a captured photo.
 *
 * Goes to /captures rather than /files because an unconfirmed capture has no
 * BillFile row to delete — it was never stored. The route handles both: drop
 * what is held, or delete the file once it exists.
 */
export async function discardCapture(fileId: string): Promise<void> {
  log.info("Discarding capture", { fileId });
  const res = await apiFetch(`/captures/${fileId}`, { method: "DELETE", timeout: 15000 });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
}

/**
 * Record the user's answer for a photo the model could not classify.
 *
 * A deliberate user decision rather than a lower confidence threshold: a price
 * tag filed as a receipt invents spending that never happened.
 */
export async function setCaptureKind(
  fileId: string,
  kind: CaptureKind
): Promise<CaptureResult> {
  const body = new FormData();
  body.append("kind", kind);
  const res = await apiFetch(`/captures/${fileId}/kind`, {
    method: "POST",
    body,
    timeout: 15000,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
