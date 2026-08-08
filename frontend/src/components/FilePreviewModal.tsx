/**
 * Previewing a stored file: a receipt photo, or a CSV.
 *
 * This layer is all data — signing a URL, reading the CSV, working out which
 * file comes next. The sheet, header and paging come from PreviewSheet, and a
 * photo is handed to PhotoViewer, which the chat attachments use too, so the
 * same receipt looks and behaves the same wherever it was tapped.
 */

import React from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Button, IconButton, Text } from "react-native-paper";
import { getSupabase } from "../lib/supabase";
import { colors, spacing, typography } from "../styles/theme";
import { createLogger } from "../lib/logger";
import { formatDate } from "../lib/format";
import { saveToDevice } from "../lib/download";
import { fileVisual } from "../lib/sourceVisual";
import type { FileItem } from "../lib/types";

import { DeleteConfirmModal } from "./DeleteConfirmModal";
import { PhotoViewer } from "./PhotoViewer";
import { PreviewSheet } from "./PreviewSheet";

const log = createLogger("FilePreviewModal");

const BUCKET = "money-rag-files";
/** Signed-URL lifetime. Long enough to read a receipt, short enough to matter. */
const URL_TTL_SECONDS = 300;

interface FilePreviewModalProps {
  file: FileItem | null;
  onClose: () => void;
  onReviewReceipt?: (file: FileItem) => void;
  /**
   * The files this one sits among, in display order. Supplying it turns on
   * previous/next, so a user can page through a month of receipts without
   * closing the sheet each time. Omit it (as the transaction screen does, where
   * there is exactly one receipt) and the controls are hidden entirely.
   */
  siblings?: FileItem[];
  /** Required for navigation to do anything — hands back the file to show. */
  onNavigate?: (file: FileItem) => void;
  /**
   * Delete this file. Called only AFTER the user confirms — the confirmation
   * is raised here, inside the sheet, so it appears over the picture being
   * deleted rather than replacing it.
   *
   * The caller still owns the deletion, because what a delete MEANS differs by
   * where you are: from the Files tab it removes a file, but from a transaction
   * the cascade takes the transaction and its line items too. Omit it and no
   * delete is offered.
   */
  onDelete?: (file: FileItem) => void;
  /** Hide or unhide this file's transactions. Omit to hide the control. */
  onToggleVisibility?: (file: FileItem) => void;
  /** True while a visibility change is in flight for this file. */
  togglingVisibility?: boolean;
  /**
   * Persist how far this photo must be turned to read it. Called when the user
   * settles on an angle. Omit and rotation still works, just for this viewing.
   */
  onRotate?: (file: FileItem, degrees: number) => void;
}

/**
 * Storage says the object is not there.
 *
 * The database row and the stored file can come apart: an upload that failed
 * after the row was written, a bucket cleared in development, a restore that
 * brought back the tables but not the objects. The row is real, the picture is
 * not.
 */
function isMissingObject(e: any): boolean {
  const message = String(e?.message ?? e).toLowerCase();
  return message.includes("not found") || message.includes("does not exist");
}

function describeLoadFailure(e: any, type: string): string {
  const noun = type === "csv" ? "file" : "photo";
  if (isMissingObject(e)) {
    return (
      `This ${noun} is no longer in storage, though its record is still here. ` +
      `Nothing else is affected — you can delete the record below.`
    );
  }
  return e?.message || `Could not load that ${noun}.`;
}

// Minimal CSV parser that respects double-quoted fields (which may contain commas).
function parseCsv(text: string, maxRows = 100): string[][] {
  const rows: string[][] = [];
  const lines = text.split(/\r\n|\n|\r/).filter((l) => l.length > 0);
  for (const line of lines.slice(0, maxRows)) {
    const cells: string[] = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (inQuotes) {
        if (ch === '"') {
          if (line[i + 1] === '"') { cur += '"'; i++; }
          else inQuotes = false;
        } else cur += ch;
      } else if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        cells.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
    cells.push(cur);
    rows.push(cells);
  }
  return rows;
}

export function FilePreviewModal({
  file,
  onClose,
  onReviewReceipt,
  siblings,
  onNavigate,
  onDelete,
  onToggleVisibility,
  togglingVisibility = false,
  onRotate,
}: FilePreviewModalProps) {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [imageUrl, setImageUrl] = React.useState<string | null>(null);
  const [rows, setRows] = React.useState<string[][] | null>(null);
  const [downloading, setDownloading] = React.useState(false);
  const [notice, setNotice] = React.useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = React.useState(false);

  // What the fetch below ACTUALLY depends on. Watching the whole `file` object
  // meant any change to it re-ran this: hiding a file replaces the object to
  // flip `is_hidden`, so the URL was re-signed and the picture reloaded from
  // scratch — the sheet appeared to restart on a tap that has nothing to do
  // with the image. These three are what identifies the bytes.
  const fileId = file?.id ?? null;
  const s3Key = file?.s3_key ?? null;
  const isCsv = file?.type === "csv";

  React.useEffect(() => {
    if (!fileId || !s3Key) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setImageUrl(null);
    setRows(null);
    setNotice(null);
    // Paging away mid-decision must not leave a confirmation pointed at the
    // file you just moved on from.
    setConfirmingDelete(false);

    (async () => {
      try {
        log.info("Loading preview", { s3Key, isCsv });
        const supabase = await getSupabase();
        const { data, error: signErr } = await supabase.storage
          .from(BUCKET)
          .createSignedUrl(s3Key, URL_TTL_SECONDS);
        if (signErr || !data?.signedUrl) {
          throw new Error(signErr?.message || "Could not get a link to the file");
        }
        if (cancelled) return;

        if (isCsv) {
          const res = await fetch(data.signedUrl);
          const text = await res.text();
          if (cancelled) return;
          setRows(parseCsv(text));
        } else {
          setImageUrl(data.signedUrl);
        }
      } catch (e: any) {
        const message = describeLoadFailure(e, isCsv ? "csv" : "bill");
        // A missing object is a data state, not a fault: the row outlived its
        // file. Logged as a warning so it does not sit in the console looking
        // like a crash every time an old record is opened.
        if (isMissingObject(e)) {
          log.warn("Stored file is gone", { s3Key });
        } else {
          log.error("Preview load failed", e);
        }
        if (!cancelled) setError(message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [fileId, s3Key, isCsv]);

  /**
   * Save the original to the device.
   *
   * A fresh signed URL is minted with Supabase's `download` option, which sets
   * Content-Disposition so the browser saves rather than renders it. The URL
   * used for the preview is deliberately not reused — it may be minutes old by
   * the time someone taps this, and an expired link fails as a blank page.
   */
  const handleDownload = async () => {
    if (!file || downloading) return;
    setDownloading(true);
    setNotice(null);
    try {
      const supabase = await getSupabase();
      const { data, error: signErr } = await supabase.storage
        .from(BUCKET)
        .createSignedUrl(file.s3_key, URL_TTL_SECONDS, { download: file.filename });
      if (signErr || !data?.signedUrl) {
        throw new Error(signErr?.message || "Could not get a download link");
      }
      await saveToDevice(data.signedUrl, file.filename);
      setNotice("Saved to your downloads.");
    } catch (e: any) {
      log.error("Download failed", e);
      setNotice(e.message || "Could not download that file.");
    } finally {
      setDownloading(false);
    }
  };

  // --- previous / next ---------------------------------------------------------
  const list = siblings ?? [];
  const index = file ? list.findIndex((f) => f.id === file.id) : -1;
  const canNavigate = Boolean(onNavigate) && index >= 0 && list.length > 1;
  const previous = canNavigate && index > 0 ? list[index - 1] : null;
  const next = canNavigate && index < list.length - 1 ? list[index + 1] : null;

  const paging = canNavigate
    ? {
        onPrevious: previous ? () => onNavigate!(previous) : undefined,
        onNext: next ? () => onNavigate!(next) : undefined,
        position: { index, total: list.length },
      }
    : {};

  const visual = file ? fileVisual(file) : undefined;
  const subtitle = [
    visual?.label,
    file?.upload_date ? formatDate(file.upload_date.slice(0, 10)) : null,
    file?.is_verified === false ? "Needs review" : null,
    // Worth a word, not just an icon: hidden means this file's spending is in
    // no total, which is a surprising thing to discover later from a number
    // that does not add up.
    file?.is_hidden ? "Hidden" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  /**
   * Actions on the record, under the content.
   *
   * ONE row, not two. Every pixel here is taken from the picture, which is what
   * the user opened this to look at, so the primary action sits between the two
   * secondary ones rather than stacked above them — half the height for the
   * same three controls.
   *
   * Not overlaid on the image, unlike the paging arrows: those are edge-anchored
   * and you tap them repeatedly, whereas a red Delete floating over the middle
   * of a receipt covers the text someone is trying to read.
   *
   * Not in the header either — that row is view controls and close, and an
   * irreversible delete one tap from the X is a mis-tap waiting to happen.
   * These also need words: a bare eye icon does not say what it hides, and
   * "Hide" versus "Unhide" is the whole state.
   */
  const footerActions =
    file && !loading ? (
      <View style={styles.actionRow}>
        {onToggleVisibility ? (
          // The glyph shows the STATE, not the action: a slashed eye means
          // this file IS hidden. Drawn the other way round — slashed for "tap
          // to hide" — every ordinary file wore a crossed-out eye and the whole
          // list looked hidden, while the same file's icon in the row behind
          // said the opposite. Identical to FileListItem, colour included; the
          // two are the same control and must not look like different ones.
          <IconButton
            icon={file.is_hidden ? "eye-off-outline" : "eye-outline"}
            size={22}
            iconColor={colors.textSecondary}
            loading={togglingVisibility}
            disabled={togglingVisibility}
            onPress={() => onToggleVisibility(file)}
            accessibilityLabel={
              file.is_hidden
                ? "Unhide this file's transactions"
                : "Hide this file's transactions"
            }
          />
        ) : null}
        {!error && file.type === "bill" && onReviewReceipt ? (
          // Takes the whole middle. It is the one thing most people came here
          // to do, and the two icons flanking it need no more than their own
          // width.
          <Button
            mode="contained"
            icon="pencil-outline"
            style={styles.reviewButton}
            onPress={() => onReviewReceipt(file)}
          >
            Review / edit
          </Button>
        ) : null}
        {onDelete ? (
          <IconButton
            icon="trash-can-outline"
            size={22}
            iconColor={colors.error}
            onPress={() => setConfirmingDelete(true)}
            accessibilityLabel="Delete this file"
          />
        ) : null}
      </View>
    ) : null;

  // Rendered INSIDE the sheet (see PreviewSheet's Portal.Host) so it sits over
  // the picture being deleted. Reuses the same confirmation the file list uses
  // rather than growing a second one that could drift from it.
  const deleteConfirmation =
    file && onDelete ? (
      <DeleteConfirmModal
        visible={confirmingDelete}
        filename={file.filename}
        onDismiss={() => setConfirmingDelete(false)}
        onConfirm={() => {
          setConfirmingDelete(false);
          onDelete(file);
        }}
      />
    ) : null;

  // Branches on the file's TYPE, not on whether its URL has arrived. Keying
  // this on `imageUrl` meant every tap of Next — which nulls the URL while the
  // next one is signed — swapped PhotoViewer's Modal for PreviewSheet's and
  // back, so the sheet slid off the bottom of the screen and returned twice per
  // page. Type is stable across navigation between photos, so the Modal stays
  // mounted and only the picture changes.
  if (file && file.type !== "csv") {
    return (
      <PhotoViewer
        visible
        uri={imageUrl}
        loading={loading}
        error={error}
        initialRotation={file.rotation ?? 0}
        onRotationChange={onRotate ? (degrees) => onRotate(file, degrees) : undefined}
        onClose={onClose}
        title={file.filename}
        subtitle={subtitle}
        icon={visual}
        onDownload={handleDownload}
        downloading={downloading}
        notice={notice}
        footer={footerActions}
        {...paging}
      >
        {deleteConfirmation}
      </PhotoViewer>
    );
  }

  // CSVs, and the empty state when nothing is selected.
  const header = rows?.[0];
  const body = rows?.slice(1);

  return (
    <PreviewSheet
      visible={!!file}
      onClose={onClose}
      title={file?.filename ?? ""}
      subtitle={subtitle}
      icon={visual}
      notice={notice}
      footer={footerActions}
      {...paging}
    >
      {loading && (
        <View style={styles.center}>
          <ActivityIndicator color={colors.primary} />
          <Text style={styles.hint}>Loading preview…</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {!loading && !error && rows && (
        /* Fills the sheet rather than a fixed fraction of the screen, so a long
           statement shows as many rows as there is room for. */
        <View style={styles.table}>
          <ScrollView>
            <ScrollView horizontal showsHorizontalScrollIndicator>
              <View>
                {header && (
                  <View style={[styles.tr, styles.headerTr]}>
                    {header.map((c, i) => (
                      <Text key={i} style={[styles.cell, styles.headerCell]} numberOfLines={1}>
                        {c}
                      </Text>
                    ))}
                  </View>
                )}
                {body?.map((r, ri) => (
                  <View key={ri} style={[styles.tr, ri % 2 ? styles.trAlt : null]}>
                    {r.map((c, ci) => (
                      <Text key={ci} style={styles.cell} numberOfLines={1}>
                        {c}
                      </Text>
                    ))}
                  </View>
                ))}
              </View>
            </ScrollView>
          </ScrollView>
          {body && (
            <Text style={styles.hint}>
              {body.length} row{body.length !== 1 ? "s" : ""} shown (first 100)
            </Text>
          )}
        </View>
      )}
      {deleteConfirmation}
    </PreviewSheet>
  );
}


const styles = StyleSheet.create({
  actionRow: {
    flexDirection: "row",
    alignItems: "center",
    // Pushes the two icons to the edges when there is no Review between them,
    // which is the CSV case.
    justifyContent: "space-between",
    gap: spacing.xs,
  },
  reviewButton: {
    flex: 1,
  },
  table: {
    flex: 1,
  },
  center: {
    padding: 40,
    alignItems: "center",
    gap: 8,
  },
  hint: {
    ...typography.caption,
    color: colors.textTertiary,
    textAlign: "center",
    padding: spacing.sm,
  },
  errorText: {
    color: colors.error,
    textAlign: "center",
  },
  tr: {
    flexDirection: "row",
  },
  trAlt: {
    backgroundColor: colors.surface,
  },
  headerTr: {
    borderBottomWidth: 1,
    borderBottomColor: colors.surfaceBorder,
    backgroundColor: colors.surface,
  },
  cell: {
    width: 120,
    paddingVertical: 8,
    paddingHorizontal: 10,
    fontSize: 13,
    color: colors.text,
  },
  headerCell: {
    fontWeight: "700",
    color: colors.textSecondary,
  },
});
