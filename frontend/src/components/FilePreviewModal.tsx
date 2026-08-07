import React from "react";
import {
  Modal,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
  View,
} from "react-native";
import { ActivityIndicator, Button, IconButton, Text } from "react-native-paper";
import { getSupabase } from "../lib/supabase";
import { colors, spacing, typography } from "../styles/theme";
import { createLogger } from "../lib/logger";
import type { FileItem } from "../lib/types";

import { ZoomableImage } from "./ZoomableImage";

const log = createLogger("FilePreviewModal");

interface FilePreviewModalProps {
  file: FileItem | null;
  onClose: () => void;
  onReviewReceipt?: (file: FileItem) => void;
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

export function FilePreviewModal({ file, onClose, onReviewReceipt }: FilePreviewModalProps) {
  const { width, height } = useWindowDimensions();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [imageUrl, setImageUrl] = React.useState<string | null>(null);
  const [rows, setRows] = React.useState<string[][] | null>(null);

  React.useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setImageUrl(null);
    setRows(null);

    (async () => {
      try {
        log.info("Loading preview", { file: file.filename, type: file.type });
        const supabase = await getSupabase();
        const { data, error: signErr } = await supabase.storage
          .from("money-rag-files")
          .createSignedUrl(file.s3_key, 300);
        if (signErr || !data?.signedUrl) {
          throw new Error(signErr?.message || "Could not get a link to the file");
        }
        if (cancelled) return;

        if (file.type === "csv") {
          const res = await fetch(data.signedUrl);
          const text = await res.text();
          if (cancelled) return;
          setRows(parseCsv(text));
        } else {
          setImageUrl(data.signedUrl);
        }
      } catch (e: any) {
        log.error("Preview load failed", e);
        if (!cancelled) setError(e.message || "Failed to load preview");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [file]);

  const header = rows?.[0];
  const body = rows?.slice(1);

  return (
    <Modal
      visible={!!file}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          <View style={styles.headerRow}>
            <Text style={styles.title} numberOfLines={1}>
              {file?.filename}
            </Text>
            <IconButton icon="close" size={24} onPress={onClose} />
          </View>

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

          {!loading && !error && imageUrl && (
            /* ScrollView's maximumZoomScale is iOS-only, so this did nothing at
               all on Android or web — the pinch simply had no effect. */
            <ZoomableImage
              uri={imageUrl}
              style={{ width: width * 0.9, height: height * 0.68 }}
            />
          )}

          {!loading && !error && rows && (
            <View style={{ maxHeight: height * 0.72 }}>
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

          {!loading && !error && file?.type === "bill" && (
            <View style={styles.receiptActions}>
              <Button
                mode="contained"
                icon="pencil-outline"
                onPress={() => onReviewReceipt?.(file)}
              >
                Review / edit extracted data
              </Button>
            </View>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: colors.background,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 24,
    maxHeight: "92%",
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingLeft: spacing.lg,
    paddingRight: spacing.xs,
    paddingTop: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.surfaceBorder,
  },
  title: {
    ...typography.subtitle2,
    color: colors.text,
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
  receiptActions: {
    paddingTop: spacing.sm,
    alignItems: "center",
  },
  errorText: {
    color: colors.error,
    textAlign: "center",
  },
  imageWrap: {
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.md,
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
