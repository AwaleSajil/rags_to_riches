import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { StyleSheet, View, SectionList, Platform, Pressable, RefreshControl } from "react-native";
import { Text, Button, Snackbar, ProgressBar, Badge, Searchbar, Chip } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as DocumentPicker from "expo-document-picker";
import { useCameraPermissions } from "expo-camera";
import { FileListItem } from "../../src/components/FileListItem";
import { FilePreviewModal } from "../../src/components/FilePreviewModal";
import { DeleteConfirmModal } from "../../src/components/DeleteConfirmModal";
import { CameraCapture } from "../../src/components/CameraCapture";
import { GlassCard } from "../../src/components/GlassCard";
import { UploadDropZone } from "../../src/components/UploadDropZone";
import { PickedFileChip } from "../../src/components/PickedFileChip";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { useFiles } from "../../src/hooks/useFiles";
import { colors, typography, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";
import { openReview } from "../../src/lib/reviewRouting";
import type { FileItem } from "../../src/lib/types";
import { useFocusEffect, useRouter } from "expo-router";

const log = createLogger("IngestScreen");

type FileFilter = "all" | "csv" | "bill";

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

// One shared empty array, so collapsing a month doesn't hand SectionList a new
// `data` identity every render and re-render the whole list.
const NO_ROWS: FileItem[] = [];

type MonthSection = {
  title: string;
  count: number;
  collapsed: boolean;
  data: FileItem[];
};

function monthLabel(dateStr?: string): string {
  if (!dateStr) return "Unknown date";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "Unknown date";
  return `${MONTH_NAMES[d.getMonth()]} ${d.getFullYear()}`;
}

export default function IngestScreen() {
  const router = useRouter();
  const {
    files,
    isLoading,
    isUploading,
    isIngesting,
    isDeleting,
    visibilityFileId,
    error,
    duplicates,
    reviewItems,
    ingestionProgress,
    uploadFiles,
    deleteFile,
    toggleFileVisibility,
    loadFiles,
    clearDuplicates,
  } = useFiles();

  // Reload whenever this tab comes into view. Tab screens stay MOUNTED when you
  // switch away, so useFiles' mount-time load runs exactly once — the first time
  // this tab is ever opened. A photo taken in Chat therefore never appeared here
  // until a manual pull-to-refresh, even though it had uploaded fine. Chat and
  // Transactions already did this; Files was the only tab that did not.
  //
  // Each screen holds its own useFiles() state, so Chat refreshing its copy does
  // nothing for this one.
  useFocusEffect(
    useCallback(() => {
      loadFiles();
    }, [loadFiles])
  );

  useEffect(() => {
    openReview(router, reviewItems);
  }, [reviewItems, router]);

  const [pickedFiles, setPickedFiles] = useState<
    { uri: string; name: string; type: string }[]
  >([]);
  const [deleteTarget, setDeleteTarget] = useState<FileItem | null>(null);
  const deleteInFlight = useRef(false);
  const [previewTarget, setPreviewTarget] = useState<FileItem | null>(null);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<FileFilter>("all");
  const [collapsedMonths, setCollapsedMonths] = useState<Record<string, boolean>>({});

  // Filter → sort newest-first → group into month buckets (already in newest order).
  const groupedFiles = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = files
      .filter((f) => typeFilter === "all" || f.type === typeFilter)
      .filter((f) => !q || f.filename.toLowerCase().includes(q))
      .sort((a, b) => (b.upload_date || "").localeCompare(a.upload_date || ""));

    const groups: { month: string; items: FileItem[] }[] = [];
    const index: Record<string, FileItem[]> = {};
    for (const f of filtered) {
      const label = monthLabel(f.upload_date);
      if (!index[label]) {
        index[label] = [];
        groups.push({ month: label, items: index[label] });
      }
      index[label].push(f);
    }
    return groups;
  }, [files, typeFilter, search]);

  // Collapsing swaps a section's rows for an empty array rather than rebuilding
  // the groups, so toggling a month never re-runs the filtering above.
  const sections = useMemo<MonthSection[]>(
    () =>
      groupedFiles.map((group) => ({
        title: group.month,
        count: group.items.length,
        collapsed: !!collapsedMonths[group.month],
        data: collapsedMonths[group.month] ? NO_ROWS : group.items,
      })),
    [groupedFiles, collapsedMonths]
  );

  const toggleMonth = useCallback(
    (month: string) => setCollapsedMonths((prev) => ({ ...prev, [month]: !prev[month] })),
    []
  );
  const [snackbar, setSnackbar] = useState({ visible: false, message: "", error: false });
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();

  const handlePickFiles = async () => {
    log.info("Opening document picker...");
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: [
          "text/csv",
          "text/comma-separated-values",
          "image/png",
          "image/jpeg",
          "application/vnd.ms-excel",
        ],
        multiple: true,
      });

      log.info("Document picker result", {
        canceled: result.canceled,
        assetCount: result.assets?.length || 0,
      });

      if (!result.canceled && result.assets) {
        const picked = result.assets.map((a) => ({
          uri: a.uri,
          name: a.name,
          type: a.mimeType || "application/octet-stream",
        }));
        log.info("Files picked", {
          files: picked.map((f) => ({ name: f.name, type: f.type })),
        });
        setPickedFiles((prev) => [...prev, ...picked]);
      }
    } catch (e: any) {
      log.error("Document picker error", e);
      setSnackbar({ visible: true, message: e.message, error: true });
    }
  };

  const handleOpenCamera = async () => {
    log.info("Camera button pressed", {
      permissionGranted: cameraPermission?.granted,
    });
    if (!cameraPermission?.granted) {
      log.info("Requesting camera permission...");
      const result = await requestCameraPermission();
      log.info("Camera permission result", { granted: result.granted });
      if (!result.granted) {
        setSnackbar({
          visible: true,
          message: "Camera permission is required to capture receipts.",
          error: true,
        });
        return;
      }
    }
    setCameraOpen(true);
  };

  const handleCameraCapture = (
    photo: { uri: string; name: string; type: string },
    place?: string | null
  ) => {
    // The batch path has no per-photo location field, so a place captured here
    // is not carried through. Receipts get their address off the receipt itself,
    // which is better than a fix anyway; a shelf price wants the chat capture.
    log.info("Camera photo captured", { name: photo.name, type: photo.type, place: !!place });
    setPickedFiles((prev) => [...prev, photo]);
  };

  const handleRemovePickedFile = (index: number) => {
    setPickedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleReviewReceipt = (file: FileItem) => {
    setPreviewTarget(null);
    // Route on what the photo actually is. This used to open the receipt form
    // on whatever was tapped, so a price tag in this list led to a form that
    // would turn a shelf price into spending that never happened. Rows from
    // before migration 014 have no kind; treat those as receipts, which is what
    // they are — the column was added precisely because everything until then
    // was one.
    openReview(router, [{ file_id: file.id, kind: file.kind ?? "receipt" }]);
  };

  const handleIngest = async () => {
    if (pickedFiles.length === 0) return;
    log.info("Ingest button pressed", {
      fileCount: pickedFiles.length,
      names: pickedFiles.map((f) => f.name),
    });
    const ok = await uploadFiles(pickedFiles);
    if (ok) {
      log.info("Upload successful, ingestion processing in background");
      setPickedFiles([]);
      setSnackbar({
        visible: true,
        message: "Files uploaded! Ingestion processing in background...",
        error: false,
      });
    } else {
      log.error("Ingest failed", { error });
      setSnackbar({ visible: true, message: error || "Upload failed", error: true });
    }
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget || deleteInFlight.current) return;
    deleteInFlight.current = true;
    const target = deleteTarget;
    // Close immediately so a double tap cannot issue a second DELETE request.
    setDeleteTarget(null);
    log.info("Delete confirmed", {
      fileId: target.id,
      filename: target.filename,
      type: target.type,
    });
    try {
      const ok = await deleteFile(target.id, target.type);
      if (ok) {
        log.info("File deleted successfully", { filename: target.filename });
        setSnackbar({
          visible: true,
          message: `Deleted ${target.filename}`,
          error: false,
        });
      }
    } finally {
      deleteInFlight.current = false;
    }
  };

  const keyExtractor = useCallback(
    (file: FileItem, index: number) => file.id ?? `file-${index}`,
    []
  );

  const renderItem = useCallback(
    ({ item }: { item: FileItem }) => (
      <FileListItem
        file={item}
        onDelete={setDeleteTarget}
        onPress={setPreviewTarget}
        onToggleVisibility={toggleFileVisibility}
        isTogglingVisibility={visibilityFileId === item.id}
      />
    ),
    [toggleFileVisibility, visibilityFileId]
  );

  const renderSectionHeader = useCallback(
    ({ section }: { section: MonthSection }) => (
      <Pressable style={styles.monthHeader} onPress={() => toggleMonth(section.title)}>
        <MaterialCommunityIcons
          name={section.collapsed ? "chevron-right" : "chevron-down"}
          size={22}
          color={colors.textSecondary}
        />
        <Text style={styles.monthTitle}>{section.title}</Text>
        <Badge style={styles.monthBadge}>{section.count}</Badge>
      </Pressable>
    ),
    [toggleMonth]
  );

  // Replaces the `marginBottom` the wrapping <View> used to give each group.
  const renderSectionFooter = useCallback(() => <View style={styles.sectionGap} />, []);

  // Only on the FIRST load. Returning the spinner on every isLoading would tear
  // the whole screen down mid-pull the moment a refresh started, taking the
  // RefreshControl's own spinner and the gesture with it.
  if (isLoading && files.length === 0) {
    return <LoadingSpinner message="Loading files..." />;
  }

  // Passed as an ELEMENT, not a function. An inline `() => <.../>` is a new
  // component type on every render, so SectionList would unmount and remount
  // this whole block — and the Searchbar would lose focus on every keystroke.
  const listHeader = (
    <>
      {/* Section 1: Upload Drop Zone */}
      <UploadDropZone
        onPickFiles={handlePickFiles}
        onOpenCamera={handleOpenCamera}
      />

      {/* Section 2: Picked Files */}
      {pickedFiles.length > 0 && (
        <GlassCard variant="elevated" style={styles.pickedSection}>
          <Text style={styles.pickedTitle}>
            {pickedFiles.length} file{pickedFiles.length > 1 ? "s" : ""} selected
          </Text>
          <View style={styles.chipRow}>
            {pickedFiles.map((f, i) => (
              <PickedFileChip
                key={`${f.name}-${i}`}
                name={f.name}
                type={f.type}
                onRemove={() => handleRemovePickedFile(i)}
              />
            ))}
          </View>
          <Button
            mode="contained"
            icon="upload"
            onPress={handleIngest}
            loading={isUploading}
            disabled={isUploading}
            style={styles.ingestButton}
            labelStyle={styles.ingestButtonLabel}
          >
            {isUploading ? "Uploading..." : `Ingest ${pickedFiles.length} File${pickedFiles.length > 1 ? "s" : ""}`}
          </Button>
        </GlassCard>
      )}

      {/* Section 3: Ingestion Progress */}
      {isIngesting && (
        <GlassCard style={styles.progressSection}>
          <Text style={styles.progressText}>
            {ingestionProgress?.stage === "parsing" && "📄 Parsing CSV..."}
            {ingestionProgress?.stage === "enriching" && "✨ Enriching merchants..."}
            {ingestionProgress?.stage === "saving" && "💾 Saving to database..."}
            {ingestionProgress?.stage === "embedding" && "🧠 Building search index..."}
            {!ingestionProgress?.stage && "Processing your files..."}
          </Text>
          <ProgressBar
            indeterminate={
              !ingestionProgress?.stage ||
              ingestionProgress.total === 0
            }
            progress={
              ingestionProgress && ingestionProgress.total > 0
                ? ingestionProgress.done / ingestionProgress.total
                : 0
            }
            color={colors.primary}
            style={styles.progressBar}
          />
          <Text style={styles.progressSubtext}>
            {ingestionProgress && ingestionProgress.total > 0
              ? `${ingestionProgress.done} / ${ingestionProgress.total}${ingestionProgress.detail ? ` — ${ingestionProgress.detail}` : ""}`
              : "Starting up..."}
          </Text>
        </GlassCard>
      )}

      {/* Section 4: Duplicate Warnings */}
      {duplicates.length > 0 && (
        <GlassCard variant="flat" style={styles.duplicateCard}>
          <Text style={styles.duplicateTitle}>
            {duplicates.length} duplicate transaction{duplicates.length > 1 ? "s" : ""} detected
          </Text>
          <Text style={styles.duplicateSubtitle}>
            These were merged/skipped during ingestion:
          </Text>
          {duplicates.slice(0, 10).map((d, i) => (
            <Text key={i} style={styles.duplicateItem}>
              {d.date} - {d.merchant} - ${Number(d.amount).toFixed(2)}
            </Text>
          ))}
          {duplicates.length > 10 && (
            <Text style={styles.duplicateSubtitle}>
              ...and {duplicates.length - 10} more
            </Text>
          )}
          <Button
            mode="text"
            onPress={clearDuplicates}
            compact
            style={styles.dismissButton}
          >
            Dismiss
          </Button>
        </GlassCard>
      )}

      {/* Section 5: Your Files */}
      <View style={styles.fileListHeader}>
        <View style={styles.fileListTitleRow}>
          <Text style={styles.fileListTitle}>Your Files</Text>
          {files.length > 0 && (
            <Badge style={styles.fileCount}>{files.length}</Badge>
          )}
        </View>
        {files.length > 0 && (
          <Text style={styles.fileListSubtitle}>
            {files.filter(f => f.type === "csv").length} CSV, {files.filter(f => f.type === "bill").length} receipt{files.filter(f => f.type === "bill").length !== 1 ? "s" : ""}
          </Text>
        )}
      </View>

      {files.length === 0 ? (
        <Text style={styles.emptyText}>No files uploaded yet.</Text>
      ) : (
        <>
          {/* Search */}
          <Searchbar
            placeholder="Search files"
            value={search}
            onChangeText={setSearch}
            style={styles.searchbar}
            inputStyle={styles.searchInput}
            icon="magnify"
          />

          {/* Type filter chips */}
          <View style={styles.filterRow}>
            {([
              { key: "all", label: "All" },
              { key: "csv", label: "CSVs" },
              { key: "bill", label: "Receipts" },
            ] as { key: FileFilter; label: string }[]).map((chip) => (
              <Chip
                key={chip.key}
                selected={typeFilter === chip.key}
                onPress={() => setTypeFilter(chip.key)}
                style={[styles.filterChip, typeFilter === chip.key && styles.filterChipActive]}
                showSelectedCheck={false}
                compact
              >
                {chip.label}
              </Chip>
            ))}
          </View>

          {/* Not SectionList's ListEmptyComponent: that also fires when every
              month happens to be collapsed. */}
          {sections.length === 0 && (
            <Text style={styles.emptyText}>No files match your search.</Text>
          )}
        </>
      )}
    </>
  );

  return (
    <View style={styles.container}>
      {/* Collapsible month groups. Virtualized, so a long upload history mounts
          only the rows on screen instead of every row at once. */}
      <SectionList
        sections={sections}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        renderSectionHeader={renderSectionHeader}
        renderSectionFooter={renderSectionFooter}
        ListHeaderComponent={listHeader}
        contentContainerStyle={styles.scrollContent}
        // The month headers have no background of their own, so sticking them
        // would let rows scroll through the text.
        stickySectionHeadersEnabled={false}
        keyboardShouldPersistTaps="handled"
        initialNumToRender={12}
        maxToRenderPerBatch={12}
        windowSize={11}
        refreshControl={
          <RefreshControl refreshing={isLoading} onRefresh={loadFiles} tintColor={colors.primary} />
        }
      />

      {/* File Preview (receipt image or CSV table) */}
      <FilePreviewModal
        file={previewTarget}
        onClose={() => setPreviewTarget(null)}
        onReviewReceipt={handleReviewReceipt}
      />

      {/* Delete Confirmation */}
      <DeleteConfirmModal
        visible={!!deleteTarget}
        filename={deleteTarget?.filename || ""}
        loading={isDeleting}
        onConfirm={handleDeleteConfirm}
        onDismiss={() => setDeleteTarget(null)}
      />

      {/* Camera Modal */}
      {Platform.OS !== "web" && (
        <CameraCapture
          visible={cameraOpen}
          onClose={() => setCameraOpen(false)}
          onCapture={handleCameraCapture}
        />
      )}

      <Snackbar
        visible={snackbar.visible}
        onDismiss={() => setSnackbar({ ...snackbar, visible: false })}
        duration={4000}
        style={{
          backgroundColor: snackbar.error ? colors.error : colors.success,
        }}
      >
        {snackbar.message}
      </Snackbar>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scrollContent: {
    padding: spacing.lg,
    paddingBottom: 40,
  },
  pickedSection: {
    marginTop: spacing.lg,
  },
  pickedTitle: {
    ...typography.subtitle2,
    color: colors.text,
    marginBottom: spacing.md,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
  },
  ingestButton: {
    borderRadius: 10,
    backgroundColor: colors.primary,
    marginTop: spacing.md,
  },
  ingestButtonLabel: {
    fontWeight: "600",
    paddingVertical: 2,
  },
  progressSection: {
    marginTop: spacing.lg,
  },
  progressText: {
    ...typography.subtitle2,
    color: colors.primary,
    marginBottom: spacing.md,
  },
  progressBar: {
    borderRadius: 4,
    height: 4,
  },
  progressSubtext: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.sm,
  },
  duplicateCard: {
    marginTop: spacing.lg,
    borderLeftWidth: 3,
    borderLeftColor: colors.warning,
  },
  duplicateTitle: {
    ...typography.subtitle2,
    color: "#b45309",
    marginBottom: spacing.xs,
  },
  duplicateSubtitle: {
    ...typography.caption,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
  },
  duplicateItem: {
    ...typography.caption,
    color: colors.text,
    paddingLeft: spacing.sm,
    paddingVertical: 2,
  },
  dismissButton: {
    alignSelf: "flex-start",
    marginTop: spacing.xs,
  },
  searchbar: {
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    borderRadius: 12,
    elevation: 0,
  },
  searchInput: {
    fontSize: 14,
    minHeight: 0,
  },
  filterRow: {
    flexDirection: "row",
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  filterChip: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  filterChipActive: {
    backgroundColor: colors.primaryLight,
    borderColor: colors.primary,
  },
  sectionGap: {
    height: spacing.sm,
  },
  monthHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.sm,
    gap: spacing.xs,
  },
  monthTitle: {
    ...typography.subtitle2,
    color: colors.textSecondary,
    flex: 1,
  },
  monthBadge: {
    backgroundColor: colors.textTertiary,
  },
  fileListHeader: {
    marginTop: spacing.xxl,
    marginBottom: spacing.lg,
  },
  fileListTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  fileListTitle: {
    ...typography.h3,
    color: colors.text,
  },
  fileCount: {
    backgroundColor: colors.primary,
  },
  fileListSubtitle: {
    ...typography.caption,
    color: colors.textTertiary,
    marginTop: spacing.xs,
  },
  emptyText: {
    ...typography.body2,
    color: colors.textSecondary,
  },
});
