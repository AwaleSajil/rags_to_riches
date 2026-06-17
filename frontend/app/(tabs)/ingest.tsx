import React, { useState } from "react";
import { StyleSheet, View, ScrollView, Platform } from "react-native";
import { Text, Button, Snackbar, ProgressBar, Badge } from "react-native-paper";
import * as DocumentPicker from "expo-document-picker";
import { useCameraPermissions } from "expo-camera";
import { FileListItem } from "../../src/components/FileListItem";
import { DeleteConfirmModal } from "../../src/components/DeleteConfirmModal";
import { CameraCapture } from "../../src/components/CameraCapture";
import { GlassCard } from "../../src/components/GlassCard";
import { UploadDropZone } from "../../src/components/UploadDropZone";
import { PickedFileChip } from "../../src/components/PickedFileChip";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { useFiles } from "../../src/hooks/useFiles";
import { colors, typography, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";
import type { FileItem } from "../../src/lib/types";

const log = createLogger("IngestScreen");

export default function IngestScreen() {
  const {
    files,
    isLoading,
    isUploading,
    isIngesting,
    isDeleting,
    error,
    duplicates,
    ingestionProgress,
    uploadFiles,
    deleteFile,
    clearDuplicates,
  } = useFiles();

  const [pickedFiles, setPickedFiles] = useState<
    { uri: string; name: string; type: string }[]
  >([]);
  const [deleteTarget, setDeleteTarget] = useState<FileItem | null>(null);
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

  const handleCameraCapture = (photo: { uri: string; name: string; type: string }) => {
    log.info("Camera photo captured", { name: photo.name, type: photo.type });
    setPickedFiles((prev) => [...prev, photo]);
  };

  const handleRemovePickedFile = (index: number) => {
    setPickedFiles((prev) => prev.filter((_, i) => i !== index));
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
    if (!deleteTarget) return;
    log.info("Delete confirmed", {
      fileId: deleteTarget.id,
      filename: deleteTarget.filename,
      type: deleteTarget.type,
    });
    const ok = await deleteFile(deleteTarget.id, deleteTarget.type);
    if (ok) {
      log.info("File deleted successfully", { filename: deleteTarget.filename });
      setSnackbar({
        visible: true,
        message: `Deleted ${deleteTarget.filename}`,
        error: false,
      });
    }
    setDeleteTarget(null);
  };

  if (isLoading) {
    return <LoadingSpinner message="Loading files..." />;
  }

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
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
          files.map((file, index) => (
            <FileListItem
              key={file.id ?? `file-${index}`}
              file={file}
              onDelete={setDeleteTarget}
            />
          ))
        )}
      </ScrollView>

      {/* Delete Confirmation */}
      <DeleteConfirmModal
        visible={!!deleteTarget}
        filename={deleteTarget?.filename || ""}
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
