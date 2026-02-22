import React, { useState } from "react";
import { StyleSheet, View, FlatList, ScrollView, Platform } from "react-native";
import { Text, Button, Snackbar, Divider } from "react-native-paper";
import * as DocumentPicker from "expo-document-picker";
import { useCameraPermissions } from "expo-camera";
import { FileListItem } from "../../src/components/FileListItem";
import { DeleteConfirmModal } from "../../src/components/DeleteConfirmModal";
import { CameraCapture } from "../../src/components/CameraCapture";
import { GlassCard } from "../../src/components/GlassCard";
import { LoadingSpinner } from "../../src/components/LoadingSpinner";
import { useFiles } from "../../src/hooks/useFiles";
import { colors } from "../../src/styles/theme";
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
        setPickedFiles(picked);
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
        {/* Upload Section */}
        <GlassCard>
          <Text style={styles.sectionTitle}>Upload Files</Text>
          <Text style={styles.sectionSubtitle}>
            Upload CSV transactions or receipt images (PNG, JPG)
          </Text>

          <View style={styles.buttonRow}>
            <Button
              mode="outlined"
              icon="file-plus"
              onPress={handlePickFiles}
              style={[styles.pickButton, styles.buttonFlex]}
            >
              Select Files
            </Button>

            {Platform.OS !== "web" && (
              <Button
                mode="outlined"
                icon="camera"
                onPress={handleOpenCamera}
                style={[styles.pickButton, styles.buttonFlex]}
              >
                Camera
              </Button>
            )}
          </View>

          {pickedFiles.length > 0 && (
            <View style={styles.pickedList}>
              {pickedFiles.map((f, i) => (
                <Text key={i} style={styles.pickedFileName}>
                  {f.name}
                </Text>
              ))}
              <Button
                mode="contained"
                icon="upload"
                onPress={handleIngest}
                loading={isUploading}
                disabled={isUploading}
                style={styles.ingestButton}
                labelStyle={styles.ingestButtonLabel}
              >
                {isUploading ? "Processing..." : "Ingest Selected Files"}
              </Button>
            </View>
          )}
        </GlassCard>

        {/* Ingestion Status */}
        {isIngesting && (
          <GlassCard>
            <View style={styles.ingestingRow}>
              <Text style={styles.ingestingText}>Ingesting files... This may take a minute.</Text>
            </View>
          </GlassCard>
        )}

        {/* Duplicate Warnings */}
        {duplicates.length > 0 && (
          <GlassCard>
            <Text style={styles.duplicateTitle}>
              {duplicates.length} duplicate transaction(s) detected
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

        <Divider style={styles.divider} />

        {/* File List */}
        <Text style={styles.sectionTitle}>Your Uploaded Files</Text>
        {files.length === 0 ? (
          <Text style={styles.emptyText}>No files uploaded yet.</Text>
        ) : (
          files.map((file) => (
            <FileListItem
              key={file.id}
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
    padding: 16,
    paddingBottom: 40,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: colors.text,
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 10,
  },
  buttonFlex: {
    flex: 1,
  },
  pickButton: {
    borderRadius: 10,
    borderColor: colors.border,
  },
  pickedList: {
    marginTop: 12,
    gap: 6,
  },
  pickedFileName: {
    fontSize: 13,
    color: colors.text,
    paddingLeft: 4,
  },
  ingestButton: {
    borderRadius: 10,
    backgroundColor: colors.primary,
    marginTop: 8,
  },
  ingestButtonLabel: {
    fontWeight: "600",
    paddingVertical: 2,
  },
  divider: {
    marginVertical: 20,
    backgroundColor: colors.surfaceBorder,
  },
  emptyText: {
    color: colors.textSecondary,
    fontSize: 14,
    marginTop: 8,
  },
  ingestingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  ingestingText: {
    fontSize: 14,
    color: colors.primary,
    fontWeight: "600",
  },
  duplicateTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#b45309",
    marginBottom: 4,
  },
  duplicateSubtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: 8,
  },
  duplicateItem: {
    fontSize: 13,
    color: colors.text,
    paddingLeft: 8,
    paddingVertical: 2,
  },
  dismissButton: {
    alignSelf: "flex-start",
    marginTop: 4,
  },
});
