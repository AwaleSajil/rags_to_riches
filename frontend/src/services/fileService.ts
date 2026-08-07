import { Platform } from "react-native";
import * as FileSystem from "expo-file-system/legacy";
import { apiJson, apiFetch, getAccessToken, API_URL } from "./api";
import { createLogger } from "../lib/logger";
import { compressImage } from "../lib/compressImage";
import type { CaptureKind, FileItem } from "../lib/types";

const log = createLogger("FileService");

export async function listFiles(): Promise<FileItem[]> {
  log.info("Fetching file list...");
  const res = await apiJson<{ files: FileItem[] }>("/files");
  log.info("Files loaded", { count: res.files.length, files: res.files.map((f) => f.filename) });
  return res.files;
}

export async function uploadFiles(
  input: { uri: string; name: string; type: string }[]
): Promise<{ message: string; file_ids: string[] }> {
  // Shrunk here rather than at each camera and picker, so no attach path can
  // skip it. CSVs pass straight through.
  const files = await Promise.all(input.map(compressImage));

  log.info("Upload starting", {
    fileCount: files.length,
    files: files.map((f) => ({ name: f.name, type: f.type })),
    platform: Platform.OS,
  });

  if (Platform.OS === "web") {
    return uploadFilesWeb(files);
  }
  return uploadFilesNative(files);
}

// Web: multipart via fetch/FormData (works reliably in browsers).
async function uploadFilesWeb(
  files: { uri: string; name: string; type: string }[]
): Promise<{ message: string; file_ids: string[] }> {
  const formData = new FormData();
  for (const file of files) {
    const response = await fetch(file.uri);
    const blob = await response.blob();
    formData.append("files", new File([blob], file.name, { type: file.type }));
  }
  const res = await apiFetch("/files/upload", {
    method: "POST",
    body: formData,
    timeout: 60000,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    log.error("Upload failed", { status: res.status, detail: error.detail });
    throw new Error(error.detail || `Upload failed: HTTP ${res.status}`);
  }
  const result = await res.json();
  log.info("Upload successful", { message: result.message, fileIds: result.file_ids });
  return result;
}

// Native (iOS/Android): stream each file with expo-file-system's uploadAsync.
// React Native's fetch/XMLHttpRequest fails multipart file bodies with
// "Network request failed", so we bypass it entirely.
async function uploadFilesNative(
  files: { uri: string; name: string; type: string }[]
): Promise<{ message: string; file_ids: string[] }> {
  const fileIds: string[] = [];
  let message = "";
  for (const file of files) {
    const result = await uploadOneNative(file);
    if (result.file_ids) fileIds.push(...result.file_ids);
    message = result.message || message;
  }
  log.info("Upload successful (native)", { fileIds });
  return { message: message || "Upload complete", file_ids: fileIds };
}

async function uploadOneNative(
  file: { uri: string; name: string; type: string },
  isRetry = false
): Promise<{ message: string; file_ids: string[] }> {
  const token = await getAccessToken(isRetry);

  // Copy to a cache path whose basename keeps the extension — the backend
  // classifies CSV vs receipt by filename, and picker/camera URIs vary.
  const safeName = (file.name || "upload").replace(/[^\w.\-]/g, "_");
  const dest = `${FileSystem.cacheDirectory}upload_${Date.now()}_${safeName}`;
  await FileSystem.copyAsync({ from: file.uri, to: dest });

  try {
    log.debug("uploadAsync", { name: safeName, type: file.type });
    const res = await FileSystem.uploadAsync(`${API_URL}/files/upload`, dest, {
      httpMethod: "POST",
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      fieldName: "files",
      mimeType: file.type,
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });

    // Refresh the token once and retry on an expired session.
    if (res.status === 401 && !isRetry) {
      log.warn("Upload 401 — refreshing token and retrying");
      return uploadOneNative(file, true);
    }
    if (res.status < 200 || res.status >= 300) {
      let detail = `Upload failed: HTTP ${res.status}`;
      try {
        detail = JSON.parse(res.body).detail || detail;
      } catch {
        /* non-JSON body */
      }
      log.error("Upload failed (native)", { status: res.status, body: res.body?.slice(0, 200) });
      throw new Error(detail);
    }
    return JSON.parse(res.body);
  } finally {
    FileSystem.deleteAsync(dest, { idempotent: true }).catch(() => {});
  }
}

// Defined in lib/types (the leaf module this file already imports from) and
// re-exported here so existing importers keep working.
export type { CaptureKind } from "../lib/types";

export interface ReviewItem {
  file_id: string;
  kind: CaptureKind;
}

export interface IngestionStatus {
  status: "idle" | "processing" | "review" | "complete" | "failed";
  error?: string | null;
  duplicates?: { date: string; merchant: string; amount: number }[];
  stage?: string;
  total?: number;
  done?: number;
  detail?: string;
  /** Photos awaiting review, each tagged with what it was recognised as. */
  review_items?: ReviewItem[];
  /** @deprecated superseded by review_items; kept so a client mid-deploy still works. */
  receipt_review_file_ids?: string[];
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  return apiJson<IngestionStatus>("/files/ingestion-status");
}

export async function deleteFile(
  fileId: string,
  fileType: string
): Promise<{ message: string }> {
  log.info("Deleting file", { fileId, fileType });
  const result = await apiJson<{ message: string }>(`/files/${fileId}?type=${fileType}`, {
    method: "DELETE",
  });
  log.info("File deleted", { message: result.message });
  return result;
}

export async function setFileVisibility(
  fileId: string,
  fileType: "csv" | "bill",
  hidden: boolean
): Promise<{ message: string; is_hidden: boolean }> {
  return apiJson(`/files/${fileId}/visibility?type=${fileType}&hidden=${hidden}`, {
    method: "PATCH",
  });
}
