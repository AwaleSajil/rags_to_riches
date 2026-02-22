import { Platform } from "react-native";
import { apiJson, apiFetch } from "./api";
import { createLogger } from "../lib/logger";
import type { FileItem } from "../lib/types";

const log = createLogger("FileService");

export async function listFiles(): Promise<FileItem[]> {
  log.info("Fetching file list...");
  const res = await apiJson<{ files: FileItem[] }>("/files");
  log.info("Files loaded", { count: res.files.length, files: res.files.map((f) => f.filename) });
  return res.files;
}

export async function uploadFiles(
  files: { uri: string; name: string; type: string }[]
): Promise<{ message: string; file_ids: string[] }> {
  log.info("Upload starting", {
    fileCount: files.length,
    files: files.map((f) => ({ name: f.name, type: f.type })),
    platform: Platform.OS,
  });

  const formData = new FormData();
  for (const file of files) {
    if (Platform.OS === "web") {
      log.debug("Converting blob for web upload", { name: file.name });
      const response = await fetch(file.uri);
      const blob = await response.blob();
      log.debug("Blob created", { name: file.name, size: blob.size });
      formData.append("files", new File([blob], file.name, { type: file.type }));
    } else {
      log.debug("Appending native file URI", { name: file.name, uri: file.uri.substring(0, 50) });
      formData.append("files", {
        uri: file.uri,
        name: file.name,
        type: file.type,
      } as any);
    }
  }

  log.debug("Sending upload request...");
  const res = await apiFetch("/files/upload", {
    method: "POST",
    body: formData,
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

export interface IngestionStatus {
  status: "idle" | "processing" | "complete" | "failed";
  error?: string | null;
  duplicates?: { date: string; merchant: string; amount: number }[];
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
