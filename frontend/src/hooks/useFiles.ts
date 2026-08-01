import { useState, useEffect, useCallback, useRef } from "react";
import * as fileService from "../services/fileService";
import { createLogger } from "../lib/logger";
import type { FileItem } from "../lib/types";

const log = createLogger("useFiles");

export interface DuplicateInfo {
  date: string;
  merchant: string;
  amount: number;
}

export interface IngestionProgress {
  stage: string;
  total: number;
  done: number;
  detail?: string;
}

const POLL_INTERVAL_MS = 2000;
// "idle" is expected for a poll or two while the worker spins up; beyond that
// the server has forgotten the job (most likely it restarted).
const MAX_IDLE_POLLS = 5;
// Absolute ceiling regardless of what the server reports, so a worker wedged in
// "processing" can't spin the UI indefinitely either.
const MAX_INGESTION_MS = 15 * 60 * 1000;

export function useFiles() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [visibilityFileId, setVisibilityFileId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [duplicates, setDuplicates] = useState<DuplicateInfo[]>([]);
  const [receiptReviewFileIds, setReceiptReviewFileIds] = useState<string[]>([]);
  const [ingestionProgress, setIngestionProgress] = useState<IngestionProgress | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // loadFiles is defined below and depends on nothing, but the poller needs to
  // call it; a ref avoids reordering the hook or a circular useCallback dep.
  const loadFilesRef = useRef<(() => Promise<void>) | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startIngestionPolling = useCallback(() => {
    stopPolling();
    setIsIngesting(true);
    setIngestionProgress(null);
    log.info("Starting ingestion status polling");

    // The backend tracks ingestion in an in-memory dict, so a restart mid-run
    // loses the job and the endpoint reports "idle" forever. Without these two
    // guards the interval never clears and the spinner never stops.
    let consecutiveIdle = 0;
    const startedAt = Date.now();

    const giveUp = (message: string) => {
      log.warn("Ingestion polling gave up", { message });
      stopPolling();
      setIsIngesting(false);
      setIngestionProgress(null);
      setError(message);
      loadFilesRef.current?.();
    };

    pollRef.current = setInterval(async () => {
      try {
        const status = await fileService.getIngestionStatus();
        log.debug("Ingestion poll result", status);

        if (Date.now() - startedAt > MAX_INGESTION_MS) {
          giveUp(
            "Ingestion is taking longer than expected. Pull to refresh to check whether it finished."
          );
          return;
        }

        // "idle" right after upload is a normal race with the worker starting;
        // sustained "idle" means the backend no longer knows about the job.
        if (status.status === "idle") {
          consecutiveIdle++;
          if (consecutiveIdle >= MAX_IDLE_POLLS) {
            giveUp(
              "Lost track of the upload — the server may have restarted. Pull to refresh to check whether it finished."
            );
          }
          return;
        }
        consecutiveIdle = 0;

        // Update progress if available
        if (status.stage) {
          setIngestionProgress({
            stage: status.stage,
            total: status.total || 0,
            done: status.done || 0,
            detail: status.detail,
          });
        }

        if (status.status === "complete") {
          stopPolling();
          setIsIngesting(false);
          setIngestionProgress(null);
          if (status.duplicates && status.duplicates.length > 0) {
            log.info("Duplicates detected", { count: status.duplicates.length });
            setDuplicates(status.duplicates);
          }
          // Reload files now that ingestion is done
          const data = await fileService.listFiles();
          setFiles(data);
        } else if (status.status === "review") {
          stopPolling();
          setIsIngesting(false);
          setIngestionProgress(null);
          setReceiptReviewFileIds(status.receipt_review_file_ids || []);
          const data = await fileService.listFiles();
          setFiles(data);
        } else if (status.status === "failed") {
          stopPolling();
          setIsIngesting(false);
          setIngestionProgress(null);
          setError(status.error || "Ingestion failed");
        }
      } catch (e: any) {
        log.error("Ingestion poll error", e);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const loadFiles = useCallback(async () => {
    log.info("Loading files...");
    setIsLoading(true);
    setError(null);
    try {
      const data = await fileService.listFiles();
      log.info("Files state updated", { count: data.length });
      setFiles(data);
    } catch (e: any) {
      log.error("loadFiles failed", e);
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  loadFilesRef.current = loadFiles;

  useEffect(() => {
    log.debug("useFiles mounted - loading files");
    loadFiles();
  }, [loadFiles]);

  const uploadFiles = async (
    pickedFiles: { uri: string; name: string; type: string }[]
  ) => {
    log.info("uploadFiles called", {
      count: pickedFiles.length,
      names: pickedFiles.map((f) => f.name),
    });
    setIsUploading(true);
    setError(null);
    setDuplicates([]);
    setReceiptReviewFileIds([]);
    try {
      await fileService.uploadFiles(pickedFiles);
      log.info("Upload complete - reloading file list and starting ingestion poll");
      await loadFiles();
      startIngestionPolling();
      return true;
    } catch (e: any) {
      log.error("uploadFiles failed", e);
      setError(e.message);
      return false;
    } finally {
      setIsUploading(false);
    }
  };

  const deleteFile = async (fileId: string, fileType: string) => {
    log.info("deleteFile called", { fileId, fileType });
    setIsDeleting(true);
    setError(null);
    try {
      await fileService.deleteFile(fileId, fileType);
      log.info("Delete complete - reloading file list");
      await loadFiles();
      return true;
    } catch (e: any) {
      log.error("deleteFile failed", e);
      setError(e.message);
      return false;
    } finally {
      setIsDeleting(false);
    }
  };

  const toggleFileVisibility = async (file: FileItem) => {
    setVisibilityFileId(file.id);
    try {
      const result = await fileService.setFileVisibility(file.id, file.type, !file.is_hidden);
      setFiles((current) => current.map((item) =>
        item.id === file.id ? { ...item, is_hidden: result.is_hidden } : item
      ));
      return true;
    } catch (e: any) {
      log.error("toggleFileVisibility failed", e);
      setError(e.message);
      return false;
    } finally {
      setVisibilityFileId(null);
    }
  };

  const clearDuplicates = useCallback(() => {
    setDuplicates([]);
  }, []);

  return {
    files,
    isLoading,
    isUploading,
    isIngesting,
    isDeleting,
    visibilityFileId,
    error,
    duplicates,
    receiptReviewFileIds,
    ingestionProgress,
    uploadFiles,
    deleteFile,
    toggleFileVisibility,
    loadFiles,
    clearDuplicates,
  };
}
