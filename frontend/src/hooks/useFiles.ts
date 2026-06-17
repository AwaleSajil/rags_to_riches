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

export function useFiles() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [duplicates, setDuplicates] = useState<DuplicateInfo[]>([]);
  const [ingestionProgress, setIngestionProgress] = useState<IngestionProgress | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

    pollRef.current = setInterval(async () => {
      try {
        const status = await fileService.getIngestionStatus();
        log.debug("Ingestion poll result", status);

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
        } else if (status.status === "failed") {
          stopPolling();
          setIsIngesting(false);
          setIngestionProgress(null);
          setError(status.error || "Ingestion failed");
        }
      } catch (e: any) {
        log.error("Ingestion poll error", e);
      }
    }, 2000);  // Poll faster for progress updates
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

  const clearDuplicates = useCallback(() => {
    setDuplicates([]);
  }, []);

  return {
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
    loadFiles,
    clearDuplicates,
  };
}
