import { useState, useEffect, useCallback } from "react";
import * as configService from "../services/configService";
import { createLogger } from "../lib/logger";
import type { AccountConfig } from "../lib/types";

const log = createLogger("useConfig");

export function useConfig() {
  const [config, setConfig] = useState<AccountConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConfig = useCallback(async () => {
    log.info("Loading config...");
    setIsLoading(true);
    setError(null);
    try {
      const data = await configService.getConfig();
      log.info("Config state updated", {
        hasConfig: !!data,
        provider: data?.llm_provider,
      });
      setConfig(data);
    } catch (e: any) {
      log.error("loadConfig failed", e);
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    log.debug("useConfig mounted - loading config");
    loadConfig();
  }, [loadConfig]);

  const saveConfig = async (data: {
    llm_provider: string;
    api_key: string;
    decode_model: string;
    embedding_model: string;
    deep_enrichment?: boolean;
  }) => {
    log.info("saveConfig called", {
      provider: data.llm_provider,
      decodeModel: data.decode_model,
    });
    setIsSaving(true);
    setError(null);
    try {
      const result = await configService.updateConfig(data);
      log.info("Config saved - state updated");
      setConfig(result as AccountConfig);
      return true;
    } catch (e: any) {
      log.error("saveConfig failed", e);
      setError(e.message);
      return false;
    } finally {
      setIsSaving(false);
    }
  };

  return { config, isLoading, isSaving, error, saveConfig, loadConfig };
}
