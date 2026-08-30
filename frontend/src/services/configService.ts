import { apiJson } from "./api";
import { createLogger } from "../lib/logger";
import type { AccountConfig } from "../lib/types";

const log = createLogger("ConfigService");

export async function getConfig(): Promise<AccountConfig | null> {
  log.info("Fetching config...");
  const result = await apiJson<AccountConfig | null>("/config");
  log.info("Config loaded", {
    hasConfig: !!result,
    provider: (result as AccountConfig)?.llm_provider,
    decodeModel: (result as AccountConfig)?.decode_model,
  });
  return result;
}

export async function updateConfig(data: {
  llm_provider: string;
  /** Omit to keep the key already stored — the server never sends it back, so
   *  there is nothing for the client to echo when only a model changed. */
  api_key?: string;
  decode_model: string;
  embedding_model: string;
  deep_enrichment?: boolean;
}): Promise<AccountConfig> {
  log.info("Saving config", {
    provider: data.llm_provider,
    decodeModel: data.decode_model,
    embeddingModel: data.embedding_model,
    hasApiKey: !!data.api_key,
  });
  const result = await apiJson<AccountConfig>("/config", {
    method: "PUT",
    body: JSON.stringify(data),
  });
  log.info("Config saved successfully");
  return result;
}
