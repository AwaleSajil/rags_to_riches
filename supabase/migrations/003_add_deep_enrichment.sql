-- Add deep_enrichment toggle to AccountConfig
-- When true, ingestion uses web search for richer merchant/item descriptions (slower)
-- When false (default), ingestion uses LLM-only enrichment (faster)
ALTER TABLE "AccountConfig"
ADD COLUMN IF NOT EXISTS deep_enrichment BOOLEAN NOT NULL DEFAULT false;
