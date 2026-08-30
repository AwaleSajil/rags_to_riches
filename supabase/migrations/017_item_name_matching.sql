-- Matching a shelf tag ("CHEERIOS TOASTED WHOLE GRAIN 12OZ") against what the
-- user actually bought ("Cheerios 12 oz") needs a shared join key on both
-- sides. normalized_name is produced by price_service.normalize_item_name, so
-- PriceObservation and TransactionDetail agree on spelling, case and packaging
-- noise.
--
-- Exact match is tried first; pg_trgm covers the near misses that OCR and
-- receipt abbreviations produce constantly.
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions;

ALTER TABLE public."TransactionDetail"
    ADD COLUMN IF NOT EXISTS normalized_name text;

CREATE INDEX IF NOT EXISTS idx_txdetail_normalized
    ON public."TransactionDetail"(user_id, normalized_name);

-- Fuzzy fallback for the exact-match miss.
CREATE INDEX IF NOT EXISTS idx_txdetail_normalized_trgm
    ON public."TransactionDetail" USING gin (normalized_name extensions.gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_priceobs_normalized_trgm
    ON public."PriceObservation" USING gin (normalized_name extensions.gin_trgm_ops);

-- Existing rows have no normalized_name until backfilled; comparisons simply
-- find nothing for them rather than erroring.
-- Run: .venv/bin/python scripts/backfill_normalized_names.py
