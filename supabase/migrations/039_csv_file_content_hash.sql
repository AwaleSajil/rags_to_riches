-- Fingerprint of an uploaded CSV's bytes, so the same statement cannot be
-- imported twice.
--
-- Row-level dedup already exists, but it cannot catch this. Each row's
-- content_hash is built from the CSV's own file id, which is new on every
-- upload — deliberately, so two genuinely different exports covering the same
-- period stay as separate durable records. The cost of that choice is that
-- re-uploading one identical file produces entirely new hashes, matches
-- nothing, and silently writes every transaction a second time.
--
-- Catching it here, at the file, is exact: two identical files need no fuzzy
-- matching and no judgement. It also saves what row-level dedup never could —
-- a second pass of LLM merchant enrichment and a second set of embeddings.
--
-- Only CSVs. A re-photographed receipt is a different picture of the same
-- paper, so its bytes differ; that case is caught at verification time by
-- receipt_content_hash instead.
ALTER TABLE "CSVFile"
  ADD COLUMN IF NOT EXISTS content_hash text;

-- Partial, so the rows that predate this column — all with a NULL hash — do
-- not collide with each other. Per user: two people uploading the same public
-- statement template are not duplicates of one another.
CREATE UNIQUE INDEX IF NOT EXISTS csvfile_user_content_hash
  ON "CSVFile" (user_id, content_hash)
  WHERE content_hash IS NOT NULL;
