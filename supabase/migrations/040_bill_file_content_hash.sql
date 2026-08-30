-- Fingerprint of a stored photo's bytes, so the same image file cannot be
-- uploaded twice.
--
-- Migration 039 did this for CSVs and deliberately left photos out, on the
-- grounds that two photos of one receipt are different bytes and that duplicate
-- is caught at verification by receipt_content_hash. Both halves of that are
-- true, and together they miss the case in between: the SAME image file sent
-- twice — picked from the gallery again, or re-sent because the first attempt
-- looked like it had failed.
--
-- That costs more than a duplicate CSV does. Every photo upload pays for a
-- vision extraction, and the duplicate is only noticed later, at verification,
-- if the user ever gets that far. Until then it sits in the Files tab as a
-- second receipt for a purchase that happened once.
--
-- Byte-identical only. A re-photographed receipt still passes through here and
-- is still caught by receipt_content_hash, which compares what was READ rather
-- than the pixels.
ALTER TABLE "BillFile"
  ADD COLUMN IF NOT EXISTS content_hash text;

-- Partial, so rows predating the column — all NULL — do not collide with each
-- other. Per user: two people photographing the same shop's receipt template
-- are not duplicates of one another.
CREATE UNIQUE INDEX IF NOT EXISTS billfile_user_content_hash
  ON "BillFile" (user_id, content_hash)
  WHERE content_hash IS NOT NULL;
